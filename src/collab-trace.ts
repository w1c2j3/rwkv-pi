import { randomUUID } from "node:crypto";
import { appendFileSync, existsSync, lstatSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, relative, resolve } from "node:path";
import { buildRecord, finalAnswer, renderMarkdown, toolActions, type Tree } from "./collaboration.js";
import type { StallReason } from "./stall-detector.js";
import { dataRoot } from "./config.js";

const ignored = new Set([".git", ".upstream", "node_modules", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".pi"]);
/** Bounded, best-effort evidence snapshot. Never copies a workspace or changes its contents. */
export function snapshot(root: string, exclude: readonly string[] = []): Tree {
  const tree: Tree = { files: {}, skipped: [] };
  const excluded = new Set(exclude.map(p => resolve(p)));
  let entries = 0, bytes = 0;
  const skip = (path: string, reason: string) => { if (tree.skipped.length < 1000) tree.skipped.push({ path, reason }); };
  const walk = (directory: string): void => {
    let names: string[];
    try { names = readdirSync(directory).sort(); } catch { skip(relative(root, directory), "unreadable directory"); return; }
    for (const name of names) {
      if (++entries > 20_000 || bytes >= 16 * 1024 * 1024) { skip(relative(root, directory), "snapshot budget"); break; }
      const path = join(directory, name), rel = relative(root, path);
      if (excluded.has(resolve(path))) { skip(rel, "own trace output"); continue; }
      if (ignored.has(name) || /^\.env(?:\.|$)|\.(?:key|pem)$/.test(name)) { skip(rel, "excluded dependency/cache/credential path"); continue; }
      try {
        const stat = lstatSync(path);
        if (stat.isSymbolicLink()) { skip(rel, "symlink (not followed)"); continue; }
        if (stat.isDirectory()) { walk(path); continue; }
        if (!stat.isFile() || stat.size > 1024 * 1024) { skip(rel, "non-file or larger than 1 MiB"); continue; }
        const raw = readFileSync(path), text = raw.toString("utf8");
        if (raw.includes(0) || !Buffer.from(text).equals(raw)) { skip(rel, "binary"); continue; }
        bytes += raw.length; tree.files[rel] = text;
      } catch { skip(rel, "unreadable or changed during snapshot"); }
    }
  };
  walk(root);
  return tree;
}

export function collaborationDataRoot(env: NodeJS.ProcessEnv = process.env): string {
  const root = dataRoot(env);
  if (!existsSync(root)) throw new Error(`Collaboration data root unavailable: ${root}`);
  return root;
}

export class CollaborationTrace {
  readonly directory: string;
  readonly original: Tree;
  readonly messages: any[] = [];
  lastRecord?: ReturnType<typeof buildRecord>;
  rwkvOnlyChangedFiles: string[] = [];
  handoff?: { reason: StallReason; tree: Tree; index: number; teacher: string };
  constructor(readonly cwd: string, readonly prompt: string, readonly rwkv: string, context: unknown[], root: string, readonly teacherLabel = "Qwen") {
    this.directory = join(root, "runs", "collaboration", "automatic", `${new Date().toISOString().replace(/[:.]/g, "-")}-${randomUUID().slice(0, 8)}`);
    mkdirSync(this.directory, { recursive: true, mode: 0o700 });
    this.original = snapshot(cwd, [this.directory]);
    this.save("original.json", this.original);
    this.save("context.json", context);
    this.event("task_start", { cwd, prompt, rwkv, trainingUse: "unverified; no automatic export" });
  }
  save(name: string, value: unknown): void { writeFileSync(join(this.directory, name), JSON.stringify(value, null, 2) + "\n", { mode: 0o600 }); }
  event(type: string, data: unknown): void { appendFileSync(join(this.directory, "events.jsonl"), JSON.stringify({ at: new Date().toISOString(), type, data }) + "\n", { mode: 0o600 }); }
  message(message: any): void {
    const copy = structuredClone(message);
    this.messages.push(copy); this.event("message", copy);
  }
  takeover(reason: StallReason, teacher: string): void {
    const tree = snapshot(this.cwd, [this.directory]);
    this.handoff = { reason, tree, index: this.messages.length, teacher };
    this.save("handoff.json", { reason, teacher, messageIndex: this.messages.length, tree });
    this.event("takeover", { reason, teacher });
  }
  finish(status: string, teacherError?: StallReason): string | undefined {
    this.event("task_end", { status, teacherError });
    this.save("status.json", { status, teacherError, handoff: this.handoff?.reason, trainingEligible: false });
    if (!this.handoff) {
      const after = snapshot(this.cwd, [this.directory]);
      const excluded = [...this.original.skipped, ...after.skipped].map(s => s.path);
      this.rwkvOnlyChangedFiles = [...new Set([...Object.keys(this.original.files), ...Object.keys(after.files)])]
        .filter(p => !excluded.some(x => !x || p === x || p.startsWith(x + "/")))
        .filter(p => this.original.files[p] !== after.files[p]);
      this.save("final.json", after);
      this.save("rwkv-file-changes.json", { files: this.rwkvOnlyChangedFiles, attribution: "Bounded before/after snapshot; concurrent user edits may affect attribution." });
      return;
    }
    const after = snapshot(this.cwd, [this.directory]), rwkvMessages = this.messages.slice(0, this.handoff.index), qwenMessages = this.messages.slice(this.handoff.index);
    const trees = [this.original, this.handoff.tree, after];
    const excluded = trees.flatMap(t => t.skipped.map(s => s.path));
    const covered = (tree: Tree) => Object.fromEntries(Object.entries(tree.files).filter(([p]) => !excluded.some(x => !x || p === x || p.startsWith(x + "/"))));
    const record = buildRecord({ mode: "interactive", task: { id: this.directory.split("/").at(-1)!, prompt: this.prompt },
      original: covered(this.original), rwkvFiles: covered(this.handoff.tree), qwenFiles: covered(after),
      rwkv: { model: this.rwkv, status: "handed-off", error: this.handoff.reason.detail, actions: toolActions(rwkvMessages), finalAnswer: finalAnswer(rwkvMessages) },
      qwen: { model: this.handoff.teacher, status, error: teacherError?.detail, actions: toolActions(qwenMessages), finalAnswer: finalAnswer(qwenMessages) },
      acceptance: { verified: false, checks: [], note: "Live shared-workspace continuation; no independent acceptance check. Concurrent user edits may affect attribution. Files outside cwd, excluded/binary/oversized files are not covered; see raw tool observations. Never auto-export as SFT targets." },
      skipped: trees.flatMap((t, i) => t.skipped.map(s => ({ ...s, path: `${["original", "rwkv", "qwen"][i]}:${s.path}` }))),
    });
    this.lastRecord = record;
    this.save("final.json", after);
    // Preserve v1's qwen keys for readers of historical records; identify the actual teacher explicitly.
    this.save("record.json", { ...record, teacher: record.qwen, teacherLabel: this.teacherLabel,
      execution: "same-pi-session-same-workspace", trigger: this.handoff.reason, teacherError });
    writeFileSync(join(this.directory, "report.md"), renderMarkdown(record, { teacherLabel: this.teacherLabel }), { mode: 0o600 });
    const changes = record.files.filter(f => f.qwen.existence !== "same");
    return [`${this.teacherLabel} 接管结束：${status}。触发原因：${this.handoff.reason.detail}`,
      `${this.teacherLabel} 修改 ${changes.length} 个已记录文件；RWKV 新增行保留 ${record.summary.rwkvLinesKept} 行、替换/移除 ${record.summary.rwkvLinesDiscarded} 行。`,
      ...changes.map(f => `- ${f.path}: ${f.verdict}（+${f.qwen.added}/-${f.qwen.removed}）`),
      changes.length ? "```diff\n" + changes.map(f => f.qwenDiff).join("\n").slice(0, 12000) + "\n```" : "工具执行及观察见原始记录。",
      `完整报告：${join(this.directory, "report.md")}\n轨迹：${join(this.directory, "events.jsonl")}\n未经过独立验收，尚不是可直接训练的数据。`].join("\n\n");
  }
}
