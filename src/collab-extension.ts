// Optional pi extension: hand RWKV's current work to the Qwen teacher and show exactly what Qwen
// changes. It is OFF unless the user loads it explicitly for one session:
//
//   rwkv-pi --collab          (or: pi --extension dist/collab-extension.js)
//
// It is not listed in package.json "pi.extensions", reads no enabling environment variable and
// persists no state, so it never becomes active by default or stays active across sessions.
// Inside a session: /collab on | off | status, and /collab [--check "COMMAND"] [note] to run.
//
// Qwen works on a sandboxed copy. Its changes are applied to the real directory only after the
// user confirms, and only if the directory did not change in the meantime.
import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { dirname, join, resolve, sep } from "node:path";
import type { ExtensionAPI, ExtensionCommandContext } from "@earendil-works/pi-coding-agent";
import { attemptEvidence, buildRecord, continuationTask, finalAnswer, readTree, renderMarkdown, toolActions, type Tree } from "./collaboration.js";
import { dataRoot, participant, writeRecord } from "./collaboration-cli.js";
import { runTask } from "./harness.js";
import { createTeacherProvider } from "./teacher.js";

const REPORT = "rwkv-collaboration";

function parseArgs(args: string): { check?: string; note: string } {
  const match = /--check\s+(?:"([^"]*)"|'([^']*)'|(\S+))/.exec(args);
  const check = match ? match[1] ?? match[2] ?? match[3] : undefined;
  return { check, note: (match ? args.replace(match[0], "") : args).trim() };
}

function messageText(message: Record<string, unknown>): string {
  if (typeof message.content === "string") return message.content;
  if (!Array.isArray(message.content)) return "";
  return (message.content as Record<string, unknown>[]).filter((p) => p?.type === "text").map((p) => String(p.text)).join("");
}

export default function collaborationExtension(pi: ExtensionAPI): void {
  let enabled = true; // loading the extension is the explicit opt-in; /collab off turns it off
  let baseline: Tree | undefined;
  let baselineEntry: string | null = null;
  let busy = false;

  const snapshot = (ctx: { cwd: string; sessionManager: { getLeafId(): string | null } }) => {
    baseline = readTree(ctx.cwd);
    baselineEntry = ctx.sessionManager.getLeafId();
  };
  const status = (ctx: { ui: { setStatus(key: string, text: string | undefined): void } }) =>
    ctx.ui.setStatus("collab", enabled ? "collab: on (/collab to hand off to Qwen)" : undefined);

  pi.on("session_start", (_event, ctx) => {
    if (enabled) snapshot(ctx);
    status(ctx);
  });
  // Reports are for the user. Keep them out of RWKV's model context (training parity).
  pi.on("context", (event) => ({ messages: event.messages.filter((m) => (m as { customType?: string }).customType !== REPORT) }));

  pi.registerCommand("collab", {
    description: "RWKV -> Qwen collaboration: /collab [--check \"cmd\"] [note] | on | off | status",
    getArgumentCompletions: (prefix) => ["on", "off", "status", "--check "].filter((x) => x.startsWith(prefix)).map((value) => ({ value, label: value })),
    handler: async (args, ctx) => {
      const word = args.trim();
      if (word === "off") { enabled = false; baseline = undefined; status(ctx); ctx.ui.notify("Collaboration off for this session.", "info"); return; }
      if (word === "on") { enabled = true; snapshot(ctx); status(ctx); ctx.ui.notify("Collaboration on. Baseline taken from the current directory.", "info"); return; }
      if (word === "status") { ctx.ui.notify(`Collaboration is ${enabled ? "on" : "off"}${baseline ? `; baseline has ${Object.keys(baseline.files).length} files` : ""}.`, "info"); return; }
      if (!enabled || !baseline) { ctx.ui.notify("Collaboration is off. Run /collab on first.", "warning"); return; }
      if (busy) { ctx.ui.notify("A collaboration is already running.", "warning"); return; }
      busy = true;
      try { await collaborate(pi, ctx, baseline, baselineEntry, parseArgs(args)); }
      catch (error) { ctx.ui.notify(`Collaboration failed: ${error instanceof Error ? error.message : String(error)}`, "error"); }
      finally { busy = false; }
    },
  });
}

async function collaborate(pi: ExtensionAPI, ctx: ExtensionCommandContext, baseline: Tree, baselineEntry: string | null,
  options: { check?: string; note: string }): Promise<void> {
  if (!ctx.isIdle()) throw new Error("wait until RWKV finishes its turn");
  // RWKV's conversation since the baseline.
  const branch = ctx.sessionManager.getBranch();
  const start = baselineEntry ? branch.findIndex((entry) => entry.id === baselineEntry) + 1 : 0;
  const messages = branch.slice(start).filter((e) => e.type === "message").map((e) => (e as { message: Record<string, unknown> }).message);
  const request = messages.filter((m) => m.role === "user").map(messageText).filter(Boolean).join("\n\n");
  if (!request && !options.note) throw new Error("no user request since the baseline; add a note: /collab <what Qwen should finish>");
  const rwkvTree = readTree(ctx.cwd);
  const rwkv = {
    model: ctx.model?.id ?? "rwkv", status: "handed-off",
    actions: toolActions(messages), finalAnswer: finalAnswer(messages), requests: messages.filter((m) => m.role === "assistant").length,
  };
  const checks = options.check
    ? [{ type: "command" as const, argv: ["bash", "-lc", options.check], timeoutMs: 600_000 }]
    // runTask requires a check; this one only proves the run finished. The record says so.
    : [{ type: "command" as const, argv: ["true"], timeoutMs: 10_000 }];
  const id = `interactive-${new Date().toISOString().replace(/[:.]/g, "-")}`;
  const task = continuationTask({
    id, family: "interactive-collaboration", split: "train", source: `Interactive rwkv-pi session ${ctx.sessionManager.getSessionId()} in ${ctx.cwd}.`,
    prompt: [request, options.note && `补充说明：${options.note}`].filter(Boolean).join("\n\n"),
    tools: ["read", "write", "edit", "bash"], checks,
  }, rwkvTree.files, attemptEvidence(rwkv, messages));

  const out = join(dataRoot(), "runs", "collaboration", "interactive", id);
  await mkdir(out, { recursive: true });
  ctx.ui.setStatus("collab", "collab: Qwen is working on a sandboxed copy…");
  ctx.ui.notify(`Qwen is continuing RWKV's work in a sandboxed copy (${Object.keys(rwkvTree.files).length} files). Output: ${out}`, "info");
  const qwen = await runTask(task, { provider: createTeacherProvider(), role: "teacher", outputDir: join(out, "qwen"),
    maxRequests: 64, timeoutMs: 3_600_000, sandbox: "bubblewrap", thinkingLevel: "high" });
  const qwenTree = readTree(join(dirname(qwen.path), "workspace"));
  const verified = Boolean(options.check) && qwen.record.status === "passed" && qwen.record.checks.every((c) => c.passed);
  const record = buildRecord({
    mode: "interactive", task: { id, prompt: task.prompt },
    original: baseline.files, rwkvFiles: rwkvTree.files, qwenFiles: qwenTree.files,
    rwkv, qwen: participant(qwen.record, qwen.path, await readFile(qwen.path)),
    acceptance: {
      verified, checks: options.check ? qwen.record.checks.filter((c) => !c.detail.startsWith("file_equals:.repair-context")) : [],
      note: options.check ? `acceptance command: ${options.check}` : "no --check command was given, so the result is unverified and not eligible for training",
    },
    skipped: [...baseline.skipped, ...rwkvTree.skipped],
  });
  await writeFile(join(out, "rwkv-session.json"), JSON.stringify({ sessionId: ctx.sessionManager.getSessionId(), cwd: ctx.cwd, messages }, null, 2) + "\n");
  const target = await writeRecord(record, out);
  const report = renderMarkdown(record);
  pi.sendMessage({ customType: REPORT, content: report, display: true, details: { record: join(target, "record.json") } });

  const changed = record.files.filter((f) => f.qwen.existence !== "same");
  if (!changed.length) { ctx.ui.notify("Qwen made no file changes.", "info"); return; }
  const apply = await ctx.ui.confirm("Apply Qwen's changes?", `${changed.length} file(s): ${changed.map((f) => f.path).join(", ").slice(0, 400)}`);
  if (!apply) { ctx.ui.notify(`Not applied. Qwen's workspace stays at ${dirname(qwen.path)}/workspace`, "info"); return; }
  // Refuse if the real directory changed since Qwen started from it.
  const now = readTree(ctx.cwd);
  for (const file of changed) {
    if (now.files[file.path] !== rwkvTree.files[file.path]) throw new Error(`${file.path} changed while Qwen was working; nothing applied`);
  }
  const root = resolve(ctx.cwd);
  for (const file of changed) {
    const target = resolve(root, file.path);
    if (!target.startsWith(root + sep)) throw new Error(`refusing path outside the directory: ${file.path}`);
    const content = qwenTree.files[file.path];
    if (content === undefined) await rm(target, { force: true });
    else { await mkdir(dirname(target), { recursive: true }); await writeFile(target, content); }
  }
  await writeFile(join(target, "applied.json"), JSON.stringify({ appliedAt: new Date().toISOString(), files: changed.map((f) => f.path) }, null, 2) + "\n");
  ctx.ui.notify(`Applied Qwen's changes to ${changed.length} file(s). Record: ${target}`, "info");
}
