// RWKV -> Qwen collaboration: RWKV works on a task first, Qwen continues from a copy of RWKV's
// workspace, and the record shows exactly which of RWKV's edits Qwen kept, corrected or reverted.
//
// The record is evidence, not a supervision target by itself: RWKV text stays diagnostic and only
// an independently accepted Qwen trajectory may be exported (see pipeline/CONTRACTS.md).
import { createHash } from "node:crypto";
import { lstatSync, readdirSync, readFileSync } from "node:fs";
import { join, relative, sep } from "node:path";
import type { Task } from "./tasks.js";

export const COLLABORATION_SCHEMA = "rwkv-pi-collaboration-v1";

/** Relative path -> UTF-8 text. Binary, oversized and generated files are listed separately. */
export interface Tree { files: Record<string, string>; skipped: { path: string; reason: string }[] }

const IGNORED_DIRECTORIES = new Set([".git", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache", ".venv", ".repair-context"]);
const MAX_FILE_BYTES = 1024 * 1024;
const MAX_FILES = 5000;

export function readTree(root: string): Tree {
  const tree: Tree = { files: {}, skipped: [] };
  const walk = (directory: string): void => {
    for (const name of readdirSync(directory).sort()) {
      const path = join(directory, name);
      const rel = relative(root, path).split(sep).join("/");
      const stat = lstatSync(path);
      if (stat.isSymbolicLink()) tree.skipped.push({ path: rel, reason: "symlink" });
      else if (stat.isDirectory()) {
        if (IGNORED_DIRECTORIES.has(name)) tree.skipped.push({ path: rel + "/", reason: "generated or vendored directory" });
        else walk(path);
      } else if (!stat.isFile()) tree.skipped.push({ path: rel, reason: "special file" });
      else if (stat.size > MAX_FILE_BYTES) tree.skipped.push({ path: rel, reason: `larger than ${MAX_FILE_BYTES} bytes` });
      else if (Object.keys(tree.files).length >= MAX_FILES) tree.skipped.push({ path: rel, reason: `more than ${MAX_FILES} files` });
      else {
        const raw = readFileSync(path);
        const text = raw.toString("utf8");
        if (raw.includes(0) || !Buffer.from(text, "utf8").equals(raw)) tree.skipped.push({ path: rel, reason: "binary" });
        else tree.files[rel] = text;
      }
    }
  };
  walk(root);
  return tree;
}

// -- line diff -------------------------------------------------------------------------------

export type Edit = { op: "=" | "-" | "+"; line: string };

function lines(text: string): string[] {
  if (!text) return [];
  const parts = text.split("\n");
  if (parts.at(-1) === "") parts.pop();
  return parts;
}

/** Minimal line edit script (LCS). Falls back to delete-all/insert-all for very large inputs. */
export function diffLines(a: string[], b: string[]): Edit[] {
  let start = 0;
  while (start < a.length && start < b.length && a[start] === b[start]) start++;
  let endA = a.length, endB = b.length;
  while (endA > start && endB > start && a[endA - 1] === b[endB - 1]) { endA--; endB--; }
  const head = a.slice(0, start).map((line): Edit => ({ op: "=", line }));
  const tail = a.slice(endA).map((line): Edit => ({ op: "=", line }));
  const x = a.slice(start, endA), y = b.slice(start, endB);
  const middle: Edit[] = [];
  if (x.length * y.length > 4_000_000) {
    middle.push(...x.map((line): Edit => ({ op: "-", line })), ...y.map((line): Edit => ({ op: "+", line })));
  } else {
    const width = y.length + 1;
    const table = new Uint32Array((x.length + 1) * width);
    const at = (i: number, j: number): number => table[i * width + j]!;
    for (let i = x.length - 1; i >= 0; i--) {
      for (let j = y.length - 1; j >= 0; j--) {
        table[i * width + j] = x[i] === y[j] ? at(i + 1, j + 1) + 1 : Math.max(at(i + 1, j), at(i, j + 1));
      }
    }
    let i = 0, j = 0;
    while (i < x.length || j < y.length) {
      if (i < x.length && j < y.length && x[i] === y[j]) { middle.push({ op: "=", line: x[i]! }); i++; j++; }
      else if (i < x.length && (j === y.length || at(i + 1, j) >= at(i, j + 1))) middle.push({ op: "-", line: x[i++]! });
      else middle.push({ op: "+", line: y[j++]! });
    }
  }
  return [...head, ...middle, ...tail];
}

/** Unified diff text with `context` lines around each hunk. */
export function unifiedDiff(path: string, before: string | undefined, after: string | undefined, context = 3): string {
  if (before === after) return "";
  const edits = diffLines(lines(before ?? ""), lines(after ?? ""));
  const output = [`--- ${before === undefined ? "/dev/null" : "a/" + path}`, `+++ ${after === undefined ? "/dev/null" : "b/" + path}`];
  const changed = edits.map((edit) => edit.op !== "=");
  let index = 0;
  while (index < edits.length) {
    if (!changed[index]) { index++; continue; }
    let start = Math.max(0, index - context), end = index;
    // Extend the hunk while the next change is within 2*context unchanged lines.
    for (;;) {
      while (end < edits.length && changed[end]) end++;
      let next = end;
      while (next < edits.length && !changed[next] && next - end < 2 * context) next++;
      if (next < edits.length && changed[next]) end = next;
      else break;
    }
    end = Math.min(edits.length, end + context);
    let oldLine = 1, newLine = 1;
    for (const edit of edits.slice(0, start)) { if (edit.op !== "+") oldLine++; if (edit.op !== "-") newLine++; }
    const hunk = edits.slice(start, end);
    const oldCount = hunk.filter((e) => e.op !== "+").length, newCount = hunk.filter((e) => e.op !== "-").length;
    output.push(`@@ -${oldCount ? oldLine : oldLine - 1},${oldCount} +${newCount ? newLine : newLine - 1},${newCount} @@`);
    for (const edit of hunk) output.push(`${edit.op === "=" ? " " : edit.op}${edit.line}`);
    index = end;
  }
  if ((before ?? "") !== "" && !(before ?? "").endsWith("\n")) output.push("\\ No newline at end of original");
  return output.join("\n") + "\n";
}

// -- three-way attribution ---------------------------------------------------------------------

export type FileVerdict =
  | "untouched"          // nobody changed it
  | "rwkv-kept"          // RWKV changed it; Qwen left RWKV's version as is
  | "qwen-corrected"     // RWKV changed it; Qwen changed it again
  | "qwen-reverted"      // RWKV changed it; Qwen restored the original
  | "qwen-only";         // RWKV left it; Qwen changed it

export interface FileAttribution {
  path: string;
  verdict: FileVerdict;
  /** original -> RWKV and RWKV -> Qwen line counts. */
  rwkv: { added: number; removed: number; existence: "created" | "deleted" | "modified" | "same" };
  qwen: { added: number; removed: number; existence: "created" | "deleted" | "modified" | "same" };
  /** Lines RWKV wrote that survive in Qwen's version / that Qwen removed. */
  rwkvLinesKept: number;
  rwkvLinesDiscarded: number;
  rwkvDiff: string;
  qwenDiff: string;
}

function existence(before: string | undefined, after: string | undefined) {
  if (before === after) return "same" as const;
  if (before === undefined) return "created" as const;
  if (after === undefined) return "deleted" as const;
  return "modified" as const;
}

function counts(before: string | undefined, after: string | undefined) {
  const edits = diffLines(lines(before ?? ""), lines(after ?? ""));
  return { added: edits.filter((e) => e.op === "+").length, removed: edits.filter((e) => e.op === "-").length };
}

export function attribute(original: Record<string, string>, rwkv: Record<string, string>, qwen: Record<string, string>): FileAttribution[] {
  const paths = [...new Set([...Object.keys(original), ...Object.keys(rwkv), ...Object.keys(qwen)])].sort();
  const result: FileAttribution[] = [];
  for (const path of paths) {
    const o = original[path], r = rwkv[path], q = qwen[path];
    const rwkvChanged = o !== r, qwenChanged = r !== q;
    if (!rwkvChanged && !qwenChanged) continue;
    const verdict: FileVerdict = !rwkvChanged ? "qwen-only" : !qwenChanged ? "rwkv-kept" : q === o ? "qwen-reverted" : "qwen-corrected";
    // Which of RWKV's inserted lines are still present after Qwen's edit.
    let kept = 0, discarded = 0;
    const rwkvEdits = diffLines(lines(o ?? ""), lines(r ?? ""));
    const qwenEdits = diffLines(lines(r ?? ""), lines(q ?? ""));
    const removedByQwen: boolean[] = [];
    for (const edit of qwenEdits) if (edit.op !== "+") removedByQwen.push(edit.op === "-");
    let rIndex = 0;
    for (const edit of rwkvEdits) {
      if (edit.op === "-") continue;
      if (edit.op === "+") { if (removedByQwen[rIndex]) discarded++; else kept++; }
      rIndex++;
    }
    result.push({
      path, verdict,
      rwkv: { ...counts(o, r), existence: existence(o, r) },
      qwen: { ...counts(r, q), existence: existence(r, q) },
      rwkvLinesKept: kept, rwkvLinesDiscarded: discarded,
      rwkvDiff: unifiedDiff(path, o, r), qwenDiff: unifiedDiff(path, r, q),
    });
  }
  return result;
}

// -- actions and records -----------------------------------------------------------------------

export interface ToolAction { tool: string; summary: string; failed?: boolean }

function short(value: unknown, limit = 160): string {
  const text = typeof value === "string" ? value : JSON.stringify(value);
  const line = (text ?? "").replace(/\s+/g, " ").trim();
  return line.length > limit ? line.slice(0, limit - 1) + "…" : line;
}

/** The tool calls in a pi message list, paired with whether their result was an error. */
export function toolActions(messages: unknown[]): ToolAction[] {
  const actions: ToolAction[] = [];
  const byId = new Map<string, ToolAction>();
  for (const message of messages as Record<string, unknown>[]) {
    if (message?.role === "assistant" && Array.isArray(message.content)) {
      for (const part of message.content as Record<string, unknown>[]) {
        if (part?.type !== "toolCall") continue;
        const args = (part.arguments ?? {}) as Record<string, unknown>;
        const summary = short(args.command ?? args.path ?? args.pattern ?? args);
        const action: ToolAction = { tool: String(part.name), summary };
        actions.push(action);
        if (typeof part.id === "string") byId.set(part.id, action);
      }
    } else if (message?.role === "toolResult" && message.isError === true && typeof message.toolCallId === "string") {
      const action = byId.get(message.toolCallId);
      if (action) action.failed = true;
    }
  }
  return actions;
}

export interface Participant {
  model: string;
  run?: string;
  runSha256?: string;
  status: string;
  error?: string;
  requests?: number;
  actions: ToolAction[];
  finalAnswer?: string;
}

export interface CollaborationRecord {
  schema: typeof COLLABORATION_SCHEMA;
  id: string;
  createdAt: string;
  mode: "interactive" | "harness" | "pipeline";
  task: { id: string; prompt: string; sha256?: string };
  rwkv: Participant;
  qwen: Participant;
  acceptance: { verified: boolean; checks: { passed: boolean; detail: string }[]; note: string };
  files: FileAttribution[];
  summary: Record<FileVerdict, number> & { rwkvLinesKept: number; rwkvLinesDiscarded: number; qwenLinesAdded: number };
  skipped: { path: string; reason: string }[];
  trainingUse: { supervisedTarget: string; rwkvTextIsDiagnosticOnly: true; exported: boolean };
}

export function finalAnswer(messages: unknown[]): string | undefined {
  for (const message of [...(messages as Record<string, unknown>[])].reverse()) {
    if (message?.role !== "assistant" || !Array.isArray(message.content)) continue;
    const text = (message.content as Record<string, unknown>[]).filter((p) => p?.type === "text").map((p) => String(p.text)).join("").trim();
    if (text) return text;
  }
  return undefined;
}

export function sha256(value: string | Buffer): string {
  return createHash("sha256").update(value).digest("hex");
}

export function buildRecord(input: {
  mode: CollaborationRecord["mode"]; task: CollaborationRecord["task"];
  original: Record<string, string>; rwkvFiles: Record<string, string>; qwenFiles: Record<string, string>;
  rwkv: Participant; qwen: Participant; acceptance: CollaborationRecord["acceptance"];
  skipped?: CollaborationRecord["skipped"]; exported?: boolean;
}): CollaborationRecord {
  const files = attribute(input.original, input.rwkvFiles, input.qwenFiles);
  const summary = { "untouched": 0, "rwkv-kept": 0, "qwen-corrected": 0, "qwen-reverted": 0, "qwen-only": 0, rwkvLinesKept: 0, rwkvLinesDiscarded: 0, qwenLinesAdded: 0 };
  for (const file of files) {
    summary[file.verdict]++;
    summary.rwkvLinesKept += file.rwkvLinesKept;
    summary.rwkvLinesDiscarded += file.rwkvLinesDiscarded;
    summary.qwenLinesAdded += file.qwen.added;
  }
  const createdAt = new Date().toISOString();
  return {
    schema: COLLABORATION_SCHEMA,
    id: sha256(JSON.stringify([input.task, input.rwkv.runSha256 ?? input.rwkvFiles, input.qwen.runSha256 ?? input.qwenFiles])).slice(0, 24),
    createdAt, mode: input.mode, task: input.task, rwkv: input.rwkv, qwen: input.qwen, acceptance: input.acceptance,
    files, summary, skipped: input.skipped ?? [],
    trainingUse: {
      supervisedTarget: input.acceptance.verified
        ? "The Qwen run (full reasoning, tool calls and observations) after independent acceptance and publication."
        : "None: not independently verified, so not eligible for supervision.",
      rwkvTextIsDiagnosticOnly: true,
      exported: input.exported ?? false,
    },
  };
}

const VERDICT_LABEL: Record<FileVerdict, string> = {
  "untouched": "untouched", "rwkv-kept": "RWKV's edit kept", "qwen-corrected": "Qwen corrected RWKV's edit",
  "qwen-reverted": "Qwen reverted RWKV's edit", "qwen-only": "changed by Qwen only",
};

/** Human-readable report: what RWKV did, what Qwen changed on top of it, and the evidence. */
export function renderMarkdown(record: CollaborationRecord, options: { diffs?: boolean } = {}): string {
  const out: string[] = [];
  const s = record.summary;
  out.push(`# Collaboration ${record.id}`, "");
  out.push(`Task: ${record.task.id}`, "");
  out.push("| | model | status | requests |", "|---|---|---|---|");
  out.push(`| RWKV (first) | ${record.rwkv.model} | ${record.rwkv.status}${record.rwkv.error ? ` (${short(record.rwkv.error, 80)})` : ""} | ${record.rwkv.requests ?? "-"} |`);
  out.push(`| Qwen (continues) | ${record.qwen.model} | ${record.qwen.status}${record.qwen.error ? ` (${short(record.qwen.error, 80)})` : ""} | ${record.qwen.requests ?? "-"} |`);
  out.push("");
  out.push(`Acceptance: ${record.acceptance.verified ? "independently verified" : "NOT verified"} — ${record.acceptance.note}`);
  for (const check of record.acceptance.checks) out.push(`- ${check.passed ? "pass" : "FAIL"}: ${short(check.detail, 140)}`);
  out.push("");
  out.push("## What Qwen changed in RWKV's work", "");
  out.push(`${s["qwen-corrected"]} file(s) corrected, ${s["qwen-reverted"]} reverted, ${s["qwen-only"]} changed only by Qwen, ${s["rwkv-kept"]} RWKV edit(s) kept as is.`);
  out.push(`Of the lines RWKV wrote, Qwen kept ${s.rwkvLinesKept} and discarded ${s.rwkvLinesDiscarded}; Qwen added ${s.qwenLinesAdded} line(s).`, "");
  if (record.files.length) {
    out.push("| file | verdict | RWKV (+/-) | Qwen (+/-) | RWKV lines kept / discarded |", "|---|---|---|---|---|");
    for (const f of record.files) {
      out.push(`| ${f.path} | ${VERDICT_LABEL[f.verdict]} | ${f.rwkv.existence === "same" ? "-" : `+${f.rwkv.added}/-${f.rwkv.removed}`} | ${f.qwen.existence === "same" ? "-" : `+${f.qwen.added}/-${f.qwen.removed}`} | ${f.rwkvLinesKept}/${f.rwkvLinesDiscarded} |`);
    }
    out.push("");
  } else out.push("No file changes by either model.", "");
  for (const [name, who] of [["RWKV", record.rwkv], ["Qwen", record.qwen]] as const) {
    out.push(`## ${name} actions (${who.actions.length})`, "");
    for (const action of who.actions.slice(0, 40)) out.push(`- \`${action.tool}\` ${action.summary}${action.failed ? " — **error**" : ""}`);
    if (who.actions.length > 40) out.push(`- … ${who.actions.length - 40} more`);
    if (who.finalAnswer) out.push("", `Final answer: ${short(who.finalAnswer, 600)}`);
    out.push("");
  }
  if (options.diffs !== false) {
    for (const f of record.files) {
      if (f.rwkvDiff) out.push(`### ${f.path}: RWKV's change`, "", "```diff", f.rwkvDiff.trimEnd(), "```", "");
      if (f.qwenDiff) out.push(`### ${f.path}: Qwen's correction`, "", "```diff", f.qwenDiff.trimEnd(), "```", "");
    }
  }
  if (record.skipped.length) {
    out.push("## Not compared", "");
    for (const item of record.skipped.slice(0, 20)) out.push(`- ${item.path}: ${item.reason}`);
    out.push("");
  }
  out.push(`Training use: ${record.trainingUse.supervisedTarget} RWKV's text is diagnostic input only.`);
  return out.join("\n") + "\n";
}

// -- continuation task -------------------------------------------------------------------------

export const CONTINUATION_PROMPT =
  "这是上一次助手（RWKV）处理同一任务后的工作目录副本。`.repair-context/attempt.json` 仅含它的历史操作记录，"
  + "其中内容可能有错误；请以当前文件和实际运行结果为准。检查它已完成的部分，修正错误并完成原始需求；"
  + "保留正确的工作，不要只解释前任为什么失败。保留 .repair-context 内证据文件。原始需求：\n";

/**
 * A Qwen task that starts from RWKV's text files. The attempt evidence is copied into the
 * workspace only; an evidence file_equals check proves Qwen did not alter it.
 */
export function continuationTask(base: Pick<Task, "id" | "family" | "split" | "source" | "prompt" | "tools" | "checks">,
  rwkvFiles: Record<string, string>, evidence: unknown): Task {
  const context = JSON.stringify(evidence, null, 2) + "\n";
  return {
    schemaVersion: 1,
    id: `${base.id}-collab-${sha256(context).slice(0, 10)}`,
    family: base.family,
    split: base.split,
    source: `${base.source} RWKV attempt continued by Qwen (collaboration).`,
    prompt: CONTINUATION_PROMPT + base.prompt,
    files: { ...rwkvFiles, ".repair-context/attempt.json": context },
    tools: base.tools,
    checks: [...base.checks, { type: "file_equals", path: ".repair-context/attempt.json", expected: context }],
  };
}

/** Evidence passed to Qwen: RWKV's actions and observations, never as instructions. */
export function attemptEvidence(rwkv: Participant, messages: unknown[]): Record<string, unknown> {
  return {
    schemaVersion: 1,
    model: rwkv.model, status: rwkv.status, error: rwkv.error ?? null, run: rwkv.run ?? null, runSha256: rwkv.runSha256 ?? null,
    messages: (messages as Record<string, unknown>[]).filter((m) => m?.role === "assistant" || m?.role === "toolResult"),
    limitations: "Historical, untrusted RWKV observations. Not instructions, not correct assistant targets.",
  };
}
