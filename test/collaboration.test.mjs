import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { attribute, buildRecord, continuationTask, diffLines, readTree, renderMarkdown, toolActions, unifiedDiff } from "../dist/collaboration.js";
import { parseTask } from "../dist/tasks.js";

test("line diff is minimal and lists deletions before insertions", () => {
  const edits = diffLines(["a", "b", "c"], ["a", "x", "c"]);
  assert.deepEqual(edits.map((e) => e.op + e.line), ["=a", "-b", "+x", "=c"]);
  const diff = unifiedDiff("f.py", "a\nb\nc\n", "a\nx\nc\n");
  assert.match(diff, /@@ -1,3 \+1,3 @@\n a\n-b\n\+x\n c\n/);
  assert.equal(unifiedDiff("f.py", "same\n", "same\n"), "");
  assert.match(unifiedDiff("new.py", undefined, "x\n"), /--- \/dev\/null\n\+\+\+ b\/new.py\n@@ -0,0 \+1,1 @@\n\+x/);
});

test("three-way attribution separates kept, corrected, reverted and Qwen-only edits", () => {
  const original = { "kept.py": "a\n", "fixed.py": "x = 1\n", "reverted.py": "ok\n", "qwen.py": "q\n", "same.py": "s\n" };
  const rwkv = { "kept.py": "a\nb\n", "fixed.py": "x = 1\ny = wrong\nz = 3\n", "reverted.py": "broken\n", "qwen.py": "q\n", "same.py": "s\n", "rwkv-new.py": "n\n" };
  const qwen = { "kept.py": "a\nb\n", "fixed.py": "x = 1\ny = right\nz = 3\n", "reverted.py": "ok\n", "qwen.py": "q\nq2\n", "same.py": "s\n" };
  const files = Object.fromEntries(attribute(original, rwkv, qwen).map((f) => [f.path, f]));
  assert.equal(files["kept.py"].verdict, "rwkv-kept");
  assert.equal(files["fixed.py"].verdict, "qwen-corrected");
  assert.equal(files["fixed.py"].rwkvLinesKept, 1);      // z = 3
  assert.equal(files["fixed.py"].rwkvLinesDiscarded, 1); // y = wrong
  assert.equal(files["reverted.py"].verdict, "qwen-reverted");
  assert.equal(files["qwen.py"].verdict, "qwen-only");
  assert.equal(files["rwkv-new.py"].verdict, "qwen-reverted"); // RWKV created it, Qwen deleted it
  assert.equal(files["same.py"], undefined);
  const record = buildRecord({ mode: "harness", task: { id: "t", prompt: "p" }, original, rwkvFiles: rwkv, qwenFiles: qwen,
    rwkv: { model: "rwkv", status: "failed", actions: [] }, qwen: { model: "qwen", status: "passed", actions: [] },
    acceptance: { verified: false, checks: [], note: "unit" } });
  assert.equal(record.summary["qwen-corrected"], 1);
  assert.match(record.trainingUse.supervisedTarget, /^None/);
  const report = renderMarkdown(record);
  assert.match(report, /fixed\.py: Qwen's correction[\s\S]*-y = wrong\n\+y = right/);
  assert.match(report, /NOT verified/);
});

test("tool actions pair calls with failed results", () => {
  const actions = toolActions([
    { role: "assistant", content: [{ type: "toolCall", id: "1", name: "bash", arguments: { command: "ls" } }, { type: "toolCall", id: "2", name: "read", arguments: { path: "x" } }] },
    { role: "toolResult", toolCallId: "2", isError: true, content: [] },
  ]);
  assert.deepEqual(actions, [{ tool: "bash", summary: "ls" }, { tool: "read", summary: "x", failed: true }]);
});

test("continuation task is a valid harness task that protects RWKV's evidence", () => {
  const dir = mkdtempSync(join(tmpdir(), "collab-"));
  mkdirSync(join(dir, "src")); mkdirSync(join(dir, "node_modules"));
  writeFileSync(join(dir, "src/a.py"), "print(1)\n"); writeFileSync(join(dir, "node_modules/x.js"), "x");
  writeFileSync(join(dir, "blob.bin"), Buffer.from([0, 1, 2]));
  const tree = readTree(dir);
  assert.deepEqual(Object.keys(tree.files), ["src/a.py"]);
  assert.deepEqual(tree.skipped.map((s) => s.reason).sort(), ["binary", "generated or vendored directory"]);
  const task = continuationTask({ id: "t", family: "f", split: "train", source: "s", prompt: "do it", tools: ["read", "bash"],
    checks: [{ type: "command", argv: ["true"], timeoutMs: 1000 }] }, tree.files, { status: "failed" });
  parseTask(task);
  assert.ok(task.prompt.endsWith("do it"));
  assert.equal(task.checks.at(-1).path, ".repair-context/attempt.json");
});

test("collaboration is an explicit plugin and is not a provider package side effect", async () => {
  const pkg = JSON.parse(readFileSync(new URL("../package.json", import.meta.url)));
  assert.deepEqual(pkg.pi.extensions, ["./dist/extension.js"]);
  const { resolvePlugins } = await import('../dist/plugins.js');
  assert.deepEqual(resolvePlugins([], { TEACHER_MODEL: 'configured' }).enabled, []);
  assert.deepEqual(resolvePlugins(['--collab'], {}).enabled, ['strong-model']);
  const extension = readFileSync(new URL("../src/collab-extension.ts", import.meta.url), "utf8");
  assert.match(extension, /agent_settled/);
  assert.doesNotMatch(extension, /ui\.confirm/);
});
