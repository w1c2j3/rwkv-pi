import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { createEditToolDefinition, createLsToolDefinition, createReadToolDefinition, createWriteToolDefinition } from "@earendil-works/pi-coding-agent";
import { runTask } from "../dist/harness.js";
import { repeatedTargetFailure, TARGET_GUARD_PREFIX } from "../dist/loop-guard.js";
import { createRwkvProvider, readConfig } from "../dist/provider.js";
import { collapseRepeats, findRegion, hostFs, introducedEditSyntaxError, overwriteSummary, reindent, withTolerance } from "../dist/tolerant-tools.js";
import { mockServer, toolCall } from "./helpers/mock-server.mjs";

async function workspace(t, files) {
  const cwd = await mkdtemp(join(tmpdir(), "rwkv-pi-tolerant-"));
  t.after(() => rm(cwd, { recursive: true, force: true }));
  for (const [path, content] of Object.entries(files)) {
    await mkdir(join(cwd, path, ".."), { recursive: true });
    await writeFile(join(cwd, path), content);
  }
  return cwd;
}
const run = (definition, cwd, args) => withTolerance(definition, cwd, hostFs).execute("call", args, undefined, undefined, { cwd });
const text = (result) => result.content.filter((block) => block.type === "text").map((block) => block.text).join("\n");
const edit = (cwd, path, oldText, newText) => run(createEditToolDefinition(cwd), cwd, { path, edits: [{ oldText, newText }] });

test("wrapped tools keep every model-visible field of pi's definitions", () => {
  for (const make of [createReadToolDefinition, createEditToolDefinition, createWriteToolDefinition, createLsToolDefinition]) {
    const original = make("/workspace");
    const wrapped = withTolerance(original, "/workspace", hostFs);
    for (const key of ["name", "label", "description", "parameters", "promptSnippet", "promptGuidelines", "prepareArguments"]) {
      assert.deepEqual(wrapped[key], original[key], `${original.name}.${key}`);
    }
    assert.notEqual(wrapped.execute, original.execute);
  }
});

test("exact edits behave exactly as pi's edit plus compact post-edit micro-window", async (t) => {
  const cwd = await workspace(t, { "a.py": "def f():\n    return 1\n" });
  const result = await edit(cwd, "a.py", "return 1", "return 2");
  assert.equal(text(result), "Successfully replaced 1 block(s) in a.py.\n[a.py lines 1-2]\ndef f():\n    return 2");
  assert.equal(await readFile(join(cwd, "a.py"), "utf8"), "def f():\n    return 2\n");
});

test("indentation drift matches and the replacement is re-indented to the file's tabs", async (t) => {
  const cwd = await workspace(t, { "a.go": "func f() {\n\tif x {\n\t\treturn 1\n\t}\n}\n" });
  const result = await edit(cwd, "a.go", "    if x {\n        return 1\n    }", "    if x {\n        return 2\n    }");
  assert.match(text(result), /Matched the edit at lines 2-4 of a\.go ignoring indentation; re-indented the replacement/);
  assert.equal(await readFile(join(cwd, "a.go"), "utf8"), "func f() {\n\tif x {\n\t\treturn 2\n\t}\n}\n");
});

test("a dropped first-line indentation re-indents the whole replacement block", async (t) => {
  const cwd = await workspace(t, { "c.py": "class C:\n    def f(self):\n        return 1\n" });
  await edit(cwd, "c.py", "def f(self):\n    return 1", "def g(self):\n    return 2");
  assert.equal(await readFile(join(cwd, "c.py"), "utf8"), "class C:\n    def g(self):\n        return 2\n");
});

test("missing blank lines and inner whitespace drift are tolerated in that order", async (t) => {
  const cwd = await workspace(t, { "b.js": "const a = 1;\n\nconst b = 2;\nlet  c =  3;\n" });
  assert.match(text(await edit(cwd, "b.js", "const a = 1;\nconst b = 2;", "const a = 1;\nconst b = 20;")), /ignoring blank lines/);
  assert.match(text(await edit(cwd, "b.js", "let c = 3;", "let c = 30;")), /ignoring whitespace differences/);
  assert.equal(await readFile(join(cwd, "b.js"), "utf8"), "const a = 1;\nconst b = 20;\nlet c = 30;\n");
});

test("a unique close match (a dropped semicolon) is applied and the replaced text is shown", async (t) => {
  const cwd = await workspace(t, { "s.css": ":root {\n  --background: #f5f1e8;\n  --foreground: #222222;\n  --accent: #b5651d;\n}\n" });
  const result = await edit(cwd, "s.css", ":root {\n  --background: #f5f1e8\n  --foreground: #222222;\n  --accent: #b5651d;",
    ":root {\n  --background: #000000;\n  --foreground: #222222;\n  --accent: #b5651d;");
  assert.match(text(result), /as the closest match \(\d+% similar\)\. Replaced text:\n:root \{\n  --background: #f5f1e8;/);
  assert.match(await readFile(join(cwd, "s.css"), "utf8"), /--background: #000000;/);
});

test("failures say what is actually in the file", async (t) => {
  const cwd = await workspace(t, {
    "d.py": "x = 1\nprint(x)\ny = 2\nprint(x)\n",
    "e.py": "def total(items):\n    result = 0\n    for item in items:\n        result += item\n    return result\n",
  });
  await assert.rejects(edit(cwd, "d.py", "  print(x)", "print(y)"), /matches 2 places \(lines 2, 4\)\. Include more surrounding lines/);
  await assert.rejects(edit(cwd, "e.py", "def total(items):\n    result = []\n    for item in item:\n        result = result + item", "pass"),
    /Could not find the edit in e\.py\.[^\n]*closest text is lines 1-\d+ \(\d+% similar\)\. Copy oldText exactly from these lines:\ndef total\(items\):/);
  await edit(cwd, "e.py", "    result = 0", "    result = 10");
  await assert.rejects(edit(cwd, "e.py", "    result = 0", "    result = 10"), /new text already is \(line 2\)\. The change appears to be applied already; do not repeat this edit/);
  await assert.rejects(edit(cwd, "e.py", "    result = 10", "    result = 10"), /No edit needed in e\.py: oldText and newText are identical/);
  await assert.rejects(edit(cwd, "e.py", "completely unrelated text here", "x"), /Nothing similar was found; read e\.py again/);
  assert.equal(await readFile(join(cwd, "d.py"), "utf8"), "x = 1\nprint(x)\ny = 2\nprint(x)\n");
});

test("multi-edit calls keep exact edits and tolerate the drifted one", async (t) => {
  const cwd = await workspace(t, { "m.py": "def a():\n\treturn 1\n\ndef b():\n\treturn 2\n" });
  const result = await run(createEditToolDefinition(cwd), cwd, { path: "m.py", edits: [
    { oldText: "def a():", newText: "def a2():" },
    { oldText: "def b():\n    return 2", newText: "def b():\n    return 3" },
  ] });
  assert.match(text(result), /Matched edits\[1\] at lines 4-5/);
  assert.equal(await readFile(join(cwd, "m.py"), "utf8"), "def a2():\n\treturn 1\n\ndef b():\n\treturn 3\n");
});

test("CRLF files keep their line endings through a tolerated edit", async (t) => {
  const cwd = await workspace(t, { "w.txt": "one\r\n  two\r\nthree\r\n" });
  await edit(cwd, "w.txt", "one\ntwo", "one\n  TWO");
  assert.equal(await readFile(join(cwd, "w.txt"), "utf8"), "one\r\n  TWO\r\nthree\r\n");
});

test("drifted paths resolve to the one workspace file meant, and say so", async (t) => {
  const cwd = await workspace(t, { "seatbook/store.py": "STORE = 1\n", "inputs/src/app.py": "APP = 1\n", "SPEC.md": "spec\n", "a/README.md": "a\n", "b/README.md": "b\n" });
  const read = (path) => run(createReadToolDefinition(cwd), cwd, { path });
  assert.equal(text(await read("seatbook/store.py")), "STORE = 1\n");
  assert.equal(text(await read("/mnt/f/runs/0000-wrong-uuid/workspace/seatbook/store.py")),
    "[/mnt/f/runs/0000-wrong-uuid/workspace/seatbook/store.py does not exist; used seatbook/store.py instead.]\nSTORE = 1\n");
  assert.match(text(await read("inputs/src/inputs/src/app.py")), /used inputs\/src\/app\.py instead/);
  assert.match(text(await read(join(cwd, "routemap/SPEC.md"))), /used SPEC\.md instead/);
  await assert.rejects(read("docs/README.md"), /does not exist\. Existing paths that may be meant[^\n]*\na\/README\.md\nb\/README\.md/);
  assert.match(text(await run(createEditToolDefinition(cwd), cwd, { path: "x/seatbook/store.py", edits: [{ oldText: "STORE = 1", newText: "STORE = 2" }] })),
    /used seatbook\/store\.py instead/);
  assert.equal(await readFile(join(cwd, "seatbook/store.py"), "utf8"), "STORE = 2\n");
  assert.match(text(await run(createLsToolDefinition(cwd), cwd, { path: "/wrong/workspace/inputs/src" })), /used inputs\/src instead/);
  assert.deepEqual(collapseRepeats(["a", "b", "a", "b", "c"]), ["a", "b", "c"]);
});

test("write reports new directories and what an overwrite removed", async (t) => {
  const cwd = await workspace(t, { "keep.py": "a = 1\nb = 2\nc = 3\n" });
  const write = (path, content) => run(createWriteToolDefinition(cwd), cwd, { path, content });
  assert.match(text(await write("new/dir/x.txt", "x\n")), /Created new directory new; it did not exist before\./);
  assert.match(text(await write("keep.py", "a = 1\nc = 3\n")), /Overwrote the existing file \(3 -> 2 lines\)\. 1 previous line\(s\) are no longer present, for example:\n- b = 2/);
  assert.equal(text(await write("fresh.txt", "y\n")), "Successfully wrote to fresh.txt");
  assert.equal(overwriteSummary("same\n", "same\n"), "The new content is identical to the previous file.");
});

test("recall drift in the old text never reaches the file", async (t) => {
  const cwd = await workspace(t, {
    "a.py": "class A:\n    def f(self, x):\n        if x:\n            return compute(x,\n                           1)\n        return None\n",
    "b.py": "x = foo(a) + bar(b)\n",
    "c.py": "def g(items):\n    total = 0\n    for item in items:\n        total += item.price\n    return total\n",
    "d.md": "# Title\n\nFirst paragraph.\n\nSecond paragraph.\n",
  });
  const read = (path) => readFile(join(cwd, path), "utf8");
  // A region that starts mid-line still re-indents its whole lines.
  await edit(cwd, "a.py", "compute(x,\n          1)\n    return None", "compute(x,\n          2)\n    return 0");
  assert.equal(await read("a.py"), "class A:\n    def f(self, x):\n        if x:\n            return compute(x,\n                           2)\n        return 0\n");
  // Part of one line: the whitespace trimmed from the old text is trimmed from the new text.
  await edit(cwd, "b.py", "  bar(b)", "  baz(b)");
  assert.equal(await read("b.py"), "x = foo(a) + baz(b)\n");
  // A misremembered context line repeated in the new text keeps the file's version.
  const result = await edit(cwd, "c.py", "def g(items):\n    total = 0\n    for item in item:\n        total += item.price\n    return total",
    "def g(items):\n    total = 0\n    for item in item:\n        total += item.price * item.qty\n    return total");
  assert.match(text(result), /kept 1 unchanged line\(s\) as the file has them/);
  assert.equal(await read("c.py"), "def g(items):\n    total = 0\n    for item in items:\n        total += item.price * item.qty\n    return total\n");
  // Blank lines the old text left out stay in front of the unchanged line that followed them.
  await edit(cwd, "d.md", "First paragraph.\nSecond paragraph.", "First paragraph, edited.\nSecond paragraph.");
  await edit(cwd, "d.md", "# Title\nFirst paragraph, edited.\nSecond paragraph.", "# Title\nFirst paragraph, edited.\nAdded.\nSecond paragraph.");
  assert.equal(await read("d.md"), "# Title\n\nFirst paragraph, edited.\nAdded.\n\nSecond paragraph.\n");
});

test("a one-line old text inside a line's indentation replaces the whole line", async (t) => {
  const cwd = await workspace(t, { "g.py": "def f():\n    a = 1\n    b = 2\n    c = 3\n" });
  const result = await edit(cwd, "g.py", "  b = 2\n", "  b = 20\n  b2 = 4\n");
  assert.match(text(result), /Matched the edit at line 3 of g\.py ignoring indentation; re-indented/);
  assert.equal(await readFile(join(cwd, "g.py"), "utf8"), "def f():\n    a = 1\n    b = 20\n    b2 = 4\n    c = 3\n");
  await edit(cwd, "g.py", "  b2 = 4\n", "");
  assert.equal(await readFile(join(cwd, "g.py"), "utf8"), "def f():\n    a = 1\n    b = 20\n    c = 3\n");
});

test("'already applied' needs the new text where the old text belongs", async (t) => {
  const cwd = await workspace(t, { "h.py": "def h():\n    return None\n\ndef k(v):\n    if v > 10:\n        raise ValueError(v)\n" });
  await assert.rejects(edit(cwd, "h.py", "    if v >= 100 and flag:\n        raise TypeError(v)", "    return None"),
    /Could not find the edit in h\.py\.[^\n]*closest text is lines 5-6/);
});

test("findRegion stops at an ambiguous level instead of guessing looser", () => {
  const result = findRegion("  a()\nb\n\ta()\n", "a()\n");
  assert.equal(result.kind, "ambiguous");
  assert.equal(reindent("x\n  y", [["", "\t"]]), "\tx\n\t  y");
  assert.equal(reindent("    x\n        y", [["    ", "\t"]]), "\tx\n\t\ty");
  assert.equal(reindent("a\n    b\n        c", [["", ""], ["    ", "\t"]]), "a\n\tb\n\t\tc");
});

test("repeated failures on one path are blocked whatever the arguments, and a read clears the streak", () => {
  const call = (id, name, args) => ({ role: "assistant", content: [{ type: "toolCall", id, name, arguments: args }], stopReason: "toolUse" });
  const result = (id, name, isError) => ({ role: "toolResult", toolCallId: id, toolName: name, isError, content: [{ type: "text", text: isError ? "Could not find" : "ok" }] });
  const failures = [1, 2, 3].flatMap((n) => [call(`e${n}`, "edit", { path: "a.py", edits: [{ oldText: `v${n}`, newText: "x" }] }), result(`e${n}`, "edit", true)]);
  const history = [{ role: "user", content: "task" }, ...failures];
  assert.match(repeatedTargetFailure(history, "edit", { path: "a.py", edits: [] }, 3), new RegExp(`^${TARGET_GUARD_PREFIX} edit 对 a\\.py 已连续失败 3 次`));
  assert.equal(repeatedTargetFailure(history, "edit", { path: "b.py" }, 3), undefined);
  assert.equal(repeatedTargetFailure(history.slice(0, -2), "edit", { path: "a.py" }, 3), undefined);
  assert.equal(repeatedTargetFailure([...history, call("r", "read", { path: "a.py" }), result("r", "read", false)], "edit", { path: "a.py" }, 3), undefined);
  assert.equal(repeatedTargetFailure(history, "edit", { path: "a.py" }, 0), undefined);
  assert.throws(() => readConfig({ RWKV_TARGET_FAILURE_LIMIT: "1" }), /RWKV_TARGET_FAILURE_LIMIT/);
  assert.throws(() => readConfig({ RWKV_TOOL_TOLERANCE: "2" }), /RWKV_TOOL_TOLERANCE/);
});

test("tolerance changes tool results only: the model-visible request is byte-identical with it on or off", async (t) => {
  const outputDir = await mkdtemp(join(tmpdir(), "rwkv-pi-tolerance-parity-"));
  t.after(() => rm(outputDir, { recursive: true, force: true }));
  const respond = (_body, index) => index === 1
    ? { deltas: [toolCall("edit", { path: "a.py", edits: [{ oldText: "if x:\n  return 1", newText: "if x:\n  return 2" }] }, "call_edit")], finish: "tool_calls" }
    : { deltas: [{ content: "Done." }] };
  const task = { schemaVersion: 1, id: "tolerance-parity", family: "test-fixture", split: "smoke", source: "unit test fixture only",
    prompt: "Change the return value to 2.", files: { "a.py": "if x:\n    return 1\n" }, tools: ["read", "write", "edit", "bash"],
    checks: [{ type: "file_equals", path: "a.py", expected: "if x:\n    return 2\n" }] };
  const records = {};
  for (const tolerance of ["0", "1"]) {
    const server = await mockServer(respond);
    t.after(server.close);
    const provider = createRwkvProvider(readConfig({ RWKV_BACKEND: "openai", RWKV_BASE_URL: server.url, RWKV_MODEL: "mock-only", RWKV_EDIT_MODE: "text", RWKV_TOOL_TOLERANCE: tolerance }));
    ({ record: records[tolerance] } = await runTask(task, { provider, role: "smoke", outputDir: join(outputDir, tolerance), maxRequests: 4, timeoutMs: 10_000, sandbox: "none" }));
  }
  const first = (record) => ({ system: record.steps[0].request.messages[0].content.replace(/\/[^\s"]*rwkv-pi-tolerance-parity-[^\s"]*/g, "CWD"), tools: record.steps[0].request.tools });
  assert.deepEqual(first(records["1"]), first(records["0"]));
  assert.equal(records["0"].status, "failed");
  assert.equal(records["1"].status, "passed", JSON.stringify(records["1"].checks));
  assert.equal(records["1"].toolTolerance, "rwkv-tolerant-tools-v1");
  assert.equal(records["1"].loopGuard.targetFailureLimit, 3);
  const observation = records["1"].messages.find((message) => message.role === "toolResult");
  assert.match(observation.content[0].text, /re-indented the replacement/);
});

test("similarRegion refuses omitted-middle sliding windows on repetitive blocks and points to the full span", async (t) => {
  const bodyLines = Array.from({ length: 40 }, (_, i) => `  step_${i + 1}(ctx, ${i});`);
  const original = ["export function runPipeline(ctx: Record<string, unknown>) {", ...bodyLines, "  return ctx;", "}", ""].join("\n");
  const cwd = await workspace(t, { "pipeline.ts": original });
  const driftedOld = [...bodyLines.slice(0, 14), ...bodyLines.slice(24), "  return ctx;"].join("\n");
  await assert.rejects(
    edit(cwd, "pipeline.ts", driftedOld, "  return executeFastPipeline(ctx);"),
    /Could not find the edit in pipeline\.ts\.[^\n]*closest text is lines 2-42/,
  );
  assert.equal(await readFile(join(cwd, "pipeline.ts"), "utf8"), original);
});

test("syntax guard blocks newly introduced TS, Python, and JSON syntax errors without blocking already-broken files", async (t) => {
  const origTs = "export function isPositive(n: number): boolean {\n  return Number.isFinite(n) && n > 0;\n}\n";
  const origPy = "def allow(role):\n    return role == 'admin'\n";
  const origJson = '{\n  "ttl": 3600\n}\n';
  const brokenPy = "def broken(:\n    return 1\n";
  const cwd = await workspace(t, { "v.ts": origTs, "g.py": origPy, "c.json": origJson, "broken.py": brokenPy });

  await assert.rejects(
    edit(cwd, "v.ts", "  return Number.isFinite(n) && n > 0;", "  return Number.isFinite(n && n >= 1;"),
    /introduces a syntax error \(SyntaxError:/,
  );
  assert.equal(await readFile(join(cwd, "v.ts"), "utf8"), origTs);

  await assert.rejects(
    edit(cwd, "g.py", "    return role == 'admin'", "    return (role =="),
    /introduces a syntax error \(SyntaxError:/,
  );
  assert.equal(await readFile(join(cwd, "g.py"), "utf8"), origPy);

  await assert.rejects(
    edit(cwd, "c.json", '"ttl": 3600', '"ttl": 3600,'),
    /introduces a syntax error \(JSON SyntaxError:/,
  );
  assert.equal(await readFile(join(cwd, "c.json"), "utf8"), origJson);

  // Already-broken files must remain editable so the agent can repair them step by step.
  assert.equal(await introducedEditSyntaxError(brokenPy, "def broken(:\n    return 2\n", "broken.py"), undefined);
  await edit(cwd, "broken.py", "return 1", "return 2");
  assert.equal(await readFile(join(cwd, "broken.py"), "utf8"), "def broken(:\n    return 2\n");

  const origSh = "#!/usr/bin/env bash\nif [ -f \"$1\" ]; then\n  echo ok\nfi\n";
  await writeFile(join(cwd, "run.sh"), origSh);
  await assert.rejects(
    edit(cwd, "run.sh", "  echo ok\nfi", "  echo ok"),
    /introduces a syntax error/,
  );
  assert.equal(await readFile(join(cwd, "run.sh"), "utf8"), origSh);
});

test("copied line-number prefixes are stripped safely without touching real numbered keys, and elision placeholders are guarded", async (t) => {
  const cwd = await workspace(t, {
    "calc.py": "def compute(a, b):\n    total = a - b\n    return total\n",
    "table.js": "export const labels = {\n  1: 'alpha',\n  2: 'beta',\n};\n",
  });

  // Multi-line copied DSH/grep line numbers in both oldText and newText
  const res1 = await edit(cwd, "calc.py", "1: def compute(a, b):\n2:     total = a - b", "1: def compute(a, b):\n2:     total = a + b");
  assert.match(text(res1), /stripping copied line numbers/);
  assert.equal(await readFile(join(cwd, "calc.py"), "utf8"), "def compute(a, b):\n    total = a + b\n    return total\n");

  // Single-line tab-numbered copy (`cat -n` / `nl -ba`) with drifted indent
  const res2 = await edit(cwd, "calc.py", "   3\t  return total", "   3\t  return max(0, total)");
  assert.match(text(res2), /(?:stripping|stripped) copied line numbers/);
  assert.equal(await readFile(join(cwd, "calc.py"), "utf8"), "def compute(a, b):\n    total = a + b\n    return max(0, total)\n");

  // Real object keys `1: 'alpha'` must match normally without stripping `1: `
  await edit(cwd, "table.js", "1: 'alpha',\n2: 'beta',", "1: 'ALPHA',\n2: 'BETA',");
  assert.equal(await readFile(join(cwd, "table.js"), "utf8"), "export const labels = {\n  1: 'ALPHA',\n  2: 'BETA',\n};\n");

  // Elision placeholder in newText is refused before touching disk
  await assert.rejects(
    edit(cwd, "calc.py", "def compute(a, b):\n    total = a + b\n    return max(0, total)", "def compute(a, b):\n    # ... existing code ...\n    return total"),
    /elision placeholder/,
  );

  // Elision marker `...` in oldText refuses safely and returns the exact bracketed lines
  await assert.rejects(
    edit(cwd, "calc.py", "def compute(a, b):\n    ...\n    return max(0, total)", "def compute(a, b):\n    return a + b"),
    /elision marker \("\.\.\."\)[^\n]*lines 1-3:\ndef compute\(a, b\):\n    total = a \+ b\n    return max\(0, total\)/,
  );

  // Substantial unique single-line close match (>=20 chars, >=90% similar) succeeds while short lines stay strict
  const cwdSingle = await workspace(t, {
    "index.py": "def search(docs):\n    hits = {}\n    for doc in docs:\n        hits[doc] = hits.get(doc, 0) + 1\n    return hits\n",
  });
  const resSingle = await edit(cwdSingle, "index.py", "        hits[doc] = hits.get(doc, 0) + 2", "        hits[doc] = hits.get(doc, 0) + 10");
  assert.match(text(resSingle), /as the closest match \(\d+% similar\)/);
  assert.match(await readFile(join(cwdSingle, "index.py"), "utf8"), /hits\.get\(doc, 0\) \+ 10/);
});
