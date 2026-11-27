import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { createEditToolDefinition, createReadToolDefinition, createWriteToolDefinition } from "@earendil-works/pi-coding-agent";
import { convertPackedSpanRow, convertTrajectoryToLineEdit } from "../dist/anneal-data.js";
import { buildDataset } from "../dist/dataset.js";
import { foldEpochRequest } from "../dist/epoch.js";
import { runTask } from "../dist/harness.js";
import {
  applyAnchoredLineEdits,
  applyLineEdit,
  createLineModeTools,
  DEFAULT_LINE_READ_LIMIT,
  LineViewTracker,
  MAX_LINE_READ_LIMIT,
  prepareLineEditArguments,
} from "../dist/line-edit.js";
import { repeatedTargetFailure } from "../dist/loop-guard.js";
import { createRwkvProvider, readConfig } from "../dist/provider.js";
import { hostFs } from "../dist/tolerant-tools.js";
import { packTrajectory } from "../pipeline/trajectory-packing.mjs";
import { mockServer, toolCall } from "./helpers/mock-server.mjs";

test("applyLineEdit and applyAnchoredLineEdits handle 3-tier self-healing, atomic multi-hunk, CRLF, reindent, and guards", () => {
  assert.equal(DEFAULT_LINE_READ_LIMIT, 150);
  assert.equal(MAX_LINE_READ_LIMIT, 200);

  const src = "import math\r\n\r\ndef calc(a, b):\r\n    return a - b\r\n";

  // 1. Replace with copied line-number prefix auto-stripped, start_text verified, & CRLF preserved
  const r1 = applyLineEdit(src, "calc.py", {
    startLine: 4,
    endLine: 4,
    startText: "    return a - b",
    newText: "4:     return a + b",
  });
  assert.ok(!("error" in r1));
  assert.equal(r1.strippedLineNumbers, true);
  assert.equal(r1.anchorVerified, true);
  assert.equal(r1.content, "import math\r\n\r\ndef calc(a, b):\r\n    return a + b\r\n");

  // 2. Insert before line 3 using endLine = startLine - 1
  const r2 = applyLineEdit(r1.content, "calc.py", {
    startLine: 3,
    endLine: 2,
    startText: "def calc(a, b):",
    newText: "SCALE = 1",
  });
  assert.ok(!("error" in r2));
  assert.equal(r2.removed, 0);
  assert.equal(r2.added, 1);
  assert.equal(r2.content, "import math\r\n\r\nSCALE = 1\r\ndef calc(a, b):\r\n    return a + b\r\n");

  // 3. Delete line 3 via empty newText
  const r3 = applyLineEdit(r2.content, "calc.py", {
    startLine: 3,
    endLine: 3,
    startText: "SCALE = 1",
    newText: "",
  });
  assert.ok(!("error" in r3));
  assert.equal(r3.removed, 1);
  assert.equal(r3.added, 0);
  assert.equal(r3.content, r1.content);

  // 4. Tier 2 local ±15-line self-healing: RWKV remembers function header line 3 instead of body line 4!
  const reloc = applyLineEdit(r1.content, "calc.py", {
    startLine: 3,
    endLine: 3,
    startText: "    return a + b",
    newText: "    return a * b",
  });
  assert.ok(!("error" in reloc));
  assert.equal(reloc.resolvedStart, 4);
  assert.equal(reloc.resolvedEnd, 4);
  assert.match(reloc.alignedNote ?? "", /Auto-aligned.*3-3 -> 4-4/);
  assert.equal(reloc.content, "import math\r\n\r\ndef calc(a, b):\r\n    return a * b\r\n");

  // 5. Tier 3 whole-file unique fallback for non-trivial anchor (>15 lines away) + auto-reindent
  const bigLines = Array.from({ length: 60 }, (_, idx) => `def fn_${idx + 1}(x):\n    return x + ${idx + 1}\n`);
  const bigSrc = bigLines.join("\n"); // 180 lines
  const tier3 = applyLineEdit(bigSrc, "big.py", {
    startLine: 5,
    endLine: 5,
    startText: "  return x + 45", // drifted 2 spaces instead of 4, and line number off by >120 lines!
    newText: "  return x * 45",
  });
  assert.ok(!("error" in tier3));
  assert.equal(tier3.reindented, true);
  assert.match(tier3.alignedNote ?? "", /unique start_text in file/);
  assert.match(tier3.content, /def fn_45\(x\):\n    return x \* 45\n/);

  // 6. Anchor mismatch refuses BEFORE corrupting disk and returns numbered window
  const badAnch = applyLineEdit(r1.content, "calc.py", {
    startLine: 4,
    endLine: 4,
    startText: "    return completely_wrong_line",
    newText: "    return 0",
  });
  assert.ok("error" in badAnch);
  assert.match(badAnch.error, /start_text .* did not match line 4/);
  assert.match(badAnch.error, /=== calc\.py \(lines 1-4 of 4\) ===/);

  // 7. Atomic multi-hunk batch edit against pre-edit coordinates + overlap rejection
  const multiSrc = "a = 1\nb = 2\nc = 3\nd = 4\ne = 5\n";
  const batchOk = applyAnchoredLineEdits(multiSrc, "m.py", [
    { startLine: 1, endLine: 1, startText: "a = 1", newText: "a = 10\na2 = 11" },
    { startLine: 5, endLine: 5, startText: "e = 5", newText: "e = 50" },
  ]);
  assert.ok(!("error" in batchOk));
  assert.equal(batchOk.content, "a = 10\na2 = 11\nb = 2\nc = 3\nd = 4\ne = 50\n");

  const overlapErr = applyAnchoredLineEdits(multiSrc, "m.py", [
    { startLine: 1, endLine: 3, startText: "a = 1", newText: "x = 1" },
    { startLine: 3, endLine: 4, startText: "c = 3", newText: "y = 2" },
  ]);
  assert.ok("error" in overlapErr);
  assert.match(overlapErr.error, /overlap/);

  // 8. Off-by-one boundary duplication, no-op, and elision placeholders are refused
  const dupErr = applyLineEdit(r1.content, "calc.py", {
    startLine: 4,
    endLine: 4,
    startText: "    return a + b",
    newText: "def calc(a, b):\n    return a * b",
  });
  assert.ok("error" in dupErr);
  assert.match(dupErr.error, /duplicates line 3/);

  const noop = applyLineEdit(r1.content, "calc.py", {
    startLine: 4,
    endLine: 4,
    startText: "    return a + b",
    newText: "    return a + b",
  });
  assert.ok("error" in noop);
  assert.match(noop.error, /No edit needed/);

  const elision = applyLineEdit(r1.content, "calc.py", {
    startLine: 3,
    endLine: 4,
    startText: "def calc(a, b):",
    newText: "# ... existing code ...",
  });
  assert.ok("error" in elision);
  assert.match(elision.error, /elision placeholder/);

  // 9. prepareLineEditArguments normalizes shorthand single-hunk into canonical edits[]
  const normalized = prepareLineEditArguments({ path: "a.py", start_line: 2, end_line: 2, start_text: "x = 1", new_text: "x = 2" });
  assert.deepEqual(normalized, {
    path: "a.py",
    edits: [{ start_line: 2, end_line: 2, start_text: "x = 1", new_text: "x = 2" }],
  });
});

test("createLineModeTools enforces neutral header, accurate Epoch fold sync, verified-anchor bypass, atomic edits, and syntax guard", async (t) => {
  const cwd = await mkdtemp(join(tmpdir(), "rwkv-line-edit-"));
  t.after(() => rm(cwd, { recursive: true, force: true }));
  await writeFile(join(cwd, "mod.py"), "def first_calc():\n    return 100\n\ndef second_calc():\n    return 200\n");

  const tracker = new LineViewTracker();
  const tools = createLineModeTools(
    [createReadToolDefinition(cwd), createWriteToolDefinition(cwd), createEditToolDefinition(cwd)],
    cwd,
    hostFs,
    { tolerance: true, tracker },
  );
  const readTool = tools.find((x) => x.name === "read");
  const editTool = tools.find((x) => x.name === "edit");

  // 1. Blind edit without non-trivial anchor before reading is refused
  await assert.rejects(
    editTool.execute("e0", { path: "mod.py", edits: [{ start_line: 2, end_line: 2, start_text: "", new_text: "    return 10" }] }),
    /Cannot edit mod\.py before reading its numbered lines/,
  );

  // 2. Neutral numbered read unlocks edit (NOT SWE-agent `[File: ...]` header!)
  const readOut = (await readTool.execute("r1", { path: "mod.py" })).content[0].text;
  assert.match(readOut, /^=== mod\.py \(lines 1-5 of 5\) ===\n1: def first_calc\(\):\n2:     return 100/);
  assert.doesNotMatch(readOut, /^\[File:/);

  // 3. Syntax-breaking edit is refused without corrupting disk file
  await assert.rejects(
    editTool.execute("e_bad_syn", {
      path: "mod.py",
      edits: [{ start_line: 2, end_line: 2, start_text: "    return 100", new_text: "    return (" }],
    }),
    /introduces a syntax error/,
  );
  assert.match(await readFile(join(cwd, "mod.py"), "utf8"), /return 100/);

  // 4. Atomic multi-hunk edit updates both functions in ONE call with pre-edit line numbers!
  const editRes1 = (await editTool.execute("e1", {
    path: "mod.py",
    edits: [
      { start_line: 2, end_line: 2, start_text: "    return 100", new_text: "    val = 10\n    return val" },
      { start_line: 5, end_line: 5, start_text: "    return 200", new_text: "    return 20" },
    ],
  })).content[0].text;
  assert.match(editRes1, /Replaced line 2 with 2 lines\./);
  assert.match(editRes1, /Replaced line 5 with 1 line\./);
  assert.equal(
    await readFile(join(cwd, "mod.py"), "utf8"),
    "def first_calc():\n    val = 10\n    return val\n\ndef second_calc():\n    return 20\n",
  );

  // 5. Simulate Epoch fold keeping ONLY lines 1-3 visible: syncWithActiveMessages MUST shrink seen to {1,2,3} (not keep "all"!)
  tracker.syncWithActiveMessages([
    { role: "system", content: "You are pi." },
    { role: "tool", tool_call_id: "x", content: "=== mod.py (lines 1-3 of 6) ===\n1: def first_calc():\n2:     val = 10\n3:     return val" },
  ]);
  // Editing line 6 with empty/trivial anchor is now blocked because line 6 was folded out!
  await assert.rejects(
    editTool.execute("e_folded_blind", {
      path: "mod.py",
      edits: [{ start_line: 6, end_line: 6, start_text: "", new_text: "    return 30" }],
    }),
    /line 6 has not been read yet/,
  );
  // Editing line 6 WITH a verified non-trivial start_text (`    return 20`, len >= 10) succeeds even after fold!
  await editTool.execute("e_folded_anchored", {
    path: "mod.py",
    edits: [{ start_line: 6, end_line: 6, start_text: "    return 20", new_text: "    return 30" }],
  });
  assert.match(await readFile(join(cwd, "mod.py"), "utf8"), /return 30/);

  // 6. Mode-aware repeatedTargetFailure gives line-mode guidance instead of verbatim oldText advice
  const mkCall = (id) => ({ role: "assistant", content: [{ type: "toolCall", id, name: "edit", arguments: { path: "mod.py", edits: [{ start_line: 1, end_line: 1, start_text: "x", new_text: "y" }] } }] });
  const mkErr = (id) => ({ role: "toolResult", toolCallId: id, toolName: "edit", isError: true, content: [{ type: "text", text: "err" }] });
  const hist = [mkCall("1"), mkErr("1"), mkCall("2"), mkErr("2"), mkCall("3"), mkErr("3")];
  const guardMsg = repeatedTargetFailure(hist, "edit", { path: "mod.py", edits: [{ start_line: 1, end_line: 1, start_text: "x", new_text: "z" }] }, 3, "line");
  assert.match(guardMsg ?? "", /1-based 行号与首行锚点（start_text）/);
});

test("default Anchored Batch Line-Edit v2 runs end-to-end in harness and anneal-data converts multi-hunk atomically + aligns CoT + synthesizes 4 recoveries", async (t) => {
  const outDir = await mkdtemp(join(tmpdir(), "rwkv-line-e2e-"));
  t.after(() => rm(outDir, { recursive: true, force: true }));

  // 1. Default readConfig() selects editMode === "line"
  const defaultCfg = readConfig({});
  assert.equal(defaultCfg.editMode, "line");

  const lineServer = await mockServer((_body, idx) => {
    if (idx === 1) return { deltas: [{ reasoning_content: "先读取 calc.py 行号。" }, toolCall("read", { path: "calc.py" }, "l1")], finish: "tool_calls" };
    if (idx === 2) {
      return {
        deltas: [
          { reasoning_content: "第 2 行是 return a - b，按锚点替换为 return a + b。" },
          toolCall("edit", {
            path: "calc.py",
            edits: [{ start_line: 2, end_line: 2, start_text: "    return a - b", new_text: "    return a + b" }],
          }, "l2"),
        ],
        finish: "tool_calls",
      };
    }
    return { deltas: [{ reasoning_content: "修改完成。", content: "Fixed calc.py." }] };
  });
  t.after(lineServer.close);

  const lineProvider = createRwkvProvider(readConfig({
    RWKV_BACKEND: "openai",
    RWKV_BASE_URL: lineServer.url,
    RWKV_MODEL: "mock-rwkv",
  }));
  const task = {
    schemaVersion: 1,
    id: "line-mode-task-1",
    family: "line-mode-family",
    split: "train",
    source: "unit-test",
    prompt: "Fix add(a, b) and sub(a, b) in calc.py.",
    files: { "calc.py": "def add(a, b):\n    return a - b\n\ndef sub(a, b):\n    return a + b\n" },
    tools: ["read", "write", "edit", "bash"],
    checks: [{ type: "file_equals", path: "calc.py", expected: "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a + b\n" }],
  };
  const { record: lineRec } = await runTask(task, {
    provider: lineProvider,
    role: "teacher",
    outputDir: join(outDir, "line"),
    maxRequests: 6,
    timeoutMs: 10_000,
    sandbox: "none",
  });
  assert.equal(lineRec.status, "passed", lineRec.error);
  assert.equal(lineRec.editMode, "line");

  // 2. Build a legacy `text`-mode multi-hunk packed span and convert it atomically via `convertPackedSpanRow`!
  const textServer = await mockServer((_body, idx) => {
    if (idx === 1) return { deltas: [{ reasoning_content: "Read calc.py." }, toolCall("read", { path: "calc.py" }, "t1")], finish: "tool_calls" };
    if (idx === 2) {
      return {
        deltas: [
          { reasoning_content: "Fix both operators in calc.py." },
          toolCall("edit", {
            path: "calc.py",
            edits: [
              { oldText: "    return a - b", newText: "    return a + b" },
              { oldText: "    return a + b", newText: "    return a - b" },
            ],
          }, "t2"),
        ],
        finish: "tool_calls",
      };
    }
    return { deltas: [{ reasoning_content: "All set.", content: "Fixed." }] };
  });
  t.after(textServer.close);

  const textProvider = createRwkvProvider(readConfig({
    RWKV_BACKEND: "openai",
    RWKV_BASE_URL: textServer.url,
    RWKV_MODEL: "mock-teacher",
    RWKV_EDIT_MODE: "text",
  }));
  const multiTask = {
    ...task,
    id: "legacy-text-multi-1",
    family: "legacy-family",
    checks: [{ type: "file_equals", path: "calc.py", expected: "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n" }],
  };
  const { record: textRec } = await runTask(multiTask, {
    provider: textProvider,
    role: "teacher",
    outputDir: join(outDir, "text"),
    maxRequests: 6,
    timeoutMs: 10_000,
    sandbox: "none",
  });
  assert.equal(textRec.status, "passed");
  const legacyDs = buildDataset([textRec]);
  const legacySpans = packTrajectory(legacyDs.samples, "xml");
  assert.equal(legacySpans.length, 1);

  for (const augKind of ["unread_first", "anchor_mismatch", "boundary_dup", "noop_recovery"]) {
    const converted = convertPackedSpanRow(
      { id: `legacy-span-${augKind}`, segments: legacySpans[0].segments },
      { augmentRecovery: augKind },
    );
    assert.equal(converted.ok, true, `convert failed for ${augKind}`);
    assert.equal(converted.convertedReads, 1);
    assert.equal(converted.convertedEdits, 2);
    const fullConvertedText = converted.row.segments.map((s) => s.text).join("");
    assert.match(fullConvertedText, /=== calc\.py \(lines 1-5 of 5\) ===\n1: def add\(a, b\):\n2:     return a - b/);
    // Multi-hunk preserved as ONE atomic edits[] call (zero synthetic "Apply edit hunk 2/2" turns!)
    assert.doesNotMatch(fullConvertedText, /Apply edit hunk/);
    assert.match(fullConvertedText, /"edits":\[\{"start_line":2,"end_line":2,"start_text":"    return a - b","new_text":"    return a \+ b"\},\{"start_line":5,"end_line":5,"start_text":"    return a \+ b","new_text":"    return a - b"\}\]/);
    // CoT-Action alignment grounded the supervised `<think>` with exact line numbers & start_text!
    assert.match(fullConvertedText, /Target anchors in calc\.py \(2 hunks: line 2 \(`return a - b`\); line 5 \(`return a \+ b`\)\)/);
    assert.doesNotMatch(fullConvertedText, /"oldText"/);

    // Off-policy recovery puts train:false on simulated mistake and train:true on recovery!
    assert.ok(converted.augmented, `expected recovery augmentation for ${augKind}`);
    const untrained = converted.augmented.segments.filter((s) => !s.train).map((s) => s.text).join("\n");
    const trained = converted.augmented.segments.filter((s) => s.train).map((s) => s.text).join("\n");
    assert.doesNotMatch(trained, /call_sim_/);
    assert.match(trained, /"start_line":2,"end_line":2,"start_text":"    return a - b"/);
    if (augKind === "unread_first") assert.match(untrained, /Cannot edit calc\.py before reading/);
    if (augKind === "anchor_mismatch") assert.match(untrained, /start_text \("# wrong anchor"\) did not match/);
    if (augKind === "boundary_dup") assert.match(untrained, /duplicates line 1/);
    if (augKind === "noop_recovery") assert.match(untrained, /No edit needed in calc\.py/);
  }

  // Verify upgraded Epoch Ledger text includes explicit re-read guidance
  const folded = foldEpochRequest({
    messages: [
      { role: "system", content: "You are pi." },
      { role: "user", content: "Fix." },
      { role: "assistant", content: null, tool_calls: [{ id: "r1", type: "function", function: { name: "read", arguments: '{"path":"big.py"}' } }] },
      { role: "tool", tool_call_id: "r1", content: "x".repeat(8000) },
      { role: "assistant", content: null, tool_calls: [{ id: "e1", type: "function", function: { name: "edit", arguments: '{"path":"big.py","edits":[{"start_line":1,"end_line":1,"start_text":"x","new_text":"y"}]}' } }] },
      { role: "tool", tool_call_id: "e1", content: "File updated." },
      { role: "assistant", content: null, tool_calls: [{ id: "b1", type: "function", function: { name: "bash", arguments: '{"command":"pytest"}' } }] },
      { role: "tool", tool_call_id: "b1", content: "ok\nexit code: 0" },
    ],
    tools: [],
  }, { budgetTokens: 1200, keepRecentPairs: 1 });
  const ledgerMsg = JSON.stringify(folded.request.messages);
  assert.match(ledgerMsg, /Inspected \(content folded; call read with offset\/limit before editing\): read big\.py/);
});

test("convertTrajectoryToLineEdit merges disjoint read windows and synthesizes post-fold read across multi-epoch spans", () => {
  const history = new Map();

  // Span 0: disjoint reads (lines 1-3 and lines 4-5) followed by an edit on line 5
  const span0 = convertTrajectoryToLineEdit({
    system: "You are pi.",
    tools: [{ name: "read", description: "r", parameters: {} }, { name: "edit", description: "e", parameters: {} }],
    messages: [
      { role: "system", content: "You are pi." },
      { role: "user", content: "Update mod.py." },
      {
        role: "assistant",
        content: null,
        reasoning_content: "Read header and body.",
        tool_calls: [
          { id: "r_a", type: "function", function: { name: "read", arguments: '{"path":"mod.py","offset":1,"limit":3}' } },
          { id: "r_b", type: "function", function: { name: "read", arguments: '{"path":"mod.py","offset":4,"limit":2}' } },
        ],
      },
      { role: "tool", tool_call_id: "r_a", content: "def a():\n    return 1\n\n[Showing lines 1-3 of 5]" },
      { role: "tool", tool_call_id: "r_b", content: "def b():\n    return 2\n[Showing lines 4-5 of 5]" },
      {
        role: "assistant",
        content: null,
        reasoning_content: "Change b().",
        tool_calls: [{
          id: "e_1",
          type: "function",
          function: { name: "edit", arguments: JSON.stringify({ path: "mod.py", edits: [{ oldText: "    return 2", newText: "    return 20" }] }) },
        }],
      },
      { role: "tool", tool_call_id: "e_1", content: "Successfully replaced text in mod.py." },
    ],
    supervisedIndices: new Set([2, 5]),
  }, { trajectoryHistory: history });

  assert.equal(span0.ok, true);
  assert.equal(span0.convertedReads, 2);
  assert.equal(span0.convertedEdits, 1);
  assert.equal(span0.synthesizedPostFoldReads, 0);

  // Span 1 (after Epoch fold): `mod.py` reads were folded into State Ledger; teacher tries a supervised edit on line 2 (`return 1` -> `return 10`).
  const span1 = convertTrajectoryToLineEdit({
    system: "You are pi.",
    tools: [{ name: "read", description: "r", parameters: {} }, { name: "edit", description: "e", parameters: {} }],
    messages: [
      { role: "system", content: "You are pi." },
      { role: "user", content: "[State Ledger (earlier steps folded):\n- Modified files: mod.py (edit: ok)\n- Inspected: read mod.py]" },
      {
        role: "assistant",
        content: null,
        reasoning_content: "Now update a() as well.",
        tool_calls: [{
          id: "e_2",
          type: "function",
          function: { name: "edit", arguments: JSON.stringify({ path: "mod.py", edits: [{ oldText: "    return 1", newText: "    return 10" }] }) },
        }],
      },
      { role: "tool", tool_call_id: "e_2", content: "Successfully replaced text in mod.py." },
    ],
    supervisedIndices: new Set([2]),
  }, { trajectoryHistory: history });

  assert.equal(span1.ok, true);
  assert.equal(span1.synthesizedPostFoldReads, 1);
  assert.equal(span1.convertedEdits, 1);
  const supTurns = [...span1.supervisedIndices].sort((a, b) => a - b).map((idx) => span1.messages[idx]);
  assert.equal(supTurns.length, 2);
  assert.equal(supTurns[0].tool_calls[0].function.name, "read");
  assert.equal(supTurns[1].tool_calls[0].function.name, "edit");
  assert.deepEqual(JSON.parse(supTurns[1].tool_calls[0].function.arguments), {
    path: "mod.py",
    edits: [{
      start_line: 2,
      end_line: 2,
      start_text: "    return 1",
      new_text: "    return 10",
    }],
  });
  assert.match(supTurns[1].reasoning_content, /Target anchor in mod\.py: line 2 \(starting with `return 1`\)/);
});

test("convertTrajectoryToLineEdit losslessly upgrades legacy v1 line-edit trajectories ([File:] + single-hunk) to v2 anchored batch format", () => {
  const res = convertTrajectoryToLineEdit({
    system: "You are pi.\n- edit: Replace, insert, or delete numbered line ranges in a file",
    tools: [{ name: "read", description: "r", parameters: {} }, { name: "edit", description: "e", parameters: {} }],
    messages: [
      { role: "system", content: "You are pi.\n- edit: Replace, insert, or delete numbered line ranges in a file" },
      { role: "user", content: "Fix calc.py." },
      {
        role: "assistant",
        content: null,
        reasoning_content: "读取 calc.py。",
        tool_calls: [{ id: "r1", type: "function", function: { name: "read", arguments: '{"path":"calc.py"}' } }],
      },
      { role: "tool", tool_call_id: "r1", content: "[File: calc.py (2 lines total)]\n1: def add(a, b):\n2:     return a - b" },
      {
        role: "assistant",
        content: null,
        reasoning_content: "修复减号为加号。",
        tool_calls: [{ id: "e1", type: "function", function: { name: "edit", arguments: '{"path":"calc.py","start_line":2,"end_line":2,"new_text":"    return a + b"}' } }],
      },
      { role: "tool", tool_call_id: "e1", content: "File updated. Replaced line 2 with 1 line.\n[File: calc.py (2 lines total)]\n1: def add(a, b):\n2:     return a + b" },
    ],
    supervisedIndices: new Set([2, 4]),
  });
  assert.equal(res.ok, true);
  assert.equal(res.convertedReads, 1);
  assert.equal(res.convertedEdits, 1);
  assert.match(res.messages[3].content, /^=== calc\.py \(lines 1-2 of 2\) ===/);
  assert.deepEqual(JSON.parse(res.messages[4].tool_calls[0].function.arguments), {
    path: "calc.py",
    edits: [{ start_line: 2, end_line: 2, start_text: "    return a - b", new_text: "    return a + b" }],
  });
  assert.match(res.messages[4].reasoning_content, /确认目标位置：calc\.py 第 2 行/);
  assert.match(res.messages[5].content, /=== calc\.py \(lines 1-2 of 2\) ===/);
});
