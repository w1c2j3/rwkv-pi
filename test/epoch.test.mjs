import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { buildDataset } from "../dist/dataset.js";
import { DEFAULT_EPOCH_BUDGET_TOKENS, foldEpochRequest } from "../dist/epoch.js";
import { runTask } from "../dist/harness.js";
import { renderPrompt } from "../dist/prompt.js";
import { createRwkvProvider, readConfig } from "../dist/provider.js";
import { createTeacherProvider, readTeacherConfig } from "../dist/teacher.js";
import { runTask as runTeacherHarnessTask } from "../pipeline/teacher-harness.mjs";
import { packTrajectory } from "../pipeline/trajectory-packing.mjs";
import { mockServer, toolCall } from "./helpers/mock-server.mjs";

const piBaseTools = [
  { type: "function", function: { name: "read", parameters: { type: "object", properties: { path: { type: "string" } }, required: ["path"] } } },
  { type: "function", function: { name: "write", parameters: { type: "object", properties: { path: { type: "string" }, content: { type: "string" } }, required: ["path", "content"] } } },
  { type: "function", function: { name: "edit", parameters: { type: "object", properties: { path: { type: "string" }, oldText: { type: "string" }, newText: { type: "string" } }, required: ["path", "oldText", "newText"] } } },
  { type: "function", function: { name: "bash", parameters: { type: "object", properties: { command: { type: "string" } }, required: ["command"] } } },
];

function callPair(id, name, args, resultText, reasoning = "") {
  return [
    {
      role: "assistant",
      content: null,
      ...(reasoning ? { reasoning_content: reasoning } : {}),
      tool_calls: [{ id, type: "function", function: { name, arguments: JSON.stringify(args) } }],
    },
    { role: "tool", tool_call_id: id, content: resultText },
  ];
}

test("epoch mode: defaults to 9600 watermark and preserves 100% untouched baseline parity below watermark", () => {
  assert.equal(DEFAULT_EPOCH_BUDGET_TOKENS, 9600);
  const cfg = readConfig({ RWKV_HISTORY_MODE: "epoch" });
  assert.equal(cfg.historyMode, "epoch");
  assert.equal(cfg.epochBudgetTokens, 9600);
  assert.equal(cfg.epochKeepRecentPairs, 2);
  assert.equal(cfg.epochMaxResultTokens, 2000);
  assert.equal(cfg.worklistBudgetTokens, 9600);
  assert.equal(cfg.worklistFoldTriggerTokens, 6144);

  const req = {
    model: "rwkv7",
    messages: [
      { role: "system", content: "You are pi." },
      { role: "user", content: "Read input.txt." },
      ...callPair("c1", "read", { path: "input.txt" }, "hello\n", "Inspect input.txt."),
    ],
    tools: piBaseTools,
  };
  const out = foldEpochRequest(req);
  assert.equal(out.folded, false);
  assert.equal(out.epochCount, 0);
  assert.equal(out.request, req, "exact request object reference preserved when under 16k watermark");
});

test("epoch mode: folds older steps into deterministic State Ledger, pins unedited read, and keeps frozen Prefix-Cache across Epoch 2", () => {
  const big = "export const linePadding = 1234567890;\n".repeat(90);
  const baseMsgs = [
    { role: "system", content: "You are pi." },
    { role: "user", content: "Fix service and tests." },
    ...callPair("c1", "read", { path: "src/missing.ts" }, "Error: ENOENT: no such file src/missing.ts", "Check missing."),
    ...callPair("c2", "read", { path: "src/service.ts" }, `${big}\n// CONSUMED_SERVICE_READ`, "Read service."),
    ...callPair("c3", "edit", { path: "src/service.ts", oldText: "a", newText: "b" }, "Edited src/service.ts", "Patch service."),
    ...callPair("c4", "bash", { command: "npm test" }, `${big}\nFAIL test/service.test.ts\nCommand exited with code 1`, "Run test."),
    ...callPair("c5", "read", { path: "src/helper.ts" }, "12: export const FLAG = false; // PINNED_UNEDITED_HELPER", "Read helper."),
    ...callPair("c6", "bash", { command: "ls src" }, "service.ts\nhelper.ts\nexit code: 0", "List src."),
    ...callPair("c7", "bash", { command: "node -v" }, "v22.22.1\nexit code: 0", "Node version."),
  ];

  const epochOpts = { budgetTokens: 1700, keepRecentPairs: 1 };
  const f1 = foldEpochRequest({ messages: baseMsgs, tools: piBaseTools }, epochOpts);
  assert.equal(f1.folded, true);
  assert.ok(f1.epochCount >= 1);
  const p1 = renderPrompt(f1.request.messages, piBaseTools, true);
  assert.doesNotMatch(p1, /CONSUMED_SERVICE_READ/);
  assert.match(p1, /\[State Ledger \(earlier steps folded\):/);
  assert.match(p1, /Modified files: src\/service\.ts \(edit: ok\)/);
  assert.match(p1, /Latest command \(1 run\): npm test -> exit 1/);
  assert.match(p1, /Failed attempts \(do not repeat verbatim\): read src\/missing\.ts/);
  assert.match(p1, /PINNED_UNEDITED_HELPER/, "unedited Working Set read stays pinned across fold");

  // Verify multiple folded bash commands retain distinct earlier bash summaries so the model never forgets prior checks
  const multiBash = foldEpochRequest({
    messages: [
      { role: "system", content: "You are pi." },
      { role: "user", content: "Create test-rwkv." },
      ...callPair("b1", "bash", { command: "git status" }, "fatal: not a git repository\nCommand exited with code 128", "Check git."),
      ...callPair("b2", "bash", { command: "ls -la test-rwkv" }, "test-rwkv 不存在\nexit code: 0", "Check dir."),
      ...callPair("b3", "bash", { command: "ls -la ." }, `${big}\ntotal 76\nexit code: 0`, "List cwd."),
      ...callPair("b4", "bash", { command: "pwd" }, "/workspace\nexit code: 0", "Print cwd."),
    ],
    tools: piBaseTools,
  }, { budgetTokens: 1200, keepRecentPairs: 1 });
  const multiPrompt = renderPrompt(multiBash.request.messages, piBaseTools, true);
  assert.match(multiPrompt, /Earlier commands \(already run, do not repeat\):.*git status -> exit 128.*ls -la test-rwkv -> exit 0 \(test-rwkv 不存在\)/);

  // Verify DeepSeek Reasoner strict API invariants: ZERO consecutive user messages, and reasoning_content="" on stripped tool turns!
  for (let i = 1; i < f1.request.messages.length; i++) {
    assert.ok(
      !(f1.request.messages[i - 1].role === "user" && f1.request.messages[i].role === "user"),
      "must never emit consecutive user messages (breaks DeepSeek thinking API)",
    );
  }
  const pinnedAssistant = f1.request.messages.find((m) => m.role === "assistant" && m.tool_calls?.[0]?.id === "c5");
  assert.equal(pinnedAssistant?.reasoning_content, "", "stripped tool-call assistant turn keeps reasoning_content: '' for DeepSeek API");

  // Next steps in current Epoch (including a new `read`!) stay under watermark -> strictly extend p1/p2 prefix bytes!
  const step2Msgs = [
    ...baseMsgs,
    ...callPair("c8", "edit", { path: "src/helper.ts", oldText: "false", newText: "true" }, "Edited src/helper.ts", "Edit helper."),
  ];
  const f2 = foldEpochRequest({ messages: step2Msgs, tools: piBaseTools }, epochOpts);
  assert.equal(f2.epochCount, f1.epochCount, "does not re-fold inside current Epoch");
  const p2 = renderPrompt(f2.request.messages, piBaseTools, true);
  const p1Body = p1.slice(0, p1.length - "Assistant: <think".length);
  assert.ok(p2.startsWith(p1Body), "Step 2 strictly extends Epoch 2 prefix bytes");

  const step3Msgs = [
    ...step2Msgs,
    ...callPair("c9", "read", { path: "src/other.ts" }, "1: export const OTHER = 1;", "Read another file in Epoch 2."),
  ];
  const f3 = foldEpochRequest({ messages: step3Msgs, tools: piBaseTools }, epochOpts);
  const p3 = renderPrompt(f3.request.messages, piBaseTools, true);
  const p2Body = p2.slice(0, p2.length - "Assistant: <think".length);
  assert.ok(p3.startsWith(p2Body), "Subsequent read inside Epoch 2 keeps frozenPinnedRead invariant (100% intra-epoch Append-Only!)");
});

test("epoch mode: end-to-end harness + buildDataset SFT export with ZERO extra tools in prompt", async () => {
  const big = "line content for realistic module ".repeat(110);
  const server = await mockServer((_body, index) => {
    if (index === 1) return { deltas: [{ reasoning_content: "Read a.txt." }, toolCall("read", { path: "a.txt" }, "e1")], finish: "tool_calls" };
    if (index === 2) return { deltas: [{ reasoning_content: "Read input.txt." }, toolCall("read", { path: "input.txt" }, "e2")], finish: "tool_calls" };
    if (index === 3) return { deltas: [{ reasoning_content: "Write output.txt." }, toolCall("write", { path: "output.txt", content: "EPOCH OK\n" }, "e3")], finish: "tool_calls" };
    if (index === 4) return { deltas: [{ reasoning_content: "Verify output.txt." }, toolCall("read", { path: "output.txt" }, "e4")], finish: "tool_calls" };
    return { deltas: [{ reasoning_content: "Done.", content: "Wrote EPOCH OK." }] };
  });
  const outDir = await mkdtemp(join(tmpdir(), "rwkv-pi-epoch-"));
  try {
    const provider = createRwkvProvider(readConfig({
      RWKV_BACKEND: "openai",
      RWKV_BASE_URL: server.url,
      RWKV_MODEL: "mock-only",
      RWKV_HISTORY_MODE: "epoch",
      RWKV_EPOCH_BUDGET_TOKENS: "1800",
      RWKV_EPOCH_KEEP_RECENT_PAIRS: "1",
    }));
    const task = {
      schemaVersion: 1,
      id: "epoch-e2e-1",
      family: "epoch-e2e-family",
      split: "train",
      source: "unit-test",
      prompt: "Inspect a.txt and input.txt, then write EPOCH OK to output.txt.",
      files: { "a.txt": `${big}\nOLD_EXPLORATION_PAYLOAD\n`, "input.txt": "go\n" },
      tools: ["read", "write"],
      checks: [{ type: "file_equals", path: "output.txt", expected: "EPOCH OK\n" }],
    };
    const { record } = await runTask(task, { provider, role: "student", outputDir: outDir, maxRequests: 8, timeoutMs: 15000, sandbox: "none" });
    assert.equal(record.status, "passed");
    assert.equal(record.historyMode, "epoch");

    // Verify ZERO todo_plan / todo_update tools were injected into the prompt!
    const toolNames = record.steps[0].request.tools.map((t) => t.function.name);
    assert.deepEqual(toolNames, ["read", "write"]);

    // Verify step 4 folded out OLD_EXPLORATION_PAYLOAD into State Ledger!
    const finalPrompt = renderPrompt(record.steps[4].request.messages, record.steps[4].request.tools, true);
    assert.doesNotMatch(finalPrompt, /OLD_EXPLORATION_PAYLOAD/);
    assert.match(finalPrompt, /\[State Ledger \(earlier steps folded\):/);

    // Verify all steps pass buildDataset's strict validateMessages() AND pack into exact 2 continuous Epochs!
    const ds = buildDataset([record]);
    assert.equal(ds.rejected.length, 0);
    assert.equal(ds.samples.length, 5);
    const spans = packTrajectory(ds.samples, "xml");
    assert.equal(spans.length, 2, "5-step single-fold run packs into exact 2 continuous Lightning SFT spans (Epoch 1 + Epoch 2)");

    // Verify Teacher collection (`createTeacherProvider`) produces 100% identical Epoch folding & 2-span packing!
    const teacherProvider = createTeacherProvider(readTeacherConfig({
      TEACHER_BASE_URL: server.url,
      TEACHER_MODEL: "mock-teacher",
      TEACHER_API_KEY: "mock",
      TEACHER_HISTORY_MODE: "epoch",
      TEACHER_EPOCH_BUDGET_TOKENS: "1800",
      TEACHER_EPOCH_KEEP_RECENT_PAIRS: "1",
    }));
    const server2 = await mockServer((_body, index) => {
      if (index === 1) return { deltas: [{ reasoning_content: "Read a.txt." }, toolCall("read", { path: "a.txt" }, "t1")], finish: "tool_calls" };
      if (index === 2) return { deltas: [{ reasoning_content: "Read input.txt." }, toolCall("read", { path: "input.txt" }, "t2")], finish: "tool_calls" };
      if (index === 3) return { deltas: [{ reasoning_content: "Write output.txt." }, toolCall("write", { path: "output.txt", content: "EPOCH OK\n" }, "t3")], finish: "tool_calls" };
      if (index === 4) return { deltas: [{ reasoning_content: "Verify output.txt." }, toolCall("read", { path: "output.txt" }, "t4")], finish: "tool_calls" };
      return { deltas: [{ reasoning_content: "Done.", content: "Wrote EPOCH OK." }] };
    });
    try {
      const tp = createTeacherProvider(readTeacherConfig({
        TEACHER_BASE_URL: server2.url,
        TEACHER_MODEL: "mock-teacher",
        TEACHER_API_KEY: "mock",
        TEACHER_HISTORY_MODE: "epoch",
        TEACHER_EPOCH_BUDGET_TOKENS: "1800",
        TEACHER_EPOCH_KEEP_RECENT_PAIRS: "1",
      }));
      void teacherProvider;
      const { record: teacherRec } = await runTask({ ...task, id: "epoch-teacher-1" }, {
        provider: tp, role: "teacher", outputDir: join(outDir, "teacher"), maxRequests: 8, timeoutMs: 15000, sandbox: "none",
      });
      assert.equal(teacherRec.status, "passed");
      assert.equal(teacherRec.historyMode, "epoch");
      const teacherDs = buildDataset([teacherRec]);
      assert.equal(packTrajectory(teacherDs.samples, "xml").length, 2, "Teacher run packs into exact 2 continuous Epochs matching Student");
    } finally {
      await server2.close();
    }

    // Verify pipeline/teacher-harness.mjs (used by data_pipeline/collect.mjs) folds Epoch requests and records historyMode="epoch":
    const server3 = await mockServer((_body, index) => {
      if (index === 1) return { deltas: [{ reasoning_content: "Read a.txt." }, toolCall("read", { path: "a.txt" }, "p1")], finish: "tool_calls" };
      if (index === 2) return { deltas: [{ reasoning_content: "Read input.txt." }, toolCall("read", { path: "input.txt" }, "p2")], finish: "tool_calls" };
      if (index === 3) return { deltas: [{ reasoning_content: "Write output.txt." }, toolCall("write", { path: "output.txt", content: "EPOCH OK\n" }, "p3")], finish: "tool_calls" };
      if (index === 4) return { deltas: [{ reasoning_content: "Verify output.txt." }, toolCall("read", { path: "output.txt" }, "p4")], finish: "tool_calls" };
      return { deltas: [{ reasoning_content: "Done.", content: "Wrote EPOCH OK." }] };
    });
    try {
      const tp3 = createTeacherProvider(readTeacherConfig({
        TEACHER_BASE_URL: server3.url,
        TEACHER_MODEL: "rwkv-ds41f-1017",
        TEACHER_KIND: "deepseek",
        TEACHER_API_KEY: "mock",
      }));
      const { record: prodTeacherRec } = await runTeacherHarnessTask({ ...task, id: "epoch-prod-teacher-1" }, {
        provider: tp3,
        role: "teacher",
        outputDir: join(outDir, "prod-teacher"),
        maxRequests: 8,
        timeoutMs: 15000,
        sandbox: "none",
        historyMode: "epoch",
        epoch: { budgetTokens: 1800, keepRecentPairs: 1 },
      });
      assert.equal(prodTeacherRec.status, "passed");
      assert.equal(prodTeacherRec.historyMode, "epoch");
      const prodDs = buildDataset([prodTeacherRec]);
      assert.equal(prodDs.rejected.length, 0);
      assert.equal(packTrajectory(prodDs.samples, "xml").length, 2, "pipeline/teacher-harness.mjs packs into exact 2 continuous Epochs");
    } finally {
      await server3.close();
    }
  } finally {
    await server.close();
    await rm(outDir, { recursive: true, force: true });
  }
});
