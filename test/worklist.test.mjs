import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { buildDataset } from "../dist/dataset.js";
import { runTask } from "../dist/harness.js";
import { repeatedWorklistSpin } from "../dist/loop-guard.js";
import { renderPrompt } from "../dist/prompt.js";
import { createRwkvProvider, readConfig } from "../dist/provider.js";
import {
  applyTodoPlan,
  applyTodoUpdate,
  foldWorklistRequest,
  formatWorklist,
  PiWorklistState,
  TODO_PLAN_TOOL,
  TODO_UPDATE_TOOL,
  WORKLIST_CHECKPOINT_NUDGE,
  WORKLIST_INIT_NUDGE,
} from "../dist/worklist.js";
import { mockServer, toolCall } from "./helpers/mock-server.mjs";

const openAiTools = [
  { type: "function", function: { name: "read", parameters: { type: "object", properties: { path: { type: "string" } }, required: ["path"] } } },
  { type: "function", function: { name: "write", parameters: { type: "object", properties: { path: { type: "string" }, content: { type: "string" } }, required: ["path", "content"] } } },
  { type: "function", function: { name: "edit", parameters: { type: "object", properties: { path: { type: "string" }, oldText: { type: "string" }, newText: { type: "string" } }, required: ["path", "oldText", "newText"] } } },
  { type: "function", function: { name: "bash", parameters: { type: "object", properties: { command: { type: "string" } }, required: ["command"] } } },
  { type: "function", function: { name: TODO_PLAN_TOOL, parameters: { type: "object", properties: { items: { type: "array", items: { type: "string" } } }, required: ["items"] } } },
  { type: "function", function: { name: TODO_UPDATE_TOOL, parameters: { type: "object", properties: { id: { type: "integer" }, status: { type: "string" }, note: { type: "string" }, content: { type: "string" } }, required: ["id", "status"] } } },
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

test("worklist state machine: plan, closed-item lock, note length cap, auto-advance, and back-to-back complete guard", () => {
  const initial = applyTodoPlan([], ["1. Update JWT TTL to 3600s", "2. Add expiry unit test"]);
  assert.ok(Array.isArray(initial));
  assert.equal(initial.length, 2);
  assert.equal(initial[0].content, "Update JWT TTL to 3600s");
  assert.equal(initial[0].status, "in_progress");
  assert.equal(initial[1].status, "pending");

  assert.match(String(applyTodoUpdate(initial, { id: 1, status: "completed" })), /note is required/);
  assert.match(String(applyTodoUpdate(initial, { id: 1, status: "completed", note: "x".repeat(601) })), /exceeds 600 characters/);

  const doneOne = applyTodoUpdate(initial, { id: 1, status: "completed", note: "src/auth/jwt.ts:18 DEFAULT_EXP 900->3600" });
  assert.equal(typeof doneOne, "object");
  assert.equal(doneOne.items[0].status, "completed");
  assert.equal(doneOne.items[1].status, "in_progress");

  // Regressing an already closed item is rejected.
  assert.match(String(applyTodoUpdate(doneOne.items, { id: 1, status: "in_progress", note: "oops" })), /already completed/);

  // PiWorklistState blocks two completions in a row without an intervening workspace tool.
  const state = new PiWorklistState();
  state.plan(["Step A", "Step B"]);
  state.update({ id: 1, status: "completed", note: "Step A verified" });
  assert.throws(() => state.update({ id: 2, status: "completed", note: "Hallucinated Step B" }), /no workspace tool call in between/);
  state.recordWorkspaceTool();
  const finalText = state.update({ id: 2, status: "completed", note: "npm test exit 0" });
  assert.match(finalText, /All todo items are finished/);
});

test("worklist folding: foldTriggerTokens preserves Prefix-Cache below threshold, then folds pre-checkpoint drafts while pinning Working Set read", () => {
  const messages = [
    { role: "system", content: "You are pi." },
    { role: "user", content: "Change JWT TTL and run tests." },
    ...callPair("c1", TODO_PLAN_TOOL, { items: ["Inspect & edit JWT", "Add test"] }, "Todo list:\n[1] (in_progress) Inspect & edit JWT\n[2] (pending) Add test", "OPENING_PLAN_REASONING"),
    ...callPair("c2", "read", { path: "src/old.ts" }, "// FOLDED_OLD_FILE\n" + "const padding = 1;\n".repeat(120), "Read old."),
    ...callPair("c3", "read", { path: "src/auth/jwt.ts" }, "18: export const DEFAULT_EXP = 900; // PINNED_WORKING_SET", "Read target."),
    ...callPair("c4", "bash", { command: "grep -rn DEFAULT_EXP src" }, "src/auth/jwt.ts:18", "Search refs."),
    ...callPair("c5", "bash", { command: "npm test" }, "passed\nexit code: 0", "Baseline test."),
    ...callPair("c6", TODO_UPDATE_TOOL, { id: 1, status: "completed", note: "src/auth/jwt.ts:18 DEFAULT_EXP=900" }, formatWorklist([
      { id: 1, status: "completed", content: "Inspect & edit JWT", note: "src/auth/jwt.ts:18 DEFAULT_EXP=900" },
      { id: 2, status: "in_progress", content: "Add test" },
    ]), "CHECKPOINT_REASONING"),
  ];

  // With foldTriggerTokens high (e.g. 50,000), prompt stays 100% Append-Only for Lightning Prefix-Cache.
  const buffered = foldWorklistRequest({ messages, tools: openAiTools }, { foldTriggerTokens: 50000, budgetTokens: 8000 });
  const bufferedPrompt = renderPrompt(buffered.request.messages, openAiTools, true);
  assert.match(bufferedPrompt, /FOLDED_OLD_FILE/);

  // Once unfolded tokens exceed foldTriggerTokens (or foldTriggerTokens=0), pre-checkpoint drafts fold out cleanly:
  const folded = foldWorklistRequest({ messages, tools: openAiTools }, { foldTriggerTokens: 0, budgetTokens: 8000, keepRecentPairs: 2 });
  const foldedPlanMsg = folded.request.messages.find((m) => m.role === "assistant" && Array.isArray(m.tool_calls) && m.tool_calls[0]?.id === "c1");
  assert.equal(foldedPlanMsg.reasoning_content, "", "stripped tool-call assistant preserves empty reasoning_content for DeepSeek thinking APIs");
  const foldedPrompt = renderPrompt(folded.request.messages, openAiTools, true);
  assert.doesNotMatch(foldedPrompt, /FOLDED_OLD_FILE/, "old exploratory read is folded out");
  assert.doesNotMatch(foldedPrompt, /OPENING_PLAN_REASONING/, "discontinuous pinned plan strips stale opening CoT");
  assert.match(foldedPrompt, /PINNED_WORKING_SET/, "unedited Working Set file read stays pinned across 2 bash steps");
  assert.match(foldedPrompt, /CHECKPOINT_REASONING/, "active checkpoint reasoning stays intact");
});

test("worklist folding: non-withdrawn budget reminders preserve strict prompt prefix continuity and cross-turn Tail Digest", () => {
  const big = "export const line = 1234567890;\n".repeat(160);
  const step1Msgs = [
    { role: "system", content: "You are pi." },
    { role: "user", content: "Turn 1: fix config." },
    ...callPair("u1", "read", { path: "big.ts" }, big, "Read big.ts."),
  ];
  const f1 = foldWorklistRequest({ messages: step1Msgs, tools: openAiTools }, { budgetTokens: 900, hardCapTokens: 8000, foldTriggerTokens: 0 });
  assert.equal(f1.nudged, true);
  const p1 = renderPrompt(f1.request.messages, openAiTools, true);
  assert.ok(p1.includes(WORKLIST_INIT_NUDGE));

  // Next step skips todo_plan and writes directly -> INIT nudge is NOT withdrawn, preserving exact prefix bytes!
  const step2Msgs = [
    ...step1Msgs,
    ...callPair("u2", "write", { path: "big.ts", content: "ok\n" }, "Wrote big.ts", "Quick 1-step fix."),
  ];
  const f2 = foldWorklistRequest({ messages: step2Msgs, tools: openAiTools }, { budgetTokens: 900, hardCapTokens: 8000, foldTriggerTokens: 0 });
  const p2 = renderPrompt(f2.request.messages, openAiTools, true);
  const p1Prefix = p1.slice(0, p1.length - "Assistant: <think".length);
  assert.ok(p2.startsWith(p1Prefix), "step 2 prompt strictly extends step 1 prefix bytes (reminder not withdrawn)");

  // Multi-turn Tail Digest when Turn 1 executes write + bash AFTER its last todo_update:
  const multiTurnMsgs = [
    { role: "system", content: "You are pi." },
    { role: "user", content: "Turn 1: update config and test." },
    ...callPair("t1p", TODO_PLAN_TOOL, { items: ["Update config", "Run tests"] }, "Todo list..."),
    ...callPair("t1u", TODO_UPDATE_TOOL, { id: 1, status: "completed", note: "config.ts:4 port=8080" }, "Todo list:\n[1] (completed) Update config\n  note: config.ts:4 port=8080\n[2] (in_progress) Run tests"),
    ...callPair("t1w", "write", { path: "test/config.test.ts", content: "ok" }, "Wrote test/config.test.ts"),
    ...callPair("t1b", "bash", { command: "npm test" }, "all green\nexit code: 0"),
    { role: "assistant", content: "Updated config and verified tests." },
    { role: "user", content: "Turn 2: also expose HOST env var." },
  ];
  const fTurn2 = foldWorklistRequest({ messages: multiTurnMsgs, tools: openAiTools }, { budgetTokens: 8000, foldTriggerTokens: 0 });
  const pTurn2 = renderPrompt(fTurn2.request.messages, openAiTools, true);
  assert.match(pTurn2, /Workspace steps after last todo update in this turn/);
  assert.match(pTurn2, /- write test\/config\.test\.ts -> ok/);
  assert.match(pTurn2, /- bash npm test -> exit 0/);
});

test("worklist loop guard blocks 3rd consecutive bookkeeping call", () => {
  const piMsgs = [
    { role: "user", content: [{ type: "text", text: "Fix bug" }] },
    { role: "assistant", stopReason: "toolUse", content: [{ type: "toolCall", id: "c1", name: TODO_PLAN_TOOL, arguments: { items: ["A", "B"] } }] },
    { role: "toolResult", toolCallId: "c1", toolName: TODO_PLAN_TOOL, isError: false, content: [{ type: "text", text: "Todo list..." }] },
    { role: "assistant", stopReason: "toolUse", content: [{ type: "toolCall", id: "c2", name: TODO_UPDATE_TOOL, arguments: { id: 1, status: "in_progress", note: "p" } }] },
    { role: "toolResult", toolCallId: "c2", toolName: TODO_UPDATE_TOOL, isError: false, content: [{ type: "text", text: "Todo list..." }] },
  ];
  const blocked = repeatedWorklistSpin(piMsgs, TODO_UPDATE_TOOL);
  assert.ok(blocked && blocked.includes("RWKV_WORKLIST_SPIN"));
  assert.equal(repeatedWorklistSpin(piMsgs, "read"), undefined);
});

test("end-to-end harness + buildDataset SFT export with RWKV_HISTORY_MODE=worklist", async (t) => {
  let server;
  try {
    server = await mockServer((_body, index) => {
      if (index === 1) return { deltas: [{ reasoning_content: "Plan two closed-loop steps." }, toolCall(TODO_PLAN_TOOL, { items: ["Read input and write uppercase output", "Verify output file"] }, "c1")], finish: "tool_calls" };
      if (index === 2) return { deltas: [{ reasoning_content: "Read input.txt." }, toolCall("read", { path: "input.txt" }, "c2")], finish: "tool_calls" };
      if (index === 3) return { deltas: [{ reasoning_content: "Write uppercase." }, toolCall("write", { path: "output.txt", content: "HELLO WORKLIST\n" }, "c3")], finish: "tool_calls" };
      if (index === 4) return { deltas: [{ reasoning_content: "Checkpoint #1." }, toolCall(TODO_UPDATE_TOOL, { id: 1, status: "completed", note: "input.txt read; wrote HELLO WORKLIST to output.txt" }, "c4")], finish: "tool_calls" };
      if (index === 5) return { deltas: [{ reasoning_content: "Verify output.txt." }, toolCall("read", { path: "output.txt" }, "c5")], finish: "tool_calls" };
      return { deltas: [{ reasoning_content: "Verified.", content: "Done: output.txt contains HELLO WORKLIST." }] };
    });
  } catch (error) {
    if (error && (error.code === "EPERM" || error.code === "EACCES")) {
      t.skip("sandbox forbids local TCP listen");
      return;
    }
    throw error;
  }
  const outDir = await mkdtemp(join(tmpdir(), "rwkv-pi-worklist-"));
  try {
    const config = readConfig({
      RWKV_BACKEND: "openai",
      RWKV_BASE_URL: server.url,
      RWKV_MODEL: "mock-only",
      RWKV_HISTORY_MODE: "worklist",
      RWKV_WORKLIST_FOLD_TRIGGER_TOKENS: "0",
      RWKV_WORKLIST_KEEP_RECENT_PAIRS: "1",
    });
    const provider = createRwkvProvider(config);
    const task = {
      schemaVersion: 1,
      id: "worklist-e2e-1",
      family: "worklist-e2e-family",
      split: "train",
      source: "unit-test",
      prompt: "Read input.txt and write its uppercase text to output.txt.",
      files: { "input.txt": "hello worklist\n" },
      tools: ["read", "write"],
      checks: [{ type: "file_equals", path: "output.txt", expected: "HELLO WORKLIST\n" }],
    };
    const { record } = await runTask(task, { provider, role: "student", outputDir: outDir, maxRequests: 8, timeoutMs: 15000, sandbox: "none" });
    assert.equal(record.status, "passed");
    assert.equal(record.historyMode, "worklist");

    // Step 5 (index 4, right after c4 todo_update) must have folded out c2 ("read input.txt") while keeping c1 (todo_plan), c3 (latest 1 pair: write), and c4 (todo_update)!
    const step5Msgs = record.steps[4].request.messages;
    assert.match(step5Msgs[0].content, /- todo_plan:/, "worklist tools appear in Pi <tools> prompt section");
    assert.match(step5Msgs[0].content, /- todo_update:/, "worklist tools appear in Pi <tools> prompt section");
    const step5Prompt = renderPrompt(step5Msgs, record.steps[4].request.tools, true);
    assert.doesNotMatch(step5Prompt, /"id":"c2"/);
    assert.doesNotMatch(step5Prompt, /hello worklist/, "folded step 2 read output is removed from step 5 prompt");
    assert.match(step5Prompt, /input\.txt read; wrote HELLO WORKLIST to output\.txt/, "checkpoint note is present");

    // Every single step in run.steps must pass buildDataset's strict validateMessages() without orphan tool calls!
    const dataset = buildDataset([record]);
    assert.equal(dataset.rejected.length, 0);
    assert.equal(dataset.samples.length, 6);
  } finally {
    await server.close();
    await rm(outDir, { recursive: true, force: true });
  }
});
