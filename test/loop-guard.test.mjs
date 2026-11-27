import assert from "node:assert/strict";
import test from "node:test";
import { repeatedIdenticalCall, repeatedToolFailure } from "../dist/loop-guard.js";
import { readConfig } from "../dist/config.js";

const failedTurn = (id, args = { path: "/project" }, text = "EISDIR: illegal operation on a directory, read", isError = true) => [
  { role: "assistant", stopReason: "toolUse", content: [
    { type: "text", text: `Attempt ${id}` },
    { type: "toolCall", id, name: "read", arguments: args },
  ] },
  { role: "toolResult", toolCallId: id, toolName: "read", isError, content: [{ type: "text", text }] },
];

test("different call IDs and assistant wording do not hide identical failed operations", () => {
  const history = [
    ...failedTurn("a", { path: "/project", limit: 5 }),
    ...failedTurn("b", { limit: 5, path: "/project" }),
    ...failedTurn("c", { path: "/project", limit: 5 }),
  ];
  assert.equal(repeatedToolFailure(history.slice(0, 4)), undefined);
  assert.match(repeatedToolFailure(history), /RWKV_TOOL_LOOP: read/);
  assert.equal(repeatedToolFailure(history, 0), undefined);
  assert.equal(repeatedToolFailure(history, 4), undefined);
});

test("user corrections, different arguments, and changed observations interrupt a failure streak", () => {
  const first = [...failedTurn("a"), ...failedTurn("b")];
  assert.equal(repeatedToolFailure([...first, { role: "user", content: "Try another approach" }]), undefined);
  assert.equal(repeatedToolFailure([...first, { role: "user", content: "Try again" }, ...failedTurn("c")]), undefined);
  assert.equal(repeatedToolFailure([...first, ...failedTurn("c", { path: "/project/README.md" })]), undefined);
  assert.equal(repeatedToolFailure([...first, ...failedTurn("c", { path: "/project" }, "ENOENT")]), undefined);
});

test("successful repeated reads, mixed batches, and unmatched results are not diagnosed as this loop", () => {
  const success = ["a", "b", "c"].flatMap(id => failedTurn(id, { path: "README.md" }, "contents", false));
  assert.equal(repeatedToolFailure(success), undefined);
  const mixed = ["a", "b", "c"].flatMap(id => failedTurn(id));
  mixed.at(-2).content.push({ type: "toolCall", id: "other", name: "bash", arguments: { command: "ls" } });
  assert.equal(repeatedToolFailure(mixed), undefined);
  const unmatched = ["a", "b", "c"].flatMap(id => failedTurn(id));
  unmatched.at(-1).toolCallId = "missing";
  assert.equal(repeatedToolFailure(unmatched), undefined);
});

test("failure-loop threshold has an explicit opt-out and validates configuration", () => {
  assert.equal(readConfig({ RWKV_BACKEND: "openai",}).repeatFailureLimit, 3);
  assert.equal(readConfig({ RWKV_BACKEND: "openai", RWKV_REPEAT_FAILURE_LIMIT: "0" }).repeatFailureLimit, 0);
  for (const value of ["1", "-1", "2.5", "NaN", "Infinity"]) {
    assert.throws(() => readConfig({ RWKV_BACKEND: "openai", RWKV_REPEAT_FAILURE_LIMIT: value }), /RWKV_REPEAT_FAILURE_LIMIT/);
  }
});

test("repeatedIdenticalCall blocks period-2 ping-pong loops (including mixed ok/error) without blocking real edit-test cycles", () => {
  const turn = (id, name, args, text, isError = false) => [
    { role: "assistant", stopReason: "toolUse", content: [{ type: "toolCall", id, name, arguments: args }] },
    { role: "toolResult", toolCallId: id, toolName: name, isError, content: [{ type: "text", text }] },
  ];

  // Ping-pong A -> B -> A -> B -> [A] (even when B failed with exit 2 and A succeeded with exit 0)
  const pingPong = [
    ...turn("1", "bash", { command: "ls -la /home/dev/" }, "dir listing", false),
    ...turn("2", "bash", { command: "ls -la /home/dev/frontend" }, "No such file\nCommand exited with code 2", true),
    ...turn("3", "bash", { command: "ls -la /home/dev/" }, "dir listing", false),
    ...turn("4", "bash", { command: "ls -la /home/dev/frontend" }, "No such file\nCommand exited with code 2", true),
  ];
  assert.equal(repeatedIdenticalCall(pingPong.slice(0, 4), "bash", { command: "ls -la /home/dev/" }, 2), undefined);
  assert.match(repeatedIdenticalCall(pingPong, "bash", { command: "ls -la /home/dev/" }, 2), /RWKV_REPEAT_GUARD: bash 与另一操作已交替重复 2 轮/);

  // Real test -> edit1 -> test -> edit2 -> [test] must NOT be blocked because edit arguments change!
  const editTestCycle = [
    ...turn("1", "bash", { command: "npm test" }, "1 failing", false),
    ...turn("2", "edit", { path: "a.js", oldText: "x", newText: "y" }, "Edited a.js", false),
    ...turn("3", "bash", { command: "npm test" }, "1 failing", false),
    ...turn("4", "edit", { path: "a.js", oldText: "y", newText: "z" }, "Edited a.js", false),
  ];
  assert.equal(repeatedIdenticalCall(editTestCycle, "bash", { command: "npm test" }, 2), undefined);

  // 3rd identical failing call is intercepted by repeatedIdenticalCall before execution (giving a recovery turn)
  const twoFails = [
    ...turn("f1", "edit", { path: "a.py", oldText: "same", newText: "same" }, "No edit needed", true),
    ...turn("f2", "edit", { path: "a.py", oldText: "same", newText: "same" }, "No edit needed", true),
  ];
  assert.equal(repeatedIdenticalCall(twoFails.slice(0, 2), "edit", { path: "a.py", oldText: "same", newText: "same" }, 2), undefined);
  assert.match(repeatedIdenticalCall(twoFails, "edit", { path: "a.py", oldText: "same", newText: "same" }, 2), /RWKV_REPEAT_GUARD: edit 已连续 2 次以完全相同的参数返回相同错误/);
});
