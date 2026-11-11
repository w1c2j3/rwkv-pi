// RWKV subagent manager: real pi subprocesses (pi-next source) against a mock Lightning endpoint.
// Protocol fixture only; says nothing about model quality.
import assert from "node:assert/strict";
import test from "node:test";
import { once } from "node:events";
import { createServer } from "node:http";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { SubagentManager } from "../dist/subagents.js";

const answer = (prompt) => {
  if (prompt.includes("WRITE-TASK")) {
    return /Successfully wrote/i.test(prompt) ? "已写入 out.txt。"
      : '<tool_call>{"name":"write","arguments":{"path":"out.txt","content":"written by subagent\\n"}}</tool_call>';
  }
  if (prompt.includes("LOOP-TASK")) return '<tool_call>{"name":"read","arguments":{"path":"missing.txt"}}</tool_call>';
  return prompt.includes("HELLO") ? "input.txt 的内容是 HELLO。"
    : '<tool_call>{"name":"read","arguments":{"path":"input.txt"}}</tool_call>';
};

async function lightning(t) {
  const server = createServer(async (req, res) => {
    let raw = ""; for await (const part of req) raw += part;
    if (req.url.endsWith("/tokens/count")) { res.setHeader("content-type", "application/json"); res.end('{"tokens":100}'); return; }
    if (req.url.endsWith("/server/stop")) { res.end("{}"); return; }
    const text = answer(JSON.parse(raw).contents[0]);
    res.writeHead(200, { "content-type": "text/event-stream" });
    const emit = (delta, finish_reason = null) => res.write(`data: ${JSON.stringify({ id: "raw", object: "chat.completion.chunk", choices: [{ index: 0, delta, finish_reason, ...(finish_reason ? { matched_stop: "<EOD>" } : {}) }] })}\n\n`);
    emit({ content: "</think>\n" + text + "<EOD>" }); emit({}, "stop"); res.write("data: [DONE]\n\n"); res.end();
  });
  server.listen(0, "127.0.0.1"); await once(server, "listening");
  t.after(() => new Promise((r) => { server.close(r); server.closeAllConnections(); }));
  return `http://127.0.0.1:${server.address().port}/v1`;
}

test("parallel read and write subagents, budget stop, copy-only changes and conflict-safe apply", { timeout: 240000 }, async (t) => {
  const url = await lightning(t);
  const base = await mkdtemp(join(tmpdir(), "rwkv-subagents-test-"));
  if (!process.env.KEEP) t.after(() => rm(base, { recursive: true, force: true }));
  const ws = join(base, "ws"), data = join(base, "data");
  await mkdir(ws); await mkdir(data);
  await writeFile(join(ws, "input.txt"), "HELLO\n");
  Object.assign(process.env, { RWKV_BASE_URL: url, RWKV_MODEL: "mock", RWKV_CONTEXT_WINDOW: "32768", RWKV_MAX_TOKENS: "4096",
    RWKV_PI_SOURCE: resolve(".upstream/pi-next"), RWKV_TOOL_CALL_RECOVERY: "1" });
  const m = new SubagentManager(ws, { concurrency: 4, maxRequests: 6, timeoutSeconds: 180, maxTasks: 8, sandbox: "none", mount: "" }, data);
  m.start([
    { name: "reader", task: "读取 input.txt 并告诉我内容。" },
    { name: "writer", task: "WRITE-TASK 请创建 out.txt。", write: true },
    { name: "looper", task: "LOOP-TASK 读取 missing.txt。", maxRequests: 3 },
  ]);
  const [reader, writer, looper] = await m.wait(undefined, 200000);
  assert.equal(reader.state, "completed", reader.error);
  assert.match(reader.answer, /HELLO/);
  assert.equal(writer.state, "completed", writer.error);
  assert.deepEqual(writer.changes, [{ path: "out.txt", kind: "added" }]);
  // The write subagent worked on a copy: nothing reaches the workspace before apply().
  await assert.rejects(readFile(join(ws, "out.txt")));
  assert.ok(["failed", "budget"].includes(looper.state), looper.state);
  assert.match(m.diff(writer.id), /\+written by subagent/);
  // A file changed in the workspace after the copy was made is a conflict, not overwritten.
  await writeFile(join(ws, "out.txt"), "user edit\n");
  assert.deepEqual(m.apply(writer.id).conflicts, ["out.txt"]);
  assert.equal(await readFile(join(ws, "out.txt"), "utf8"), "user edit\n");
  await rm(join(ws, "out.txt"));
  assert.deepEqual(m.apply(writer.id), { applied: [{ path: "out.txt", kind: "added" }], conflicts: [] });
  assert.equal(await readFile(join(ws, "out.txt"), "utf8"), "written by subagent\n");
  const result = JSON.parse(await readFile(join(writer.dir, "result.json"), "utf8"));
  assert.equal(result.applied, true);
});
