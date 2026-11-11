import assert from "node:assert/strict";
import { once } from "node:events";
import { mkdtemp, rm } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { budgetRwkvRequest } from "../dist/token-budget.js";
import { createRwkvProvider, readConfig } from "../dist/provider.js";
import { runTask } from "../dist/harness.js";

async function endpoint(t) {
  const calls = [];
  const state = { count: 12000, context: 32768 };
  const server = createServer(async (req, res) => {
    let raw = "";
    for await (const chunk of req) raw += chunk;
    calls.push({ path: req.url, body: JSON.parse(raw) });
    if (req.url === "/tokenize") {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({ count: state.count, max_model_len: state.context, tokens: Array(state.count).fill(1) }));
    } else {
      res.writeHead(200, { "content-type": "text/event-stream" });
      res.write(`data: ${JSON.stringify({ id: "budget-mock", object: "chat.completion.chunk", model: "mock-only", created: 1,
        choices: [{ index: 0, delta: { content: "Done." }, finish_reason: "stop" }] })}\n\n`);
      res.end("data: [DONE]\n\n");
    }
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  t.after(() => new Promise(resolve => { server.close(resolve); server.closeAllConnections(); }));
  const config = readConfig({ RWKV_BACKEND: "openai", RWKV_BASE_URL: `http://127.0.0.1:${server.address().port}/v1`, RWKV_MODEL: "mock-only",
    RWKV_CONTEXT_WINDOW: "32768", RWKV_MAX_TOKENS: "24576" });
  return { calls, state, config };
}

test("counts the full chat prompt and reduces only output when history fills the context", async t => {
  const { config, calls, state } = await endpoint(t);
  const request = { model: "mock-only", messages: [{ role: "user", content: "中文😀" }],
    tools: [{ type: "function", function: { name: "read", parameters: { type: "object" } } }],
    chat_template_kwargs: { fake_cot_mode: false }, max_tokens: 24576, stream: true };
  const result = await budgetRwkvRequest(request, config, new AbortController().signal);
  assert.equal(result.request.max_tokens, 20768);
  assert.deepEqual(result.request.messages, request.messages);
  assert.deepEqual(result.request.tools, request.tools);
  assert.equal(request.max_tokens, 24576);
  assert.deepEqual(calls[0].body.chat_template_kwargs, request.chat_template_kwargs);
  assert.equal(calls[0].body.add_generation_prompt, true);
  assert.equal(result.budget.inputTokens + result.budget.outputTokens, 32768);
  state.context = 16384;
  await assert.rejects(budgetRwkvRequest(request, config, new AbortController().signal), /context mismatch/);
  state.context = state.count = 32768;
  await assert.rejects(budgetRwkvRequest(request, config, new AbortController().signal), /context exhausted/);
  const aborted = AbortSignal.abort();
  await assert.rejects(budgetRwkvRequest(request, config, aborted), { name: "AbortError" });
});

test("pi harness captures the budget actually sent and records Fake Think for an explicit off run", async t => {
  const { config, calls, state } = await endpoint(t);
  const outputDir = await mkdtemp(join(tmpdir(), "rwkv-pi-budget-"));
  t.after(() => rm(outputDir, { recursive: true, force: true }));
  const { record } = await runTask({ schemaVersion: 1, id: "budget-fixture", family: "budget-fixture", split: "smoke",
    source: "unit test fixture only", prompt: "Reply Done.", tools: ["read"], files: { "file.txt": "unchanged\n" },
    checks: [{ type: "file_equals", path: "file.txt", expected: "unchanged\n" }],
  }, { provider: createRwkvProvider(config), role: "smoke", outputDir, maxRequests: 1, timeoutMs: 5000,
    sandbox: "none", thinkingLevel: "off", contextBudget: config });
  assert.equal(record.status, "passed", record.error);
  assert.deepEqual(calls.map(call => call.path), ["/tokenize", "/v1/chat/completions"]);
  assert.equal(calls[1].body.max_tokens, 20768);
  assert.deepEqual(JSON.parse(JSON.stringify(record.steps[0].request)), calls[1].body);
  assert.equal(record.steps[0].tokenBudget.inputTokens, 12000);
  assert.equal(record.generation.thinkingLevel, "off");
  assert.equal(calls[1].body.chat_template_kwargs.fake_cot_mode, true);
  state.context = 16384;
  const mismatch = await runTask(record.task, { provider: createRwkvProvider(config), role: "smoke", outputDir,
    maxRequests: 1, timeoutMs: 5000, sandbox: "none", contextBudget: config });
  assert.equal(mismatch.record.status, "error");
  assert.match(mismatch.record.error, /context mismatch/);
  assert.equal(mismatch.record.steps.length, 0);
  assert.equal(calls.filter(call => call.path === "/v1/chat/completions").length, 1);
});
