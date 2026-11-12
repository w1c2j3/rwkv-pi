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

// A Lightning engine mock: the native tokenizer route and the raw continuation route.
async function endpoint(t) {
  const calls = [];
  const state = { count: 12000 };
  const server = createServer(async (req, res) => {
    let raw = "";
    for await (const chunk of req) raw += chunk;
    calls.push({ path: req.url, body: JSON.parse(raw) });
    if (req.url === "/v1/tokens/count") {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({ count: state.count }));
      return;
    }
    res.writeHead(200, { "content-type": "text/event-stream" });
    const emit = (delta, finish_reason = null) => res.write(`data: ${JSON.stringify({ id: "budget-mock", object: "chat.completion.chunk",
      choices: [{ index: 0, delta, finish_reason, ...(finish_reason ? { matched_stop: "<EOD>" } : {}) }] })}\n\n`);
    emit({ content: "Done.<EOD>" });
    emit({}, "stop");
    res.end("data: [DONE]\n\n");
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  t.after(() => new Promise(resolve => { server.close(resolve); server.closeAllConnections(); }));
  const config = readConfig({ RWKV_BASE_URL: `http://127.0.0.1:${server.address().port}/v1`, RWKV_MODEL: "mock-only",
    RWKV_CONTEXT_WINDOW: "32768", RWKV_MAX_TOKENS: "24576" });
  return { calls, state, config };
}

const request = { model: "mock-only", messages: [{ role: "user", content: "中文😀" }],
  tools: [{ type: "function", function: { name: "read", parameters: { type: "object" } } }], max_tokens: 24576, stream: true };

test("counts the rendered continuation with the Lightning-native tokenizer and reduces only output", async t => {
  const { config, calls, state } = await endpoint(t);
  const result = await budgetRwkvRequest(request, config, new AbortController().signal);
  assert.equal(result.request.max_tokens, 20768);
  assert.deepEqual(result.request.messages, request.messages);
  assert.equal(request.max_tokens, 24576);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].path, "/v1/tokens/count");
  assert.equal(typeof calls[0].body.text, "string");
  assert.match(calls[0].body.text, /中文😀/);
  assert.equal(result.budget.inputTokens + result.budget.outputTokens, 32768);
  state.count = 32768;
  await assert.rejects(budgetRwkvRequest(request, config, new AbortController().signal), /context exhausted/);
  await assert.rejects(budgetRwkvRequest(request, config, AbortSignal.abort()), { name: "AbortError" });
});

test("refuses any tokenizer other than Lightning's", async () => {
  const config = readConfig({ RWKV_BACKEND: "openai", RWKV_MODEL: "mock-only" });
  await assert.rejects(budgetRwkvRequest(request, config, new AbortController().signal), /Lightning-native tokenizer/);
});

test("pi harness records the budget actually sent and renders Fake Think for an explicit off run", async t => {
  const { config, calls } = await endpoint(t);
  const outputDir = await mkdtemp(join(tmpdir(), "rwkv-pi-budget-"));
  t.after(() => rm(outputDir, { recursive: true, force: true }));
  const { record } = await runTask({ schemaVersion: 1, id: "budget-fixture", family: "budget-fixture", split: "smoke",
    source: "unit test fixture only", prompt: "Reply Done.", tools: ["read"], files: { "file.txt": "unchanged\n" },
    checks: [{ type: "file_equals", path: "file.txt", expected: "unchanged\n" }],
  }, { provider: createRwkvProvider(config), role: "smoke", outputDir, maxRequests: 1, timeoutMs: 5000,
    sandbox: "none", thinkingLevel: "off", contextBudget: config });
  assert.equal(record.status, "passed", record.error);
  const generations = calls.filter(call => call.path === "/v1/batch/completions");
  assert.equal(generations.length, 1);
  assert.ok(calls.filter(call => call.path === "/v1/tokens/count").length >= 1);
  assert.ok(calls.every(call => call.path === "/v1/tokens/count" || call.path === "/v1/batch/completions"));
  assert.equal(generations[0].body.max_tokens, 20768);
  assert.ok(generations[0].body.contents[0].endsWith("Assistant: <think></think>\n"));
  assert.equal(record.steps[0].tokenBudget.inputTokens, 12000);
  assert.equal(record.generation.thinkingLevel, "off");
});
