import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { once } from "node:events";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import test from "node:test";
import { createModels } from "@earendil-works/pi-ai";
import { createRwkvProvider, readConfig } from "../dist/provider.js";

const run = promisify(execFile);
const cli = fileURLToPath(new URL("../dist/cli.js", import.meta.url));
const prompt = { systemPrompt: "You are an assistant.", messages: [{ role: "user", content: "Hello", timestamp: 1 }] };

function sendStream(res, deltas, finishReason = "stop") {
  res.writeHead(200, { "content-type": "text/event-stream" });
  const chunk = (delta, finish_reason) => res.write(`data: ${JSON.stringify({
    id: "chatcmpl-rwkv-test", object: "chat.completion.chunk", created: 1, model: "rwkv-test",
    choices: [{ index: 0, delta, finish_reason }],
  })}\n\n`);
  for (const delta of deltas) chunk(delta, null);
  chunk({}, finishReason);
  res.end("data: [DONE]\n\n");
}

async function endpoint(t, handler, overrides = {}) {
  const requests = [];
  const server = createServer(async (req, res) => {
    let body = "";
    for await (const chunk of req) body += chunk;
    requests.push({ url: req.url, headers: req.headers, body: JSON.parse(body) });
    handler(req, res, requests.length);
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  t.after(() => new Promise((resolve) => { server.close(resolve); server.closeAllConnections(); }));
  const config = readConfig({ RWKV_BACKEND: "openai",
    RWKV_BASE_URL: `http://127.0.0.1:${server.address().port}/v1`,
    RWKV_MODEL: "rwkv-test", RWKV_API_KEY: "test-key",
    ...overrides,
  });
  const models = createModels();
  const provider = createRwkvProvider(config);
  models.setProvider(provider);
  return { config, requests, models, model: provider.getModels()[0] };
}

test("streams RWKV reasoning and maps thinking off to Fake Think without overriding artifact sampling", async (t) => {
  const { requests, models, model } = await endpoint(t, (_req, res) => {
    sendStream(res, [{ reasoning: "Let me think." }, { content: "Hello!" }]);
  });
  for (const reasoning of ["high", undefined]) {
    const stream = models.streamSimple(model, prompt, { reasoning });
    const events = [];
    for await (const event of stream) events.push(event.type);
    const result = await stream.result();
    assert.equal(result.stopReason, "stop");
    assert.ok(events.includes("thinking_delta"));
    assert.ok(events.includes("text_delta"));
    assert.equal(result.content.find((c) => c.type === "thinking").thinking, "Let me think.");
    assert.equal(result.content.find((c) => c.type === "text").text, "Hello!");
  }
  for (const [index, request] of requests.entries()) {
    assert.equal(request.url, "/v1/chat/completions");
    assert.equal(request.headers.authorization, "Bearer test-key");
    assert.equal(request.body.model, "rwkv-test");
    assert.equal(request.body.messages[0].role, "system");
    assert.equal(request.body.stream, true);
    assert.deepEqual(request.body.chat_template_kwargs, { fake_cot_mode: index === 1, thinking: index === 0, bos_token: "" });
    assert.equal(request.body.max_tokens, 8192);
    for (const key of ["store", "reasoning_effort", "temperature", "top_p", "top_k", "presence_penalty", "frequency_penalty", "penalty_decay"]) {
      assert.equal(request.body[key], undefined, `${key} should be left to irwkv`);
    }
  }
});

test("XML template receives thinking switches and omits the converted tokenizer BOS", async t => {
  const { requests, models, model } = await endpoint(t, (_req, res) => sendStream(res, [{ content: "Done." }]), { RWKV_PROMPT_TEMPLATE: "xml" });
  await models.completeSimple(model, prompt, { reasoning: "high" });
  await models.completeSimple(model, prompt);
  assert.deepEqual(requests[0].body.chat_template_kwargs, { fake_cot_mode: false, thinking: true, bos_token: "" });
  assert.deepEqual(requests[1].body.chat_template_kwargs, { fake_cot_mode: true, thinking: false, bos_token: "" });
});

test("real pi CLI executes a streamed read tool call and replays reasoning plus the tool result", { timeout: 30000 }, async (t) => {
  const cwd = await mkdtemp(join(tmpdir(), "rwkv-pi-test-"));
  t.after(() => rm(cwd, { recursive: true, force: true }));
  await writeFile(join(cwd, "message.txt"), "RWKV_PI_TOOL_ROUNDTRIP_OK\n");
  const { config, requests } = await endpoint(t, (_req, res, count) => {
    if (count === 1) {
      sendStream(res, [
        { reasoning_content: "I should read the file." },
        { tool_calls: [{ index: 0, id: "call_rwkv_1", type: "function", function: { name: "read", arguments: '{"path":' } }] },
        { tool_calls: [{ index: 0, function: { arguments: '"message.txt"}' } }] },
      ], "tool_calls");
    } else {
      sendStream(res, [{ content: "RWKV_PI_TOOL_ROUNDTRIP_OK" }]);
    }
  });
  // Exercise .env loading, model selection, and forwarding of native pi flags.
  await writeFile(join(cwd, ".env"), `RWKV_BACKEND=openai\nRWKV_TOOL_FORMAT=native\nRWKV_PROMPT_TEMPLATE=xml\nRWKV_BASE_URL=${config.baseUrl}\nRWKV_MODEL=rwkv-test\nRWKV_API_KEY=test-key\n`);
  const env = { ...process.env, PI_CODING_AGENT_DIR: join(cwd, "agent"), PI_OFFLINE: "1" };
  for (const key of Object.keys(env)) if (key.startsWith("RWKV_") && key !== "RWKV_PI_SOURCE") delete env[key];
  const execution = run(process.execPath, [cli, "--no-collab",
    "--offline", "--no-approve", "--no-extensions", "--no-skills", "--no-prompt-templates", "--no-themes",
    "--no-session", "--tools", "read", "--thinking", "off", "--print", "Read message.txt and report its contents.",
  ], { cwd, env, timeout: 25000 });
  execution.child.stdin.end();
  const { stdout } = await execution;
  assert.match(stdout, /RWKV_PI_TOOL_ROUNDTRIP_OK/);
  assert.equal(requests.length, 2);
  assert.equal(requests[0].body.model, "rwkv-test");
  assert.equal(requests[0].headers.authorization, "Bearer test-key");
  assert.deepEqual(requests[0].body.tools.map((tool) => tool.function.name), ["read"]);
  assert.deepEqual(requests[0].body.chat_template_kwargs, { fake_cot_mode: true, thinking: false, bos_token: "" });
  const assistant = requests[1].body.messages.find((message) => message.role === "assistant");
  assert.equal(assistant.reasoning_content, "I should read the file.");
  assert.equal(assistant.tool_calls[0].id, "call_rwkv_1");
  const result = requests[1].body.messages.find((message) => message.role === "tool");
  assert.equal(result.tool_call_id, "call_rwkv_1");
  assert.match(result.content, /RWKV_PI_TOOL_ROUNDTRIP_OK/);
});

test("surfaces authentication failures instead of returning a successful empty answer", async (t) => {
  const { models, model } = await endpoint(t, (_req, res) => {
    res.writeHead(401, { "content-type": "application/json" });
    res.end(JSON.stringify({ error: { message: "Invalid API key", type: "authentication_error" } }));
  });
  const result = await models.completeSimple(model, prompt);
  assert.equal(result.stopReason, "error");
  assert.match(result.errorMessage, /401|Invalid API key/);
});

test("real pi CLI stops repeated directory-read errors; disabling the guard permits another request", { timeout: 30000 }, async (t) => {
  for (const limit of [3, 0]) {
    const cwd = await mkdtemp(join(tmpdir(), "rwkv-pi-loop-test-"));
    t.after(() => rm(cwd, { recursive: true, force: true }));
    const { config, requests } = await endpoint(t, (_req, res, count) => {
      if (count <= 4) sendStream(res, [{ tool_calls: [{ index: 0, id: `call_${count}`, type: "function",
        function: { name: "read", arguments: '{"path":"."}' } }] }], "tool_calls");
      else sendStream(res, [{ content: "STOPPED_BY_TEST_FIXTURE" }]);
    });
    const env = { ...process.env, PI_CODING_AGENT_DIR: join(cwd, "agent"), PI_OFFLINE: "1",
      RWKV_BACKEND: "openai", RWKV_TOOL_FORMAT: "native", RWKV_PROMPT_TEMPLATE: "xml", RWKV_BASE_URL: config.baseUrl, RWKV_MODEL: config.modelId, RWKV_API_KEY: "test-key", RWKV_REPEAT_FAILURE_LIMIT: String(limit) };
    const execution = run(process.execPath, [cli, "--no-collab", "--offline", "--no-approve", "--no-extensions", "--no-skills",
      "--no-mcp", "--no-context-files", "--no-session", "--tools", "read", "--thinking", "off", "--print", "Inspect this directory."],
      { cwd, env, timeout: 20000 });
    execution.child.stdin.end();
    if (limit) await assert.rejects(execution, error => {
      assert.match(error.stdout + error.stderr, /RWKV_TOOL_LOOP/);
      return true;
    });
    else assert.match((await execution).stdout, /STOPPED_BY_TEST_FIXTURE/);
    assert.equal(requests.length, limit ? 3 : 5);
  }
});

test("cancels a pending generation through pi's abort signal", { timeout: 5000 }, async (t) => {
  const controller = new AbortController();
  const { models, model } = await endpoint(t, (_req, res) => {
    res.writeHead(200, { "content-type": "text/event-stream" });
    res.flushHeaders();
    controller.abort();
  });
  const result = await models.completeSimple(model, prompt, { signal: controller.signal });
  assert.equal(result.stopReason, "aborted");
});
