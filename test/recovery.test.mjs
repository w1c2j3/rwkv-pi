import assert from "node:assert/strict";
import test from "node:test";
import { once } from "node:events";
import { createServer } from "node:http";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createRwkvProvider, readConfig } from "../dist/provider.js";
import { parseCalls } from "../dist/xml.js";
import { runTask } from "../dist/harness.js";

const tools = [{ type: "function", function: { name: "read", parameters: { type: "object", properties: { path: { type: "string" }, offset: { type: "number" } }, required: ["path"], additionalProperties: false } } }];
const args = (calls, i = 0) => calls[i].function.arguments;

test("recover mode hands invalid but enveloped calls to pi verbatim; strict mode still rejects them", () => {
  const stringNumber = '<tool_call>{"name":"read","arguments":{"path":"a.py","offset":"1"}}</tool_call>';
  assert.throws(() => parseCalls(stringNumber, tools), /Invalid arguments/);
  assert.equal(args(parseCalls(stringNumber, tools, "auto", true)), '{"path":"a.py","offset":"1"}');
  const unknown = '<tool_call>{"name":"finish","arguments":{"reason":"done"}}</tool_call>';
  assert.throws(() => parseCalls(unknown, tools), /not allowed/);
  assert.equal(parseCalls(unknown, tools, "auto", true)[0].function.name, "finish");
  // One extra closing brace: the model's own argument text is passed on unrepaired.
  const extra = '<tool_call>{"name":"read","arguments":{"path":"a.py"}}}\n</tool_call>';
  assert.throws(() => parseCalls(extra, tools), /closing tag/);
  assert.equal(args(parseCalls(extra, tools, "auto", true)), '{"path":"a.py"}}');
  // Missing closing tag at the end of the turn.
  assert.equal(args(parseCalls('<tool_call>{"name":"read","arguments":{"path":"a.py"}}', tools, "auto", true)), '{"path":"a.py"}');
  // A following valid call is still parsed.
  const two = parseCalls(extra + '\n<tool_call>{"name":"read","arguments":{"path":"b.py"}}</tool_call>', tools, "auto", true);
  assert.equal(two.length, 2); assert.equal(args(two, 1), '{"path":"b.py"}');
  // Unnamed frames, text outside envelopes and tool_choice none remain fatal.
  for (const bad of ['<tool_call>{"arguments":{"path":"a"}}</tool_call>', '<tool_call>garbage</tool_call>', stringNumber + ' trailing']) {
    assert.throws(() => parseCalls(bad, tools, "auto", true));
  }
  assert.throws(() => parseCalls(stringNumber, tools, "none", true));
});

async function endpoint(t, answers, env = {}) {
  const requests = [];
  const server = createServer(async (req, res) => {
    let raw = ""; for await (const part of req) raw += part;
    if (req.url === "/v1/tokens/count") { res.setHeader("content-type", "application/json"); res.end('{"tokens":100}'); return; }
    if (req.url === "/v1/server/stop") { res.end("{}"); return; }
    requests.push(JSON.parse(raw));
    const text = answers[Math.min(requests.length - 1, answers.length - 1)];
    res.writeHead(200, { "content-type": "text/event-stream" });
    const emit = (delta, finish_reason = null) => res.write(`data: ${JSON.stringify({ id: "raw", object: "chat.completion.chunk", choices: [{ index: 0, delta, finish_reason, ...(finish_reason ? { matched_stop: "<EOD>" } : {}) }] })}\n\n`);
    for (const content of (text + "<EOD>").match(/[\s\S]{1,5}/g)) emit({ content });
    emit({}, "stop"); res.write("data: [DONE]\n\n"); res.end();
  });
  server.listen(0, "127.0.0.1"); await once(server, "listening");
  t.after(() => new Promise(resolve => { server.close(resolve); server.closeAllConnections(); }));
  return { config: readConfig({ RWKV_BASE_URL: `http://127.0.0.1:${server.address().port}/v1`, ...env }), requests };
}

async function run(t, answers, env, maxRequests = 6) {
  const { config, requests } = await endpoint(t, answers, env);
  const outputDir = await mkdtemp(join(tmpdir(), "rwkv-recovery-test-")); t.after(() => rm(outputDir, { recursive: true, force: true }));
  const { record } = await runTask({ schemaVersion: 1, id: "recovery-fixture", family: "recovery-fixture", split: "smoke", source: "mock protocol fixture only",
    prompt: "Read input.txt.", tools: ["read"], files: { "input.txt": "LINE1\nLINE2\n" }, checks: [{ type: "file_equals", path: "input.txt", expected: "LINE1\nLINE2\n" }] },
  { provider: createRwkvProvider(config), role: "smoke", outputDir, maxRequests, timeoutMs: 15000, sandbox: "none", thinkingLevel: "off" });
  return { record, requests, results: record.messages.filter(m => m.role === "toolResult") };
}

test("pi coerces string numbers and returns a native error result for broken JSON, then the model continues", async t => {
  const { record, results, requests } = await run(t, [
    '<tool_call>{"name":"read","arguments":{"path":"input.txt","offset":"2"}}</tool_call>',
    '<tool_call>{"name":"read","arguments":{"path":"input.txt"}}}\n</tool_call>',
    '<tool_call>{"name":"finish","arguments":{}}</tool_call>',
    "Done.",
  ]);
  assert.equal(record.status, "passed", record.error);
  assert.equal(record.toolCallRecovery, "pi-native-validation");
  assert.equal(results.length, 3);
  assert.equal(results[0].isError, false);
  assert.match(results[0].content[0].text, /LINE2/);
  assert.doesNotMatch(results[0].content[0].text, /LINE1/);
  assert.equal(results[1].isError, true);
  assert.match(results[1].content[0].text, /Invalid arguments for tool "read"/);
  assert.equal(results[2].isError, true);
  assert.match(results[2].content[0].text, /Tool finish not found/);
  // The error observations are part of the next raw continuation the model sees.
  assert.match(requests[2].contents[0], /Invalid arguments for tool/);
});

test("strict mode (RWKV_TOOL_CALL_RECOVERY=0) still aborts on the first malformed call", async t => {
  const { record } = await run(t, ['<tool_call>{"name":"read","arguments":{"path":"input.txt","offset":"2"}}</tool_call>', "Done."], { RWKV_TOOL_CALL_RECOVERY: "0" });
  assert.equal(record.status, "error");
  assert.match(record.error, /Invalid arguments for read/);
  assert.equal(record.toolCallRecovery, undefined);
});

test("the third identical call with an identical result is blocked and recorded as a tool error", async t => {
  const same = '<tool_call>{"name":"read","arguments":{"path":"input.txt"}}</tool_call>';
  const { record, results } = await run(t, [same, same, same, "Done."]);
  assert.equal(record.status, "passed", record.error);
  assert.deepEqual(record.loopGuard, { repeatFailureLimit: 3, identicalCallLimit: 2, targetFailureLimit: 3 });
  assert.deepEqual(results.map(r => r.isError), [false, false, true]);
  assert.match(results[2].content[0].text, /RWKV_REPEAT_GUARD: read/);
});

test("continued identical repeats end the run through the repeated-failure guard instead of burning the budget", async t => {
  const same = '<tool_call>{"name":"read","arguments":{"path":"input.txt"}}</tool_call>';
  const { record } = await run(t, [same], {}, 20);
  assert.equal(record.status, "error");
  assert.match(record.error, /RWKV_TOOL_LOOP/);
  assert.ok(record.steps.length < 10, String(record.steps.length));
});
