import assert from "node:assert/strict";
import test from "node:test";
import { once } from "node:events";
import { createServer } from "node:http";
import { existsSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createModels } from "@earendil-works/pi-ai";
import { createRwkvProvider, readConfig } from "../dist/provider.js";
import { countTokensLocal } from "../dist/local-tokenizer.js";
import { nativeTokenize, nativeTokenizerPaths } from "../dist/doctor.js";
import { ContinuationParser, parseCalls, parseJsonCalls } from "../dist/xml.js";
import { renderPrompt, PROMPT_VERSION, JSON_PROMPT_VERSION } from "../dist/prompt.js";
import { runTask } from "../dist/harness.js";

const tools = [{ type: "function", function: { name: "read", parameters: { type: "object", properties: { path: { type: "string" } }, required: ["path"], additionalProperties: false } } }];
const xml = '<tool_call>{"name":"read","arguments":{"path":"input.txt"}}</tool_call>';
const nativeJson = '{"type":"toolCall","id":"call_1","name":"read","arguments":{"path":"input.txt"}}';

test("native pi JSON preserves IDs and requires complete schema-valid calls", () => {
  assert.equal(parseJsonCalls(nativeJson, tools)[0].id, "call_1");
  assert.deepEqual(JSON.parse(parseJsonCalls(nativeJson.replace('input.txt', '</tool_call> { \\"quoted\\" }'), tools)[0].function.arguments), {path:'</tool_call> { "quoted" }'});
  for (const bad of [nativeJson.slice(0,-1), nativeJson+' trailing', nativeJson.replace('"read"','"unknown"'),
    nativeJson.replace('"input.txt"','5'),nativeJson.replace('"path":"input.txt"','"path":"a","path":"b"'),
    nativeJson.replace('"type":"toolCall",',''), '['+nativeJson+','+nativeJson+']']) assert.throws(()=>parseJsonCalls(bad,tools));
  assert.throws(()=>parseJsonCalls(nativeJson,tools,'none'));
  assert.deepEqual(parseJsonCalls('{"answer":42}',tools),[]);
  for (let size=1;size<30;size++) {
    const p=new ContinuationParser(true,tools,'auto','json'), deltas=[];
    const source='>检查目录。😀</think>\n'+nativeJson;
    for(let i=0;i<source.length;i+=size)deltas.push(...p.push(source.slice(i,i+size)));
    assert.ok(deltas.every(d=>!d.tool_calls));
    deltas.push(...p.finish('stop').deltas);
    assert.equal(deltas.flatMap(d=>d.tool_calls??[])[0].id,'call_1');
    assert.equal(deltas.map(d=>d.reasoning_content??'').join(''),'检查目录。😀');
  }
  for(const text of ['{"answer":42}', 'Example:\n```json\n'+nativeJson+'\n```']) {
    const p=new ContinuationParser(false,tools,'auto','json');
    const deltas=[...p.push(text),...p.finish('stop').deltas];
    assert.equal(deltas.map(d=>d.content??'').join(''),text);
    assert.ok(deltas.every(d=>!d.tool_calls));
  }
  const p=new ContinuationParser(false,tools,'auto','json');p.push(nativeJson);
  assert.throws(()=>p.finish('length'),/Truncated/);
});

test("client XML parsing rejects malformed calls and preserves nested tag strings and examples", () => {
  assert.deepEqual(JSON.parse(parseCalls(xml.replace("input.txt", "</tool_call>"), tools)[0].function.arguments), { path: "</tool_call>" });
  for (const bad of [xml.slice(0, -1), xml.replace('"read"', '"bash"'), xml.replace('"input.txt"', '5'), xml.replace('"path":"input.txt"', '"path":"input.txt","path":"second"'), xml + " trailing text", '<tool_call>{"name":"read","arguments":{"path":"a","x":1e309}}</tool_call>']) assert.throws(() => parseCalls(bad, tools));
  assert.throws(() => parseCalls(xml, tools, "none"));
  const p = new ContinuationParser(false, tools);
  const text = 'Example:\n```xml\n' + xml + '\n```\n{"name":"read","arguments":{"path":"x"}}';
  const deltas = [...p.push(text), ...p.finish("stop").deltas];
  assert.equal(deltas.map(d => d.content ?? "").join(""), text);
  assert.ok(deltas.every(d => !d.tool_calls));
});

test("arbitrary chunk boundaries preserve reasoning and withhold incomplete calls", () => {
  for (let size = 1; size < 30; size++) {
    const p = new ContinuationParser(true, tools), source = '>先检查文件。😀</think>\n' + xml, deltas = [];
    for (let i = 0; i < source.length; i += size) deltas.push(...p.push(source.slice(i, i + size)));
    assert.ok(deltas.every(d => !d.tool_calls));
    const result = p.finish("stop"); deltas.push(...result.deltas);
    assert.equal(result.reason, "tool_calls");
    assert.equal(deltas.map(d => d.reasoning_content ?? "").join(""), "先检查文件。😀");
    assert.equal(deltas.flatMap(d => d.tool_calls ?? []).length, 1);
  }
  const p = new ContinuationParser(false, tools); p.push(xml);
  assert.throws(() => p.finish("length"), /Truncated/);
});

async function endpoint(t, answers) {
  const requests = [], stops = [], paths = [], headers = [];
  const server = createServer(async (req, res) => {
    let raw = ""; for await (const part of req) raw += part;
    const body = JSON.parse(raw);
    if (req.url === "/v1/tokens/count") { res.setHeader("content-type", "application/json"); res.end('{"tokens":100}'); return; }
    if (req.url === "/v1/server/stop") { stops.push(body); res.end('{}'); return; }
    paths.push(req.url); requests.push(body); headers.push(req.headers);
    const result = answers[requests.length - 1];
    res.writeHead(200, { "content-type": "text/event-stream" });
    const emit = (delta, finish_reason = null) => res.write(`data: ${JSON.stringify({ id: "raw-native", object: "chat.completion.chunk", choices: [{ index: 0, delta, finish_reason, ...(finish_reason && result.matchedStop ? { matched_stop: result.matchedStop } : {}) }] })}\n\n`);
    for (const content of result.text.match(/[\s\S]{1,3}/g) ?? []) emit({ content });
    if (result.hold) return;
    if (!result.incomplete) { emit({}, result.finish ?? "stop"); res.write("data: [DONE]\n\n"); }
    res.end();
  });
  server.listen(0, "127.0.0.1"); await once(server, "listening");
  t.after(() => new Promise(resolve => { server.close(resolve); server.closeAllConnections(); }));
  return { config: readConfig({ RWKV_BASE_URL: `http://127.0.0.1:${server.address().port}/v1` }), requests, stops, paths, headers };
}

test("real pi harness executes client-parsed XML and replays observations in raw continuation", async t => {
  const { config, requests, paths } = await endpoint(t, [{ text: xml + "\n\nUser:", matchedStop: "\nUser:" }, { text: "Read successfully.<EOD>", matchedStop: "<EOD>" }]);
  const outputDir = await mkdtemp(join(tmpdir(), "rwkv-lightning-test-")); t.after(() => rm(outputDir, { recursive: true, force: true }));
  const { record } = await runTask({ schemaVersion: 1, id: "continuation-fixture", family: "continuation-fixture", split: "smoke", source: "mock protocol fixture only", prompt: "Read input.txt.", tools: ["read"], files: { "input.txt": "OBSERVATION_SENTINEL\n" }, checks: [{ type: "file_equals", path: "input.txt", expected: "OBSERVATION_SENTINEL\n" }] }, { provider: createRwkvProvider(config), role: "smoke", outputDir, maxRequests: 2, timeoutMs: 15000, sandbox: "none", thinkingLevel: "off" });
  assert.equal(record.status, "passed", record.error);
  assert.equal(record.responseAdapter, "rwkv-xml-continuation-v2");
  assert.equal(record.promptVersion, PROMPT_VERSION);
  assert.equal(record.steps[0].response.stopReason, "toolUse");
  assert.equal(requests.length, 2);
  assert.deepEqual(paths, ["/v1/batch/completions", "/v1/batch/completions"]);
  assert.equal(requests[0].session_id, undefined);
  assert.equal(requests[0].think_type, undefined);
  assert.equal(requests[0].force_reasoning, false);
  assert.match(requests[1].contents[0], /<tool_response>\n.*OBSERVATION_SENTINEL/s);
  assert.equal(requests[0].messages, undefined); assert.equal(requests[0].tools, undefined);
  assert.deepEqual(requests[0].stop, ["<EOD>", "✿", "\nUser:", "\n### User"]); assert.equal(requests[0].stop_tokens, undefined);
  assert.equal(requests[0].contents[0], renderPrompt(record.steps[0].request.messages, record.steps[0].request.tools, false));
  assert.ok(requests[1].contents[0].startsWith(requests[0].contents[0] + xml + "\n\nUser: <tool_response>\n"));
});

test("native pi JSON executes through the real harness and matches canonical replay",async t=>{
  const {config,requests}=await endpoint(t,[{text:nativeJson+'\n\nUser:',matchedStop:'\nUser:'},{text:'Read successfully.<EOD>',matchedStop:'<EOD>'}]);
  const outputDir=await mkdtemp(join(tmpdir(),'native-json-test-'));t.after(()=>rm(outputDir,{recursive:true,force:true}));
  const jsonConfig={...config,toolFormat:'json',promptTemplate:'json',editMode:'text'};
  const {record}=await runTask({schemaVersion:1,id:'json-fixture',family:'json-fixture',split:'smoke',source:'mock protocol test; not model evaluation',prompt:'Read input.txt.',tools:['read'],files:{'input.txt':'NATIVE_JSON_OBSERVATION\n'},checks:[{type:'file_equals',path:'input.txt',expected:'NATIVE_JSON_OBSERVATION\n'}]},
    {provider:createRwkvProvider(jsonConfig),role:'smoke',outputDir,maxRequests:2,timeoutMs:15000,sandbox:'bubblewrap',thinkingLevel:'off'});
  assert.equal(record.status,'passed',record.error);
  assert.equal(record.promptVersion,JSON_PROMPT_VERSION);
  assert.equal(record.responseAdapter,'rwkv-json-continuation-v1');
  assert.equal(record.steps[0].response.content.find(b=>b.type==='toolCall').id,'call_1');
  assert.match(requests[1].contents[0],/"role":"tool","tool_call_id":"call_1","content":"NATIVE_JSON_OBSERVATION/);
  assert.equal(requests[0].contents[0],renderPrompt(record.steps[0].request.messages,record.steps[0].request.tools,false,true,'json'));
  assert.ok(requests[1].contents[0].startsWith(requests[0].contents[0]+nativeJson+'\n\nUser: '));
  assert.ok(!requests[0].contents[0].includes('<tool_call>'));
  assert.equal(requests[0].messages,undefined);assert.equal(requests[0].tools,undefined);
});

test("JSON transport withholds calls on incomplete streams and cancellation",async t=>{
  for(const answer of [{text:nativeJson,incomplete:true},{text:nativeJson}]) {
    const {config}=await endpoint(t,[answer]);
    const models=createModels(),provider=createRwkvProvider({...config,toolFormat:'json',promptTemplate:'json'});models.setProvider(provider);
    const result=await models.completeSimple(provider.getModels()[0],{messages:[{role:'user',content:'Read.',timestamp:1}],tools:[{name:'read',description:'read',parameters:tools[0].function.parameters}]});
    assert.equal(result.stopReason,'error');assert.ok(result.content.every(c=>c.type!=='toolCall'));
  }
});

test("canonical tool-only history preserves the generated turn without an added blank line", () => {
  const history = [{ role: "user", content: "Read input.txt." }];
  const assistant = { role: "assistant", content: null, tool_calls: [{ function: { name: "read", arguments: '{"path":"input.txt"}' } }] };
  assert.equal(renderPrompt([...history, assistant], tools, false, false), renderPrompt(history, tools, false) + xml + "\n\n");
  const open = renderPrompt(history, tools, true);
  assert.equal(renderPrompt([...history, { ...assistant, reasoning_content: "Check the file." }], tools, true, false), open + ">Check the file.</think>\n" + xml + "\n\n");
});

test("native stop metadata must describe an exact suffix; internal mentions remain text", () => {
  const source = "This is an example containing <EOD> inside a sentence. ".repeat(6);
  const p = new ContinuationParser(false, []), deltas = [];
  for (const char of source + "<EOD>") deltas.push(...p.push(char));
  deltas.push(...p.finish("stop", "<EOD>").deltas);
  assert.equal(deltas.map(d => d.content ?? "").join(""), source);
  for (const [text, reason, stop] of [[xml + "\nUser: ", "stop", "\nUser:"], [xml + "<EOD>", "length", "<EOD>"], [xml, "stop", "</tool_call>"]]) {
    const invalid = new ContinuationParser(false, tools); invalid.push(text);
    assert.throws(() => invalid.finish(reason, stop), /Invalid native stop metadata/);
  }
});

test("incomplete SSE cannot expose a complete XML call", async t => {
  const { config } = await endpoint(t, [{ text: xml, incomplete: true }]);
  const models = createModels(), provider = createRwkvProvider(config); models.setProvider(provider);
  const result = await models.completeSimple(provider.getModels()[0], { messages: [{ role: "user", content: "Read.", timestamp: 1 }], tools: [{ name: "read", description: "read", parameters: tools[0].function.parameters }] });
  assert.equal(result.stopReason, "error");
  assert.ok(result.content.every(c => c.type !== "toolCall"));
});

test("native cancellation cannot authorize tools merely by reporting finish_reason stop", async t => {
  const { config } = await endpoint(t, [{ text: xml }]);
  const models = createModels(), provider = createRwkvProvider(config); models.setProvider(provider);
  const result = await models.completeSimple(provider.getModels()[0], { messages: [{ role: "user", content: "Read.", timestamp: 1 }], tools: [{ name: "read", description: "read", parameters: tools[0].function.parameters }] });
  assert.equal(result.stopReason, "error");
  assert.match(result.errorMessage, /without a configured text boundary/);
  assert.ok(result.content.every(c => c.type !== "toolCall"));
});

test("Open Think content and registered adapter selection survive the client transport", async t => {
  const { config, requests } = await endpoint(t, [{ text: ">检查一下。</think>完成。<EOD>", matchedStop: "<EOD>" }]);
  const models = createModels(), provider = createRwkvProvider({ ...config, adapterId: "trained", adapterVersion: "revision", adapterScale: 0.5 }); models.setProvider(provider);
  const result = await models.completeSimple(provider.getModels()[0], { messages: [{ role: "user", content: "Hi", timestamp: 1 }] }, { reasoning: "high", temperature: 0 });
  assert.equal(result.stopReason, "stop", result.errorMessage);
  assert.equal(result.content.find(c => c.type === "thinking").thinking, "检查一下。");
  assert.equal(result.content.find(c => c.type === "text").text, "完成。");
  assert.equal(requests[0].adapter_id, "trained");
  assert.equal(requests[0].adapter_version, "revision");
  assert.equal(requests[0].adapter_scale, 0.5);
  assert.equal(requests[0].temperature, 1); assert.equal(requests[0].top_k, 1);
});

test("client abort stops only the native request ID", async t => {
  const { config, stops } = await endpoint(t, [{ text: "Generating a long answer. ".repeat(16), hold: true }]);
  const models = createModels(), provider = createRwkvProvider(config); models.setProvider(provider);
  const controller = new AbortController();
  const stream = models.streamSimple(provider.getModels()[0], { messages: [{ role: "user", content: "Hi", timestamp: 1 }] }, { signal: controller.signal });
  for await (const event of stream) if (event.type === "text_delta") { controller.abort(); break; }
  const result = await stream.result();
  assert.equal(result.stopReason, "aborted");
  for (let i = 0; i < 50 && !stops.length; i++) await new Promise(resolve => setTimeout(resolve, 20));
  assert.deepEqual(stops, [{ request_id: "raw-native" }]);
});

test("Lightning XML continuation is default; legacy native APIs require explicit selection", () => {
  assert.equal(readConfig({}).toolFormat, "xml");
  assert.equal(readConfig({}).backend, "lightning");
  assert.throws(() => readConfig({ RWKV_TOOL_FORMAT: "native" }), /Lightning/);
  assert.equal(readConfig({ RWKV_BACKEND: "openai" }).toolFormat, "native");
  assert.throws(() => readConfig({ RWKV_ADAPTER_VERSION: "v1" }), /ADAPTER_ID/);
  assert.throws(() => readConfig({ RWKV_ADAPTER_ID: "a", RWKV_ADAPTER_SCALE: "NaN" }), /finite/);
  const model = createRwkvProvider(readConfig({ RWKV_ADAPTER_ID: "a", RWKV_ADAPTER_VERSION: "v1", RWKV_ADAPTER_SCALE: "0" })).getModels()[0];
  assert.equal(model.samplingParams.adapter_id, "a");
  assert.equal(model.samplingParams.adapter_scale, 0);
});

test("stabilized tool-use sampling defaults, env overrides, and X-RWKV-Session-Id header forwarding", async t => {
  const { config, requests, headers } = await endpoint(t, [
    { text: "Tool turn.<EOD>", matchedStop: "<EOD>" },
    { text: ">想一想。</think>Open think.<EOD>", matchedStop: "<EOD>" },
    { text: "Fake think.<EOD>", matchedStop: "<EOD>" },
    { text: "Env override.<EOD>", matchedStop: "<EOD>" },
    { text: "Explicit greedy.<EOD>", matchedStop: "<EOD>" },
  ]);
  const toolDef = [{ name: "read", description: "read", parameters: tools[0].function.parameters }];
  const userMsg = [{ role: "user", content: "Hi", timestamp: 1 }];

  const models = createModels();
  const baseProvider = createRwkvProvider({ ...config, sessionId: "sticky-gpu-0" });
  models.setProvider(baseProvider);
  await models.completeSimple(baseProvider.getModels()[0], { messages: userMsg, tools: toolDef }, { reasoning: "high" });
  await models.completeSimple(baseProvider.getModels()[0], { messages: userMsg }, { reasoning: "high" });
  await models.completeSimple(baseProvider.getModels()[0], { messages: userMsg }, { reasoning: "off" });

  // 1. Stabilized tool-use profile
  assert.equal(requests[0].temperature, 0.5);
  assert.equal(requests[0].top_p, 0.7);
  assert.equal(requests[0].top_k, 65536);
  assert.equal(requests[0].alpha_presence, 0.8);
  assert.equal(requests[0].alpha_frequency, 0.2);
  assert.equal(requests[0].alpha_decay, 0.996);
  assert.equal(requests[0].session_id, undefined);
  assert.equal(headers[0]["x-rwkv-session-id"], "sticky-gpu-0");

  // 2. Non-tool Open Think & Fake Think profiles remain unchanged
  assert.equal(requests[1].temperature, 0.96);
  assert.equal(requests[1].top_p, 0.76);
  assert.equal(requests[1].top_k, 32);
  assert.equal(requests[1].alpha_presence, 1);
  assert.equal(requests[1].alpha_frequency, 0.1);
  assert.equal(requests[1].alpha_decay, 0.988);

  assert.equal(requests[2].temperature, 1.0);
  assert.equal(requests[2].top_p, 0.28);
  assert.equal(requests[2].top_k, 32);
  assert.equal(requests[2].alpha_presence, 0);
  assert.equal(requests[2].alpha_frequency, 0);
  assert.equal(requests[2].alpha_decay, 0.996);

  // 3. Env sampling overrides apply, while explicit temperature: 0 still wins
  const envCfg = readConfig({
    RWKV_BASE_URL: config.baseUrl,
    RWKV_TEMPERATURE: "0.4",
    RWKV_TOP_P: "0.6",
    RWKV_TOP_K: "128",
    RWKV_ALPHA_PRESENCE: "0.8",
    RWKV_ALPHA_FREQUENCY: "0.2",
    RWKV_ALPHA_DECAY: "0.99",
  });
  const envModels = createModels();
  const envProvider = createRwkvProvider(envCfg);
  envModels.setProvider(envProvider);
  await envModels.completeSimple(envProvider.getModels()[0], { messages: userMsg, tools: toolDef }, { reasoning: "high" });
  await envModels.completeSimple(envProvider.getModels()[0], { messages: userMsg, tools: toolDef }, { reasoning: "high", temperature: 0 });

  assert.equal(requests[3].temperature, 0.4);
  assert.equal(requests[3].top_p, 0.6);
  assert.equal(requests[3].top_k, 128);
  assert.equal(requests[3].alpha_presence, 0.8);
  assert.equal(requests[3].alpha_frequency, 0.2);
  assert.equal(requests[3].alpha_decay, 0.99);

  assert.equal(requests[4].temperature, 1);
  assert.equal(requests[4].top_k, 1);
  assert.equal(requests[4].top_p, 1);
});

test("in-memory local Lightning tokenizer matches C++ tokenize-native byte-for-byte and eliminates /v1/tokens/count RTTs", async t => {
  if (!existsSync(nativeTokenizerPaths().binary)) { t.skip("native tokenizer not built; run npm run lightning:setup && npm run lightning:tokenizer"); return; }
  const samples = [
    "System: doctor probe\n\nUser: 你好，RWKV 😀 <tool_call>{\"name\":\"read\"}</tool_call>\n\nAssistant: <think",
    "Hello, world!\n\t\"escaped\" \\ backslash 'single' <EOD> trailing",
    "def fib(n: int) -> int:\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n# 中文注释 🚀🎉🔥\n",
    "\x00\x01\x7f\r\n" + xml + "\n\nUser: <tool_response>\n1: line one\n</tool_response>\n\nAssistant: <think></think>\n",
  ];
  for (const sample of samples) {
    assert.equal(countTokensLocal(sample), nativeTokenize(sample).length, `mismatch on: ${JSON.stringify(sample.slice(0, 40))}`);
  }

  let remoteCountHits = 0;
  const server = createServer(async (req, res) => {
    for await (const _ of req) { /* drain */ }
    if (req.url === "/v1/tokens/count") {
      remoteCountHits++;
      res.setHeader("content-type", "application/json");
      res.end('{"tokens":100}');
      return;
    }
    res.writeHead(200, { "content-type": "text/event-stream" });
    res.write(`data: ${JSON.stringify({ id: "fast-1", object: "chat.completion.chunk", choices: [{ index: 0, delta: { content: ">ok</think>Done.<EOD>" }, finish_reason: null }] })}\n\n`);
    res.write(`data: ${JSON.stringify({ id: "fast-1", object: "chat.completion.chunk", choices: [{ index: 0, delta: {}, finish_reason: "stop", matched_stop: "<EOD>" }] })}\n\n`);
    res.end("data: [DONE]\n\n");
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  t.after(() => new Promise(resolve => { server.close(resolve); server.closeAllConnections(); }));

  const cfg = readConfig({ RWKV_BASE_URL: `http://127.0.0.1:${server.address().port}/v1` });
  const models = createModels(), provider = createRwkvProvider(cfg);
  models.setProvider(provider);
  const res = await models.completeSimple(provider.getModels()[0], { messages: [{ role: "user", content: "测试免往返本地分词", timestamp: 1 }] }, { reasoning: "high" });
  assert.equal(res.stopReason, "stop");
  assert.equal(remoteCountHits, 0, "expected zero remote /v1/tokens/count HTTP requests when local tokenizer is active");
  assert.ok(res.usage.input > 0 && res.usage.output > 0);
});
