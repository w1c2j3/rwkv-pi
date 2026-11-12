#!/usr/bin/env node
// Conformance fixtures for the standard RWKV agent wire, rwkv-pi-xml-continuation-v3.
//
// Any other harness integration (for example a DSH plugin) must reproduce these bytes exactly:
//   render.json  chat request -> exact prompt text, including the generation prefix
//   parse.json   raw engine continuation -> reasoning, visible text, tool calls, finish reason, or rejection
//   tokens.json  prompt -> Lightning-native token ids (the only tokenizer used for RWKV)
//
//   npm run conformance            # check the committed fixtures against this build
//   npm run conformance -- --write # regenerate them after an intentional, versioned protocol change
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { PROMPT_VERSION, promptForRequest } from "../dist/prompt.js";
import { ContinuationParser } from "../dist/xml.js";
import { nativeTokenize, nativeTokenizerPaths } from "../dist/doctor.js";

const root = fileURLToPath(new URL("../", import.meta.url));
export const FIXTURES = join(root, "conformance", PROMPT_VERSION);

const fn = (name, description, properties, required) => ({ type: "function", function: { name, description, parameters: { type: "object", properties, required, additionalProperties: false } } });
const TOOLS = [
  fn("read", "Read a file.", { path: { type: "string" } }, ["path"]),
  fn("write", "Write a file.", { path: { type: "string" }, content: { type: "string" } }, ["path", "content"]),
  fn("bash", "Run a shell command.", { command: { type: "string" } }, ["command"]),
];
const call = (id, name, args) => ({ id, type: "function", function: { name, arguments: JSON.stringify(args) } });
const off = { fake_cot_mode: true, thinking: false };
const on = { fake_cot_mode: false, thinking: true };

export const RENDER_CASES = [
  { name: "user-only-fake-think", request: { messages: [{ role: "user", content: "你好" }], chat_template_kwargs: off } },
  { name: "system-tools-open-think", request: { messages: [{ role: "system", content: "You are a coding agent." }, { role: "user", content: "修复 app.py 中的日期解析错误。" }], tools: TOOLS, chat_template_kwargs: on } },
  { name: "tool-round-trip", request: { tools: TOOLS, chat_template_kwargs: on, messages: [
    { role: "system", content: "You are a coding agent." },
    { role: "user", content: "运行测试并修复失败。" },
    { role: "assistant", reasoning_content: "先运行测试。", content: "", tool_calls: [call("c1", "bash", { command: "pytest -q" })] },
    { role: "tool", tool_call_id: "c1", content: "1 failed, 3 passed\nE  assert parse('2026-11-12') == date(2026, 11, 12)" },
  ] } },
  { name: "parallel-calls-and-results", request: { tools: TOOLS, chat_template_kwargs: on, messages: [
    { role: "user", content: "Compare a.txt and b.txt." },
    { role: "assistant", reasoning_content: "Read both.", content: "Reading both files.", tool_calls: [call("c1", "read", { path: "a.txt" }), call("c2", "read", { path: "b.txt" })] },
    { role: "tool", tool_call_id: "c1", content: "alpha" },
    { role: "tool", tool_call_id: "c2", content: "beta </tool_response> 😀 \"quoted\"" },
    { role: "assistant", reasoning_content: "They differ.", content: "a.txt says alpha; b.txt says beta." },
    { role: "user", content: "Thanks." },
  ] } },
];

const raw = (name, args) => `<tool_call>${JSON.stringify({ name, arguments: args })}</tool_call>`;
export const PARSE_CASES = [
  { name: "answer-fake-think", thinking: false, text: "任务完成。<EOD>", finish: "stop", matchedStop: "<EOD>" },
  { name: "reasoning-then-call", thinking: true, text: `>先读文件。</think>\n${raw("read", { path: "app.py" })}\n\nUser:`, finish: "stop", matchedStop: "\nUser:" },
  { name: "two-calls", thinking: true, text: `>Both.</think>\n${raw("read", { path: "a.txt" })}\n${raw("read", { path: "b.txt" })}\n\nUser:`, finish: "stop", matchedStop: "\nUser:" },
  { name: "closing-tag-inside-string", thinking: false, text: `${raw("write", { path: "x.md", content: "</tool_call> & <b>" })}\n\nUser:`, finish: "stop", matchedStop: "\nUser:" },
  { name: "example-in-fence-is-text", thinking: false, text: "Example:\n```\n" + raw("read", { path: "a" }) + "\n```<EOD>", finish: "stop", matchedStop: "<EOD>" },
  { name: "reject-unknown-tool", thinking: false, text: `${raw("powershell", { command: "dir" })}\n\nUser:`, finish: "stop", matchedStop: "\nUser:" },
  { name: "reject-schema-violation", thinking: false, text: `${raw("read", { path: 12 })}\n\nUser:`, finish: "stop", matchedStop: "\nUser:" },
  { name: "reject-truncated-call", thinking: false, text: '<tool_call>{"name":"read","arguments":{"pa', finish: "length" },
];

export const TOKEN_CASES = ["user-only-fake-think", "tool-round-trip"];

function render(c) { return promptForRequest(c.request).prompt; }

function parse(c) {
  const parser = new ContinuationParser(c.thinking, TOOLS);
  const deltas = [];
  try {
    for (const piece of c.text.match(/[\s\S]{1,5}/gu) ?? []) deltas.push(...parser.push(piece));
    const done = parser.finish(c.finish, c.matchedStop);
    deltas.push(...done.deltas);
    const calls = deltas.flatMap(d => d.tool_calls ?? []).map(t => ({ name: t.function.name, arguments: JSON.parse(t.function.arguments) }));
    return { reasoning: deltas.map(d => d.reasoning_content ?? "").join(""), text: deltas.map(d => d.content ?? "").join(""), calls, finish: done.reason };
  } catch (error) { return { rejected: true }; }
}

export function build({ tokens = true } = {}) {
  const renderOut = RENDER_CASES.map(c => ({ ...c, prompt: render(c) }));
  const out = {
    "render.json": { version: PROMPT_VERSION, tools: TOOLS, cases: renderOut },
    "parse.json": { version: PROMPT_VERSION, tools: TOOLS, cases: PARSE_CASES.map(c => ({ ...c, expect: parse(c) })) },
  };
  if (tokens) {
    const lock = JSON.parse(readFileSync(join(root, "lightning-source.lock.json"), "utf8"));
    out["tokens.json"] = { version: PROMPT_VERSION, tokenizer: { engine: lock.repository, revision: lock.revision, vocabSha256: lock.vocabSha256 },
      cases: TOKEN_CASES.map(name => { const prompt = renderOut.find(c => c.name === name).prompt; return { name, prompt, ids: nativeTokenize(prompt) }; }) };
  }
  return out;
}

export function check({ tokens = existsSync(nativeTokenizerPaths().binary) } = {}) {
  const want = build({ tokens }), problems = [];
  for (const [file, value] of Object.entries(want)) {
    const path = join(FIXTURES, file);
    if (!existsSync(path)) { problems.push(`${file}: missing`); continue; }
    if (JSON.stringify(JSON.parse(readFileSync(path, "utf8"))) !== JSON.stringify(value)) problems.push(`${file}: differs from this build`);
  }
  return problems;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const { values } = parseArgs({ options: { write: { type: "boolean", default: false } } });
  if (values.write) {
    mkdirSync(FIXTURES, { recursive: true });
    for (const [file, value] of Object.entries(build())) writeFileSync(join(FIXTURES, file), JSON.stringify(value, null, 1) + "\n");
    console.log(`wrote ${FIXTURES}`);
  } else {
    const problems = check();
    for (const p of problems) console.error(`FAIL ${p}`);
    if (!problems.length) console.log(`${PROMPT_VERSION} conformance fixtures match this build`);
    process.exitCode = problems.length ? 1 : 0;
  }
}
