import assert from "node:assert/strict";
import { execFile, execFileSync } from "node:child_process";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { promisify } from "node:util";
import test from "node:test";
import { buildDataset } from "../dist/dataset.js";
import { runTask } from "../dist/harness.js";
import { createRwkvProvider, readConfig } from "../dist/provider.js";
import { parseTasks } from "../dist/tasks.js";
import { copyFileResponses, mockServer, toolCall } from "./helpers/mock-server.mjs";

const task = {
  schemaVersion: 1, id: "copy-test", family: "copy-family", split: "train", source: "unit test fixture only",
  prompt: "Read input.txt and write its uppercase contents to output.txt.",
  files: { "input.txt": "hello pi\n" }, tools: ["read", "write"],
  checks: [
    { type: "file_equals", path: "output.txt", expected: "HELLO PI\n" },
    { type: "command", argv: [process.execPath, "-e", 'const hiddenAcceptanceMarker = 1; require("node:assert/strict").equal(require("node:fs").readFileSync("output.txt", "utf8"), "HELLO PI\\n")'], timeoutMs: 1000 },
  ],
};

async function setup(t, respond) {
  const outputDir = await mkdtemp(join(tmpdir(), "rwkv-pi-harness-"));
  t.after(() => rm(outputDir, { recursive: true, force: true }));
  const server = await mockServer(respond);
  t.after(server.close);
  const provider = createRwkvProvider(readConfig({ RWKV_BACKEND: "openai", RWKV_BASE_URL: server.url, RWKV_MODEL: "mock-only" }));
  return { server, options: { provider, role: "teacher", outputDir, maxRequests: 5, timeoutMs: 5000, sandbox: "none" } };
}

test("pi harness executes tools, verifies files, captures exact prompts, and exports target-only SFT", async (t) => {
  const { server, options } = await setup(t, copyFileResponses);
  const { record, path } = await runTask(task, options);
  assert.equal(record.status, "passed", record.error ?? JSON.stringify(record.checks));
  assert.equal(record.steps.length, 3);
  assert.equal(record.provenance.piMode, process.env.RWKV_PI_SOURCE ? "source" : "npm");
  assert.ok(record.steps[0].request.messages.some((message) => message.role === "system" && message.content.length > 100));
  assert.deepEqual(record.steps[0].request.messages, server.requests[0].messages);
  assert.deepEqual(record.steps[0].request.tools, server.requests[0].tools);
  assert.doesNotMatch(JSON.stringify(server.requests), /hiddenAcceptanceMarker/);
  assert.match(record.steps[1].request.messages.find((message) => message.role === "tool").content, /hello pi/);
  assert.equal(JSON.parse(await readFile(path, "utf8")).status, "passed");
  const result = buildDataset([record]);
  assert.equal(result.samples.length, 3);
  assert.equal(result.rejected.length, 0);
  assert.equal(result.samples[0].messages.at(-1).tool_calls[0].function.name, "read");
  assert.equal(result.samples[0].messages.at(-1).reasoning_content, "Read the input first.");
  assert.equal(result.samples[2].loss.messageIndex, result.samples[2].messages.length - 1);
  assert.equal(buildDataset([record, record]).samples.length, 3);
  const author={...record,role:'question-author'};
  assert.equal(buildDataset([author]).samples.length,0);
  assert.equal(buildDataset([author]).rejected[0].reason,'question_author_needs_independent_acceptance');
  assert.ok(buildDataset([author],false,new Set([record.runId])).samples.every(s=>s.role==='question-author'));

  const holdout = structuredClone(record);
  holdout.task.split = "test";
  assert.throws(() => buildDataset([record, holdout]), /leakage/);
  const smoke = { ...record, role: "smoke" };
  assert.equal(buildDataset([smoke]).samples.length, 0);
  assert.ok(buildDataset([smoke], true).samples.every((sample) => sample.split === "smoke"));
  const malformed = structuredClone(record);
  malformed.steps[0].response.content.find((part) => part.type === "toolCall").name = "not_a_pi_tool";
  assert.equal(buildDataset([malformed]).samples.length, 0);
  assert.match(buildDataset([malformed]).rejected[0].reason, /undeclared/);
});

test("a model's success claim fails acceptance when it did not modify the file", async (t) => {
  const { options } = await setup(t, () => ({ deltas: [{ content: "Done, all tests pass." }] }));
  const { record } = await runTask(task, options);
  assert.equal(record.status, "failed");
  assert.equal(buildDataset([record]).samples.length, 0);
});

test("offline teacher CLI runs the same pi tools without sending RWKV-specific parameters", async (t) => {
  const { server, options } = await setup(t, copyFileResponses);
  const manifest = join(options.outputDir, "tasks.jsonl");
  await writeFile(manifest, JSON.stringify({ ...task, split: "smoke" }) + "\n");
  const source = process.env.RWKV_PI_SOURCE;
  const preload = source ? ["--import", pathToFileURL(join(source, "packages/coding-agent/src/experimental/source-resolver.ts")).href] : [];
  const { stdout } = await promisify(execFile)(process.execPath, [
    ...preload, fileURLToPath(new URL("../dist/harness-cli.js", import.meta.url)),
    "--tasks", manifest, "--out", join(options.outputDir, "cli"), "--backend", "teacher", "--timeout-ms", "5000", "--sandbox", "none",
  ], { cwd: options.outputDir, env: { ...process.env, TEACHER_BASE_URL: server.url, TEACHER_MODEL: "mock-teacher-only", TEACHER_API_KEY: "mock" }, timeout: 15000 });
  const summary = JSON.parse(stdout.trim().split("\n").at(-1));
  assert.equal(summary.counts.passed, 1);
  assert.equal(server.requests[0].model, "mock-teacher-only");
  assert.equal(server.requests[0].chat_template_kwargs, undefined);
  assert.equal(server.requests[0].reasoning_effort, undefined);
});

test("request budget stops a tool loop without overwriting the last real response", async (t) => {
  const { server, options } = await setup(t, () => ({ deltas: [toolCall("read", { path: "input.txt" })], finish: "tool_calls" }));
  const { record } = await runTask(task, { ...options, maxRequests: 1 });
  assert.equal(record.status, "budget_exhausted");
  assert.equal(server.requests.length, 1);
  assert.equal(record.steps[0].response.stopReason, "toolUse");
  assert.equal(buildDataset([record]).samples.length, 0);
});

test("timeout aborts pending model generation and excludes the run", async (t) => {
  const { options } = await setup(t, () => null);
  const { record } = await runTask(task, { ...options, timeoutMs: 100 });
  assert.equal(record.status, "timeout");
  assert.equal(buildDataset([record]).samples.length, 0);
});

test("identical real tool failures stop before a fourth request and retain the refusal reason", async (t) => {
  const { server, options } = await setup(t, (_body, index) => ({
    deltas: [toolCall("read", { path: "." }, `directory-${index}`)], finish: "tool_calls",
  }));
  const { record } = await runTask(task, options);
  assert.equal(server.requests.length, 3);
  assert.equal(record.steps.length, 3);
  assert.equal(record.steps.at(-1).response.stopReason, "toolUse");
  assert.equal(record.status, "error");
  assert.match(record.error, /RWKV_TOOL_LOOP/);
  assert.deepEqual(record.loopGuard, { repeatFailureLimit: 3, identicalCallLimit: 2, targetFailureLimit: 3 });
  assert.equal(record.messages.filter(m => m.role === "toolResult" && m.isError).length, 3);
  assert.equal(buildDataset([record]).samples.length, 0);
});

test("a length-truncated native tool response stops the task and cannot become a successful trajectory", async (t) => {
  const { server, options } = await setup(t, (body, index) => ({ ...copyFileResponses(body, index), ...(index === 1 ? { finish: "length" } : {}) }));
  const { record } = await runTask(task, options);
  assert.equal(record.status, "error");
  assert.match(record.error, /truncated/);
  assert.equal(server.requests.length, 1);
  assert.equal(buildDataset([record]).samples.length, 0);
});

test("task manifests reject path escapes, missing verification, duplicate IDs, and family leakage", () => {
  const parse = (entries) => parseTasks(entries.map((entry) => JSON.stringify(entry)).join("\n"));
  assert.throws(() => parse([{ ...task, files: { "../escape": "bad" } }]), /escapes/);
  assert.throws(() => parse([{ ...task, checks: [] }]), /acceptance/);
  assert.throws(() => parse([task, task]), /Duplicate/);
  assert.throws(() => parse([task, { ...task, id: "other", split: "test" }]), /crosses splits/);
});

test("unsandboxed bash keeps exposeSessionEnvironment=false parity and sequential followups record conversation turns", async (t) => {
  const { options } = await setup(t, (_body, index) => {
    if (index === 1) return { deltas: [toolCall("write", { path: "output.txt", content: "TURN1\n" }, "f1")], finish: "tool_calls" };
    if (index === 2) return { deltas: [{ content: "Turn 1 complete." }] };
    if (index === 3) return { deltas: [toolCall("write", { path: "output.txt", content: "HELLO PI\n" }, "f2")], finish: "tool_calls" };
    return { deltas: [{ content: "Turn 2 complete." }] };
  });
  const bashTask = { ...task, tools: ["read", "write", "bash"] };
  const { record } = await runTask(bashTask, {
    ...options,
    maxRequests: 6,
    historyMode: "full",
    appendSystemPrompt: "TEST_APPEND_CONSTRAINT",
    promptSuffix: "TEST_SUFFIX",
    followups: ["Now overwrite output.txt with HELLO PI."],
  });
  assert.equal(record.status, "passed", record.error);
  assert.equal(record.historyMode, "full");
  assert.equal(record.appendSystemPrompt, "TEST_APPEND_CONSTRAINT");
  assert.equal(record.promptSuffix, "TEST_SUFFIX");
  assert.equal(record.conversation?.allTurnsCompleted, true);
  assert.equal(record.conversation?.turns.length, 2);
  const sysText = record.steps[0].request.messages[0].content;
  assert.doesNotMatch(sysText, /PI_\* environment variables/, "unsandboxed bash matches sandbox prompt rules");
  assert.match(sysText, /TEST_APPEND_CONSTRAINT/);
});

test("provenance refuses to label npm code as a source checkout just because an env variable is set", () => {
  const root = fileURLToPath(new URL("../", import.meta.url));
  assert.throws(() => execFileSync(process.execPath, ["--input-type=module", "-e",
    'import { runtimeProvenance } from "./dist/provenance.js"; runtimeProvenance();',
  ], {
    cwd: root, env: { ...process.env, RWKV_PI_SOURCE: join(root, ".upstream/pi") }, stdio: "pipe",
  }), (error) => {
    assert.match(error.stderr.toString(), /does not match the loaded pi runtime/);
    return true;
  });
});
