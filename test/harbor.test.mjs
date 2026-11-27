import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { existsSync } from "node:fs";
import { cp, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { dirname, join, posix } from "node:path";
import test from "node:test";
import {
  createBashToolDefinition, createEditToolDefinition, createLsToolDefinition,
  createReadToolDefinition, createWriteToolDefinition,
} from "@earendil-works/pi-coding-agent";
import { readConfig } from "../dist/config.js";
import {
  discoverHarborTasks, EndpointPool, loadHarborTask, parseHarborToml,
  runHarborBenchmark, splitReusableDockerfile,
} from "../dist/harbor.js";
import { hostFs } from "../dist/tolerant-tools.js";

test("parseHarborToml and loadHarborTask support single-turn, turns.json, and v1.4 multi-step tasks", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "rwkv-harbor-parse-"));
  t.after(() => rm(root, { recursive: true, force: true }));

  // 1. Single-turn task
  const singleDir = join(root, "single-probe");
  await mkdir(join(singleDir, "environment"), { recursive: true });
  await mkdir(join(singleDir, "tests"), { recursive: true });
  await writeFile(join(singleDir, "task.toml"), [
    'version = "1.0"',
    "[metadata]",
    'category = "probe"',
    "[agent]",
    "timeout_sec = 120.0",
    "[verifier]",
    "timeout_sec = 60.0",
    "[environment]",
    "cpus = 2",
    'memory = "2G"',
    "allow_internet = false",
  ].join("\n"));
  await writeFile(join(singleDir, "instruction.md"), "Solve single probe.\n");
  await writeFile(join(singleDir, "environment", "Dockerfile"), "FROM ubuntu:24.04\nWORKDIR /app\n");
  await writeFile(join(singleDir, "tests", "test.sh"), "#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n");

  const single = await loadHarborTask(singleDir);
  assert.equal(single.id, "single-probe");
  assert.equal(single.steps.length, 1);
  assert.equal(single.steps[0].instruction, "Solve single probe.");
  assert.equal(single.environment.memoryMb, 2048);
  assert.equal(single.environment.allowInternet, false);
  assert.ok(single.steps[0].testsDir?.endsWith("/tests"));

  // 2. Multi-turn task via environment/turns.json
  const turnsDir = join(root, "turns-task");
  await mkdir(join(turnsDir, "environment"), { recursive: true });
  await mkdir(join(turnsDir, "tests"), { recursive: true });
  await writeFile(join(turnsDir, "task.toml"), 'version = "1.0"\n[environment]\ndocker_image = "python:3.12-slim"\n');
  await writeFile(join(turnsDir, "environment", "turns.json"), JSON.stringify(["Turn one.", { prompt: "Turn two." }]));
  await writeFile(join(turnsDir, "tests", "test.sh"), "#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n");

  const multiTurn = await loadHarborTask(turnsDir);
  assert.equal(multiTurn.steps.length, 2);
  assert.equal(multiTurn.steps[0].instruction, "Turn one.");
  assert.equal(multiTurn.steps[1].instruction, "Turn two.");
  assert.equal(multiTurn.steps[0].testsDir, undefined);
  assert.ok(multiTurn.steps[1].testsDir?.endsWith("/tests"));

  // 3. Schema 1.4 multi-step task with [[steps]], [steps.agent], workdir/setup.sh, and step tests
  const stepTaskDir = join(root, "multistep-v14");
  await mkdir(join(stepTaskDir, "environment", "workspace"), { recursive: true });
  await mkdir(join(stepTaskDir, "steps", "step-1", "tests"), { recursive: true });
  await mkdir(join(stepTaskDir, "steps", "step-2", "tests"), { recursive: true });
  await mkdir(join(stepTaskDir, "steps", "step-2", "workdir"), { recursive: true });
  const v14Toml = [
    'schema_version = "1.4"',
    'multi_step_reward_strategy = "final"',
    "[[steps]]",
    'name = "step-1"',
    "[steps.agent]",
    "timeout_sec = 300.0",
    "[[steps]]",
    'name = "step-2"',
    "[steps.agent]",
    "timeout_sec = 450.0",
    "[steps.verifier]",
    "timeout_sec = 90.0",
    "[environment]",
    "cpus = 4",
    "memory_mb = 4096",
    "[environment.env]",
    'HTTP_PROXY = ""',
  ].join("\n");
  const parsedToml = parseHarborToml(v14Toml);
  assert.equal(parsedToml.schema_version, "1.4");
  assert.equal(parsedToml.steps.length, 2);
  assert.equal(parsedToml.steps[1].agent.timeout_sec, 450);

  await writeFile(join(stepTaskDir, "task.toml"), v14Toml);
  await writeFile(join(stepTaskDir, "environment", "Dockerfile"), "FROM ubuntu:24.04\nWORKDIR /app\nCOPY workspace/ /app/\n");
  await writeFile(join(stepTaskDir, "steps", "step-1", "instruction.md"), "Implement module.\n");
  await writeFile(join(stepTaskDir, "steps", "step-1", "tests", "test.sh"), "#!/bin/sh\n");
  await writeFile(join(stepTaskDir, "steps", "step-2", "instruction.md"), "Fix regression.\n");
  await writeFile(join(stepTaskDir, "steps", "step-2", "workdir", "setup.sh"), "#!/bin/sh\necho planted > fault.txt\n");
  await writeFile(join(stepTaskDir, "steps", "step-2", "tests", "test.sh"), "#!/bin/sh\n");

  const v14 = await loadHarborTask(stepTaskDir);
  assert.equal(v14.steps.length, 2);
  assert.equal(v14.steps[0].agentTimeoutSec, 300);
  assert.equal(v14.steps[1].agentTimeoutSec, 450);
  assert.ok(v14.steps[1].setupScriptPath?.endsWith("setup.sh"));
  assert.equal(v14.environment.env.HTTP_PROXY, "");

  const discovered = await discoverHarborTasks(root);
  assert.deepEqual(discovered.map((d) => d.id), ["multistep-v14", "single-probe", "turns-task"]);
});

test("splitReusableDockerfile deduplicates base layers across workspace-only tasks", () => {
  const plain = "FROM ubuntu:24.04\nRUN apt-get update && apt-get install -y python3\nWORKDIR /app\n";
  assert.deepEqual(splitReusableDockerfile(plain), { baseDockerfile: plain, copiesWorkspaceOnly: false });

  const harborExportStyle = [
    "FROM ubuntu:24.04",
    "RUN apt-get update \\",
    "    && apt-get install -y python3",
    "WORKDIR /app",
    "COPY workspace/ /app/",
    "COPY turns.json /rae/turns.json",
    "COPY web.json /rae/web.json",
    "",
  ].join("\n");
  const split = splitReusableDockerfile(harborExportStyle);
  assert.equal(split?.copiesWorkspaceOnly, true);
  assert.ok(!split?.baseDockerfile.includes("COPY workspace/"));
  assert.ok(!split?.baseDockerfile.includes("turns.json"));
  assert.ok(split?.baseDockerfile.includes("WORKDIR /app"));

  const gitInitStyle = [
    "FROM python:3.12-slim",
    "RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*",
    "WORKDIR /app",
    "COPY workspace/ /app/",
    "COPY init.sh /tmp/_init.sh",
    "RUN bash /tmp/_init.sh && rm -f /tmp/_init.sh",
  ].join("\n");
  const gitSplit = splitReusableDockerfile(gitInitStyle);
  assert.equal(gitSplit?.copiesWorkspaceOnly, true);
  assert.equal(gitSplit?.runsInitScript, true);
  assert.ok(!gitSplit?.baseDockerfile.includes("_init.sh"));

  const customContext = "FROM ubuntu:24.04\nCOPY fixture.db /opt/fixture.db\nWORKDIR /app\n";
  assert.equal(splitReusableDockerfile(customContext), undefined);
});

test("EndpointPool assigns trials least-in-flight with sticky session IDs", () => {
  const pool = new EndpointPool([
    "http://127.0.0.1:8000/v1/",
    "http://127.0.0.1:8001/v1",
    "http://127.0.0.1:8002/v1",
  ]);
  const a = pool.acquire("task-a");
  const b = pool.acquire("task-b");
  const c = pool.acquire("task-c");
  assert.deepEqual([a.endpoint, b.endpoint, c.endpoint], [
    "http://127.0.0.1:8000/v1",
    "http://127.0.0.1:8001/v1",
    "http://127.0.0.1:8002/v1",
  ]);
  assert.match(a.sessionId, /^harbor-task-a-[0-9a-f]{8}$/);
  // Release endpoint 8001; next acquire must pick 8001 (load 0 vs load 1 on 8000 and 8002).
  b.release();
  const d = pool.acquire("task-d");
  assert.equal(d.endpoint, "http://127.0.0.1:8001/v1");
  a.release();
  c.release();
  d.release();
});

/** Local directory-backed HarborSandbox so end-to-end runner tests run inside Standard Sandbox without Docker daemon. */
async function createLocalTestSandbox(rootDir) {
  const containerRoot = await mkdtemp(join(rootDir, "container-"));
  const appDir = join(containerRoot, "app");
  await mkdir(appDir, { recursive: true });
  await mkdir(join(containerRoot, "logs", "agent"), { recursive: true });
  await mkdir(join(containerRoot, "logs", "verifier"), { recursive: true });

  const mapPath = (containerPath) => {
    const rel = posix.normalize(containerPath).replace(/^\/+/, "");
    return join(containerRoot, rel);
  };
  const remapScriptPaths = (text) =>
    text.includes(containerRoot)
      ? text
      : text
          .replaceAll("/logs/", `${join(containerRoot, "logs")}/`)
          .replace(/\/tests(?=\/|\s|$)/g, join(containerRoot, "tests"))
          .replaceAll("/tmp/.rwkv-step-setup.sh", join(containerRoot, "tmp", ".rwkv-step-setup.sh"))
          .replaceAll("/app", appDir);

  const exec = (argv, opts = {}) =>
    new Promise((resolveExec, rejectExec) => {
      const mappedArgv = argv.map((arg, i) => (i >= 2 ? remapScriptPaths(arg) : arg));
      const execCwd = !opts.cwd || opts.cwd === "/app"
        ? appDir
        : opts.cwd.startsWith(containerRoot)
          ? opts.cwd
          : mapPath(opts.cwd);
      const child = spawn(mappedArgv[0], mappedArgv.slice(1), {
        cwd: execCwd,
        env: { ...process.env, ...(opts.env ?? {}) },
        stdio: "pipe",
      });
      const out = [], err = [];
      child.stdout.on("data", (c) => { out.push(c); opts.onData?.(c); });
      child.stderr.on("data", (c) => { err.push(c); opts.onData?.(c); });
      child.once("error", rejectExec);
      child.stdin.end(opts.input);
      child.once("close", (code) => resolveExec({ exitCode: code ?? 1, stdout: Buffer.concat(out), stderr: Buffer.concat(err) }));
    });

  const sandbox = {
    cwd: appDir,
    fs: hostFs,
    exec,
    readFile: (p) => readFile(p.startsWith(containerRoot) ? p : mapPath(p)),
    writeFile: async (p, content) => {
      const target = p.startsWith(containerRoot) ? p : mapPath(p);
      await mkdir(dirname(target), { recursive: true });
      const text = Buffer.isBuffer(content) ? content.toString("utf8") : String(content);
      await writeFile(target, target.endsWith(".sh") ? remapScriptPaths(text) : content);
    },
    uploadDir: async (src, dest) => {
      const target = mapPath(dest);
      await mkdir(target, { recursive: true });
      await cp(src, target, { recursive: true });
      const testSh = join(target, "test.sh");
      if (existsSync(testSh)) {
        await writeFile(testSh, remapScriptPaths(await readFile(testSh, "utf8")));
      }
    },
    downloadFile: async (p) => {
      const target = mapPath(p);
      return existsSync(target) ? readFile(target) : undefined;
    },
    tools: (names) => {
      const make = {
        read: () => createReadToolDefinition(appDir),
        write: () => createWriteToolDefinition(appDir),
        edit: () => createEditToolDefinition(appDir),
        ls: () => createLsToolDefinition(appDir),
        bash: () => createBashToolDefinition(appDir, { exposeSessionEnvironment: false }),
      };
      return names.map((n) => make[n]());
    },
    close: async () => {
      await rm(containerRoot, { recursive: true, force: true });
    },
    hasTestsMounted: () => existsSync(join(containerRoot, "tests")),
  };
  return sandbox;
}

test("runHarborBenchmark executes concurrent multi-step trials, enforces anti-leak /tests, and resumes cleanly", async (t) => {
  const seenHeaders = [];
  const seenSampling = [];
  const server = createServer(async (req, res) => {
    let raw = "";
    for await (const chunk of req) raw += chunk;
    const body = JSON.parse(raw);
    if (req.url === "/v1/tokens/count") {
      res.setHeader("content-type", "application/json");
      res.end('{"tokens":120}');
      return;
    }
    seenHeaders.push(req.headers["x-rwkv-session-id"]);
    seenSampling.push({
      temperature: body.temperature,
      top_p: body.top_p,
      top_k: body.top_k,
      alpha_presence: body.alpha_presence,
      alpha_frequency: body.alpha_frequency,
    });
    const prompt = body.contents[0];
    let reply;
    const lastToolResp = prompt.lastIndexOf("User: <tool_response>");
    if (prompt.includes("Fix fault.txt")) {
      reply = lastToolResp < prompt.lastIndexOf("Fix fault.txt")
        ? '>修复故障。</think>\n<tool_call>{"name":"write","arguments":{"path":"fault.txt","content":"fixed\\n"}}</tool_call>\n\nUser:'
        : ">完成。</think>\nFINAL_ANSWER_OK<EOD>";
    } else if (prompt.includes("Write answer.txt") && lastToolResp < prompt.lastIndexOf("Write answer.txt")) {
      reply = '>写文件。</think>\n<tool_call>{"name":"write","arguments":{"path":"answer.txt","content":"42\\n"}}</tool_call>\n\nUser:';
    } else {
      reply = ">完成。</think>\nFINAL_ANSWER_OK<EOD>";
    }
    const matchedStop = reply.endsWith("User:") ? "\nUser:" : "<EOD>";
    res.writeHead(200, { "content-type": "text/event-stream" });
    res.write(`data: ${JSON.stringify({ id: "req-1", object: "chat.completion.chunk", choices: [{ index: 0, delta: { content: reply }, finish_reason: null }] })}\n\n`);
    res.write(`data: ${JSON.stringify({ id: "req-1", object: "chat.completion.chunk", choices: [{ index: 0, delta: {}, finish_reason: "stop", matched_stop: matchedStop }] })}\n\n`);
    res.write("data: [DONE]\n\n");
    res.end();
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  t.after(() => new Promise((r) => { server.close(r); server.closeAllConnections(); }));

  const tmp = await mkdtemp(join(tmpdir(), "rwkv-harbor-e2e-"));
  t.after(() => rm(tmp, { recursive: true, force: true }));
  const tasksDir = join(tmp, "tasks");
  const outDir = join(tmp, "out");

  // Task A: single-turn writing answer.txt + checking /logs/agent/final-1.txt, ws-1.json, wsfiles-1.json & events-1.jsonl (`"tool":"write"`)
  const taskADir = join(tasksDir, "task-a");
  await mkdir(join(taskADir, "environment"), { recursive: true });
  await mkdir(join(taskADir, "tests"), { recursive: true });
  await writeFile(join(taskADir, "task.toml"), 'version = "1.0"\n[environment]\ndocker_image = "mock:latest"\n');
  await writeFile(join(taskADir, "instruction.md"), "Write answer.txt with 42.\n");
  await writeFile(
    join(taskADir, "tests", "test.sh"),
    '#!/bin/sh\nif [ "$(cat answer.txt)" = "42" ] && grep -q "FINAL_ANSWER_OK" /logs/agent/final-1.txt && grep -q \'"tool":"write"\' /logs/agent/events-1.jsonl && grep -q "answer.txt" /logs/agent/ws-1.json && grep -q "42" /logs/agent/wsfiles-1.json; then echo 1.0 > /logs/verifier/reward.txt; echo \'{"score":1.0}\' > /logs/verifier/details.json; else echo 0.0 > /logs/verifier/reward.txt; fi\n',
  );

  // Task B: 2-step v1.4 task with setup.sh fault injection on step 2 + reward.json
  const taskBDir = join(tasksDir, "task-b");
  await mkdir(join(taskBDir, "environment"), { recursive: true });
  await mkdir(join(taskBDir, "steps", "step-1", "tests"), { recursive: true });
  await mkdir(join(taskBDir, "steps", "step-2", "workdir"), { recursive: true });
  await mkdir(join(taskBDir, "steps", "step-2", "tests"), { recursive: true });
  await writeFile(join(taskBDir, "task.toml"), 'schema_version = "1.4"\n[environment]\ndocker_image = "mock:latest"\n');
  await writeFile(join(taskBDir, "steps", "step-1", "instruction.md"), "Write answer.txt with 42.\n");
  await writeFile(join(taskBDir, "steps", "step-1", "tests", "test.sh"), "#!/bin/sh\n");
  await writeFile(join(taskBDir, "steps", "step-2", "workdir", "setup.sh"), "#!/bin/sh\necho broken > /app/fault.txt\n");
  await writeFile(join(taskBDir, "steps", "step-2", "instruction.md"), "Fix fault.txt.\n");
  await writeFile(
    join(taskBDir, "steps", "step-2", "tests", "test.sh"),
    '#!/bin/sh\nif [ "$(cat fault.txt)" = "fixed" ]; then echo \'{"reward": 1.0, "tests_pass": 1.0}\' > /logs/verifier/reward.json; else echo \'{"reward": 0.0}\' > /logs/verifier/reward.json; fi\n',
  );

  let failTaskBFirstRun = true;
  let antiLeakVerified = true;
  const baseUrl = `http://127.0.0.1:${server.address().port}/v1`;
  const baseConfig = readConfig({ RWKV_BASE_URL: baseUrl });

  const summary1 = await runHarborBenchmark({
    tasksPath: tasksDir,
    outputDir: outDir,
    baseConfig,
    endpoints: [baseUrl],
    concurrency: 2,
    thinkingLevel: "high",
    sandboxFactory: async (task) => {
      if (task.id === "task-b" && failTaskBFirstRun) {
        failTaskBFirstRun = false;
        throw new Error("Simulated transient container start timeout");
      }
      const sb = await createLocalTestSandbox(tmp);
      if (sb.hasTestsMounted()) antiLeakVerified = false;
      return { sandbox: sb };
    },
  });

  assert.equal(antiLeakVerified, true);
  assert.equal(summary1.totals.planned, 2);
  assert.equal(summary1.totals.valid, 1);
  assert.equal(summary1.totals.infraErrors, 1);
  assert.equal(summary1.totals.meanReward, 1);
  assert.ok(seenHeaders.every((h) => typeof h === "string" && h.startsWith("harbor-")));
  assert.deepEqual(seenSampling[0], {
    temperature: 0.5,
    top_p: 0.7,
    top_k: 65536,
    alpha_presence: 0.8,
    alpha_frequency: 0.2,
  });

  // Second run with resume=true: task-a (completed) must be skipped, task-b (infra_error) must be retried and pass!
  const requestsBeforeResume = seenHeaders.length;
  const summary2 = await runHarborBenchmark({
    tasksPath: tasksDir,
    outputDir: outDir,
    baseConfig,
    endpoints: [baseUrl],
    concurrency: 2,
    thinkingLevel: "high",
    resume: true,
    sandboxFactory: async () => ({ sandbox: await createLocalTestSandbox(tmp) }),
  });

  assert.equal(summary2.totals.valid, 2);
  assert.equal(summary2.totals.infraErrors, 0);
  assert.equal(summary2.totals.passed, 2);
  assert.equal(summary2.totals.meanReward, 1);
  assert.equal(summary2.totals.passAtK, 1);
  // Task A took 2 requests; resume should only execute Task B's 4 requests (2 steps x 2 requests).
  assert.equal(seenHeaders.length - requestsBeforeResume, 4);
  assert.ok(existsSync(join(outDir, "trials", "task-a", "details.json")));
  assert.equal(summary2.totals.totalPoints, 20);
  assert.equal(summary2.totals.maxPoints, 20);
  assert.ok(existsSync(join(outDir, "SCOREBOARD.md")));
  assert.match(await readFile(join(outDir, "SCOREBOARD.md"), "utf8"), /100\.00%/);
});
