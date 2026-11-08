#!/usr/bin/env node
// rwkv-pi-collab: RWKV works on a task first, Qwen continues from RWKV's workspace, and a record
// shows exactly what Qwen changed. Nothing here runs unless this command is invoked explicitly.
//
//   collab run    --task T.jsonl [--out DIR]            RWKV attempt, then Qwen continuation, then report
//   collab report --rwkv-run R/run.json --qwen-run Q/run.json [--out DIR]
//   collab scan   --runs WAVE_RUNS_DIR [--out DIR]      records for every pipeline RWKV->Qwen repair pair
import { existsSync, readdirSync, statSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { loadEnvFile } from "node:process";
import { parseArgs } from "node:util";
import {
  attemptEvidence, buildRecord, continuationTask, finalAnswer, readTree, renderMarkdown, sha256, toolActions,
  type CollaborationRecord, type Participant,
} from "./collaboration.js";
import { runTask, type RunRecord } from "./harness.js";
import { createRwkvProvider, readConfig } from "./provider.js";
import { parseTasks } from "./tasks.js";
import { createTeacherProvider } from "./teacher.js";

const USAGE = `Usage:
  npm run collab -- run --task task.jsonl [--out DIR] [--max-requests 32] [--timeout-ms 900000] [--thinking high|off] [--qwen-always]
  npm run collab -- report --rwkv-run RWKV/run.json --qwen-run QWEN/run.json [--out DIR]
  npm run collab -- scan --runs /mnt/f/rwkv-pi-data/runs/<wave> [--out DIR]
Output defaults to $RWKV_PI_DATA_ROOT/runs/collaboration (RWKV_PI_DATA_ROOT defaults to /mnt/f/rwkv-pi-data).`;

export function dataRoot(): string {
  const root = resolve(process.env.RWKV_PI_DATA_ROOT ?? "/mnt/f/rwkv-pi-data");
  // Never fall back to another disk when the data root is unavailable.
  if (!existsSync(root)) throw new Error(`Data root ${root} is not available; set RWKV_PI_DATA_ROOT.`);
  return root;
}

export function participant(record: RunRecord, path: string, raw: Buffer): Participant {
  return {
    model: record.model.id, run: path, runSha256: sha256(raw), status: record.status, error: record.error,
    requests: record.steps.length, actions: toolActions(record.messages), finalAnswer: finalAnswer(record.messages),
  };
}

async function loadRun(path: string): Promise<{ record: RunRecord; raw: Buffer; path: string }> {
  const full = resolve(path);
  const raw = await readFile(full);
  return { record: JSON.parse(raw.toString()) as RunRecord, raw, path: full };
}

/** Compare two finished runs: RWKV's run (original task files -> its workspace) and Qwen's continuation. */
export async function reportFromRuns(rwkvPath: string, qwenPath: string, mode: CollaborationRecord["mode"], exported = false): Promise<CollaborationRecord> {
  const rwkv = await loadRun(rwkvPath), qwen = await loadRun(qwenPath);
  const rwkvTree = readTree(join(dirname(rwkv.path), "workspace"));
  const qwenTree = readTree(join(dirname(qwen.path), "workspace"));
  const verified = qwen.record.status === "passed" && qwen.record.checks.length > 0 && qwen.record.checks.every((c) => c.passed);
  return buildRecord({
    mode, task: { id: rwkv.record.task.id, prompt: rwkv.record.task.prompt, sha256: rwkv.record.taskSha256 },
    original: rwkv.record.task.files, rwkvFiles: rwkvTree.files, qwenFiles: qwenTree.files,
    rwkv: participant(rwkv.record, rwkv.path, rwkv.raw), qwen: participant(qwen.record, qwen.path, qwen.raw),
    acceptance: {
      verified, checks: qwen.record.checks,
      note: verified ? "every independent task check passed on Qwen's final workspace" : "Qwen's final workspace did not pass every independent check",
    },
    skipped: [...rwkvTree.skipped.map((s) => ({ ...s, path: "rwkv:" + s.path })), ...qwenTree.skipped.map((s) => ({ ...s, path: "qwen:" + s.path }))],
    exported,
  });
}

export async function writeRecord(record: CollaborationRecord, directory: string): Promise<string> {
  const target = join(directory, record.id);
  await mkdir(target, { recursive: true });
  await writeFile(join(target, "record.json"), JSON.stringify(record, null, 2) + "\n");
  await writeFile(join(target, "report.md"), renderMarkdown(record));
  return target;
}

/** Pipeline repair runs (trajectoryStrategy student-failure-repair) paired with their RWKV source run. */
function repairRuns(runs: string): string[] {
  const found: string[] = [];
  const registry = join(runs, "teacher", "registry");
  if (!existsSync(registry)) return found;
  for (const job of readdirSync(registry)) {
    const jobDir = join(registry, job);
    if (!statSync(jobDir).isDirectory()) continue;
    for (const attempt of readdirSync(jobDir).filter((name) => name.startsWith("attempt-"))) {
      const runsDir = join(jobDir, attempt, "runs");
      if (!existsSync(runsDir)) continue;
      for (const run of readdirSync(runsDir)) {
        const path = join(runsDir, run, "run.json");
        if (existsSync(path)) found.push(path);
      }
    }
  }
  return found;
}

async function main(): Promise<void> {
  if (existsSync(".env")) loadEnvFile(".env");
  const [command, ...rest] = process.argv.slice(2);
  const { values } = parseArgs({ args: rest, options: {
    task: { type: "string" }, out: { type: "string" }, "rwkv-run": { type: "string" }, "qwen-run": { type: "string" },
    runs: { type: "string" }, "max-requests": { type: "string", default: "32" }, "timeout-ms": { type: "string", default: "900000" },
    "teacher-max-requests": { type: "string", default: "64" }, "teacher-timeout-ms": { type: "string", default: "3600000" },
    thinking: { type: "string", default: "high" }, sandbox: { type: "string", default: "bubblewrap" },
    "qwen-always": { type: "boolean", default: false }, help: { type: "boolean" },
  } });
  if (values.help || !command || command === "help") { console.log(USAGE); return; }
  const out = resolve(values.out ?? join(dataRoot(), "runs", "collaboration", command === "scan" ? "pipeline" : "harness"));

  if (command === "report") {
    if (!values["rwkv-run"] || !values["qwen-run"]) throw new Error("--rwkv-run and --qwen-run are required.");
    const record = await reportFromRuns(values["rwkv-run"], values["qwen-run"], "harness");
    console.log(renderMarkdown(record));
    console.log(`Record: ${await writeRecord(record, out)}`);
    return;
  }

  if (command === "scan") {
    if (!values.runs) throw new Error("--runs is required.");
    let written = 0;
    for (const path of repairRuns(resolve(values.runs))) {
      const { record } = await loadRun(path);
      const repairOf = (record.task as unknown as { repairOf?: { sourceRun?: string } }).repairOf;
      if (!repairOf?.sourceRun || !existsSync(repairOf.sourceRun)) continue;
      // The attempt's job file says whether the teacher scheduler accepted and published it.
      const job = join(dirname(dirname(dirname(dirname(path)))), "job.json");
      const status = existsSync(job) ? (JSON.parse(await readFile(job, "utf8")) as { status?: string }).status : undefined;
      const collaboration = await reportFromRuns(repairOf.sourceRun, path, "pipeline", status === "verified_and_compiled");
      const target = await writeRecord(collaboration, out);
      console.log(JSON.stringify({ task: collaboration.task.id, rwkv: collaboration.rwkv.status, qwen: collaboration.qwen.status,
        verified: collaboration.acceptance.verified, exported: collaboration.trainingUse.exported, record: target }));
      written++;
    }
    console.log(JSON.stringify({ records: written, out }));
    return;
  }

  if (command === "run") {
    if (!values.task) throw new Error("--task is required.");
    if (values.thinking !== "high" && values.thinking !== "off") throw new Error("--thinking must be high or off.");
    if (values.sandbox !== "bubblewrap" && values.sandbox !== "none") throw new Error("--sandbox must be bubblewrap or none.");
    const tasks = parseTasks(await readFile(values.task, "utf8"));
    if (tasks.length !== 1) throw new Error("collab run takes exactly one task.");
    const task = tasks[0]!;
    const rwkvConfig = readConfig();
    const teacher = createTeacherProvider();
    const common = { sandbox: values.sandbox, thinkingLevel: values.thinking } as const;
    const rwkv = await runTask(task, { ...common, provider: createRwkvProvider(rwkvConfig), role: "student",
      outputDir: join(out, "rwkv"), maxRequests: Number(values["max-requests"]), timeoutMs: Number(values["timeout-ms"]), contextBudget: rwkvConfig });
    console.log(JSON.stringify({ rwkv: rwkv.record.status, requests: rwkv.record.steps.length, run: rwkv.path }));
    if (rwkv.record.status === "passed" && !values["qwen-always"]) {
      console.log("RWKV passed every check; no Qwen continuation needed (use --qwen-always to have Qwen review it anyway).");
      return;
    }
    const rwkvRaw = await readFile(join(rwkv.path));
    const rwkvTree = readTree(join(dirname(rwkv.path), "workspace"));
    const continuation = continuationTask(task, rwkvTree.files, attemptEvidence(participant(rwkv.record, rwkv.path, rwkvRaw), rwkv.record.messages));
    const qwen = await runTask(continuation, { ...common, provider: teacher, role: "teacher",
      outputDir: join(out, "qwen"), maxRequests: Number(values["teacher-max-requests"]), timeoutMs: Number(values["teacher-timeout-ms"]) });
    console.log(JSON.stringify({ qwen: qwen.record.status, requests: qwen.record.steps.length, run: qwen.path }));
    const record = await reportFromRuns(rwkv.path, qwen.path, "harness");
    console.log(renderMarkdown(record));
    console.log(`Record: ${await writeRecord(record, join(out, "records"))}`);
    if (!record.acceptance.verified) process.exitCode = 1;
    return;
  }
  throw new Error(`Unknown command ${command}.\n${USAGE}`);
}

if (import.meta.url === `file://${process.argv[1]}` || process.argv[1]?.endsWith("collaboration-cli.js")) {
  main().catch((error: unknown) => {
    console.error(`rwkv-pi collab: ${error instanceof Error ? error.message : String(error)}`);
    process.exitCode = 1;
  });
}
