#!/usr/bin/env node
// rwkv-pi-orchestrate: interactive pi with a teacher main agent and parallel RWKV subagents.
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { loadEnvFile } from "node:process";
import { fileURLToPath, pathToFileURL } from "node:url";
import { readConfig } from "./config.js";
import { subagentLimits, subagentRoot } from "./subagents.js";
import { readTeacherConfig } from "./teacher.js";

const ORCHESTRATOR_TOOLS = ["read", "grep", "find", "ls", "subagent_plan",
  "subagent_start", "subagent_status", "subagent_wait", "subagent_diff", "subagent_apply", "subagent_cancel", "project_participation"];

/** The server's own context length when its /models reports one (vLLM max_model_len and similar). */
async function serverContext(baseUrl: string, apiKey: string, modelId: string): Promise<number | undefined> {
  try {
    const response = await fetch(baseUrl + "/models", { headers: { Authorization: `Bearer ${apiKey}` }, signal: AbortSignal.timeout(8000) });
    if (!response.ok) return;
    const body: any = await response.json();
    const values = (body?.data ?? []).filter((m: any) => m.id === modelId).map((m: any) => Number(m.max_model_len ?? m.context_length ?? m.context_window ?? m.meta?.n_ctx_train)).filter((n: number) => Number.isSafeInteger(n) && n > 0);
    return values.length ? Math.max(...values) : undefined;
  } catch { return; }
}

try {
  // The working directory is usually the user's project, not rwkv-pi: also read rwkv-pi's own .env.
  // Earlier files and explicitly exported variables take precedence.
  const projectEnv = fileURLToPath(new URL("../.env", import.meta.url));
  for (const file of [resolve(".env"), projectEnv]) if (existsSync(file)) loadEnvFile(file);
  readConfig(); subagentLimits(); subagentRoot();          // fail before starting pi, not mid-session
  const teacher = readTeacherConfig();
  if (teacher.kind === "deepseek" && !process.env.TEACHER_BUDGET_LEDGER) throw new Error("Use scripts/run-deepseek-subagents.py for the shared CNY 10 budget.");
  const probed = await serverContext(teacher.baseUrl, teacher.apiKey, teacher.modelId);
  if (probed && probed < teacher.contextWindow) {
    // A context larger than the server's makes long sessions fail instead of compacting.
    process.env.TEACHER_CONTEXT_WINDOW = String(probed);
    if (teacher.maxTokens >= probed) process.env.TEACHER_MAX_TOKENS = String(Math.floor(probed / 4));
    console.error(`rwkv-pi-orchestrate: ${teacher.modelId} reports context ${probed}; using it instead of ${teacher.contextWindow}.`);
  } else if (!probed) console.error(`rwkv-pi-orchestrate: ${teacher.modelId} context not reported; using TEACHER_CONTEXT_WINDOW=${teacher.contextWindow}, TEACHER_MAX_TOKENS=${teacher.maxTokens}.`);
  console.error(`rwkv-pi-orchestrate: main=${teacher.modelId}; subagents=${readConfig().modelId}; concurrency limit=${subagentLimits().concurrency}.`);
  // The seven-tool pi used for RWKV training (grep/find spawn hooks) is the pi-next source checkout.
  const piNext = fileURLToPath(new URL("../.upstream/pi-next", import.meta.url));
  if (!process.env.RWKV_PI_SOURCE && existsSync(join(piNext, "packages/coding-agent/src/cli.ts"))) process.env.RWKV_PI_SOURCE = piNext;
  const source = process.env.RWKV_PI_SOURCE ? resolve(process.env.RWKV_PI_SOURCE) : undefined;
  const piEntry = fileURLToPath(import.meta.resolve("@earendil-works/pi-coding-agent"));
  const piCli = source ? join(source, "packages/coding-agent/src/cli.ts") : join(dirname(piEntry), "bundle", "cli.js");
  const preload = source ? ["--import", pathToFileURL(join(source, "packages/coding-agent/src/experimental/source-resolver.ts")).href] : [];
  const userArgs = process.argv.slice(2);
  const userTools = userArgs.some((arg) => arg === "--tools" || arg === "-t" || arg.startsWith("--tools="));
  const child = spawn(process.execPath, [
    ...preload, piCli,
    "--extension", fileURLToPath(new URL("./orchestrator-extension.js", import.meta.url)),
    "--provider", "teacher", "--model", teacher.modelId, "--thinking", "high",
    ...(userTools ? [] : ["--tools", ORCHESTRATOR_TOOLS.join(",")]),
    ...userArgs,
  ], { stdio: "inherit", env: process.env });
  const forward = (signal: NodeJS.Signals) => child.kill(signal);
  process.on("SIGINT", forward); process.on("SIGTERM", forward);
  child.once("error", (error) => { console.error(`rwkv-pi-orchestrate: ${error.message}`); process.exitCode = 1; });
  child.once("exit", (code, signal) => {
    process.removeListener("SIGINT", forward); process.removeListener("SIGTERM", forward);
    if (signal) process.kill(process.pid, signal); else process.exitCode = code ?? 1;
  });
} catch (error) {
  console.error(`rwkv-pi-orchestrate: ${error instanceof Error ? error.message : String(error)}`);
  process.exitCode = 1;
}
