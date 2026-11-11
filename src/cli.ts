#!/usr/bin/env node
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { loadEnvFile } from "node:process";
import { fileURLToPath, pathToFileURL } from "node:url";
import { readConfig } from "./config.js";
import { DEFAULT_PI_TOOLS, parseToolList } from "./tools.js";
import { readTeacherConfig } from "./teacher.js";
import { stallLimits } from "./stall-detector.js";
import { collaborationDataRoot } from "./collab-trace.js";
import { resolvePlugins } from "./plugins.js";

try {
  // Preserve the user's cwd, while supporting launches outside this checkout.
  for (const file of [resolve(".env"), fileURLToPath(new URL("../.env", import.meta.url))]) {
    if (existsSync(file)) loadEnvFile(file);
  }
  const config = readConfig();
  const source = process.env.RWKV_PI_SOURCE ? resolve(process.env.RWKV_PI_SOURCE) : undefined;
  const piEntry = fileURLToPath(import.meta.resolve("@earendil-works/pi-coding-agent"));
  const piCli = source ? join(source, "packages/coding-agent/src/cli.ts") : join(dirname(piEntry), "bundle", "cli.js");
  const preload = source ? ["--import", pathToFileURL(join(source, "packages/coding-agent/src/experimental/source-resolver.ts")).href] : [];
  const extension = fileURLToPath(new URL("./extension.js", import.meta.url));
  // An explicit allowlist keeps the prompt independent of local pi settings; RWKV_TOOLS must match training data.
  const plugins = resolvePlugins(process.argv.slice(2));
  const collab = plugins.enabled.includes("strong-model");
  const subagents = plugins.enabled.includes("subagents");
  // Fail visibly before starting pi, rather than silently losing an optional extension.
  if (collab) {
    const teacher = readTeacherConfig();
    if (teacher.kind === "deepseek" && !process.env.TEACHER_BUDGET_LEDGER) throw new Error("Use rwkv-pi-repair for the shared paid trial budget.");
    stallLimits(); collaborationDataRoot();
  }
  let subagentTools: string[] = [];
  if (subagents) {
    const { subagentLimits, subagentRoot } = await import("./subagents.js");
    const { SUBAGENT_CONTROL_TOOLS } = await import("./orchestrator-extension.js");
    subagentLimits(); subagentRoot();
    subagentTools = [...SUBAGENT_CONTROL_TOOLS];
  }
  const userArgs = plugins.args;
  const userTools = userArgs.some((arg) => arg === "--tools" || arg === "-t" || arg.startsWith("--tools="));
  const tools = userTools ? [] : ["--tools", [...parseToolList(process.env.RWKV_TOOLS?.trim() || DEFAULT_PI_TOOLS), ...subagentTools].join(",")];
  // Training parity: the harness renders no skills, context files or prompt templates.
  // RWKV_PI_PARITY=0 restores pi's discovery for exploratory use (the model never saw those prompts).
  const parity = process.env.RWKV_PI_PARITY?.trim() !== "0";
  const parityArgs = parity ? ["--no-skills", "--no-context-files", "--no-prompt-templates"] : [];
  const child = spawn(process.execPath, [
    ...preload,
    piCli,
    // Optional capabilities come from explicit configuration, never auto-discovered extensions.
    "--no-extensions",
    "--extension", extension,
    ...(collab ? ["--extension", fileURLToPath(new URL("./collab-extension.js", import.meta.url))] : []),
    ...(subagents ? ["--extension", fileURLToPath(new URL("./subagent-extension.js", import.meta.url))] : []),
    "--provider", "rwkv",
    "--model", config.modelId,
    "--thinking", "high",
    ...tools,
    ...parityArgs,
    ...userArgs,
  ], { stdio: "inherit", env: process.env });

  const forward = (signal: NodeJS.Signals) => child.kill(signal);
  process.on("SIGINT", forward);
  process.on("SIGTERM", forward);
  child.once("error", (error) => {
    console.error(`rwkv-pi: ${error.message}`);
    process.exitCode = 1;
  });
  child.once("exit", (code, signal) => {
    process.removeListener("SIGINT", forward);
    process.removeListener("SIGTERM", forward);
    if (signal) process.kill(process.pid, signal);
    else process.exitCode = code ?? 1;
  });
} catch (error) {
  console.error(`rwkv-pi: ${error instanceof Error ? error.message : String(error)}`);
  process.exitCode = 1;
}
