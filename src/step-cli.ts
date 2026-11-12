#!/usr/bin/env node
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { loadEnvFile } from "node:process";
import { fileURLToPath, pathToFileURL } from "node:url";
import { readStepConfig } from "./step-teacher.js";
import { PI_TOOL_NAMES } from "./tools.js";

try {
  const projectEnv = fileURLToPath(new URL("../.env", import.meta.url));
  for (const file of [resolve(".env"), projectEnv]) if (existsSync(file)) loadEnvFile(file);
  const config = readStepConfig();
  const piNext = fileURLToPath(new URL("../.upstream/pi", import.meta.url));
  const source = process.env.RWKV_PI_SOURCE ? resolve(process.env.RWKV_PI_SOURCE)
    : existsSync(join(piNext, "packages/coding-agent/src/cli.ts")) ? piNext : undefined;
  const entry = fileURLToPath(import.meta.resolve("@earendil-works/pi-coding-agent"));
  const preload = source ? ["--import", pathToFileURL(join(source, "packages/coding-agent/src/experimental/source-resolver.ts")).href] : [];
  const piCli = source ? join(source, "packages/coding-agent/src/cli.ts") : join(dirname(entry), "bundle/cli.js");
  const args = process.argv.slice(2);
  const customTools = args.some(a => a === "--tools" || a === "-t" || a.startsWith("--tools="));
  const child = spawn(process.execPath, [...preload, piCli,
    "--no-extensions", "--no-skills", "--no-context-files", "--no-prompt-templates",
    "--extension", fileURLToPath(new URL("./step-extension.js", import.meta.url)),
    "--provider", "stepfun", "--model", config.modelId, "--thinking", config.reasoningEffort || "high",
    ...(customTools ? [] : ["--tools", PI_TOOL_NAMES.join(",")]), ...args,
  ], { stdio: "inherit", env: process.env });
  const forward = (signal: NodeJS.Signals) => child.kill(signal);
  process.on("SIGINT", forward); process.on("SIGTERM", forward);
  child.once("error", error => { console.error(`rwkv-pi-step: ${error.message}`); process.exitCode = 1; });
  child.once("exit", (code, signal) => {
    process.removeListener("SIGINT", forward); process.removeListener("SIGTERM", forward);
    if (signal) process.kill(process.pid, signal); else process.exitCode = code ?? 1;
  });
} catch (error) {
  console.error(`rwkv-pi-step: ${error instanceof Error ? error.message : String(error)}`);
  process.exitCode = 1;
}
