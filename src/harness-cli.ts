#!/usr/bin/env node
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { loadEnvFile } from "node:process";
import { parseArgs } from "node:util";
import { createRwkvProvider, readConfig } from "./provider.js";
import { runTask } from "./harness.js";
import { createTeacherProvider } from "./teacher.js";
import { parseTasks } from "./tasks.js";
import { sha256 } from "./provenance.js";

try {
  if (existsSync(".env")) loadEnvFile(".env");
  const { values } = parseArgs({ options: {
    tasks: { type: "string" }, out: { type: "string" }, backend: { type: "string", default: "rwkv" },
    "max-requests": { type: "string", default: "32" }, "timeout-ms": { type: "string", default: "300000" },
    sandbox: { type: "string", default: "bubblewrap" },
    "workspace-mount": { type: "string" },
    deployment: { type: "string" },
    thinking: { type: "string", default: "high" },
    help: { type: "boolean" },
  } });
  if (values.help) {
    console.log("Usage: npm run harness -- --tasks tasks.jsonl --out runs/batch --backend rwkv|teacher [--max-requests 32 --timeout-ms 300000 --thinking high|off --workspace-mount /srv/run-id/project]");
  } else {
    if (!values.tasks || !values.out) throw new Error("--tasks and --out are required.");
    if (values.backend !== "rwkv" && values.backend !== "teacher") throw new Error("--backend must be rwkv or teacher.");
    if (values.sandbox !== "bubblewrap" && values.sandbox !== "none") throw new Error("--sandbox must be bubblewrap or none.");
    if (values.thinking !== "high" && values.thinking !== "off") throw new Error("--thinking must be high or off.");
    const tasks = parseTasks(await readFile(values.tasks, "utf8"));
    const maxRequests = Number(values["max-requests"]);
    const timeoutMs = Number(values["timeout-ms"]);
    let inference;
    if (values.deployment) {
      if (values.backend !== "rwkv") throw new Error("--deployment identifies the RWKV server; do not attach it to teacher runs.");
      const raw = await readFile(values.deployment);
      const deployment = JSON.parse(raw.toString());
      if (deployment.schemaVersion !== 1 || typeof deployment.engineRevision !== "string" || !deployment.metadataSha256) throw new Error("Invalid deployment record.");
      inference = { deploymentSha256: sha256(raw), model: deployment.artifact as string, engineRevision: deployment.engineRevision as string, metadataSha256: deployment.metadataSha256 as Record<string, string> };
    }
    let provider;
    const rwkvConfig = values.backend === "rwkv" ? readConfig() : undefined;
    if (rwkvConfig) provider = createRwkvProvider(rwkvConfig);
    else {
      provider = createTeacherProvider();
    }
    const counts: Record<string, number> = {};
    for (const task of tasks) {
      const { record, path } = await runTask(task, {
        provider, role: task.split === "smoke" ? "smoke" : values.backend === "teacher" ? "teacher" : "student",
        outputDir: values.out, maxRequests, timeoutMs,
        sandbox: values.sandbox,
        workspaceMount: values["workspace-mount"],
        inference,
        contextBudget: rwkvConfig,
        thinkingLevel: values.thinking,
      });
      counts[record.status] = (counts[record.status] ?? 0) + 1;
      console.log(JSON.stringify({ task: task.id, status: record.status, requests: record.steps.length, path }));
    }
    console.log(JSON.stringify({ tasks: tasks.length, counts }));
    if (Object.keys(counts).some((status) => status !== "passed")) process.exitCode = 1;
  }
} catch (error) {
  console.error(`rwkv-pi harness: ${error instanceof Error ? error.message : String(error)}`);
  process.exitCode = 1;
}
