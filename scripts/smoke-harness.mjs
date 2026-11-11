import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { runTask } from "../dist/harness.js";
import { createRwkvProvider, readConfig } from "../dist/provider.js";
import { parseTasks } from "../dist/tasks.js";
import { copyFileResponses, mockServer } from "../test/helpers/mock-server.mjs";

const tasks = parseTasks(await readFile(new URL("../tasks/smoke.jsonl", import.meta.url), "utf8"));
const server = await mockServer(copyFileResponses);
try {
  const provider = createRwkvProvider(readConfig({ RWKV_BACKEND: "openai", RWKV_BASE_URL: server.url, RWKV_MODEL: "scripted-smoke-only" }));
  const result = await runTask(tasks[0], {
    provider, role: "smoke", maxRequests: 4, timeoutMs: 10000,
    outputDir: fileURLToPath(new URL("../runs/smoke", import.meta.url)),
  });
  console.log(JSON.stringify({ status: result.record.status, requests: result.record.steps.length, path: result.path, model: "scripted smoke; no real model inference" }, null, 2));
  if (result.record.status !== "passed") process.exitCode = 1;
} finally {
  await server.close();
}
