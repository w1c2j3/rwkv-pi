#!/usr/bin/env node
import { existsSync } from "node:fs";
import { loadEnvFile } from "node:process";
import { parseArgs } from "node:util";
import { readConfig } from "./config.js";
import { runHarborBenchmark } from "./harbor.js";
import { parseToolList } from "./tools.js";

function parseKeyValueFlags(items: readonly string[] | undefined): Record<string, string> {
  const out: Record<string, string> = {};
  for (const item of items ?? []) {
    const eq = item.indexOf("=");
    if (eq < 1) throw new Error(`Invalid --verifier-env entry "${item}" (expected KEY=VALUE).`);
    out[item.slice(0, eq)] = item.slice(eq + 1);
  }
  return out;
}

try {
  if (existsSync(".env")) loadEnvFile(".env");
  const { values } = parseArgs({
    options: {
      tasks: { type: "string" },
      out: { type: "string" },
      endpoints: { type: "string" },
      concurrency: { type: "string", default: "4" },
      k: { type: "string", default: "1" },
      filter: { type: "string" },
      limit: { type: "string" },
      tools: { type: "string", default: "read,write,edit,bash" },
      web: { type: "boolean", default: false },
      "edit-mode": { type: "string" },
      "history-mode": { type: "string" },
      thinking: { type: "string", default: "high" },
      "max-requests": { type: "string", default: "48" },
      "max-agent-timeout-sec": { type: "string" },
      "max-verifier-timeout-sec": { type: "string" },
      "container-network": { type: "string", default: "auto" },
      "verifier-env": { type: "string", multiple: true },
      "append-system-prompt": { type: "string" },
      "cleanup-context-images": { type: "boolean", default: false },
      "no-resume": { type: "boolean", default: false },
      help: { type: "boolean", default: false },
    },
  });

  if (values.help) {
    console.log(
      "Usage: npm run harbor -- --tasks <harbor-tasks-dir> --out <output-dir> " +
      "[--endpoints http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1] [--concurrency 16] [--k 1] " +
      "[--filter substr] [--limit N] [--tools read,write,edit,bash,ls] [--web] " +
      "[--history-mode epoch|full|worklist] [--thinking high|off] [--max-requests 48] " +
      "[--container-network auto|bridge|none|host] [--verifier-env KEY=VAL] [--no-resume]",
    );
  } else {
    if (!values.tasks || !values.out) throw new Error("--tasks and --out are required.");
    if (values.thinking !== "high" && values.thinking !== "off") throw new Error("--thinking must be high or off.");
    const editRaw = values["edit-mode"] ?? process.env.RWKV_EDIT_MODE ?? "line";
    if (editRaw !== "line" && editRaw !== "text") {
      throw new Error("--edit-mode must be line or text.");
    }
    const historyRaw = values["history-mode"] ?? process.env.RWKV_HISTORY_MODE ?? "epoch";
    if (historyRaw !== "full" && historyRaw !== "epoch" && historyRaw !== "worklist") {
      throw new Error("--history-mode must be full, epoch, or worklist.");
    }
    const netRaw = values["container-network"] ?? "auto";
    if (netRaw !== "auto" && netRaw !== "bridge" && netRaw !== "none" && netRaw !== "host") {
      throw new Error("--container-network must be auto, bridge, none, or host.");
    }
    const baseConfig = readConfig();
    const endpoints = values.endpoints
      ? values.endpoints.split(",").map((s) => s.trim()).filter(Boolean)
      : [baseConfig.baseUrl];
    const concurrency = Number(values.concurrency);
    const repetitions = Number(values.k);
    const maxRequests = Number(values["max-requests"]);
    if (!Number.isSafeInteger(concurrency) || concurrency < 1) throw new Error("--concurrency must be a positive integer.");
    if (!Number.isSafeInteger(repetitions) || repetitions < 1) throw new Error("--k must be a positive integer.");
    if (!Number.isSafeInteger(maxRequests) || maxRequests < 1) throw new Error("--max-requests must be a positive integer.");
    const limit = values.limit !== undefined ? Number(values.limit) : undefined;
    if (limit !== undefined && (!Number.isSafeInteger(limit) || limit < 1)) throw new Error("--limit must be a positive integer.");
    const maxAgentTimeoutSec = values["max-agent-timeout-sec"] !== undefined ? Number(values["max-agent-timeout-sec"]) : undefined;
    const maxVerifierTimeoutSec = values["max-verifier-timeout-sec"] !== undefined ? Number(values["max-verifier-timeout-sec"]) : undefined;

    const summary = await runHarborBenchmark({
      tasksPath: values.tasks,
      outputDir: values.out,
      baseConfig,
      endpoints,
      concurrency,
      repetitions,
      filter: values.filter,
      limit,
      tools: parseToolList(values.tools),
      enableWebTools: values.web,
      editMode: editRaw,
      historyMode: historyRaw,
      thinkingLevel: values.thinking,
      maxRequests,
      maxAgentTimeoutSec,
      maxVerifierTimeoutSec,
      containerNetwork: netRaw,
      verifierEnv: parseKeyValueFlags(values["verifier-env"]),
      appendSystemPrompt: values["append-system-prompt"],
      cleanupContextImages: values["cleanup-context-images"],
      resume: !values["no-resume"],
      onTrialComplete: (rec, sum) => {
        console.log(JSON.stringify({
          trial: rec.trialId,
          status: rec.status,
          reward: rec.reward,
          requests: rec.requests,
          durationSec: rec.durationSec,
          endpoint: rec.endpoint,
          progress: `${sum.totals.finished}/${sum.totals.planned}`,
          points: `${sum.totals.totalPoints}/${sum.totals.maxPoints}`,
          meanReward: sum.totals.meanReward,
        }));
      },
    });
    console.log(JSON.stringify({ summary: summary.totals, byStatus: summary.byStatus }));
    if (summary.totals.infraErrors > 0) process.exitCode = 2;
  }
} catch (error) {
  console.error(`rwkv-pi harbor: ${error instanceof Error ? error.message : String(error)}`);
  process.exitCode = 1;
}
