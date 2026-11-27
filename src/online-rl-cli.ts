#!/usr/bin/env node
import { parseArgs } from "node:util";
import { OnlineRlCluster, resolveClusterUrls } from "./online-rl-client.js";
import { runOnlineRl, type OnlineRlOptions, type RlAlgorithm } from "./online-rl.js";

function num(value: string | undefined, fallback: number, name: string): number {
  const n = value === undefined ? fallback : Number(value);
  if (!Number.isFinite(n)) throw new Error(`Invalid numeric option --${name}: ${value}`);
  return n;
}

function printHelp(): void {
  console.log(`Usage: npm run online-rl -- --tasks tasks.jsonl --out runs/rl-drgrpo [options]

Connects to Alic-Li v1.8.15 rwkv_nora_tune_server / rwkv_nora_tune_server_dp
and runs multi-turn Agentic Dr.GRPO + Reference KL online RL inside bwrap sandboxes.

Options:
  --url URL                   Base server URL for rank 0 (default: http://127.0.0.1:8850)
  --urls URL1,URL2,...        Explicit comma-separated per-rank URLs (overrides --url/--dp-world-size)
  --dp-world-size N           Auto-expand --url port..port+N-1 for NCCL DP server (default: 1)
  --tasks PATH                Training task manifest (.jsonl or .jsonl.gz) [required]
  --eval-tasks PATH           Optional holdout evaluation task manifest
  --out DIR                   Output directory for metrics.jsonl & scratch workspaces [required]
  --steps N                   Number of optimizer steps (default: 100)
  --group-size G              Rollouts per task (default: 4)
  --tasks-per-rank K          Informative task groups per DP rank per step (default: 1)
  --max-turns T               Max assistant turns per episode (default: 16)
  --max-tokens N              Per-turn generation safety cap (default: 2048)
  --temperature F             Sampling & training temperature (default: 1.0)
  --top-p F                   Optional rollout nucleus sampling top_p (0 < top_p <= 1)
  --top-k N                   Optional rollout top_k filter (0 disables)
  --presence-penalty F        Optional RWKV rollout presence penalty (0 disables)
  --repetition-penalty F      Optional RWKV rollout repetition penalty (0 disables)
  --penalty-decay F           Optional RWKV rollout penalty decay (default: 0.996)
  --require-pass-variance     Skip all-fail (0%) and all-pass (100%) groups during dynamic sampling
  --algorithm dr_grpo|grpo    Advantage estimator (default: dr_grpo)
  --dr-grpo-group-budget N    Override Dr.GRPO constant group token divisor
  --lr F                      Peak learning rate (default: 2e-5)
  --lr-min F                  Minimum cosine LR (default: 3e-6)
  --warmup-steps N            Linear warmup steps (default: 10)
  --lr-schedule cosine|constant (default: cosine)
  --kl-beta F                 Per-token KL penalty vs reference_version (default: 0.03; 0 disables)
  --kl-clip F                 Clip per-token log-ratio (default: 10.0)
  --max-resample N            Dynamic sampling retries per zero-variance group (default: 3)
  --sandbox bubblewrap|none   Workspace isolation mode (default: bubblewrap)
  --edit-mode line|text       Tool edit protocol (default: line)
  --max-result-chars N        Max chars per tool output before elision (default: 6000)
  --tool-timeout-ms N         Per-tool timeout ms (default: 60000)
  --min-gen-headroom N        Min free context tokens required to continue (default: 512)
  --save-every N              Save adapter/state every N steps (default: 10)
  --checkpoint-every N        Save full Adam checkpoint every N steps (default: 20)
  --eval-every N              Evaluate holdout every N steps (default: 20)
  --eval-limit N              Max eval tasks per sweep (default: 64)
  --timeout-sec N             HTTP request timeout in seconds (default: 3600)
  --seed N                    PRNG seed (default: 1234)
  --keep-workspaces           Keep temp rollout workspaces for debugging
  --help                      Show this help message`);
}

try {
  const { values } = parseArgs({
    options: {
      url: { type: "string", default: "http://127.0.0.1:8850" },
      urls: { type: "string" },
      "dp-world-size": { type: "string", default: "1" },
      tasks: { type: "string" },
      "eval-tasks": { type: "string" },
      out: { type: "string" },
      steps: { type: "string", default: "100" },
      "group-size": { type: "string", default: "4" },
      "tasks-per-rank": { type: "string", default: "1" },
      "max-turns": { type: "string", default: "16" },
      "max-tokens": { type: "string", default: "2048" },
      temperature: { type: "string", default: "1.0" },
      "top-p": { type: "string" },
      "top-k": { type: "string" },
      "presence-penalty": { type: "string" },
      "repetition-penalty": { type: "string" },
      "penalty-decay": { type: "string" },
      "require-pass-variance": { type: "boolean", default: false },
      algorithm: { type: "string", default: "dr_grpo" },
      "dr-grpo-group-budget": { type: "string" },
      lr: { type: "string", default: "2e-5" },
      "lr-min": { type: "string", default: "3e-6" },
      "warmup-steps": { type: "string", default: "10" },
      "lr-schedule": { type: "string", default: "cosine" },
      "kl-beta": { type: "string", default: "0.03" },
      "kl-clip": { type: "string", default: "10.0" },
      "max-resample": { type: "string", default: "3" },
      sandbox: { type: "string", default: "bubblewrap" },
      "edit-mode": { type: "string", default: "line" },
      "max-result-chars": { type: "string", default: "6000" },
      "tool-timeout-ms": { type: "string", default: "60000" },
      "min-gen-headroom": { type: "string", default: "512" },
      "save-every": { type: "string", default: "10" },
      "checkpoint-every": { type: "string", default: "20" },
      "eval-every": { type: "string", default: "20" },
      "eval-limit": { type: "string", default: "64" },
      "timeout-sec": { type: "string", default: "3600" },
      seed: { type: "string", default: "1234" },
      "keep-workspaces": { type: "boolean", default: false },
      help: { type: "boolean", default: false },
    },
  });

  if (values.help) {
    printHelp();
  } else {
    if (!values.tasks || !values.out) throw new Error("--tasks and --out are required.");
    const algorithm = values.algorithm as RlAlgorithm;
    if (algorithm !== "dr_grpo" && algorithm !== "grpo") throw new Error("--algorithm must be dr_grpo or grpo.");
    const lrSchedule = values["lr-schedule"];
    if (lrSchedule !== "cosine" && lrSchedule !== "constant") throw new Error("--lr-schedule must be cosine or constant.");
    const sandbox = values.sandbox;
    if (sandbox !== "bubblewrap" && sandbox !== "none") throw new Error("--sandbox must be bubblewrap or none.");
    const editMode = values["edit-mode"];
    if (editMode !== "line" && editMode !== "text") throw new Error("--edit-mode must be line or text.");

    const dpWorldSize = num(values["dp-world-size"], 1, "dp-world-size");
    const urls = resolveClusterUrls(values.urls, values.url, dpWorldSize);
    const cluster = new OnlineRlCluster(urls, num(values["timeout-sec"], 3600, "timeout-sec") * 1000);
    const options: OnlineRlOptions = {
      tasksPath: values.tasks,
      evalTasksPath: values["eval-tasks"],
      outputDir: values.out,
      steps: num(values.steps, 100, "steps"),
      groupSize: num(values["group-size"], 4, "group-size"),
      tasksPerRank: num(values["tasks-per-rank"], 1, "tasks-per-rank"),
      maxTurns: num(values["max-turns"], 16, "max-turns"),
      maxTokensPerTurn: num(values["max-tokens"], 2048, "max-tokens"),
      temperature: num(values.temperature, 1.0, "temperature"),
      ...(values["top-p"] !== undefined ? { topP: num(values["top-p"], 1.0, "top-p") } : {}),
      ...(values["top-k"] !== undefined ? { topK: num(values["top-k"], 0, "top-k") } : {}),
      ...(values["presence-penalty"] !== undefined ? { presencePenalty: num(values["presence-penalty"], 0.0, "presence-penalty") } : {}),
      ...(values["repetition-penalty"] !== undefined ? { repetitionPenalty: num(values["repetition-penalty"], 0.0, "repetition-penalty") } : {}),
      ...(values["penalty-decay"] !== undefined ? { penaltyDecay: num(values["penalty-decay"], 0.996, "penalty-decay") } : {}),
      requirePassVariance: values["require-pass-variance"],
      algorithm,
      drGrpoGroupBudget: values["dr-grpo-group-budget"] ? num(values["dr-grpo-group-budget"], 0, "dr-grpo-group-budget") : undefined,
      lr: num(values.lr, 2e-5, "lr"),
      lrMin: num(values["lr-min"], 3e-6, "lr-min"),
      warmupSteps: num(values["warmup-steps"], 10, "warmup-steps"),
      lrSchedule,
      klBeta: num(values["kl-beta"], 0.03, "kl-beta"),
      klClip: num(values["kl-clip"], 10.0, "kl-clip"),
      maxResampleAttempts: num(values["max-resample"], 3, "max-resample"),
      sandbox,
      editMode,
      maxToolResultChars: num(values["max-result-chars"], 6000, "max-result-chars"),
      toolTimeoutMs: num(values["tool-timeout-ms"], 60000, "tool-timeout-ms"),
      minGenHeadroom: num(values["min-gen-headroom"], 512, "min-gen-headroom"),
      identicalCallLimit: 2,
      targetFailureLimit: 3,
      saveEvery: num(values["save-every"], 10, "save-every"),
      checkpointEvery: num(values["checkpoint-every"], 20, "checkpoint-every"),
      evalEvery: num(values["eval-every"], 20, "eval-every"),
      evalLimit: num(values["eval-limit"], 64, "eval-limit"),
      seed: num(values.seed, 1234, "seed"),
      keepWorkspaces: values["keep-workspaces"],
    };
    try {
      const summary = await runOnlineRl(cluster, options);
      console.log(JSON.stringify(summary));
    } finally {
      await cluster.close();
    }
  }
} catch (error) {
  console.error(`rwkv-pi online-rl: ${error instanceof Error ? error.message : String(error)}`);
  process.exitCode = 1;
}
