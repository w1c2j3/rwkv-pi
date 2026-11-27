import { Agent, fetch as undiciFetch } from "undici";

export interface HealthPolicyEntry {
  version: string;
  step: number;
  cache_enabled: boolean;
  cached_prefixes: number;
  cache_bytes: number;
}

export interface ServerHealth {
  status: string;
  mode: "nora" | "miss" | "state";
  dtype: string;
  gemm_mode: string;
  step: number;
  policy_version: string;
  published_step: number;
  reference_version: string;
  reference_step: number;
  chunk: number;
  ctx: number;
  vocab_size: number;
  dp_rank: number;
  dp_world_size: number;
  max_batch: number;
  micro_bsz: number;
  default_max_tokens: number;
  default_stop: string[];
  policies: HealthPolicyEntry[];
  cache?: {
    hits: number;
    misses: number;
    cached_prefixes: number;
    used_bytes: number;
    effective_capacity: number;
  };
}

export interface CompletionItem {
  tokens: number[];
  logprobs: number[];
  text: string;
  finish_reason: "stop" | "length";
  truncated: boolean;
  truncation_reason?: "max_tokens" | "context_length";
  matched_stop?: string | number | number[];
  cached_prompt_tokens: number;
  prefill_tokens: number;
  max_tokens: number;
}

export interface CompletionsRequest {
  policy_version?: string;
  prompts: number[][];
  max_tokens: number;
  temperature: number;
  top_p?: number;
  top_k?: number;
  presence_penalty?: number;
  repetition_penalty?: number;
  penalty_decay?: number;
  seed?: number;
  stop?: string[];
  use_cache?: boolean;
}

export interface CompletionsResponse {
  policy_version: string;
  step: number;
  temperature: number;
  max_tokens: number;
  completions: CompletionItem[];
  cached_prompt_tokens: number;
  prefill_tokens: number;
  generated_tokens: number;
  seconds: number;
}

export interface LogprobsResponse {
  policy_version: string;
  step: number;
  temperature: number;
  logprobs: number[][];
  seconds: number;
}

export interface TrainTurn {
  tokens: number[];
  loss_mask: (0 | 1)[];
  advantages?: number | number[];
}

export interface TrainSample {
  turns: TrainTurn[];
  advantages: number;
  temperature?: number;
}

export interface TrainResponse {
  policy_version: string;
  trained_tokens: number;
  total_tokens: number;
  learning_rate: number;
  updated: boolean;
  publish_required: boolean;
  step: number;
  loss: number;
  grad_norm: number;
  seconds: number;
}

export interface PublishResponse {
  policy_version: string;
  step: number;
  directory?: string;
  checkpoint?: string;
}

export interface LrScheduleConfig {
  lr: number;
  lrMin: number;
  warmupSteps: number;
  totalSteps: number;
  schedule: "cosine" | "constant";
}

/** Cosine or constant LR schedule with linear warmup; step is 1-based optimizer step. */
export function computeLearningRate(step: number, cfg: LrScheduleConfig): number {
  const s = Math.max(1, Math.floor(step));
  if (cfg.warmupSteps > 0 && s <= cfg.warmupSteps) {
    return Math.max(1e-12, cfg.lr * (s / cfg.warmupSteps));
  }
  if (cfg.schedule === "constant" || cfg.totalSteps <= cfg.warmupSteps) {
    return Math.max(1e-12, cfg.lr);
  }
  const progress = Math.min(1, Math.max(0, (s - cfg.warmupSteps) / Math.max(1, cfg.totalSteps - cfg.warmupSteps)));
  const cosine = 0.5 * (1 + Math.cos(Math.PI * progress));
  return Math.max(1e-12, cfg.lrMin + (cfg.lr - cfg.lrMin) * cosine);
}

/** Expand `--url http://host:8850 --dp-world-size 8` or explicit comma-separated URLs. */
export function resolveClusterUrls(urlsArg?: string, baseUrl = "http://127.0.0.1:8850", dpWorldSize = 1): string[] {
  if (urlsArg && urlsArg.trim()) {
    const list = urlsArg.split(",").map((u) => u.trim().replace(/\/+$/, "")).filter(Boolean);
    if (!list.length) throw new Error("Empty --urls list.");
    return list;
  }
  const parsed = new URL(baseUrl);
  const basePort = parsed.port ? Number(parsed.port) : parsed.protocol === "https:" ? 443 : 80;
  if (!Number.isSafeInteger(basePort) || basePort < 1) throw new Error(`Invalid server URL port: ${baseUrl}`);
  if (!Number.isSafeInteger(dpWorldSize) || dpWorldSize < 1 || dpWorldSize > 64) {
    throw new Error(`Invalid dpWorldSize: ${dpWorldSize}`);
  }
  const urls: string[] = [];
  for (let r = 0; r < dpWorldSize; r++) {
    const u = new URL(baseUrl);
    u.port = String(basePort + r);
    urls.push(u.toString().replace(/\/+$/, ""));
  }
  return urls;
}

export class OnlineRlRankClient {
  readonly url: string;
  readonly rankIndex: number;
  private readonly dispatcher: Agent;
  private readonly timeoutMs: number;
  policyVersion = "";
  referenceVersion = "";
  health!: ServerHealth;

  constructor(url: string, rankIndex: number, timeoutMs = 3_600_000) {
    this.url = url.replace(/\/+$/, "");
    this.rankIndex = rankIndex;
    this.timeoutMs = timeoutMs;
    this.dispatcher = new Agent({ headersTimeout: timeoutMs, bodyTimeout: timeoutMs, connections: 8 });
  }

  async close(): Promise<void> {
    await this.dispatcher.close();
  }

  async call<T>(path: string, body?: Record<string, unknown>): Promise<T> {
    const init: RequestInit & { dispatcher: Agent } = {
      method: body === undefined ? "GET" : "POST",
      headers: { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: AbortSignal.timeout(this.timeoutMs),
      dispatcher: this.dispatcher,
    };
    let response: Response;
    try {
      response = await (undiciFetch as unknown as typeof fetch)(this.url + path, init);
    } catch (error) {
      throw new Error(
        `Connection failed on rank ${this.rankIndex} (${this.url}${path}): ${error instanceof Error ? error.message : String(error)}. ` +
        "Request was NOT retried; check /health before resuming.",
      );
    }
    const text = await response.text();
    if (!response.ok) {
      throw new Error(`HTTP ${response.status} on rank ${this.rankIndex} (${this.url}${path}): ${text}`);
    }
    return JSON.parse(text) as T;
  }

  async refreshHealth(): Promise<ServerHealth> {
    const h = await this.call<ServerHealth>("/health");
    this.health = h;
    this.policyVersion = h.policy_version;
    this.referenceVersion = h.reference_version;
    return h;
  }

  async tokenize(text: string): Promise<number[]> {
    const res = await this.call<{ tokens: number[] }>("/v1/tokenize", { text });
    return res.tokens;
  }

  async detokenize(tokens: number[]): Promise<string> {
    const res = await this.call<{ text: string }>("/v1/detokenize", { tokens });
    return res.text;
  }

  async completions(req: CompletionsRequest): Promise<CompletionsResponse> {
    return this.call<CompletionsResponse>("/v1/completions", {
      policy_version: req.policy_version ?? this.policyVersion,
      prompts: req.prompts,
      max_tokens: req.max_tokens,
      temperature: req.temperature,
      ...(req.top_p !== undefined ? { top_p: req.top_p } : {}),
      ...(req.top_k !== undefined ? { top_k: req.top_k } : {}),
      ...(req.presence_penalty !== undefined ? { presence_penalty: req.presence_penalty } : {}),
      ...(req.repetition_penalty !== undefined ? { repetition_penalty: req.repetition_penalty } : {}),
      ...(req.penalty_decay !== undefined ? { penalty_decay: req.penalty_decay } : {}),
      ...(req.seed !== undefined ? { seed: req.seed } : {}),
      ...(req.stop !== undefined ? { stop: req.stop } : {}),
      ...(req.use_cache !== undefined ? { use_cache: req.use_cache } : {}),
    });
  }

  async logprobs(sequences: number[][], temperature: number, policyVersion = this.referenceVersion): Promise<LogprobsResponse> {
    return this.call<LogprobsResponse>("/v1/logprobs", {
      policy_version: policyVersion,
      temperature,
      sequences,
    });
  }
}

export interface ClusterTrainSummary {
  step: number;
  updated: boolean;
  loss: number;
  gradNorm: number;
  learningRate: number;
  trainedTokens: number;
  totalTokens: number;
  seconds: number;
  perRank: TrainResponse[];
}

export interface ClusterPublishSummary {
  step: number;
  directory?: string;
  checkpoint?: string;
  policyVersions: string[];
}

/** Coordinates 1..N ranks of `rwkv_nora_tune_server` / `rwkv_nora_tune_server_dp`. */
export class OnlineRlCluster {
  readonly ranks: OnlineRlRankClient[];
  ctx = 0;
  maxBatch = 0;
  dpWorldSize = 1;
  mode: ServerHealth["mode"] = "nora";
  step = 0;

  constructor(urls: readonly string[], timeoutMs = 3_600_000) {
    if (!urls.length) throw new Error("OnlineRlCluster requires at least one server URL.");
    this.ranks = urls.map((u, i) => new OnlineRlRankClient(u, i, timeoutMs));
  }

  async close(): Promise<void> {
    await Promise.all(this.ranks.map((r) => r.close()));
  }

  async connect(): Promise<ServerHealth[]> {
    const healths = await Promise.all(this.ranks.map((r) => r.refreshHealth()));
    const first = healths[0]!;
    if (first.dp_world_size !== this.ranks.length) {
      throw new Error(
        `Server reports dp_world_size=${first.dp_world_size}, but client was configured with ${this.ranks.length} URL(s).`,
      );
    }
    for (let i = 0; i < healths.length; i++) {
      const h = healths[i]!;
      if (h.status !== "ok") throw new Error(`Rank ${i} unhealthy: ${JSON.stringify(h)}`);
      if (h.step !== h.published_step) {
        throw new Error(
          `Rank ${i} has unpublished step=${h.step} (published_step=${h.published_step}); POST /v1/publish with expected_step=${h.step} first.`,
        );
      }
      if (h.step !== first.step) throw new Error(`Step mismatch between rank 0 (${first.step}) and rank ${i} (${h.step}).`);
      if (h.dp_rank !== i) throw new Error(`Expected dp_rank=${i} on ${this.ranks[i]!.url}, got dp_rank=${h.dp_rank}.`);
      if (h.ctx !== first.ctx || h.max_batch !== first.max_batch) {
        throw new Error(`Config mismatch on rank ${i}: ctx=${h.ctx}, max_batch=${h.max_batch}.`);
      }
    }
    this.ctx = first.ctx;
    this.maxBatch = first.max_batch;
    this.dpWorldSize = first.dp_world_size;
    this.mode = first.mode;
    this.step = first.step;
    return healths;
  }

  /**
   * Executes one synchronized on-policy optimizer step across all ranks.
   * In DP mode (`dpWorldSize > 1`), EVERY rank receives a concurrent `/v1/train` call with
   * identical `global_trained_tokens` so NCCL collectives never deadlock.
   */
  async trainStep(
    samplesByRank: readonly (readonly TrainSample[])[],
    learningRate: number,
    temperature: number,
  ): Promise<ClusterTrainSummary> {
    if (samplesByRank.length !== this.ranks.length) {
      throw new Error(`Expected ${this.ranks.length} rank sample shards, got ${samplesByRank.length}.`);
    }
    let globalTrainedTokens = 0;
    for (const shard of samplesByRank) {
      for (const sample of shard) {
        let pos = 0;
        for (const turn of sample.turns) {
          for (let i = 0; i < turn.loss_mask.length; i++) {
            if (turn.loss_mask[i] === 1 && pos > 0) globalTrainedTokens++;
            pos++;
          }
        }
      }
    }
    if (globalTrainedTokens <= 0) {
      throw new Error("trainStep received zero trainable tokens across all ranks.");
    }
    const perRank = await Promise.all(
      this.ranks.map((rank, i) => {
        const body: Record<string, unknown> = {
          policy_version: rank.policyVersion,
          learning_rate: learningRate,
          temperature,
          samples: samplesByRank[i]!,
        };
        if (this.dpWorldSize > 1) {
          body.global_trained_tokens = globalTrainedTokens;
        }
        return rank.call<TrainResponse>("/v1/train", body);
      }),
    );
    const primary = perRank[0]!;
    const trainedTokens = perRank.reduce((acc, r) => acc + r.trained_tokens, 0);
    const totalTokens = perRank.reduce((acc, r) => acc + r.total_tokens, 0);
    const seconds = Math.max(...perRank.map((r) => r.seconds));
    this.step = primary.step;
    return {
      step: primary.step,
      updated: primary.updated,
      loss: primary.loss,
      gradNorm: primary.grad_norm,
      learningRate: primary.learning_rate,
      trainedTokens,
      totalTokens,
      seconds,
      perRank,
    };
  }

  async publishStep(expectedStep: number, options: { save?: boolean; checkpoint?: boolean } = {}): Promise<ClusterPublishSummary> {
    const body = {
      expected_step: expectedStep,
      save: Boolean(options.save || options.checkpoint),
      checkpoint: Boolean(options.checkpoint),
    };
    const perRank = await Promise.all(this.ranks.map((rank) => rank.call<PublishResponse>("/v1/publish", body)));
    for (let i = 0; i < this.ranks.length; i++) {
      this.ranks[i]!.policyVersion = perRank[i]!.policy_version;
    }
    this.step = perRank[0]!.step;
    return {
      step: perRank[0]!.step,
      directory: perRank[0]!.directory,
      checkpoint: perRank[0]!.checkpoint,
      policyVersions: perRank.map((r) => r.policy_version),
    };
  }
}
