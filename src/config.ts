import { existsSync, mkdirSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join, resolve } from "node:path";

export interface RwkvSamplingConfig {
  temperature?: number;
  topP?: number;
  topK?: number;
  alphaPresence?: number;
  alphaFrequency?: number;
  alphaDecay?: number;
}

export interface RwkvConfig {
  backend: "lightning" | "openai";
  baseUrl: string;
  modelId: string;
  apiKey: string;
  contextWindow: number;
  maxTokens: number;
  toolFormat: "xml" | "json" | "native";
  promptTemplate: "xml" | "json" | "g1";
  repeatFailureLimit: number;
  /** Pass envelope-complete but invalid calls to pi's native validation (tool error result) instead of aborting the turn. */
  toolCallRecovery: boolean;
  /** Block an identical tool call after this many identical consecutive calls with identical results (0 disables). */
  identicalCallLimit: number;
  /** Block a path-based call after this many consecutive failures of that tool on that path (0 disables). */
  targetFailureLimit: number;
  /** Recall-tolerant read/edit/write/ls execution (src/tolerant-tools.ts); prompts are unchanged. */
  toolTolerance: boolean;
  /** full (default): complete conversation history; epoch: zero-tool-intrusion deterministic State Ledger folding at 9.6k; worklist: Stateful Work List folding (`todo_plan` + `todo_update`). */
  historyMode: "full" | "epoch" | "worklist";
  epochBudgetTokens: number;
  epochKeepRecentPairs: number;
  epochMaxResultTokens: number;
  worklistBudgetTokens: number;
  worklistKeepRecentPairs: number;
  worklistFoldTriggerTokens: number;
  sampling?: RwkvSamplingConfig;
  /** Sticky session affinity header (`X-RWKV-Session-Id`) for RWKV_Lightning_CUDA_router prefix caching. */
  sessionId?: string;
  adapterId?: string;
  adapterVersion?: string;
  adapterScale?: number;
}

/** Unsigned decimal integer from the environment, at least `min`. */
/** Where traces and participation records go: RWKV_PI_DATA_ROOT, else $XDG_DATA_HOME/rwkv-pi, else ~/.local/share/rwkv-pi. */
export function dataRoot(env: NodeJS.ProcessEnv = process.env): string {
  const explicit = env.RWKV_PI_DATA_ROOT?.trim();
  if (explicit) return resolve(explicit);
  return resolve(env.XDG_DATA_HOME?.trim() || join(homedir(), ".local", "share"), "rwkv-pi");
}

/** The data root, ready for writing. An explicit root must already exist: an unmounted disk never silently falls back. */
export function requireDataRoot(env: NodeJS.ProcessEnv = process.env): string {
  const root = dataRoot(env);
  if (env.RWKV_PI_DATA_ROOT?.trim()) {
    if (!existsSync(root)) throw new Error(`Data root ${root} is not available; mount it or change RWKV_PI_DATA_ROOT.`);
  } else mkdirSync(root, { recursive: true });
  return root;
}

export function envInteger(env: NodeJS.ProcessEnv, name: string, fallback: number, min: number): number {
  const raw = env[name]?.trim();
  if (!raw) return fallback;
  const value = Number(raw);
  if (!/^\d+$/.test(raw) || !Number.isSafeInteger(value) || value < min) {
    throw new Error(`${name} must be ${min === 0 ? "0 (disabled) or " : ""}a positive integer.`);
  }
  return value;
}

function optionalEnvFloat(
  env: NodeJS.ProcessEnv,
  name: string,
  valid: (value: number) => boolean,
  requirement: string,
): number | undefined {
  const raw = env[name]?.trim();
  if (!raw) return undefined;
  const value = Number(raw);
  if (!Number.isFinite(value) || !valid(value)) throw new Error(`${name} ${requirement}`);
  return value;
}

export function readSamplingConfig(env: NodeJS.ProcessEnv = process.env): RwkvSamplingConfig | undefined {
  const temperature = optionalEnvFloat(env, "RWKV_TEMPERATURE", (v) => v >= 0, "must be a finite number >= 0.");
  const topP = optionalEnvFloat(env, "RWKV_TOP_P", (v) => v > 0 && v <= 1, "must be in (0, 1].");
  const topK = env.RWKV_TOP_K?.trim() ? envInteger(env, "RWKV_TOP_K", 1, 1) : undefined;
  const alphaPresence = optionalEnvFloat(env, "RWKV_ALPHA_PRESENCE", (v) => v >= 0, "must be a finite number >= 0.");
  const alphaFrequency = optionalEnvFloat(env, "RWKV_ALPHA_FREQUENCY", (v) => v >= 0, "must be a finite number >= 0.");
  const alphaDecay = optionalEnvFloat(env, "RWKV_ALPHA_DECAY", (v) => v > 0 && v <= 1, "must be in (0, 1].");
  const sampling: RwkvSamplingConfig = {
    ...(temperature !== undefined ? { temperature } : {}),
    ...(topP !== undefined ? { topP } : {}),
    ...(topK !== undefined ? { topK } : {}),
    ...(alphaPresence !== undefined ? { alphaPresence } : {}),
    ...(alphaFrequency !== undefined ? { alphaFrequency } : {}),
    ...(alphaDecay !== undefined ? { alphaDecay } : {}),
  };
  return Object.keys(sampling).length > 0 ? sampling : undefined;
}

export function readConfig(env: NodeJS.ProcessEnv = process.env): RwkvConfig {
  const rawUrl = env.RWKV_BASE_URL?.trim() || "http://127.0.0.1:8000/v1";
  let url: URL;
  try {
    url = new URL(rawUrl);
  } catch {
    throw new Error("RWKV_BASE_URL must be an absolute http(s) API URL, ending in /v1.");
  }
  if (!["http:", "https:"].includes(url.protocol) || url.username || url.password || url.search || url.hash) {
    throw new Error("RWKV_BASE_URL must use http(s) without credentials, query parameters, or fragments.");
  }
  if (url.pathname.replace(/\/+$/, "").endsWith("/chat/completions")) {
    throw new Error("RWKV_BASE_URL must point to the API root (for example /v1), not /chat/completions.");
  }

  const positiveInteger = (name: string, fallback: number) => envInteger(env, name, fallback, 1);
  const contextWindow = positiveInteger("RWKV_CONTEXT_WINDOW", 131072);
  const backend = env.RWKV_BACKEND?.trim() || "lightning";
  if (!["lightning", "openai"].includes(backend)) throw new Error("RWKV_BACKEND must be lightning or openai.");
  const toolFormat = env.RWKV_TOOL_FORMAT?.trim() || (backend === "lightning" ? "xml" : "native");
  if (backend === "lightning" ? !["xml", "json"].includes(toolFormat) : toolFormat !== "native") throw new Error("Lightning requires client XML or JSON parsing; legacy OpenAI requires native tool calls.");
  const promptTemplate = env.RWKV_PROMPT_TEMPLATE?.trim() || (toolFormat === "json" ? "json" : "xml");
  if (!["xml", "json", "g1"].includes(promptTemplate)) throw new Error("RWKV_PROMPT_TEMPLATE must be xml, json or g1.");
  if (backend === "lightning" && promptTemplate !== toolFormat) throw new Error("Lightning prompt template must match its tool format.");
  const maxTokens = positiveInteger("RWKV_MAX_TOKENS", 8192);
  const repeatFailureLimit = envInteger(env, "RWKV_REPEAT_FAILURE_LIMIT", 3, 0);
  if (repeatFailureLimit === 1) throw new Error("RWKV_REPEAT_FAILURE_LIMIT must be 0 (disabled) or an integer >= 2.");
  const recoveryRaw = env.RWKV_TOOL_CALL_RECOVERY?.trim() || "1";
  if (!["0", "1"].includes(recoveryRaw)) throw new Error("RWKV_TOOL_CALL_RECOVERY must be 0 or 1.");
  const identicalCallLimit = envInteger(env, "RWKV_IDENTICAL_CALL_LIMIT", 2, 0);
  const targetFailureLimit = envInteger(env, "RWKV_TARGET_FAILURE_LIMIT", 3, 0);
  if (targetFailureLimit === 1) throw new Error("RWKV_TARGET_FAILURE_LIMIT must be 0 (disabled) or an integer >= 2.");
  const toleranceRaw = env.RWKV_TOOL_TOLERANCE?.trim() || "1";
  if (!["0", "1"].includes(toleranceRaw)) throw new Error("RWKV_TOOL_TOLERANCE must be 0 or 1.");
  const historyMode = env.RWKV_HISTORY_MODE?.trim() || "full";
  if (!["full", "epoch", "worklist"].includes(historyMode)) throw new Error("RWKV_HISTORY_MODE must be full, epoch, or worklist.");
  const epochBudgetTokens = positiveInteger("RWKV_EPOCH_BUDGET_TOKENS", 9600);
  const epochKeepRecentPairs = envInteger(env, "RWKV_EPOCH_KEEP_RECENT_PAIRS", 2, 1);
  const epochMaxResultTokens = envInteger(env, "RWKV_EPOCH_MAX_RESULT_TOKENS", 2000, 0);
  const worklistBudgetTokens = positiveInteger("RWKV_WORKLIST_BUDGET_TOKENS", 9600);
  const worklistKeepRecentPairs = envInteger(env, "RWKV_WORKLIST_KEEP_RECENT_PAIRS", 2, 0);
  const worklistFoldTriggerTokens = envInteger(env, "RWKV_WORKLIST_FOLD_TRIGGER_TOKENS", 6144, 0);
  if (maxTokens >= contextWindow) {
    throw new Error("RWKV_MAX_TOKENS must be smaller than RWKV_CONTEXT_WINDOW to leave room for the prompt.");
  }
  const sampling = readSamplingConfig(env);
  const sessionId = env.RWKV_SESSION_ID?.trim() || undefined;
  if (sessionId && /[\r\n\0]/.test(sessionId)) throw new Error("RWKV_SESSION_ID must not contain control characters.");
  const adapterId = env.RWKV_ADAPTER_ID?.trim() || undefined;
  const adapterVersion = env.RWKV_ADAPTER_VERSION?.trim() || undefined;
  const adapterScale = env.RWKV_ADAPTER_SCALE?.trim() ? Number(env.RWKV_ADAPTER_SCALE) : undefined;
  if ((adapterVersion || adapterScale !== undefined) && !adapterId) throw new Error("RWKV_ADAPTER_ID is required for adapter version or scale.");
  if (adapterScale !== undefined && !Number.isFinite(adapterScale)) throw new Error("RWKV_ADAPTER_SCALE must be finite.");
  if (adapterId && backend !== "lightning") throw new Error("Adapter selection requires the Lightning backend.");
  return {
    backend: backend as RwkvConfig["backend"],
    baseUrl: url.toString().replace(/\/+$/, ""),
    modelId: env.RWKV_MODEL?.trim() || "rwkv7",
    apiKey: env.RWKV_API_KEY?.trim() || (env.RWKV_API_KEY_FILE ? readFileSync(env.RWKV_API_KEY_FILE, "utf8").trim() : "rwkv-local"),
    contextWindow,
    maxTokens,
    repeatFailureLimit,
    toolCallRecovery: recoveryRaw === "1",
    identicalCallLimit,
    targetFailureLimit,
    toolTolerance: toleranceRaw === "1",
    historyMode: historyMode as RwkvConfig["historyMode"],
    epochBudgetTokens,
    epochKeepRecentPairs,
    epochMaxResultTokens,
    worklistBudgetTokens,
    worklistKeepRecentPairs,
    worklistFoldTriggerTokens,
    ...(sampling ? { sampling } : {}),
    ...(sessionId ? { sessionId } : {}),
    ...(adapterId ? { adapterId } : {}),
    ...(adapterVersion ? { adapterVersion } : {}),
    ...(adapterScale !== undefined ? { adapterScale } : {}),
    toolFormat: toolFormat as RwkvConfig["toolFormat"],
    promptTemplate: promptTemplate as RwkvConfig["promptTemplate"],
  };
}
