import { readFileSync } from "node:fs";

export interface RwkvConfig {
  backend: "lightning" | "openai";
  baseUrl: string;
  modelId: string;
  apiKey: string;
  contextWindow: number;
  maxTokens: number;
  toolFormat: "xml" | "json" | "native";
  promptTemplate: "xml" | "json" | "irwkv" | "g1";
  repeatFailureLimit: number;
  /** Pass envelope-complete but invalid calls to pi's native validation (tool error result) instead of aborting the turn. */
  toolCallRecovery: boolean;
  /** Block an identical tool call after this many identical consecutive calls with identical results (0 disables). */
  identicalCallLimit: number;
  /** Block a path-based call after this many consecutive failures of that tool on that path (0 disables). */
  targetFailureLimit: number;
  /** Recall-tolerant read/edit/write/ls execution (src/tolerant-tools.ts); prompts are unchanged. */
  toolTolerance: boolean;
  adapterId?: string;
  adapterVersion?: string;
  adapterScale?: number;
}

/** Unsigned decimal integer from the environment, at least `min`. */
export function envInteger(env: NodeJS.ProcessEnv, name: string, fallback: number, min: number): number {
  const raw = env[name]?.trim();
  if (!raw) return fallback;
  const value = Number(raw);
  if (!/^\d+$/.test(raw) || !Number.isSafeInteger(value) || value < min) {
    throw new Error(`${name} must be ${min === 0 ? "0 (disabled) or " : ""}a positive integer.`);
  }
  return value;
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
  if (!["xml", "json", "irwkv", "g1"].includes(promptTemplate)) throw new Error("RWKV_PROMPT_TEMPLATE must be xml, json, irwkv or g1.");
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
  if (maxTokens >= contextWindow) {
    throw new Error("RWKV_MAX_TOKENS must be smaller than RWKV_CONTEXT_WINDOW to leave room for the prompt.");
  }
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
    ...(adapterId ? { adapterId } : {}),
    ...(adapterVersion ? { adapterVersion } : {}),
    ...(adapterScale !== undefined ? { adapterScale } : {}),
    toolFormat: toolFormat as RwkvConfig["toolFormat"],
    promptTemplate: promptTemplate as RwkvConfig["promptTemplate"],
  };
}
