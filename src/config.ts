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
  adapterId?: string;
  adapterVersion?: string;
  adapterScale?: number;
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

  function positiveInteger(name: string, fallback: number): number {
    const raw = env[name]?.trim();
    if (!raw) return fallback;
    const value = Number(raw);
    if (!/^\d+$/.test(raw) || !Number.isSafeInteger(value) || value <= 0) {
      throw new Error(`${name} must be a positive integer.`);
    }
    return value;
  }

  const contextWindow = positiveInteger("RWKV_CONTEXT_WINDOW", 32768);
  const backend = env.RWKV_BACKEND?.trim() || "lightning";
  if (!["lightning", "openai"].includes(backend)) throw new Error("RWKV_BACKEND must be lightning or openai.");
  const toolFormat = env.RWKV_TOOL_FORMAT?.trim() || (backend === "lightning" ? "xml" : "native");
  if (backend === "lightning" ? !["xml", "json"].includes(toolFormat) : toolFormat !== "native") throw new Error("Lightning requires client XML or JSON parsing; legacy OpenAI requires native tool calls.");
  const promptTemplate = env.RWKV_PROMPT_TEMPLATE?.trim() || (toolFormat === "json" ? "json" : "xml");
  if (!["xml", "json", "irwkv", "g1"].includes(promptTemplate)) throw new Error("RWKV_PROMPT_TEMPLATE must be xml, json, irwkv or g1.");
  if (backend === "lightning" && promptTemplate !== toolFormat) throw new Error("Lightning prompt template must match its tool format.");
  const maxTokens = positiveInteger("RWKV_MAX_TOKENS", 8192);
  const repeatRaw = env.RWKV_REPEAT_FAILURE_LIMIT?.trim() || "3";
  const repeatFailureLimit = Number(repeatRaw);
  if (!/^\d+$/.test(repeatRaw) || !Number.isSafeInteger(repeatFailureLimit)
    || (repeatFailureLimit !== 0 && repeatFailureLimit < 2)) {
    throw new Error("RWKV_REPEAT_FAILURE_LIMIT must be 0 (disabled) or an integer >= 2.");
  }
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
    ...(adapterId ? { adapterId } : {}),
    ...(adapterVersion ? { adapterVersion } : {}),
    ...(adapterScale !== undefined ? { adapterScale } : {}),
    toolFormat: toolFormat as RwkvConfig["toolFormat"],
    promptTemplate: promptTemplate as RwkvConfig["promptTemplate"],
  };
}
