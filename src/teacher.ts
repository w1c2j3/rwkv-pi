// The offline Qwen teacher as a pi provider. Used only by explicit collection or collaboration
// commands; never a fallback in RWKV's production inference path.
import { createProvider, type Model, type Provider } from "@earendil-works/pi-ai";
import { openAICompletionsApi } from "@earendil-works/pi-ai/api/openai-completions.lazy";
import { readConfig, type RwkvConfig } from "./config.js";

export function readTeacherConfig(env: NodeJS.ProcessEnv = process.env): RwkvConfig {
  if (!env.TEACHER_BASE_URL || !env.TEACHER_MODEL) throw new Error("Set TEACHER_BASE_URL and TEACHER_MODEL for the Qwen teacher.");
  return readConfig({
    RWKV_BACKEND: "openai",
    RWKV_BASE_URL: env.TEACHER_BASE_URL, RWKV_MODEL: env.TEACHER_MODEL,
    RWKV_API_KEY: env.TEACHER_API_KEY, RWKV_API_KEY_FILE: env.TEACHER_API_KEY_FILE,
    RWKV_CONTEXT_WINDOW: env.TEACHER_CONTEXT_WINDOW, RWKV_MAX_TOKENS: env.TEACHER_MAX_TOKENS,
  });
}

export function createTeacherProvider(config: RwkvConfig = readTeacherConfig()): Provider<"openai-completions"> {
  const model: Model<"openai-completions"> = {
    id: config.modelId, name: config.modelId, provider: "teacher", api: "openai-completions", baseUrl: config.baseUrl,
    reasoning: true, input: ["text"], contextWindow: config.contextWindow, maxTokens: config.maxTokens,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    compat: { supportsStore: false, supportsDeveloperRole: false, supportsReasoningEffort: false, supportsStrictMode: false, maxTokensField: "max_tokens" },
  };
  return createProvider({
    id: "teacher", models: [model], baseUrl: config.baseUrl, api: openAICompletionsApi(),
    auth: { apiKey: { name: "Offline teacher", resolve: async () => ({ auth: { apiKey: config.apiKey } }) } },
  });
}
