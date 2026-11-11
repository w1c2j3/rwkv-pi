import { createProvider, type Model, type Provider } from "@earendil-works/pi-ai";
import { openAICompletionsApi } from "@earendil-works/pi-ai/api/openai-completions.lazy";
import { readConfig, type RwkvConfig } from "./config.js";
import { withRwkvTransport } from "./transport.js";

export { readConfig, type RwkvConfig } from "./config.js";

export function createRwkvModel(config: RwkvConfig = readConfig()): Model<"openai-completions"> {
  const templateKwargs = (thinking: boolean) => config.promptTemplate !== "irwkv"
    // The author's tokenizer has no BOS. Override our converted tokenizer's EOS-as-BOS default.
    ? { fake_cot_mode: !thinking, thinking, bos_token: "" }
    : { fake_cot_mode: !thinking };
  return {
    id: config.modelId,
    name: `RWKV (${config.modelId})`,
    provider: "rwkv",
    api: "openai-completions",
    baseUrl: config.baseUrl,
    reasoning: true,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: config.contextWindow,
    maxTokens: config.maxTokens,
    // The continuation adapter supplies the recorded RWKV sampling profile.
    samplingParams: { chat_template_kwargs: templateKwargs(true),
      ...(config.adapterId ? { adapter_id: config.adapterId } : {}),
      ...(config.adapterVersion ? { adapter_version: config.adapterVersion } : {}),
      ...(config.adapterScale !== undefined ? { adapter_scale: config.adapterScale } : {}),
    },
    samplingParamsByThinkingLevel: {
      off: { chat_template_kwargs: templateKwargs(false) },
    },
    compat: {
      supportsStore: false,
      supportsDeveloperRole: false,
      supportsReasoningEffort: false,
      supportsStrictMode: false,
      maxTokensField: "max_tokens",
    },
  };
}

export function createRwkvProvider(config: RwkvConfig = readConfig()): Provider<"openai-completions"> & {
  rwkvPiToolFormat: RwkvConfig["toolFormat"]; rwkvPiRepeatFailureLimit: number; rwkvPiToolCallRecovery: boolean; rwkvPiIdenticalCallLimit: number;
  rwkvPiTargetFailureLimit: number; rwkvPiToolTolerance: boolean;
} {
  const api = withRwkvTransport(openAICompletionsApi(), config.repeatFailureLimit, config);
  return Object.assign(createProvider({
    id: "rwkv",
    name: config.backend === "lightning" ? "RWKV / Lightning CUDA" : "RWKV / OpenAI compatibility",
    baseUrl: config.baseUrl,
    auth: {
      apiKey: {
        name: "RWKV API key",
        async resolve({ credential, signal }) {
          signal.throwIfAborted();
          return {
            auth: { apiKey: credential?.key || config.apiKey },
            source: credential?.key ? "stored credential" : "RWKV configuration",
          };
        },
      },
    },
    models: [createRwkvModel(config)],
    api,
  }), { rwkvPiToolFormat: config.toolFormat, rwkvPiRepeatFailureLimit: config.repeatFailureLimit,
    rwkvPiToolCallRecovery: config.backend === "lightning" && config.toolCallRecovery, rwkvPiIdenticalCallLimit: config.identicalCallLimit,
    rwkvPiTargetFailureLimit: config.targetFailureLimit, rwkvPiToolTolerance: config.toolTolerance });
}
