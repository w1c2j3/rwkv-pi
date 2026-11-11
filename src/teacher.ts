// Teacher as a pi provider; Qwen remains the default, DeepSeek is explicit opt-in.
// Automatic handoff is owned by the collaboration extension, never the RWKV transport.
import { createProvider, type Model, type Provider } from "@earendil-works/pi-ai";
import { openAICompletionsApi } from "@earendil-works/pi-ai/api/openai-completions.lazy";
import { readConfig, type RwkvConfig } from "./config.js";

export interface TeacherConfig extends RwkvConfig { kind?: "qwen" | "deepseek"; reasoningEffort?: "low" | "high" }

export const DEEPSEEK_CHINESE_SUFFIX = "语言与协作约束：主 agent 的进度说明、思考内容、工具结果分析和最终答复均使用简体中文，不要用英文叙述规划或推理；代码、命令、路径、字段名和必须引用的原文保持原样。给 RWKV 子 agent 的任务必须是具体、完整的中文请求。依据实际工具观察判断，核验子 agent 的结论与修改；失败如实说明，不把主 agent 补做算作子 agent 成功。完成必要验证后简洁答复并结束。";

export function withDeepSeekChinese(prompt: string): string {
  const text = prompt.trimEnd();
  return text.endsWith(DEEPSEEK_CHINESE_SUFFIX) ? text : `${text}\n\n${DEEPSEEK_CHINESE_SUFFIX}`;
}

export function readTeacherConfig(env: NodeJS.ProcessEnv = process.env): TeacherConfig {
  if (!env.TEACHER_BASE_URL || !env.TEACHER_MODEL) throw new Error("Set TEACHER_BASE_URL and TEACHER_MODEL for the teacher.");
  const kind = env.TEACHER_KIND || "qwen";
  if (kind !== "qwen" && kind !== "deepseek") throw new Error("TEACHER_KIND must be qwen or deepseek.");
  const config = readConfig({
    RWKV_BACKEND: "openai",
    RWKV_BASE_URL: env.TEACHER_BASE_URL, RWKV_MODEL: env.TEACHER_MODEL,
    RWKV_API_KEY: env.TEACHER_API_KEY, RWKV_API_KEY_FILE: env.TEACHER_API_KEY_FILE,
    RWKV_CONTEXT_WINDOW: env.TEACHER_CONTEXT_WINDOW, RWKV_MAX_TOKENS: env.TEACHER_MAX_TOKENS,
  });
  if (kind === "deepseek") {
    if (config.baseUrl !== "https://api.deepseek.com" || config.modelId !== "deepseek-flash") throw new Error("This paid profile requires official deepseek-flash.");
    const effort = env.TEACHER_REASONING_EFFORT || "high";
    if (effort !== "low" && effort !== "high") throw new Error("DeepSeek thinking stays enabled: use low or high.");
    if (config.maxTokens > 16384 || config.contextWindow > 131072) throw new Error("Paid trial context/output limits exceeded.");
    return { ...config, kind, reasoningEffort: effort };
  }
  return { ...config, kind };
}

export function createTeacherProvider(config: TeacherConfig = readTeacherConfig()): Provider<"openai-completions"> {
  const deepseek = config.kind === "deepseek";
  const model: Model<"openai-completions"> = {
    id: config.modelId, name: config.modelId, provider: "teacher", api: "openai-completions", baseUrl: config.baseUrl,
    reasoning: true, input: ["text"], contextWindow: config.contextWindow, maxTokens: config.maxTokens,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    ...(deepseek ? { samplingParams: { thinking: { type: "enabled" }, reasoning_effort: config.reasoningEffort || "high", parallel_tool_calls: false } } : {}),
    compat: { supportsStore: false, supportsDeveloperRole: false, supportsReasoningEffort: false, supportsStrictMode: false, maxTokensField: "max_tokens",
      ...(deepseek ? { requiresReasoningContentOnAssistantMessages: true } : {}) },
  };
  return createProvider({
    id: "teacher", models: [model], baseUrl: config.baseUrl, api: openAICompletionsApi(),
    auth: { apiKey: { name: deepseek ? "DeepSeek official API" : "Offline teacher", resolve: async () => ({ auth: { apiKey: config.apiKey } }) } },
  });
}
