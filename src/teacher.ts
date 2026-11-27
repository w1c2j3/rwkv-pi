// Teacher as a pi provider; Qwen remains the default, DeepSeek is explicit opt-in.
// Automatic handoff is owned by the collaboration extension, never the RWKV transport.
import { randomUUID } from "node:crypto";
import { createProvider, type Model, type Provider } from "@earendil-works/pi-ai";
import { openAICompletionsApi } from "@earendil-works/pi-ai/api/openai-completions.lazy";
import { readConfig, type RwkvConfig } from "./config.js";

export interface TeacherConfig extends RwkvConfig {
  kind?: "qwen" | "deepseek";
  reasoningEffort?: "low" | "high";
  sessionId?: string;
}

export const DEEPSEEK_CHINESE_SUFFIX = "语言与协作约束：最终答复的语言必须与用户输入的问题或提示词语言保持一致；代码、命令、路径、字段名和必须引用的原文保持原样。依据实际工具观察判断，核验子 agent 的结论与修改；失败如实说明，不把主 agent 补做算作子 agent 成功。完成必要验证后简洁答复并结束。";

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
    RWKV_EDIT_MODE: env.TEACHER_EDIT_MODE ?? env.RWKV_EDIT_MODE,
    RWKV_HISTORY_MODE: env.TEACHER_HISTORY_MODE ?? env.RWKV_HISTORY_MODE,
    RWKV_EPOCH_BUDGET_TOKENS: env.TEACHER_EPOCH_BUDGET_TOKENS ?? env.RWKV_EPOCH_BUDGET_TOKENS,
    RWKV_EPOCH_HARD_SLACK_TOKENS: env.TEACHER_EPOCH_HARD_SLACK_TOKENS ?? env.RWKV_EPOCH_HARD_SLACK_TOKENS,
    RWKV_EPOCH_KEEP_RECENT_PAIRS: env.TEACHER_EPOCH_KEEP_RECENT_PAIRS ?? env.RWKV_EPOCH_KEEP_RECENT_PAIRS,
    RWKV_EPOCH_MAX_RESULT_TOKENS: env.TEACHER_EPOCH_MAX_RESULT_TOKENS ?? env.RWKV_EPOCH_MAX_RESULT_TOKENS,
  });
  const sessionId = env.TEACHER_SESSION_ID?.trim() || randomUUID();
  if (kind === "deepseek") {
    const official = config.baseUrl === "https://api.deepseek.com" && config.modelId === "deepseek-flash";
    const selfHosted = config.modelId === "rwkv-ds41f-1017" || env.TEACHER_SELF_HOSTED === "1";
    if (!official && !selfHosted) throw new Error("This paid profile requires official deepseek-flash or self-hosted rwkv-ds41f-1017.");
    const effort = env.TEACHER_REASONING_EFFORT || "high";
    if (effort !== "low" && effort !== "high") throw new Error("DeepSeek thinking stays enabled: use low or high.");
    if (official && (config.maxTokens > 16384 || config.contextWindow > 131072)) throw new Error("Paid trial context/output limits exceeded.");
    return { ...config, kind, reasoningEffort: effort, sessionId };
  }
  return { ...config, kind, sessionId };
}

export function createTeacherProvider(config: TeacherConfig = readTeacherConfig()): Provider<"openai-completions"> {
  const deepseek = config.kind === "deepseek";
  const sessionId = config.sessionId || randomUUID();
  const headers = { "x-litellm-session-id": sessionId };
  const model: Model<"openai-completions"> = {
    id: config.modelId, name: config.modelId, provider: "teacher", api: "openai-completions", baseUrl: config.baseUrl,
    reasoning: true, input: ["text"], contextWindow: config.contextWindow, maxTokens: config.maxTokens,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    headers,
    ...(deepseek ? { samplingParams: {
      thinking: { type: "enabled" },
      reasoning_effort: config.reasoningEffort || "high",
      parallel_tool_calls: false,
      ...(config.modelId === "rwkv-ds41f-1017" ? { chat_template_kwargs: { enable_thinking: true } } : {}),
    } } : {}),
    compat: { supportsStore: false, supportsDeveloperRole: false, supportsReasoningEffort: false, supportsStrictMode: false, maxTokensField: "max_tokens",
      ...(deepseek ? { requiresReasoningContentOnAssistantMessages: true } : {}) },
  };
  return Object.assign(createProvider({
    id: "teacher", models: [model], baseUrl: config.baseUrl, api: openAICompletionsApi(),
    auth: { apiKey: { name: deepseek ? "DeepSeek API" : "Offline teacher", resolve: async () => ({ auth: { apiKey: config.apiKey, headers } }) } },
  }), {
    rwkvPiSessionId: sessionId,
    rwkvPiEditMode: config.editMode,
    rwkvPiHistoryMode: config.historyMode,
    ...(config.historyMode === "epoch" ? {
      rwkvPiEpochConfig: {
        budgetTokens: config.epochBudgetTokens,
        keepRecentPairs: config.epochKeepRecentPairs,
        maxResultTokens: config.epochMaxResultTokens,
      },
    } : {}),
    ...(config.historyMode === "worklist" ? {
      rwkvPiWorklistConfig: {
        budgetTokens: config.worklistBudgetTokens,
        keepRecentPairs: config.worklistKeepRecentPairs,
        foldTriggerTokens: config.worklistFoldTriggerTokens,
      },
    } : {}),
  });
}
