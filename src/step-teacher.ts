// Official Step endpoints only. This opt-in provider does not change the local Qwen teacher.
import { createProvider, type Model, type Provider } from "@earendil-works/pi-ai";
import { openAICompletionsApi } from "@earendil-works/pi-ai/api/openai-completions.lazy";
import { readConfig, type RwkvConfig } from "./config.js";

export const CHINESE_REASONING_SUFFIX = "语言要求：请使用中文进行推理、分析工具结果并给出最终答复；代码、命令、路径、字段名及必要的原文保持原样。依据实际工具观察推进任务，完成后说明结果和验证情况。";
export const STEP_PROMPT_VERSION = "pi-native-user-zh-reasoning-v1";

/** Append only to the user task. The native pi system prompt and tool schemas are untouched. */
export function withChineseReasoning(prompt: string): string {
  const text = prompt.trimEnd();
  return text.endsWith(CHINESE_REASONING_SUFFIX) ? text : `${text}\n\n${CHINESE_REASONING_SUFFIX}`;
}

export interface StepConfig extends RwkvConfig {
  billing: "api" | "plan";
  reasoningEffort?: "low" | "medium" | "high";
}

export function readStepConfig(env: NodeJS.ProcessEnv = process.env): StepConfig {
  const billing = env.STEP_BILLING?.trim() || "api";
  if (billing !== "api" && billing !== "plan") throw new Error("STEP_BILLING must be api or plan.");
  const model = env.STEP_MODEL?.trim() || "step-3.5-flash";
  if (!["step-3.5-flash", "step-3.5-flash-2603", "step-3.7-flash", "step-5-preview"].includes(model)) throw new Error("Unsupported STEP_MODEL; use step-3.5-flash, step-3.5-flash-2603, step-3.7-flash or step-5-preview.");
  const effort = env.STEP_REASONING_EFFORT?.trim();
  if (effort && !["low", "medium", "high"].includes(effort)) throw new Error("STEP_REASONING_EFFORT must be low, medium or high; thinking stays enabled.");
  if (effort && model === "step-3.5-flash") throw new Error("STEP_REASONING_EFFORT is supported only by newer Step models; omit it for the original model.");
  if (effort === "medium" && model === "step-3.5-flash-2603") throw new Error("step-3.5-flash-2603 supports low or high, not medium.");
  if (!env.STEP_API_KEY?.trim() && !env.STEP_API_KEY_FILE?.trim()) throw new Error("Set STEP_API_KEY or STEP_API_KEY_FILE.");
  const config = readConfig({
    RWKV_BACKEND: "openai", RWKV_MODEL: model,
    RWKV_BASE_URL: billing === "plan" ? "https://api.stepfun.com/step_plan/v1" : "https://api.stepfun.com/v1",
    RWKV_API_KEY: env.STEP_API_KEY, RWKV_API_KEY_FILE: env.STEP_API_KEY_FILE,
    RWKV_CONTEXT_WINDOW: env.STEP_CONTEXT_WINDOW || (model === "step-5-preview" ? "1000000" : "256000"),
    RWKV_MAX_TOKENS: env.STEP_MAX_TOKENS || "16384",
    RWKV_HISTORY_MODE: env.STEP_HISTORY_MODE ?? env.RWKV_HISTORY_MODE,
    RWKV_EPOCH_BUDGET_TOKENS: env.STEP_EPOCH_BUDGET_TOKENS ?? env.RWKV_EPOCH_BUDGET_TOKENS,
    RWKV_EPOCH_HARD_SLACK_TOKENS: env.STEP_EPOCH_HARD_SLACK_TOKENS ?? env.RWKV_EPOCH_HARD_SLACK_TOKENS,
    RWKV_EPOCH_KEEP_RECENT_PAIRS: env.STEP_EPOCH_KEEP_RECENT_PAIRS ?? env.RWKV_EPOCH_KEEP_RECENT_PAIRS,
    RWKV_EPOCH_MAX_RESULT_TOKENS: env.STEP_EPOCH_MAX_RESULT_TOKENS ?? env.RWKV_EPOCH_MAX_RESULT_TOKENS,
  });
  if (!config.apiKey) throw new Error("The Step API key file is empty.");
  const contextLimit = model === "step-5-preview" ? 1000000 : 256000;
  if (config.contextWindow > contextLimit) throw new Error(`STEP_CONTEXT_WINDOW must not exceed ${contextLimit}.`);
  return { ...config, billing, ...(model !== "step-3.5-flash" ? { reasoningEffort: (effort || "high") as "low" | "medium" | "high" } : {}) };
}

export function createStepProvider(config: StepConfig = readStepConfig()): Provider<"openai-completions"> {
  const model: Model<"openai-completions"> = {
    id: config.modelId, name: config.modelId, provider: "stepfun", api: "openai-completions", baseUrl: config.baseUrl,
    reasoning: true, input: ["text"], contextWindow: config.contextWindow, maxTokens: config.maxTokens,
    // pi's cost unit is USD. Do not display RMB prices or subscription Credits as dollar charges.
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    samplingParams: { reasoning_format: "deepseek-style", parallel_tool_calls: false,
      ...(config.reasoningEffort ? { reasoning_effort: config.reasoningEffort } : {}) },
    compat: { supportsStore: false, supportsDeveloperRole: false, supportsReasoningEffort: false,
      supportsStrictMode: false, maxTokensField: "max_tokens" },
  };
  return Object.assign(createProvider({
    id: "stepfun", models: [model], baseUrl: config.baseUrl, api: openAICompletionsApi(),
    auth: { apiKey: { name: "StepFun official API key", resolve: async () => ({ auth: { apiKey: config.apiKey } }) } },
  }), {
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
