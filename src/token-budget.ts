import type { RwkvConfig } from "./config.js";
import { promptForRequest } from "./prompt.js";
import { countPrompt } from "./lightning.js";

export interface TokenBudget {
  inputTokens: number;
  contextWindow: number;
  requestedOutputTokens: number;
  outputTokens: number;
}

/** Count the complete deployed chat prompt, including tools and the generation prefix. */
export async function budgetRwkvRequest(
  request: Record<string, unknown>, config: RwkvConfig, signal: AbortSignal,
): Promise<{ request: Record<string, unknown>; budget: TokenBudget }> {
  if (request.model !== config.modelId || !Array.isArray(request.messages)) throw new Error("Token budget requires this RWKV model's chat request.");
  if (request.max_completion_tokens !== undefined || request.truncate_prompt_tokens !== undefined) {
    throw new Error("Expected max_tokens without prompt truncation.");
  }
  const desired = request.max_tokens ?? config.maxTokens;
  if (typeof desired !== "number" || !Number.isSafeInteger(desired) || desired < 1) throw new Error("Invalid requested output budget.");
  // RWKV prompts are measured only with the Lightning-native tokenizer, on the exact rendered continuation text.
  if (config.backend !== "lightning") throw new Error("RWKV token budgets use the Lightning-native tokenizer; set RWKV_BACKEND=lightning.");
  const inputTokens = await countPrompt(promptForRequest(request, config.toolFormat === "json" ? "json" : "xml").prompt, config, fetch, signal);
  const outputTokens = Math.min(desired, config.maxTokens, config.contextWindow - inputTokens);
  if (outputTokens < 1) throw new Error("RWKV context exhausted; input truncation is disabled.");
  return { request: { ...request, max_tokens: outputTokens }, budget: { inputTokens, outputTokens, contextWindow: config.contextWindow, requestedOutputTokens: desired } };
}
