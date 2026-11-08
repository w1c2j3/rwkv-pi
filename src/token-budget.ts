import type { RwkvConfig } from "./config.js";
import { object } from "./tasks.js";
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
  if (config.backend === "lightning") {
    const inputTokens = await countPrompt(promptForRequest(request, config.toolFormat === "json" ? "json" : "xml").prompt, config, fetch, signal);
    const outputTokens = Math.min(desired, config.maxTokens, config.contextWindow - inputTokens);
    if (outputTokens < 1) throw new Error("RWKV context exhausted; input truncation is disabled.");
    return { request: { ...request, max_tokens: outputTokens }, budget: { inputTokens, outputTokens, contextWindow: config.contextWindow, requestedOutputTokens: desired } };
  }
  const body: Record<string, unknown> = { model: request.model, messages: request.messages, add_generation_prompt: true };
  for (const key of ["tools", "chat_template", "chat_template_kwargs", "add_special_tokens", "continue_final_message", "add_generation_prompt"]) {
    if (request[key] !== undefined) body[key] = request[key];
  }
  const response = await fetch(config.baseUrl.replace(/\/v1$/, "") + "/tokenize", {
    method: "POST", headers: { "Content-Type": "application/json", Authorization: `Bearer ${config.apiKey}` },
    body: JSON.stringify(body), signal: AbortSignal.any([signal, AbortSignal.timeout(30000)]),
  });
  if (!response.ok) throw new Error(`Context tokenization returned HTTP ${response.status}.`);
  const count: unknown = await response.json();
  if (!object(count) || !Number.isSafeInteger(count.count) || (count.count as number) < 1
    || !Array.isArray(count.tokens) || count.tokens.length !== count.count
    || !Number.isSafeInteger(count.max_model_len)) throw new Error("Invalid tokenizer budget response.");
  if (count.max_model_len !== config.contextWindow) throw new Error(`RWKV context mismatch: client ${config.contextWindow}, server ${count.max_model_len}.`);
  const remaining = config.contextWindow - (count.count as number);
  if (remaining < 1) throw new Error("RWKV context exhausted by the prompt; input truncation is disabled.");
  const outputTokens = Math.min(desired, config.maxTokens, remaining);
  return { request: { ...request, max_tokens: outputTokens }, budget: {
    inputTokens: count.count as number, contextWindow: config.contextWindow,
    requestedOutputTokens: desired, outputTokens,
  } };
}
