import { Agent, fetch as undiciFetch } from "undici";
import type { ProviderStreams } from "@earendil-works/pi-ai";
import { REPEAT_GUARD_PREFIX, repeatedToolFailure, TARGET_GUARD_PREFIX, WORKLIST_SPIN_PREFIX } from "./loop-guard.js";
import type { RwkvConfig } from "./config.js";
import { foldEpochRequest } from "./epoch.js";
import { lightningFetch } from "./lightning.js";
import { object } from "./tasks.js";
import { foldWorklistRequest } from "./worklist.js";

function sanitizeRwkvPayload(payload: Record<string, unknown>): Record<string, unknown> {
  const raw = Array.isArray(payload.messages) ? (payload.messages as Record<string, any>[]) : undefined;
  if (!raw || raw.length < 2) return payload;
  const declaredNames = Array.isArray(payload.tools)
    ? payload.tools.map((t: any) => t?.function?.name ?? t?.name).filter((n: unknown): n is string => typeof n === "string" && n.length > 0)
    : [];
  const declaredHint = declaredNames.length > 0
    ? ` Use only the declared tools: ${declaredNames.map((n) => (n === "write" ? "write (creates parent directories automatically)" : n)).join(", ")}.`
    : "";

  const clearAssistantIndices = new Set<number>();
  let mutated = false;
  const markStreak = (toolIdx: number, depth: number) => {
    for (let step = 0, cur = toolIdx - 1; step < depth && cur >= 0; cur--) {
      const m = raw[cur];
      if (m?.role === "user") break;
      if (m?.role === "assistant" && Array.isArray(m.tool_calls)) {
        clearAssistantIndices.add(cur);
        step++;
      }
    }
  };
  const withHints = raw.map((msg, idx) => {
    if (msg?.role === "system" && typeof msg.content === "string" && /[A-Za-z]:\\/.test(msg.content)) {
      const normalized = msg.content.replace(/[A-Za-z]:\\[^\s<>"'`)]+/g, (m: string) => m.replaceAll("\\", "/"));
      if (normalized !== msg.content) {
        mutated = true;
        return { ...msg, content: normalized };
      }
    }
    if (msg?.role !== "tool" || typeof msg.tool_call_id !== "string") return msg;
    const text = typeof msg.content === "string"
      ? msg.content
      : Array.isArray(msg.content) ? msg.content.map((p: any) => (typeof p?.text === "string" ? p.text : "")).join("\n") : "";
    const trimmed = text.trim();
    if (trimmed.startsWith(REPEAT_GUARD_PREFIX) || trimmed.startsWith(TARGET_GUARD_PREFIX) || trimmed.startsWith(WORKLIST_SPIN_PREFIX)) {
      markStreak(idx, 3);
      return msg;
    }
    // Also clear cloned `<think>` after 2 consecutive identical tool-call observations so Request #3 gets a clean recovery shot.
    if (idx >= 3 && raw[idx - 1]?.role === "assistant" && raw[idx - 2]?.role === "tool" && raw[idx - 3]?.role === "assistant") {
      const prevContent = typeof raw[idx - 2]!.content === "string" ? raw[idx - 2]!.content.trim() : "";
      const c1 = JSON.stringify((raw[idx - 1]!.tool_calls ?? []).map((c: any) => [c?.function?.name, c?.function?.arguments]));
      const c0 = JSON.stringify((raw[idx - 3]!.tool_calls ?? []).map((c: any) => [c?.function?.name, c?.function?.arguments]));
      if (c1 !== "[]" && c1 === c0 && trimmed && trimmed === prevContent) {
        markStreak(idx, 2);
      }
    }
    if (/^Tool \S+ not found\.?$/.test(trimmed) && declaredHint) {
      markStreak(idx, 1);
      mutated = true;
      return { ...msg, content: `${trimmed.replace(/\.$/, "")}.${declaredHint}` };
    }
    return msg;
  });
  if (clearAssistantIndices.size === 0) return mutated ? { ...payload, messages: withHints } : payload;
  const next = withHints.map((msg, idx) => {
    if (!clearAssistantIndices.has(idx)) return msg;
    if (!msg.reasoning_content && !msg.reasoning) return msg;
    mutated = true;
    const copy: Record<string, any> = { ...msg, reasoning_content: "" };
    delete copy.reasoning;
    return copy;
  });
  return mutated ? { ...payload, messages: next } : payload;
}

// A long prefill may emit no SSE content. AbortSignal still cancels immediately.
const timeoutMs = 7_200_000;
const dispatcher = new Agent({ headersTimeout: timeoutMs, bodyTimeout: timeoutMs });
// Node's bundled fetch and an independently installed dispatcher can use different protocols.
const fetchWithDispatcher = undiciFetch as unknown as typeof fetch;
export const rwkvFetch: typeof fetch = (input, init) => fetchWithDispatcher(input, { ...init, dispatcher } as RequestInit);

export function withRwkvTransport(
  api: ProviderStreams,
  repeatFailureLimit = 3,
  config?: RwkvConfig,
  onPayloadMessages?: (messages: readonly Record<string, any>[]) => void,
): ProviderStreams {
  const wrap = (method: "stream" | "streamSimple"): ProviderStreams["stream"] => (model, context, options) =>
    api[method](model, context, {
      ...options, timeoutMs: options?.timeoutMs ?? timeoutMs,
      fetch: config?.backend === "lightning" ? lightningFetch(config, options?.fetch ?? rwkvFetch) : options?.fetch ?? rwkvFetch,
      onPayload: async (payload, requestModel) => {
        const failure = repeatedToolFailure(context.messages, repeatFailureLimit);
        // Throw inside pi-ai's stream error handling, before sending another request.
        if (failure) throw new Error(failure);
        const sanitized = object(payload) ? sanitizeRwkvPayload(payload) : payload;
        const nextPayload = !object(sanitized)
          ? sanitized
          : config?.historyMode === "epoch"
            ? foldEpochRequest(sanitized, {
                budgetTokens: config.epochBudgetTokens,
                keepRecentPairs: config.epochKeepRecentPairs,
                maxResultTokens: config.epochMaxResultTokens,
              }).request
            : config?.historyMode === "worklist"
              ? foldWorklistRequest(sanitized, {
                  budgetTokens: config.worklistBudgetTokens,
                  keepRecentPairs: config.worklistKeepRecentPairs,
                  foldTriggerTokens: config.worklistFoldTriggerTokens,
                }).request
              : sanitized;
        if (object(nextPayload) && Array.isArray(nextPayload.messages)) {
          onPayloadMessages?.(nextPayload.messages as Record<string, any>[]);
        }
        return options?.onPayload ? options.onPayload(nextPayload, requestModel) : nextPayload;
      },
    });
  return { ...api, stream: wrap("stream"), streamSimple: wrap("streamSimple") };
}
