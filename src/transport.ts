import { Agent, fetch as undiciFetch } from "undici";
import type { ProviderStreams } from "@earendil-works/pi-ai";
import { repeatedToolFailure } from "./loop-guard.js";
import type { RwkvConfig } from "./config.js";
import { lightningFetch } from "./lightning.js";

// A long prefill may emit no SSE content. AbortSignal still cancels immediately.
const timeoutMs = 7_200_000;
const dispatcher = new Agent({ headersTimeout: timeoutMs, bodyTimeout: timeoutMs });
// Node's bundled fetch and an independently installed dispatcher can use different protocols.
const fetchWithDispatcher = undiciFetch as unknown as typeof fetch;
const rwkvFetch: typeof fetch = (input, init) => fetchWithDispatcher(input, { ...init, dispatcher } as RequestInit);

export function withRwkvTransport(api: ProviderStreams, repeatFailureLimit = 3, config?: RwkvConfig): ProviderStreams {
  const wrap = (method: "stream" | "streamSimple"): ProviderStreams["stream"] => (model, context, options) =>
    api[method](model, context, {
      ...options, timeoutMs: options?.timeoutMs ?? timeoutMs,
      fetch: config?.backend === "lightning" ? lightningFetch(config, options?.fetch ?? rwkvFetch) : options?.fetch ?? rwkvFetch,
      onPayload: async (payload, requestModel) => {
        const failure = repeatedToolFailure(context.messages, repeatFailureLimit);
        // Throw inside pi-ai's stream error handling, before sending another request.
        if (failure) throw new Error(failure);
        return options?.onPayload?.(payload, requestModel);
      },
    });
  return { ...api, stream: wrap("stream"), streamSimple: wrap("streamSimple") };
}
