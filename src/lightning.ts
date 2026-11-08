import type { RwkvConfig } from "./config.js";
import { promptForRequest, STOP_TEXTS } from "./prompt.js";
import { ContinuationParser } from "./xml.js";

export async function countPrompt(prompt: string, config: RwkvConfig, fetcher: typeof fetch, signal?: AbortSignal | null): Promise<number> {
  const response = await fetcher(config.baseUrl + "/tokens/count", { method: "POST", headers: { "Content-Type": "application/json", Authorization: `Bearer ${config.apiKey}` }, body: JSON.stringify({ text: prompt }), signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(30000)]) : AbortSignal.timeout(30000) });
  if (!response.ok) throw new Error(`Lightning tokenizer returned HTTP ${response.status}.`);
  const data = await response.json() as any;
  const count = data.count ?? data.token_count ?? data.tokens;
  if (!Number.isSafeInteger(count) || count < 1) throw new Error("Invalid Lightning token count.");
  return count;
}

/** Local fetch adapter: the remote server receives only raw continuation text. */
export function lightningFetch(config: RwkvConfig, fetcher: typeof fetch): typeof fetch {
  return async (_input, init) => {
    if (typeof init?.body !== "string") throw new Error("Expected serialized chat request.");
    const chat = JSON.parse(init.body);
    const format = config.toolFormat === "json" ? "json" : "xml";
    const { prompt, thinking } = promptForRequest(chat, format);
    const count = await countPrompt(prompt, config, fetcher, init.signal);
    const max = Math.min(chat.max_tokens ?? config.maxTokens, config.maxTokens, config.contextWindow - count);
    if (!Number.isSafeInteger(max) || max < 1) throw new Error("RWKV context exhausted; input truncation is disabled.");
    const hasTools = Array.isArray(chat.tools) && chat.tools.length > 0;
    const wire: Record<string, unknown> = {
      model: config.modelId, contents: [prompt], stream: true, max_tokens: max,
      // A raw continuation must hand control back before a generated User turn.
      // This server includes matched text stops; the client removes that suffix.
      stop: STOP_TEXTS, force_reasoning: false, chunk_size: 2, metrics: true,
      // Preserve the base artifact's Tools / Open Think / Fake Think profiles.
      temperature: hasTools || thinking ? 0.96 : 1.0, top_p: hasTools || thinking ? 0.76 : 0.28, top_k: 32,
      alpha_presence: !hasTools && thinking ? 1 : 0,
      alpha_frequency: !hasTools && thinking ? 0.1 : 0,
      alpha_decay: !hasTools && thinking ? 0.988 : 0.996,
    };
    for (const key of ["temperature", "top_p", "top_k", "alpha_presence", "alpha_frequency", "alpha_decay", "adapter_id", "adapter_version", "adapter_scale", "state_id"]) if (chat[key] !== undefined) wire[key] = chat[key];
    // OpenAI-style temperature zero means greedy; native Lightning requires > 0.
    if (wire.temperature === 0) { wire.temperature = 1; wire.top_k = 1; wire.top_p = 1; }
    const controller = new AbortController();
    const signal = init.signal ? AbortSignal.any([init.signal, controller.signal]) : controller.signal;
    // The pinned batch route enables the engine's native prefix cache, including
    // for one prompt. Send full history: the engine restores a matching snapshot
    // and prefills only its token suffix. Chat's compatibility route bypasses it.
    const response = await fetcher(config.baseUrl + "/batch/completions", { ...init, body: JSON.stringify(wire), signal });
    if (!response.ok || !response.body) return response;
    const parser = new ContinuationParser(thinking, chat.tools ?? [], chat.tool_choice ?? "auto", format);
    let requestId: string | undefined, finished = false, stopSent = false;
    const encoder = new TextEncoder();
    const reader = response.body.getReader();
    async function cancelNative() {
      controller.abort();
      // Never issue the server's global stop; cancel this request ID only.
      if (requestId && !finished && !stopSent) {
        stopSent = true;
        await fetcher(config.baseUrl + "/server/stop", { method: "POST", headers: init?.headers, body: JSON.stringify({ request_id: requestId }), signal: AbortSignal.timeout(5000) }).catch(() => {});
      }
    }
    const body = new ReadableStream<Uint8Array>({
      async start(output) {
        const send = (value: unknown) => output.enqueue(encoder.encode(`data: ${JSON.stringify(value)}\n\n`));
        const chunk = (delta: Record<string, unknown>, reason: string | null = null) => ({ id: requestId ?? "rwkv-continuation", object: "chat.completion.chunk", created: Math.floor(Date.now() / 1000), model: config.modelId, choices: [{ index: 0, delta, finish_reason: reason }] });
        const decoder = new TextDecoder();
        let buffer = "", finish: string | undefined, matchedStop: string | undefined, done = false;
        try {
          while (true) {
            const next = await reader.read();
            if (next.done) break;
            buffer = (buffer + decoder.decode(next.value, { stream: true })).replaceAll("\r\n", "\n");
            let boundary: number;
            while ((boundary = buffer.indexOf("\n\n")) >= 0) {
              const frame = buffer.slice(0, boundary); buffer = buffer.slice(boundary + 2);
              const data = frame.split("\n").filter(l => l.startsWith("data:")).map(l => l.slice(5).trimStart()).join("\n");
              if (!data) continue;
              if (data === "[DONE]") { done = true; continue; }
              const event = JSON.parse(data);
              if (event.error) throw new Error(typeof event.error === "string" ? event.error : event.error.message ?? "Lightning generation error.");
              if (done || finish) throw new Error("Unexpected content after generation finished.");
              if (event.id) requestId = event.id;
              for (const choice of event.choices ?? []) {
                if (choice.index !== 0 || choice.delta?.tool_calls) throw new Error("Expected one raw text continuation.");
                if (choice.delta?.content) for (const delta of parser.push(choice.delta.content)) send(chunk(delta));
                if (choice.finish_reason) { finish = choice.finish_reason; matchedStop = choice.matched_stop; }
              }
            }
          }
          if (!done || !finish || buffer.trim()) throw new Error("Incomplete Lightning stream; tool execution refused.");
          // We always request text stops. Native cancellation also reports
          // 'stop', but without matched_stop; it must not authorize tools.
          if (finish === "stop" && matchedStop === undefined) throw new Error("Lightning stopped without a configured text boundary; tool execution refused.");
          const generated = parser.raw;
          const parsed = parser.finish(finish, matchedStop);
          // Text-stop tokens were generated and consumed by the native model,
          // even though their display suffix is removed by the client parser.
          const completionTokens = generated ? await countPrompt(generated, config, fetcher, signal) : 0;
          for (const delta of parsed.deltas) send(chunk(delta));
          send(chunk({}, parsed.reason));
          send({ ...chunk({}), choices: [], usage: { prompt_tokens: count, completion_tokens: completionTokens, total_tokens: count + completionTokens } });
          output.enqueue(encoder.encode("data: [DONE]\n\n"));
          finished = true;
          output.close();
        } catch (error) {
          await cancelNative();
          output.error(error);
        } finally { reader.releaseLock(); }
      },
      async cancel() { await cancelNative(); await reader.cancel().catch(() => {}); },
    });
    return new Response(body, { status: response.status, headers: { "content-type": "text/event-stream", "cache-control": "no-cache" } });
  };
}
