import type { Message } from "@earendil-works/pi-ai";

function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value !== null && typeof value === "object") {
    return `{${Object.entries(value).sort(([a], [b]) => a.localeCompare(b))
      .map(([key, item]) => `${JSON.stringify(key)}:${canonical(item)}`).join(",")}}`;
  }
  return JSON.stringify(value) ?? "null";
}

/** Stop unchanged failed single-tool turns, not repeated words or successful polling.
 * Read native messages before OpenAI serialization drops toolResult.isError.
 * A user message, different operation/result, success, or multi-tool batch breaks the streak.
 */
export function repeatedToolFailure(messages: readonly Message[], limit = 3): string | undefined {
  if (limit === 0) return;
  let fingerprint: string | undefined;
  let toolName = "";
  for (let count = 0, index = messages.length - 1; count < limit; count++, index -= 2) {
    const result = messages[index];
    const assistant = messages[index - 1];
    if (result?.role !== "toolResult" || !result.isError || assistant?.role !== "assistant"
      || assistant.stopReason !== "toolUse") return;
    const calls = assistant.content.filter(block => block.type === "toolCall");
    const call = calls[0];
    if (calls.length !== 1 || !call || call.id !== result.toolCallId || call.name !== result.toolName) return;
    const current = canonical([call.name, call.arguments, result.content]);
    if (fingerprint !== undefined && current !== fingerprint) return;
    fingerprint = current;
    toolName = call.name;
  }
  return `RWKV_TOOL_LOOP: ${toolName} 连续 ${limit} 次以相同参数返回相同错误，已停止本轮。请根据错误更换操作或补充指令后继续。`;
}
