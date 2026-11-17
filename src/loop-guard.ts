import type { Message } from "@earendil-works/pi-ai";

export function canonical(value: unknown): string {
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

/** Block a tool call identical (name + arguments) to the immediately preceding `limit` single-call
 * turns when all of those succeeded with the same result: repeating it cannot produce new information.
 * Returns the reason pi records as the blocked call's tool error result; polling with changing
 * results, different arguments, or multi-call batches never matches.
 */
export const REPEAT_GUARD_PREFIX = "RWKV_REPEAT_GUARD:";
export const TARGET_GUARD_PREFIX = "RWKV_TARGET_GUARD:";
export const WORKLIST_SPIN_PREFIX = "RWKV_WORKLIST_SPIN:";
const WORKLIST_NAMES = new Set(["todo_plan", "todo_update"]);

const targetOf = (args: unknown): string | undefined => {
  const path = (args as { path?: unknown } | undefined)?.path;
  return typeof path === "string" && path.trim() ? path.trim().replace(/^\.\//, "") : undefined;
};

/** Block a path-based call after the latest `limit` tool results were all failures of this tool on this
 * same path, whatever the other arguments were. This catches what repeatedToolFailure cannot: an edit
 * retried with slightly different, still wrong oldText each time. Any other tool result (a read of the
 * file, a success, another path) or a user message ends the streak.
 */
export function repeatedTargetFailure(messages: readonly Message[], name: string, args: unknown, limit = 3): string | undefined {
  const target = targetOf(args);
  if (limit === 0 || !target) return;
  const calls = new Map<string, { name: string; arguments: unknown }>();
  for (const message of messages) if (message.role === "assistant") {
    for (const block of message.content) if (block.type === "toolCall") calls.set(block.id, block);
  }
  let count = 0;
  for (let index = messages.length - 1; index >= 0; index--) {
    const message = messages[index]!;
    if (message.role === "assistant") continue;
    if (message.role !== "toolResult") break;
    const call = calls.get(message.toolCallId);
    if (!message.isError || call?.name !== name || targetOf(call.arguments) !== target) break;
    if (++count >= limit) {
      const next = name === "edit" || name === "write"
        ? `请先重新读取 ${target}，按文件原文逐字复制要替换的内容后再修改。`
        : "请先用 ls 或 find 确认路径后再试。";
      return `${TARGET_GUARD_PREFIX} ${name} 对 ${target} 已连续失败 ${count} 次，本次未执行。${next}`;
    }
  }
  return;
}

export function repeatedIdenticalCall(messages: readonly Message[], name: string, args: unknown, limit = 2): string | undefined {
  if (limit === 0) return;
  const target = canonical([name, args]);
  let index = messages.length - 1;
  // The current assistant turn (containing this call) may already be in the context.
  if (messages[index]?.role === "assistant") index--;
  let result: string | undefined;
  for (let count = 0; count < limit; count++, index -= 2) {
    const toolResult = messages[index];
    const assistant = messages[index - 1];
    if (toolResult?.role !== "toolResult" || assistant?.role !== "assistant") return;
    const calls = assistant.content.filter(block => block.type === "toolCall");
    const call = calls[0];
    if (calls.length !== 1 || !call || call.id !== toolResult.toolCallId || canonical([call.name, call.arguments]) !== target) return;
    // An earlier block by this guard is neutral: it carries no new observation.
    const blocked = toolResult.isError && toolResult.content.some(part => part.type === "text" && part.text.startsWith(REPEAT_GUARD_PREFIX));
    if (blocked) continue;
    // Identical failures are left to repeatedToolFailure, which ends the turn.
    if (toolResult.isError) return;
    const current = canonical([toolResult.isError, toolResult.content]);
    if (result !== undefined && current !== result) return;
    result = current;
  }
  return `${REPEAT_GUARD_PREFIX} ${name} 已连续 ${limit} 次以完全相同的参数调用并得到相同结果，本次未执行。请根据已有结果换一种操作，或给出最终答案。`;
}

/** Block a 3rd consecutive worklist bookkeeping call (`todo_plan` / `todo_update`) when no workspace tool executed in between. */
export function repeatedWorklistSpin(messages: readonly Message[], name: string, limit = 2): string | undefined {
  if (limit === 0 || !WORKLIST_NAMES.has(name)) return;
  let index = messages.length - 1;
  if (messages[index]?.role === "assistant") index--;
  let consecutive = 0;
  while (index >= 1 && consecutive < limit) {
    const toolResult = messages[index];
    const assistant = messages[index - 1];
    if (toolResult?.role !== "toolResult" || assistant?.role !== "assistant") return;
    const calls = assistant.content.filter((block) => block.type === "toolCall");
    if (!calls.length || !calls.every((call) => WORKLIST_NAMES.has(call.name))) return;
    consecutive++;
    index -= 2;
  }
  if (consecutive < limit) return;
  return `${WORKLIST_SPIN_PREFIX} 已连续调用 ${consecutive} 次工单工具而未执行工作区工具，本次未执行。请先使用工作区工具（read/edit/write/bash）推进当前子目标，若全部子目标均已完成请直接作答。`;
}
