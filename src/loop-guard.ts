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
  let callFingerprint: string | undefined;
  let resultFingerprint: string | undefined;
  let toolName = "";
  for (let count = 0, index = messages.length - 1; count < limit; count++, index -= 2) {
    const result = messages[index];
    const assistant = messages[index - 1];
    if (result?.role !== "toolResult" || !result.isError || assistant?.role !== "assistant"
      || assistant.stopReason !== "toolUse") return;
    const calls = assistant.content.filter(block => block.type === "toolCall");
    const call = calls[0];
    if (calls.length !== 1 || !call || call.id !== result.toolCallId || call.name !== result.toolName) return;
    const curCall = canonical([call.name, call.arguments]);
    if (callFingerprint !== undefined && curCall !== callFingerprint) return;
    callFingerprint = curCall;
    const blockedByRepeatGuard = result.content.some(
      part => part.type === "text" && part.text.startsWith(REPEAT_GUARD_PREFIX),
    );
    if (!blockedByRepeatGuard) {
      const curRes = canonical(result.content);
      if (resultFingerprint !== undefined && curRes !== resultFingerprint) return;
      resultFingerprint = curRes;
    }
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
  const obj = args as { path?: unknown; url?: unknown } | undefined;
  const path = obj?.path;
  if (typeof path === "string" && path.trim()) return path.trim().replace(/^\.\//, "");
  const url = obj?.url;
  if (typeof url === "string" && url.trim()) return url.trim();
  return undefined;
};

/** Block a path-based call after the latest `limit` tool results were all failures of this tool on this
 * same path, whatever the other arguments were. This catches what repeatedToolFailure cannot: an edit
 * retried with slightly different, still wrong oldText each time. Any other tool result (a read of the
 * file, a success, another path) or a user message ends the streak.
 */
export function repeatedTargetFailure(
  messages: readonly Message[],
  name: string,
  args: unknown,
  limit = 3,
  editMode?: "line" | "text",
): string | undefined {
  const target = targetOf(args);
  if (limit === 0 || !target) return;
  const calls = new Map<string, { name: string; arguments: unknown }>();
  for (const message of messages) if (message.role === "assistant") {
    for (const block of message.content) if (block.type === "toolCall") calls.set(block.id, block);
  }
  const argObj = args as Record<string, any> | undefined;
  const looksLineMode = editMode === "line"
    || (editMode === undefined && (
      typeof argObj?.start_line === "number"
      || (Array.isArray(argObj?.edits) && typeof argObj.edits[0]?.start_line === "number")
    ));
  let count = 0;
  for (let index = messages.length - 1; index >= 0; index--) {
    const message = messages[index]!;
    if (message.role === "assistant") continue;
    if (message.role !== "toolResult") break;
    const call = calls.get(message.toolCallId);
    if (!message.isError || call?.name !== name || targetOf(call.arguments) !== target) break;
    if (++count >= limit) {
      const next = name === "edit" || name === "write"
        ? (looksLineMode
            ? `请先用 read 重新读取 ${target}，核对当前的 1-based 行号与首行锚点（start_text）后再调用 edit。`
            : `请先重新读取 ${target}，按文件原文逐字复制要替换的内容后再修改。`)
        : name === "web_fetch"
          ? "该 URL 无法访问（可能不存在或已被目标站点拦截），请改用搜索结果中的其他官方 URL 或结合已有证据作答。"
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

  // Collect recent unblocked single-tool turns from newest to oldest, stopping at any user message or multi-tool batch.
  const recent: { callKey: string; resultKey: string; isError: boolean }[] = [];
  for (let cur = index; cur >= 1 && recent.length < limit * 2; cur -= 2) {
    const toolResult = messages[cur];
    const assistant = messages[cur - 1];
    if (toolResult?.role !== "toolResult" || assistant?.role !== "assistant") break;
    const calls = assistant.content.filter(block => block.type === "toolCall");
    const call = calls[0];
    if (calls.length !== 1 || !call || call.id !== toolResult.toolCallId) break;
    // An earlier block by this guard is neutral: it carries no new observation.
    const blocked = toolResult.isError && toolResult.content.some(part => part.type === "text" && part.text.startsWith(REPEAT_GUARD_PREFIX));
    if (blocked) continue;
    recent.push({
      callKey: canonical([call.name, call.arguments]),
      resultKey: canonical([toolResult.isError, toolResult.content]),
      isError: Boolean(toolResult.isError),
    });
  }

  // 1. Consecutive identical calls (`A -> A -> [A]`): intercepts both repeated identical successes and
  // a 3rd identical failure before execution so reasoning sanitization can break `<think>` cloning;
  // if the model still repeats 3 blocked calls in a row, repeatedToolFailure terminates the turn cleanly.
  if (recent.length >= limit) {
    const streak = recent.slice(0, limit);
    if (streak.every(entry => !entry.isError && entry.callKey === target && entry.resultKey === streak[0]!.resultKey)) {
      return `${REPEAT_GUARD_PREFIX} ${name} 已连续 ${limit} 次以完全相同的参数调用并得到相同结果，本次未执行。请根据已有结果换一种操作，或给出最终答案。`;
    }
    if (streak.every(entry => entry.isError && entry.callKey === target && entry.resultKey === streak[0]!.resultKey)) {
      return `${REPEAT_GUARD_PREFIX} ${name} 已连续 ${limit} 次以完全相同的参数返回相同错误，本次未执行。请仔细根据上一条报错修改参数（如先用 read 核对真实内容或更换路径/命令），不要原样重试。`;
    }
  }

  // 2. Period-2 ping-pong loop (`A -> B -> A -> B -> [A]`): catches alternating read/check loops (including mixed ok/error pairs)
  // when neither operation's arguments nor observations changed across `limit` full cycles.
  if (limit >= 2 && recent.length >= limit * 2) {
    const window = recent.slice(0, limit * 2); // newest-first: [B2, A2, B1, A1] when proposing A3
    const partnerCall = window[0]!.callKey;
    const partnerRes = window[0]!.resultKey;
    const selfRes = window[1]!.resultKey;
    if (partnerCall !== target) {
      let pingPong = true;
      for (let i = 0; i < limit * 2; i++) {
        const entry = window[i]!;
        if (i % 2 === 0) {
          if (entry.callKey !== partnerCall || entry.resultKey !== partnerRes) { pingPong = false; break; }
        } else {
          if (entry.callKey !== target || entry.resultKey !== selfRes) { pingPong = false; break; }
        }
      }
      if (pingPong) {
        return `${REPEAT_GUARD_PREFIX} ${name} 与另一操作已交替重复 ${limit} 轮且参数与结果完全未变，本次未执行。不要再重复探查同一目录或路径，请直接执行下一步创建/修改操作，或给出最终答案。`;
      }
    }
  }
  return;
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
