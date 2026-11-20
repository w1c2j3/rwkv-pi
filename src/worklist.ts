import type { ToolDefinition } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { object } from "./tasks.js";

export const TODO_PLAN_TOOL = "todo_plan";
export const TODO_UPDATE_TOOL = "todo_update";
export const WORKLIST_TOOL_NAMES = [TODO_PLAN_TOOL, TODO_UPDATE_TOOL] as const;
export const WORKLIST_TOOLS: ReadonlySet<string> = new Set<string>(WORKLIST_TOOL_NAMES);

export function isWorklistTool(name: string): boolean {
  return WORKLIST_TOOLS.has(name);
}

export type WorklistStatus = "pending" | "in_progress" | "completed" | "dropped";

export interface WorklistItem {
  readonly id: number;
  readonly content: string;
  readonly status: WorklistStatus;
  readonly note?: string;
}

export const TODO_PLAN_DESCRIPTION =
  "Create or revise the ordered work list for multi-step tasks (touching multiple methods/files, code + docs/tests, or 3+ distinct subgoals). "
  + "Skip this tool for casual chat, quick single-goal lookups (1-3 reads or searches), or a trivial one-spot edit. "
  + "For multi-step changes, you may inspect 1-2 files first to understand the layout, then call todo_plan BEFORE making your first file edit — do not wait until after editing to plan. "
  + "Group reading and editing the same file into one closed-loop item. "
  + "Calling todo_plan mid-turn keeps completed/dropped items and their notes intact and replaces only unfinished items.";

export const TODO_UPDATE_DESCRIPTION =
  "Update one item in the current work list by id. "
  + "As soon as an item is finished, mark it completed with a concise factual note (<= 120 words: exact file paths, updated line ranges, signatures, and test exit codes — never mark unexecuted work completed). "
  + "Older exploratory tool outputs before the recent workspace actions are folded once note is saved, while your initial todo_plan and the latest workspace results stay visible. "
  + "The next pending item becomes in_progress automatically; you may also checkpoint an in_progress item with an interim note before context fills up.";

export const WORKLIST_SYSTEM_GUIDANCE =
  "For multi-step tasks (touching multiple files or combining code, docs, and tests), call todo_plan early before editing, and call todo_update with a concise note (exact file paths, lines, and verified outcomes) as subgoals progress; skip todo_plan for casual chat, quick lookups, or trivial one-spot edits.";

export const WORKLIST_CHECKPOINT_NUDGE =
  'Context budget reached for the current step. Call `todo_update` once to checkpoint verified facts into a concise `note` (<= 120 words: exact file paths, line ranges, findings, and exit codes; keep `status: "in_progress"` if unfinished, or `"completed"` only if verified), or finish directly if ready to answer.';

export const WORKLIST_INIT_NUDGE =
  "Context budget reached while exploring this task. If multi-step work remains, call `todo_plan` to list the remaining steps (then checkpoint verified paths and findings with `todo_update`); if only 1-2 quick steps or the final answer remain, finish directly without `todo_plan`.";

export const DEFAULT_WORKLIST_BUDGET_TOKENS = 9600;
export const DEFAULT_WORKLIST_HARD_SLACK_TOKENS = 2400;
export const DEFAULT_BACKBONE_MAX_TOKENS = 3600;
export const DEFAULT_KEEP_RECENT_PAIRS = 2;
export const DEFAULT_FOLD_TRIGGER_TOKENS = 6144;
export const DEFAULT_WORKLIST_MAX_RESULT_TOKENS = 2000;
export const MAX_WORKLIST_NOTE_CHARS = 600;

const MAX_SECOND_RECENT_PAIR_TOKENS = 2200;
const MAX_PINNED_UNEDITED_READ_TOKENS = 1800;
const MAX_PRIOR_TURN_FALLBACK_RESULT_CHARS = 1600;
const INIT_NUDGE_COOLDOWN_STEPS = 4;
const CHECKPOINT_NUDGE_COOLDOWN_STEPS = 3;

const WORKLIST_PROTOCOL_TAG = /<\/?(?:think|tool_call|tool_response)(?:\s[^>]*)?>/gi;
const WORKLIST_LEADING_BULLET = /^\s*(?:\[\d+\]|\d+[.)]|[-*•])\s*/;

export function sanitizeWorklistField(text: string): string {
  return text.replace(WORKLIST_PROTOCOL_TAG, "").replace(/\s*\n+\s*/g, " | ").trim();
}

export function parsePlanItems(raw: unknown): string[] | undefined {
  if (Array.isArray(raw)) {
    const out: string[] = [];
    for (const entry of raw) {
      if (typeof entry === "string") {
        const cleaned = sanitizeWorklistField(entry.replace(WORKLIST_LEADING_BULLET, ""));
        if (cleaned !== "") out.push(cleaned);
      } else if (object(entry)) {
        const candidate = entry.content ?? entry.title ?? entry.text ?? entry.step;
        if (typeof candidate === "string") {
          const cleaned = sanitizeWorklistField(candidate.replace(WORKLIST_LEADING_BULLET, ""));
          if (cleaned !== "") out.push(cleaned);
        }
      } else {
        return undefined;
      }
    }
    return out;
  }
  if (typeof raw === "string") {
    const lines = raw
      .split(/\r?\n/)
      .map((line) => sanitizeWorklistField(line.replace(WORKLIST_LEADING_BULLET, "")))
      .filter((line) => line !== "");
    return lines.length > 0 ? lines : undefined;
  }
  return undefined;
}

export function normalizeWorklistStatus(raw: unknown): WorklistStatus | undefined {
  if (typeof raw !== "string") return undefined;
  const lower = raw.trim().toLowerCase();
  if (lower === "pending" || lower === "todo" || lower === "not_started") return "pending";
  if (lower === "in_progress" || lower === "in-progress" || lower === "doing" || lower === "active") return "in_progress";
  if (lower === "completed" || lower === "complete" || lower === "done" || lower === "finished") return "completed";
  if (lower === "dropped" || lower === "cancelled" || lower === "canceled" || lower === "skipped") return "dropped";
  return undefined;
}

export function applyTodoPlan(current: readonly WorklistItem[], rawItems: unknown): WorklistItem[] | string {
  const parsed = parsePlanItems(rawItems);
  if (parsed === undefined || parsed.length === 0) {
    return "items must be a non-empty array of step strings";
  }
  const locked = current
    .filter((item) => item.status === "completed" || item.status === "dropped")
    .map((item, idx) => ({ ...item, id: idx + 1 }));
  const lockedTitles = new Set(locked.map((item) => item.content));
  const remaining = parsed.filter((title) => !lockedTitles.has(title));
  if (locked.length === 0 && remaining.length < 2) {
    return "todo_plan requires at least 2 steps; execute single-step tasks directly without todo_plan";
  }
  if (remaining.length === 0) {
    return "todo_plan requires at least 1 new unfinished step after completed items";
  }
  if (locked.length + remaining.length > 12) {
    return `todo_plan accepts at most 12 total items (got ${locked.length + remaining.length}); group micro-actions into coarser subgoals`;
  }
  const seen = new Set<string>();
  for (const [idx, title] of remaining.entries()) {
    if (seen.has(title)) return `items[${idx}] duplicates ${JSON.stringify(title)}`;
    seen.add(title);
  }
  const tail: WorklistItem[] = remaining.map((content, idx) => ({
    id: locked.length + idx + 1,
    content,
    status: idx === 0 ? "in_progress" : "pending",
  }));
  return [...locked, ...tail];
}

export interface TodoUpdateInput {
  readonly id?: unknown;
  readonly status?: unknown;
  readonly note?: unknown;
  readonly content?: unknown;
  readonly title?: unknown;
}

export function applyTodoUpdate(
  current: readonly WorklistItem[],
  input: TodoUpdateInput,
): { items: WorklistItem[]; noted: boolean } | string {
  if (current.length === 0) {
    return "no active todo list in this turn; call todo_plan first if this task has multiple subgoals";
  }
  const rawId = typeof input.id === "string" ? Number(input.id.trim()) : input.id;
  if (typeof rawId !== "number" || !Number.isInteger(rawId) || rawId < 1 || rawId > current.length) {
    return `id must be an integer between 1 and ${current.length} (got ${JSON.stringify(input.id)})`;
  }
  const status = normalizeWorklistStatus(input.status);
  if (status === undefined || status === "pending") {
    return "status must be one of: in_progress, completed, dropped";
  }
  const target = current[rawId - 1]!;
  const targetClosed = target.status === "completed" || target.status === "dropped";
  if (targetClosed && status === "in_progress") {
    return `item #${rawId} is already ${target.status}; update the active unfinished item or call todo_plan to add a follow-up step`;
  }
  const providedNote = typeof input.note === "string" ? sanitizeWorklistField(input.note) : "";
  if (providedNote.length > MAX_WORKLIST_NOTE_CHARS) {
    return `note exceeds ${MAX_WORKLIST_NOTE_CHARS} characters (got ${providedNote.length}); keep it to <= 120 words of concrete paths, line numbers, edits, and exit codes without pasting source blocks`;
  }
  const effectiveNote = providedNote !== "" ? providedNote : target.note;
  if ((status === "completed" || status === "dropped") && (!effectiveNote || effectiveNote === "")) {
    return "note is required when status is completed or dropped: record concrete findings, exact file paths/lines, applied edits, or test results";
  }
  const rawTitle = input.content ?? input.title;
  const updatedContent = !targetClosed && typeof rawTitle === "string" && sanitizeWorklistField(rawTitle) !== ""
    ? sanitizeWorklistField(rawTitle)
    : target.content;

  const next = current.map((item, index): WorklistItem => {
    if (index === rawId - 1) {
      return {
        id: item.id,
        content: updatedContent,
        status,
        ...(effectiveNote ? { note: effectiveNote } : {}),
      };
    }
    if (status === "in_progress" && item.status === "in_progress") {
      return { ...item, status: "pending" };
    }
    return item;
  });

  if (!next.some((item) => item.status === "in_progress")) {
    const afterIdx = next.findIndex((item, idx) => idx > rawId - 1 && item.status === "pending");
    const promoteIdx = afterIdx >= 0 ? afterIdx : next.findIndex((item) => item.status === "pending");
    if (promoteIdx >= 0) {
      next[promoteIdx] = { ...next[promoteIdx]!, status: "in_progress" };
    }
  }

  return { items: next, noted: providedNote !== "" };
}

export function formatWorklist(items: readonly WorklistItem[]): string {
  const lines = ["Todo list:"];
  for (const item of items) {
    lines.push(`[${item.id}] (${item.status}) ${item.content}`);
    if (item.note) lines.push(`  note: ${item.note}`);
  }
  if (items.length > 0 && items.every((item) => item.status === "completed" || item.status === "dropped")) {
    lines.push("All todo items are finished. Give the final answer directly now unless verification failed.");
  }
  return lines.join("\n");
}

/**
 * Calibrated RWKV World token estimator aligned with rwkv_vocab_v20230424
 * (CJK/fullwidth ~1.04 chars/tok, ASCII punctuation/digits ~2.0 chars/tok,
 * ASCII identifiers ~4.0 chars/tok); overrideable in tests.
 */
export function estimateRwkvTokens(text: string): number {
  if (!text) return 0;
  let weight = 0;
  let spaceRun = 0;
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i);
    if (code === 32) {
      spaceRun++;
      weight += spaceRun > 1 ? 0.22 : 0.26;
      continue;
    }
    spaceRun = 0;
    if ((code >= 0x4e00 && code <= 0x9fff) || (code >= 0x3000 && code <= 0x303f) || (code >= 0xff00 && code <= 0xffef)) {
      weight += 0.96;
    } else if (code === 10 || code === 9) {
      weight += 0.55;
    } else if (code >= 48 && code <= 57) {
      weight += 0.50;
    } else if ((code >= 65 && code <= 90) || (code >= 97 && code <= 122)) {
      weight += 0.25;
    } else if (code < 0x80) {
      weight += 0.50;
    } else {
      weight += 0.85;
    }
  }
  return Math.max(1, Math.ceil(weight));
}

function messageText(content: unknown): string {
  if (content == null) return "";
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => (object(part) && typeof part.text === "string" ? part.text : ""))
      .filter(Boolean)
      .join("\n");
  }
  return "";
}

type OpenAIMessage = Record<string, any>;

interface ToolCallEntry {
  readonly id: string;
  readonly name: string;
  readonly args: Record<string, unknown>;
}

function extractToolCalls(msg: OpenAIMessage): ToolCallEntry[] {
  const rawCalls = Array.isArray(msg.tool_calls) ? msg.tool_calls : [];
  const out: ToolCallEntry[] = [];
  for (const call of rawCalls) {
    if (!object(call)) continue;
    const fn = object(call.function) ? call.function : call;
    const name = typeof fn.name === "string" ? fn.name : "";
    let args: Record<string, unknown> = {};
    if (typeof fn.arguments === "string") {
      try {
        const parsed = JSON.parse(fn.arguments);
        if (object(parsed)) args = parsed;
      } catch {
        args = {};
      }
    } else if (object(fn.arguments)) {
      args = fn.arguments;
    }
    out.push({ id: typeof call.id === "string" ? call.id : "", name, args });
  }
  return out;
}

function renderSingleMessageEstimate(msg: OpenAIMessage): string {
  if (msg.role === "system") return `System: ${messageText(msg.content)}\n\n`;
  if (msg.role === "user") return `User: ${messageText(msg.content)}\n\n`;
  if (msg.role === "tool") return `User: <tool_response>\n${messageText(msg.content)}\n</tool_response>\n\n`;
  if (msg.role === "assistant") {
    const reasoning = messageText(msg.reasoning_content ?? msg.reasoning);
    const body = messageText(msg.content);
    const calls = extractToolCalls(msg).map((c) => `<tool_call>${JSON.stringify({ name: c.name, arguments: c.args })}</tool_call>`).join("\n");
    return `Assistant: <think>${reasoning}</think>\n${[body, calls].filter(Boolean).join("\n")}\n\n`;
  }
  return "";
}

function isErrorResultText(text: string): boolean {
  const trimmed = text.trim();
  return /^(?:Error:|RWKV_|Tool .* failed|invalid |unknown )/i.test(trimmed);
}

interface TurnGroup {
  readonly kind: "user" | "final_assistant" | "tool_group";
  readonly messages: OpenAIMessage[];
  readonly calls: ToolCallEntry[];
  readonly ok: boolean;
  readonly isWorklist: boolean;
  readonly notedUpdate: boolean;
}

function isNotedUpdateEntry(call: ToolCallEntry, ok: boolean): boolean {
  if (!ok || call.name !== TODO_UPDATE_TOOL) return false;
  const note = call.args.note ?? call.args.notes ?? call.args.summary ?? call.args.findings ?? call.args.conclusion;
  return typeof note === "string" && sanitizeWorklistField(note) !== "";
}

function slimOversizedToolMessage(
  msg: OpenAIMessage,
  maxResultTokens: number,
  count: (text: string) => number,
): OpenAIMessage {
  if (msg.role !== "tool" || maxResultTokens <= 0) return msg;
  const raw = messageText(msg.content);
  if (!raw || count(raw) <= maxResultTokens) return msg;
  const maxChars = Math.max(1200, Math.floor(maxResultTokens * 3.1));
  if (raw.length <= maxChars) return msg;
  const headChars = Math.floor(maxChars * 0.75);
  const tailChars = maxChars - headChars;
  const omitted = raw.length - maxChars;
  const slimmed = `${raw.slice(0, headChars)}\n\n[... ${omitted} characters omitted to preserve context sweet spot; use read with offset/limit or grep for specific lines ...]\n\n${raw.slice(raw.length - tailChars)}`;
  return { ...msg, content: slimmed };
}

function parseTurnGroups(
  slice: readonly OpenAIMessage[],
  maxResultTokens = DEFAULT_WORKLIST_MAX_RESULT_TOKENS,
  count: (text: string) => number = estimateRwkvTokens,
): TurnGroup[] {
  const groups: TurnGroup[] = [];
  let i = 0;
  while (i < slice.length) {
    const msg = slice[i]!;
    if (msg.role === "assistant" && Array.isArray(msg.tool_calls) && msg.tool_calls.length > 0) {
      const calls = extractToolCalls(msg);
      const pending = new Set(calls.map((c) => c.id).filter(Boolean));
      const groupMsgs: OpenAIMessage[] = [msg];
      let j = i + 1;
      let allOk = true;
      while (j < slice.length && slice[j]!.role === "tool") {
        const rawTool = slice[j]!;
        const toolMsg = slimOversizedToolMessage(rawTool, maxResultTokens, count);
        groupMsgs.push(toolMsg);
        if (typeof rawTool.tool_call_id === "string") pending.delete(rawTool.tool_call_id);
        if (rawTool.isError === true || isErrorResultText(messageText(rawTool.content))) allOk = false;
        j++;
        if (pending.size === 0 && (j >= slice.length || slice[j]!.role !== "tool")) break;
      }
      const isWl = calls.length > 0 && calls.every((c) => isWorklistTool(c.name));
      const noted = calls.some((c) => isNotedUpdateEntry(c, allOk));
      groups.push({ kind: "tool_group", messages: groupMsgs, calls, ok: allOk, isWorklist: isWl, notedUpdate: noted });
      i = j;
      continue;
    }
    if (msg.role === "assistant") {
      groups.push({ kind: "final_assistant", messages: [msg], calls: [], ok: true, isWorklist: false, notedUpdate: false });
    } else {
      groups.push({ kind: "user", messages: [msg], calls: [], ok: true, isWorklist: false, notedUpdate: false });
    }
    i++;
  }
  return groups;
}

function clearAssistantReasoning(msg: OpenAIMessage): OpenAIMessage {
  const copy = { ...msg };
  const hadReasoning = copy.reasoning_content !== undefined || copy.reasoning !== undefined;
  delete copy.reasoning;
  // Preserve empty `reasoning_content: ""` on tool-call assistant turns so strict reasoning APIs
  // (e.g. DeepSeek thinking mode) accept the message while `renderPrompt` emits `<think></think>`.
  if (Array.isArray(copy.tool_calls) && copy.tool_calls.length > 0 && hadReasoning) {
    copy.reasoning_content = "";
  } else {
    delete copy.reasoning_content;
  }
  return copy;
}

function stripGroupReasoning(group: TurnGroup): OpenAIMessage[] {
  return group.messages.map((msg, idx) => (idx === 0 && msg.role === "assistant" ? clearAssistantReasoning(msg) : msg));
}

function summarizeGroup(group: TurnGroup): string[] {
  const toolResults = group.messages.filter((m) => m.role === "tool");
  return group.calls.map((call, idx) => {
    const rawTarget = call.args.path ?? call.args.file_path ?? call.args.pattern ?? call.args.command ?? call.args.query;
    let target = "";
    if (typeof rawTarget === "string" && rawTarget.trim() !== "") {
      const clean = rawTarget.trim().replace(/\s+/g, " ");
      target = clean.length > 80 ? `${clean.slice(0, 77)}...` : clean;
    }
    const resultMsg = toolResults[idx];
    const resultText = resultMsg ? messageText(resultMsg.content) : "";
    const failed = !group.ok || (resultMsg && (resultMsg.isError === true || isErrorResultText(resultText)));
    let outcome = failed ? "error" : "ok";
    if (!failed && call.name === "bash") {
      const exitMatch = /(?:exit code[:\s]+|exited with code )(\d+)/i.exec(resultText);
      outcome = exitMatch ? `exit ${exitMatch[1]}` : "exit 0";
    }
    return `- ${call.name}${target ? ` ${target}` : ""} -> ${outcome}`;
  });
}

function formatFoldedLedger(groups: readonly TurnGroup[], header: string): OpenAIMessage {
  const lines = groups.flatMap(summarizeGroup);
  return { role: "user", content: `[${header}:\n${lines.join("\n")}]` };
}

function compactPriorToolGroup(group: TurnGroup): OpenAIMessage[] {
  return group.messages.map((msg, idx) => {
    if (idx === 0 && msg.role === "assistant") {
      return clearAssistantReasoning(msg);
    }
    if (msg.role === "tool") {
      const content = messageText(msg.content);
      if (content.length <= MAX_PRIOR_TURN_FALLBACK_RESULT_CHARS) return msg;
      const trimmed = content.trim();
      if (trimmed.startsWith("{") || trimmed.startsWith("[")) return msg;
      return {
        ...msg,
        content: `${content.slice(0, 1100)}\n...(prior-turn output folded)...\n${content.slice(content.length - 400)}`,
      };
    }
    return msg;
  });
}

function foldCompletedOpenAITurn(
  slice: readonly OpenAIMessage[],
  maxResultTokens = DEFAULT_WORKLIST_MAX_RESULT_TOKENS,
  count: (text: string) => number = estimateRwkvTokens,
): OpenAIMessage[] {
  const groups = parseTurnGroups(slice, maxResultTokens, count);
  const toolGroups = groups.filter((g) => g.kind === "tool_group");
  const lastWorklist = [...toolGroups].reverse().find((g) => g.ok && g.isWorklist);
  const fallbackWs = lastWorklist === undefined
    ? ([...toolGroups].reverse().find((g) => g.ok) ?? toolGroups.at(-1))
    : undefined;
  const earlierUnplanned = lastWorklist === undefined && fallbackWs !== undefined
    ? toolGroups.filter((g) => g !== fallbackWs)
    : [];
  const trailingAfterWorklist = lastWorklist !== undefined
    ? toolGroups.slice(toolGroups.indexOf(lastWorklist) + 1).filter((g) => !g.isWorklist)
    : [];
  const hasFinal = groups.some((g) => g.kind === "final_assistant");

  const out: OpenAIMessage[] = [];
  let insertedLedger = false;
  for (const group of groups) {
    if (group.kind === "user") {
      out.push(...group.messages);
      continue;
    }
    if (group.kind === "final_assistant") {
      out.push(...stripGroupReasoning(group));
      continue;
    }
    if (lastWorklist !== undefined && group === lastWorklist) {
      out.push(...stripGroupReasoning(group));
      if (trailingAfterWorklist.length > 0) {
        out.push(formatFoldedLedger(trailingAfterWorklist, "Workspace steps after last todo update in this turn"));
      }
    } else if (fallbackWs !== undefined && group === fallbackWs) {
      if (!insertedLedger && earlierUnplanned.length > 0) {
        out.push(formatFoldedLedger(earlierUnplanned, "Earlier workspace steps in this turn (raw outputs folded)"));
        insertedLedger = true;
      }
      out.push(...(hasFinal ? compactPriorToolGroup(group) : group.messages));
    }
  }
  return out;
}

const READ_NAMES = new Set(["read", "read_file"]);
const WRITE_NAMES = new Set(["edit", "edit_file", "write", "write_file"]);

function groupPrimaryPath(group: TurnGroup): string | undefined {
  const first = group.calls[0];
  if (!first) return undefined;
  const raw = first.args.path ?? first.args.file_path;
  return typeof raw === "string" && raw.trim() !== "" ? raw.trim().replace(/^\.\//, "") : undefined;
}

function findLatestUneditedReadGroup(
  preWs: readonly TurnGroup[],
  groupTokens: ReadonlyMap<TurnGroup, number>,
): TurnGroup | undefined {
  for (let i = preWs.length - 1; i >= 0; i--) {
    const group = preWs[i]!;
    if (group.ok && group.calls.some((c) => WRITE_NAMES.has(c.name))) {
      return undefined;
    }
    const path = groupPrimaryPath(group);
    if (group.ok && path && group.calls.length === 1 && READ_NAMES.has(group.calls[0]!.name)) {
      return (groupTokens.get(group) ?? 0) <= MAX_PINNED_UNEDITED_READ_TOKENS ? group : undefined;
    }
  }
  return undefined;
}

export interface FoldOpenAIWorklistOptions {
  readonly budgetTokens?: number;
  readonly hardCapTokens?: number;
  readonly backboneMaxTokens?: number;
  readonly keepRecentPairs?: number;
  readonly foldTriggerTokens?: number;
  readonly maxResultTokens?: number;
  readonly keepInitialPlan?: boolean;
  readonly systemGuidance?: boolean;
  readonly countTokens?: (text: string) => number;
}

export interface FoldOpenAIWorklistResult {
  readonly request: Record<string, unknown>;
  readonly promptTokens: number;
  readonly folded: boolean;
  readonly nudged: boolean;
}

type NudgeKind = "init" | "checkpoint";

interface TriggeredGroupNudge {
  readonly toolGroupIndex: number;
  readonly kind: NudgeKind;
}

export function foldWorklistRequest(
  request: Record<string, unknown>,
  options: FoldOpenAIWorklistOptions = {},
): FoldOpenAIWorklistResult {
  const count = options.countTokens ?? estimateRwkvTokens;
  const rawMessages = Array.isArray(request.messages) ? (request.messages as OpenAIMessage[]) : [];
  const rawTools = Array.isArray(request.tools) ? request.tools : [];
  const budgetTokens = options.budgetTokens ?? DEFAULT_WORKLIST_BUDGET_TOKENS;
  const hardCapTokens = options.hardCapTokens ?? (budgetTokens + DEFAULT_WORKLIST_HARD_SLACK_TOKENS);
  const backboneMaxTokens = options.backboneMaxTokens ?? DEFAULT_BACKBONE_MAX_TOKENS;
  const keepRecentPairs = Math.max(0, options.keepRecentPairs ?? DEFAULT_KEEP_RECENT_PAIRS);
  const foldTriggerTokens = Math.max(0, options.foldTriggerTokens ?? DEFAULT_FOLD_TRIGGER_TOKENS);
  const maxResultTokens = options.maxResultTokens ?? DEFAULT_WORKLIST_MAX_RESULT_TOKENS;
  const keepInitialPlan = options.keepInitialPlan !== false;

  const hasTodoPlanTool = rawTools.some((t) => {
    if (!object(t)) return false;
    const fn = object(t.function) ? t.function : t;
    return fn.name === TODO_PLAN_TOOL;
  });

  const systemMessages: OpenAIMessage[] = [];
  const conversation: OpenAIMessage[] = [];
  for (const msg of rawMessages) {
    if (msg.role === "system") systemMessages.push({ ...msg });
    else conversation.push(msg);
  }
  if (hasTodoPlanTool && options.systemGuidance !== false) {
    if (systemMessages.length > 0) {
      const lastSys = systemMessages[systemMessages.length - 1]!;
      const sysText = messageText(lastSys.content);
      if (!sysText.includes(WORKLIST_SYSTEM_GUIDANCE)) {
        lastSys.content = sysText ? `${sysText}\n\n${WORKLIST_SYSTEM_GUIDANCE}` : WORKLIST_SYSTEM_GUIDANCE;
      }
    } else {
      systemMessages.push({ role: "system", content: WORKLIST_SYSTEM_GUIDANCE });
    }
  }

  const toolsHeaderTokens = rawTools.length > 0 ? count(`Tools:\n${JSON.stringify(rawTools)}\n`) + 45 : 0;
  const systemTokens = systemMessages.reduce((sum, m) => sum + count(renderSingleMessageEstimate(m)), 0) + toolsHeaderTokens;

  const userIndices: number[] = [];
  for (let i = 0; i < conversation.length; i++) {
    if (conversation[i]!.role === "user") userIndices.push(i);
  }
  if (userIndices.length === 0) {
    const tok = systemTokens + conversation.reduce((s, m) => s + count(renderSingleMessageEstimate(m)), 0);
    return { request: { ...request, messages: [...systemMessages, ...conversation] }, promptTokens: tok, folded: false, nudged: false };
  }

  const priorTurns: OpenAIMessage[][] = [];
  for (let t = 0; t < userIndices.length - 1; t++) {
    const from = t === 0 ? 0 : userIndices[t]!;
    const to = userIndices[t + 1]!;
    priorTurns.push(foldCompletedOpenAITurn(conversation.slice(from, to), maxResultTokens, count));
  }
  const turnTokens = (msgs: readonly OpenAIMessage[]) => msgs.reduce((s, m) => s + count(renderSingleMessageEstimate(m)), 0);
  let priorTokens = priorTurns.reduce((s, turn) => s + turnTokens(turn), 0);
  while (priorTurns.length > 0 && priorTokens > backboneMaxTokens) {
    const dropped = priorTurns.shift()!;
    priorTokens -= turnTokens(dropped);
  }
  const priorFlat = priorTurns.flat();

  const activeStart = userIndices.length === 1 ? 0 : userIndices[userIndices.length - 1]!;
  const activeGroups = parseTurnGroups(conversation.slice(activeStart), maxResultTokens, count);
  const toolGroups = activeGroups.filter((g) => g.kind === "tool_group");

  const fullGroupTokens = new Map<TurnGroup, number>();
  const bareGroupTokens = new Map<TurnGroup, number>();
  for (const g of activeGroups) {
    fullGroupTokens.set(g, turnTokens(g.messages));
    bareGroupTokens.set(g, g.kind === "tool_group" ? turnTokens(stripGroupReasoning(g)) : fullGroupTokens.get(g)!);
  }

  const nudgeTok = {
    init: count(`User: ${WORKLIST_INIT_NUDGE}\n\n`),
    checkpoint: count(`User: ${WORKLIST_CHECKPOINT_NUDGE}\n\n`),
  };

  const buildWindow = (
    subToolGroupsLen: number,
    includeTailFinal: boolean,
    nudges: readonly TriggeredGroupNudge[],
  ): { messages: OpenAIMessage[]; tokens: number; folded: boolean; liveNudge: boolean } => {
    const allowedToolGroups = new Set(toolGroups.slice(0, subToolGroupsLen));
    const subGroups = activeGroups.filter((g) => {
      if (g.kind === "user") return true;
      if (g.kind === "final_assistant") return includeTailFinal;
      return allowedToolGroups.has(g);
    });
    const subTools = toolGroups.slice(0, subToolGroupsLen);
    // Checkpoint-edge-triggered folding: decide at each noted `todo_update` step
    // whether total prompt tokens have reached `foldTriggerTokens`, so folding
    // never fires mid-item on an ordinary `read`/`edit` step!
    let activeCheckpoint: TurnGroup | undefined;
    for (let idx = 0; idx < subTools.length; idx++) {
      const candidate = subTools[idx]!;
      if (!candidate.notedUpdate) continue;
      if (foldTriggerTokens === 0) {
        activeCheckpoint = candidate;
        continue;
      }
      const cpSubIdx = subGroups.indexOf(candidate);
      let tokensAtCandidate = systemTokens + priorTokens;
      const prevCpIdx = activeCheckpoint ? subGroups.indexOf(activeCheckpoint) : -1;
      for (let gIdx = 0; gIdx <= cpSubIdx; gIdx++) {
        const g = subGroups[gIdx]!;
        if (prevCpIdx < 0 || gIdx >= prevCpIdx || g.kind === "user") {
          tokensAtCandidate += fullGroupTokens.get(g) ?? 0;
        }
      }
      if (tokensAtCandidate >= foldTriggerTokens) {
        activeCheckpoint = candidate;
      }
    }
    const lastCheckpoint = activeCheckpoint;
    const firstPlan = subTools.find((g) => g.ok && g.calls.some((c) => c.name === TODO_PLAN_TOOL));
    const unclosedWorklist = lastCheckpoint === undefined ? [...subTools].reverse().find((g) => g.ok && g.isWorklist) : undefined;

    let keptGroups: TurnGroup[];
    let didFold = priorTurns.length > 0;

    if (lastCheckpoint !== undefined) {
      didFold = true;
      const checkpointIdxInSub = subGroups.indexOf(lastCheckpoint);
      const preWs = subTools.filter((g) => subGroups.indexOf(g) < checkpointIdxInSub && !g.isWorklist);
      const keptTail: TurnGroup[] = [];
      if (keepRecentPairs >= 1 && preWs.length >= 1) {
        const lastWs = preWs[preWs.length - 1]!;
        keptTail.unshift(lastWs);
        if (keepRecentPairs >= 2 && preWs.length >= 2) {
          const prevWs = preWs[preWs.length - 2]!;
          const pairSum = (fullGroupTokens.get(prevWs) ?? 0) + (fullGroupTokens.get(lastWs) ?? 0);
          if (pairSum <= MAX_SECOND_RECENT_PAIR_TOKENS) keptTail.unshift(prevWs);
        }
      }
      const keepPre = new Set<TurnGroup>(keptTail);
      if (keepInitialPlan && firstPlan && subGroups.indexOf(firstPlan) < checkpointIdxInSub) {
        keepPre.add(firstPlan);
      }
      if (keepRecentPairs >= 1 && !keptTail.some((g) => g.ok && g.calls.some((c) => READ_NAMES.has(c.name)))) {
        const pinnedRead = findLatestUneditedReadGroup(preWs, fullGroupTokens);
        if (pinnedRead) keepPre.add(pinnedRead);
      }
      keptGroups = subGroups.filter((g, idx) => idx >= checkpointIdxInSub || g.kind === "user" || keepPre.has(g));
    } else {
      keptGroups = [...subGroups];
    }

    const keptSet = new Set<TurnGroup>(keptGroups);
    const clearedReasoning = new Set<TurnGroup>();
    if (firstPlan && keptSet.has(firstPlan)) {
      const planIdx = subTools.indexOf(firstPlan);
      if (planIdx >= 0 && planIdx + 1 < subTools.length && !keptSet.has(subTools[planIdx + 1]!)) {
        clearedReasoning.add(firstPlan);
      }
    }

    const protectedGroups = new Set<TurnGroup>();
    if (lastCheckpoint) protectedGroups.add(lastCheckpoint);
    if (unclosedWorklist) protectedGroups.add(unclosedWorklist);
    if (firstPlan) protectedGroups.add(firstPlan);

    const droppedByHardCap: TurnGroup[] = [];

    const assemble = (): { messages: OpenAIMessage[]; tokens: number; liveNudge: boolean } => {
      const out: OpenAIMessage[] = [];
      let tok = systemTokens + priorTokens;
      let ledgerInserted = false;
      let liveNudge = false;
      const nudgeByGroup = new Map<TurnGroup, NudgeKind>();
      for (const n of nudges) {
        const g = subTools[n.toolGroupIndex];
        if (g && keptSet.has(g)) nudgeByGroup.set(g, n.kind);
      }

      for (const g of keptGroups) {
        if (!ledgerInserted && droppedByHardCap.length > 0 && g.kind !== "user") {
          const ledgerMsg = formatFoldedLedger(droppedByHardCap, "Earlier workspace steps in this turn (raw outputs folded)");
          out.push(ledgerMsg);
          tok += count(renderSingleMessageEstimate(ledgerMsg));
          ledgerInserted = true;
        }
        if (g.kind === "tool_group" && clearedReasoning.has(g)) {
          out.push(...stripGroupReasoning(g));
          tok += bareGroupTokens.get(g) ?? 0;
        } else {
          out.push(...g.messages);
          tok += fullGroupTokens.get(g) ?? 0;
        }
        const postKind = nudgeByGroup.get(g);
        if (postKind !== undefined) {
          const text = postKind === "checkpoint" ? WORKLIST_CHECKPOINT_NUDGE : WORKLIST_INIT_NUDGE;
          out.push({ role: "user", content: text });
          tok += nudgeTok[postKind];
          if (g === subTools[subTools.length - 1] && !includeTailFinal) liveNudge = true;
        }
      }
      return { messages: out, tokens: tok, liveNudge };
    };

    let built = assemble();

    // Phase 1 hardCap: strip bloated reasoning on older unfolded tool groups before dropping source reads.
    if (built.tokens > hardCapTokens) {
      didFold = true;
      for (const g of keptGroups) {
        if (built.tokens <= hardCapTokens) break;
        if (g.kind === "tool_group" && !clearedReasoning.has(g)) {
          const hasReasoning = Boolean(messageText(g.messages[0]?.reasoning_content ?? g.messages[0]?.reasoning));
          if (hasReasoning) {
            clearedReasoning.add(g);
            built = assemble();
          }
        }
      }
    }

    // Phase 2 hardCap: fold oldest non-protected workspace tool groups into a compact ledger.
    const droppable = subTools.filter((g) => keptSet.has(g) && !protectedGroups.has(g));
    while (droppable.length > 1 && built.tokens > hardCapTokens) {
      didFold = true;
      const oldest = droppable.shift()!;
      droppedByHardCap.push(oldest);
      keptSet.delete(oldest);
      keptGroups = keptGroups.filter((g) => g !== oldest);
      built = assemble();
    }

    return { messages: built.messages, tokens: built.tokens, folded: didFold, liveNudge: built.liveNudge };
  };

  const triggeredNudges: TriggeredGroupNudge[] = [];
  for (let k = 0; k < toolGroups.length; k++) {
    const winAtK = buildWindow(k + 1, false, triggeredNudges);
    const subTools = toolGroups.slice(0, k + 1);
    const lastCp = [...subTools].reverse().find((g) => g.notedUpdate);
    const cpIdx = lastCp ? subTools.indexOf(lastCp) : -1;
    const unfoldedWs = subTools.slice(cpIdx + 1).filter((g) => !g.isWorklist).length;
    const hasPlan = subTools.some((g) => g.ok && g.isWorklist);
    const kind: NudgeKind = hasPlan ? "checkpoint" : "init";
    const cooldown = kind === "init" ? INIT_NUDGE_COOLDOWN_STEPS : CHECKPOINT_NUDGE_COOLDOWN_STEPS;
    const prevSame = [...triggeredNudges].reverse().find((n) => n.toolGroupIndex > cpIdx && n.kind === kind);
    const onCooldown = prevSame !== undefined && (k - prevSame.toolGroupIndex) < cooldown;
    if (!onCooldown && unfoldedWs >= 1 && winAtK.tokens > budgetTokens) {
      triggeredNudges.push({ toolGroupIndex: k, kind });
    }
  }

  const hasFinalAssistant = activeGroups.some((g) => g.kind === "final_assistant");
  const finalWin = buildWindow(toolGroups.length, hasFinalAssistant, triggeredNudges);
  return {
    request: {
      ...request,
      messages: [...systemMessages, ...priorFlat, ...finalWin.messages],
    },
    promptTokens: finalWin.tokens,
    folded: finalWin.folded,
    nudged: finalWin.liveNudge,
  };
}

/** Replays worklist state for the active user turn from Pi branch entries (or falls back to per-session memory). */
export class PiWorklistState {
  private turn = 0;
  private items: WorklistItem[] = [];
  private completedSinceWorkspaceStep = false;

  private hydrateFromBranch(branch: readonly Record<string, any>[] | undefined): {
    turn: number;
    items: WorklistItem[];
    backToBackComplete: boolean;
  } | undefined {
    if (!Array.isArray(branch)) return undefined;
    const msgs = branch.flatMap((entry) => (entry && entry.type === "message" && object(entry.message) ? [entry.message] : []));
    let turn = 0;
    let lastUserIdx = -1;
    for (let i = 0; i < msgs.length; i++) {
      if (msgs[i]!.role === "user") {
        turn++;
        lastUserIdx = i;
      }
    }
    if (turn === 0) return { turn: 1, items: [], backToBackComplete: false };
    let items: WorklistItem[] = [];
    let completedSinceWs = false;
    const callsById = new Map<string, { name: string; args: Record<string, unknown> }>();
    for (let i = lastUserIdx + 1; i < msgs.length; i++) {
      const m = msgs[i]!;
      if (m.role === "assistant" && Array.isArray(m.content)) {
        for (const block of m.content) {
          if (object(block) && block.type === "toolCall" && typeof block.id === "string" && typeof block.name === "string") {
            callsById.set(block.id, { name: block.name, args: object(block.arguments) ? block.arguments : {} });
          }
        }
      } else if (m.role === "toolResult" && typeof m.toolCallId === "string") {
        const call = callsById.get(m.toolCallId);
        if (!call) continue;
        if (!isWorklistTool(call.name)) {
          completedSinceWs = false;
          continue;
        }
        if (m.isError) continue;
        if (call.name === TODO_PLAN_TOOL) {
          const next = applyTodoPlan(items, call.args.items ?? call.args.todos ?? call.args.steps ?? call.args.plan);
          if (Array.isArray(next)) items = next;
        } else if (call.name === TODO_UPDATE_TOOL) {
          const statusNorm = normalizeWorklistStatus(call.args.status);
          const next = applyTodoUpdate(items, {
            id: call.args.id ?? call.args.index ?? call.args.step,
            status: call.args.status,
            note: call.args.note ?? call.args.notes ?? call.args.summary ?? call.args.findings,
            content: call.args.content ?? call.args.title,
          });
          if (typeof next !== "string") {
            items = next.items;
            if (statusNorm === "completed") completedSinceWs = true;
          }
        }
      }
    }
    return { turn, items, backToBackComplete: completedSinceWs };
  }

  recordWorkspaceTool(): void {
    this.completedSinceWorkspaceStep = false;
  }

  plan(rawItems: unknown, branch?: readonly Record<string, any>[]): string {
    const hydrated = this.hydrateFromBranch(branch);
    if (hydrated && hydrated.turn !== this.turn) {
      this.turn = hydrated.turn;
      this.items = hydrated.items;
      this.completedSinceWorkspaceStep = hydrated.backToBackComplete;
    } else if (hydrated && this.items.length === 0 && hydrated.items.length > 0) {
      this.items = hydrated.items;
      this.completedSinceWorkspaceStep = hydrated.backToBackComplete;
    }
    const updated = applyTodoPlan(this.items, rawItems);
    if (typeof updated === "string") throw new Error(updated);
    this.items = updated;
    return formatWorklist(updated);
  }

  update(input: TodoUpdateInput, branch?: readonly Record<string, any>[]): string {
    const hydrated = this.hydrateFromBranch(branch);
    const backToBack = hydrated ? hydrated.backToBackComplete : this.completedSinceWorkspaceStep;
    if (hydrated && hydrated.turn !== this.turn) {
      this.turn = hydrated.turn;
      this.items = hydrated.items;
    } else if (hydrated && this.items.length === 0 && hydrated.items.length > 0) {
      this.items = hydrated.items;
    }
    const statusNorm = normalizeWorklistStatus(input.status);
    if (statusNorm === "completed" && backToBack) {
      throw new Error(
        "Cannot mark another item completed immediately after a previous completion with no workspace tool call in between. Execute or verify this item with a workspace tool first (read, edit, write, or bash), or use status=\"in_progress\" / \"dropped\".",
      );
    }
    const updated = applyTodoUpdate(this.items, input);
    if (typeof updated === "string") throw new Error(updated);
    this.items = updated.items;
    if (statusNorm === "completed") this.completedSinceWorkspaceStep = true;
    return formatWorklist(updated.items);
  }
}

export function createWorklistTools(state = new PiWorklistState()): ToolDefinition[] {
  const planTool: ToolDefinition = {
    name: TODO_PLAN_TOOL,
    label: "Plan work list",
    description: TODO_PLAN_DESCRIPTION,
    promptSnippet: "Create or revise the ordered multi-step work list before editing",
    parameters: Type.Object({
      items: Type.Array(Type.String(), {
        description: "Ordered list of concrete subgoals (2-10 short action lines, without numbers). Mid-turn calls keep completed/dropped items and their notes and replace only unfinished items.",
      }),
    }),
    async execute(_id: string, params: { items: string[] }, _signal: unknown, _update: unknown, ctx: unknown) {
      const branch = (ctx as { sessionManager?: { getBranch?: () => Record<string, any>[] } } | undefined)?.sessionManager?.getBranch?.();
      const text = state.plan(params.items, branch);
      return { content: [{ type: "text" as const, text }], details: { summary: text } };
    },
  } as unknown as ToolDefinition;

  const updateTool: ToolDefinition = {
    name: TODO_UPDATE_TOOL,
    label: "Update work list item",
    description: TODO_UPDATE_DESCRIPTION,
    promptSnippet: "Mark a work list item in_progress/completed/dropped with verified notes",
    parameters: Type.Object({
      id: Type.Integer({ description: "1-based item number from the current todo list." }),
      status: Type.Union([Type.Literal("in_progress"), Type.Literal("completed"), Type.Literal("dropped")], {
        description: "in_progress | completed | dropped. Marking completed or dropped automatically sets the next pending item to in_progress.",
      }),
      note: Type.Optional(Type.String({
        description: "Concrete findings, exact file paths/lines, applied edits, or test outcomes (<= 120 words). Required when status is completed or dropped; also used on in_progress to checkpoint progress.",
      })),
      content: Type.Optional(Type.String({
        description: "Optional refined title for an unfinished item.",
      })),
    }),
    async execute(_id: string, params: { id: number; status: string; note?: string; content?: string }, _signal: unknown, _update: unknown, ctx: unknown) {
      const branch = (ctx as { sessionManager?: { getBranch?: () => Record<string, any>[] } } | undefined)?.sessionManager?.getBranch?.();
      const text = state.update(params, branch);
      return { content: [{ type: "text" as const, text }], details: { summary: text } };
    },
  } as unknown as ToolDefinition;

  return [planTool, updateTool];
}
