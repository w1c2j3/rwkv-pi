import { object } from "./tasks.js";
import { estimateRwkvTokens } from "./worklist.js";

export const DEFAULT_EPOCH_BUDGET_TOKENS = 16384;
export const DEFAULT_EPOCH_HARD_SLACK_TOKENS = 4096;
export const DEFAULT_EPOCH_BACKBONE_MAX_TOKENS = 6144;
export const DEFAULT_EPOCH_KEEP_RECENT_PAIRS = 3;
export const DEFAULT_EPOCH_MAX_RESULT_TOKENS = 4000;

const MAX_PINNED_UNEDITED_READ_TOKENS = 2400;
const MAX_PRIOR_TURN_LAST_RESULT_CHARS = 1800;

const READ_NAMES = new Set(["read", "read_file"]);
const SEARCH_NAMES = new Set(["grep", "find", "ls"]);
const WRITE_NAMES = new Set(["edit", "edit_file", "write", "write_file"]);

type OpenAIMessage = Record<string, any>;

interface ToolCallEntry {
  readonly id: string;
  readonly name: string;
  readonly args: Record<string, unknown>;
}

interface EpochGroup {
  readonly kind: "user" | "final_assistant" | "tool_group";
  readonly messages: OpenAIMessage[];
  readonly calls: ToolCallEntry[];
  readonly ok: boolean;
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

function isErrorResultText(text: string): boolean {
  const trimmed = text.trim();
  return /^(?:Error:|RWKV_|Tool .* failed|invalid |unknown |ENOENT)/i.test(trimmed)
    || /Command exited with code [1-9]\d*/i.test(trimmed);
}

function cleanOneLine(text: string, maxLen = 96): string {
  const clean = text.trim().replace(/\s+/g, " ");
  return clean.length > maxLen ? `${clean.slice(0, maxLen - 3)}...` : clean;
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
    const calls = extractToolCalls(msg)
      .map((c) => `<tool_call>${JSON.stringify({ name: c.name, arguments: c.args })}</tool_call>`)
      .join("\n");
    return `Assistant: <think>${reasoning}</think>\n${[body, calls].filter(Boolean).join("\n")}\n\n`;
  }
  return "";
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
  const slimmed = `${raw.slice(0, headChars)}\n\n[... ${omitted} characters omitted to preserve 16k context window; use read with offset/limit or grep for specific lines ...]\n\n${raw.slice(raw.length - tailChars)}`;
  return { ...msg, content: slimmed };
}

function parseEpochGroups(
  slice: readonly OpenAIMessage[],
  maxResultTokens: number,
  count: (text: string) => number,
): EpochGroup[] {
  const groups: EpochGroup[] = [];
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
      groups.push({ kind: "tool_group", messages: groupMsgs, calls, ok: allOk });
      i = j;
      continue;
    }
    if (msg.role === "assistant") {
      groups.push({ kind: "final_assistant", messages: [msg], calls: [], ok: true });
    } else {
      groups.push({ kind: "user", messages: [msg], calls: [], ok: true });
    }
    i++;
  }
  return groups;
}

function stripGroupReasoning(group: EpochGroup): OpenAIMessage[] {
  return group.messages.map((msg, idx) => {
    if (idx !== 0 || msg.role !== "assistant") return msg;
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
  });
}

function appendLedgerToUserMessage(msg: OpenAIMessage, ledgerText: string): OpenAIMessage {
  const existing = messageText(msg.content).trimEnd();
  return { ...msg, role: "user", content: existing ? `${existing}\n\n${ledgerText}` : ledgerText };
}

function pathOfArgs(args: Record<string, unknown>): string | undefined {
  const raw = args.path ?? args.file_path;
  return typeof raw === "string" && raw.trim() !== "" ? raw.trim().replace(/^\.\//, "") : undefined;
}

/**
 * Deterministic Ground-Truth State Ledger extracted from folded `(assistant + tool)` groups.
 */
export function buildOpenAIEpochStateLedger(
  groups: readonly EpochGroup[],
  header = "State Ledger (earlier steps folded)",
): OpenAIMessage {
  const modifiedByPath = new Map<string, string>();
  const succeededKeys = new Set<string>();
  const inspected: string[] = [];
  const inspectedSeen = new Set<string>();
  const failed: string[] = [];
  let latestBash: string | undefined;
  let bashRuns = 0;

  for (const group of groups) {
    const toolResults = group.messages.filter((m) => m.role === "tool");
    for (let idx = 0; idx < group.calls.length; idx++) {
      const call = group.calls[idx]!;
      const resMsg = toolResults[idx];
      const resText = resMsg ? messageText(resMsg.content) : "";
      const callFailed = !group.ok || Boolean(resMsg && (resMsg.isError === true || isErrorResultText(resText)));
      const filePath = pathOfArgs(call.args);

      if (call.name === "bash") {
        bashRuns++;
        const cmd = cleanOneLine(String(call.args.command ?? ""), 80);
        const exitMatch = /(?:exit code[:\s]+|exited with code )(\d+)/i.exec(resText);
        const exitCode = exitMatch ? exitMatch[1]! : (callFailed ? "1" : "0");
        const bodyLines = resText
          .split("\n")
          .map((l) => l.trim())
          .filter((l) => l !== "" && !/^(?:\[?exit code|Command exited with code)/i.test(l));
        const tailSummary = bodyLines.length > 0 ? ` (${cleanOneLine(bodyLines[bodyLines.length - 1]!, 80)})` : "";
        latestBash = `${cmd} -> exit ${exitCode}${tailSummary}`;
      }

      if (callFailed) {
        const target = filePath ?? cleanOneLine(String(call.args.command ?? call.args.pattern ?? ""), 72);
        const firstErr = cleanOneLine(resText.split("\n").find((l) => l.trim() !== "") ?? "failed", 96);
        if (call.name !== "bash") failed.push(`${call.name}${target ? ` ${target}` : ""} -> ${firstErr}`);
        continue;
      }

      if (filePath !== undefined) succeededKeys.add(`${call.name}:${filePath}`);

      if (WRITE_NAMES.has(call.name) && filePath !== undefined) {
        modifiedByPath.set(filePath, `${filePath} (${call.name}: ok)`);
        continue;
      }

      if (READ_NAMES.has(call.name) && filePath !== undefined) {
        const offset = call.args.offset;
        const limit = call.args.limit;
        const range = typeof offset === "number"
          ? ` (L${offset}${typeof limit === "number" ? `-${offset + limit - 1}` : "+"})`
          : "";
        const entry = `read ${filePath}${range}`;
        if (!inspectedSeen.has(entry)) {
          inspectedSeen.add(entry);
          inspected.push(entry);
        }
        continue;
      }

      if (SEARCH_NAMES.has(call.name)) {
        const q = cleanOneLine(String(call.args.pattern ?? filePath ?? "."), 64);
        const entry = `${call.name} ${JSON.stringify(q)}`;
        if (!inspectedSeen.has(entry)) {
          inspectedSeen.add(entry);
          inspected.push(entry);
        }
        continue;
      }
    }
  }

  const unresolvedFailed = failed.filter((line) => {
    for (const key of succeededKeys) {
      const [toolName, path] = key.split(":", 2);
      if (toolName && path && line.startsWith(`${toolName} ${path} ->`)) return false;
    }
    return true;
  });

  const sections: string[] = [`[${header}:`];
  if (modifiedByPath.size > 0) {
    sections.push(`- Modified files: ${[...modifiedByPath.values()].join("; ")}`);
  }
  if (inspected.length > 0) {
    const shown = inspected.slice(-12);
    const omitted = inspected.length - shown.length;
    sections.push(`- Inspected: ${shown.join(", ")}${omitted > 0 ? ` (+${omitted} earlier)` : ""}`);
  }
  if (latestBash !== undefined) {
    sections.push(`- Latest command (${bashRuns} run): ${latestBash}`);
  }
  if (unresolvedFailed.length > 0) {
    sections.push(`- Failed attempts (do not repeat verbatim): ${unresolvedFailed.slice(-5).join("; ")}`);
  }
  sections[sections.length - 1] = `${sections[sections.length - 1]!}]`;
  return { role: "user", content: sections.join("\n") };
}

function findLatestUneditedReadGroup(
  foldedGroups: readonly EpochGroup[],
  groupTokens: ReadonlyMap<EpochGroup, number>,
): EpochGroup | undefined {
  for (let i = foldedGroups.length - 1; i >= 0; i--) {
    const g = foldedGroups[i]!;
    if (g.ok && g.calls.some((c) => WRITE_NAMES.has(c.name))) return undefined;
    const first = g.calls[0];
    if (g.ok && g.calls.length === 1 && first && READ_NAMES.has(first.name) && pathOfArgs(first.args) !== undefined) {
      return (groupTokens.get(g) ?? 0) <= MAX_PINNED_UNEDITED_READ_TOKENS ? g : undefined;
    }
  }
  return undefined;
}

function compactPriorLastToolGroup(group: EpochGroup): OpenAIMessage[] {
  return stripGroupReasoning(group).map((msg) => {
    if (msg.role !== "tool") return msg;
    const raw = messageText(msg.content);
    if (raw.length <= MAX_PRIOR_TURN_LAST_RESULT_CHARS) return msg;
    const trimmed = raw.trim();
    if (trimmed.startsWith("{") || trimmed.startsWith("[")) return msg;
    return {
      ...msg,
      content: `${raw.slice(0, 1200)}\n...(prior-turn output folded)...\n${raw.slice(raw.length - 450)}`,
    };
  });
}

function foldCompletedPriorOpenAITurn(
  slice: readonly OpenAIMessage[],
  maxResultTokens: number,
  count: (text: string) => number,
): OpenAIMessage[] {
  const groups = parseEpochGroups(slice, maxResultTokens, count);
  const toolGroups = groups.filter((g) => g.kind === "tool_group");
  if (toolGroups.length === 0) {
    return groups.flatMap((g) => (g.kind === "final_assistant" ? stripGroupReasoning(g) : g.messages));
  }
  const lastTool = [...toolGroups].reverse().find((g) => g.ok) ?? toolGroups.at(-1)!;
  const earlierTools = toolGroups.filter((g) => g !== lastTool);
  const hasFinal = groups.some((g) => g.kind === "final_assistant");

  const out: OpenAIMessage[] = [];
  const priorLedgerText = earlierTools.length > 0
    ? messageText(buildOpenAIEpochStateLedger(earlierTools, "Prior-turn State Ledger").content)
    : undefined;
  let ledgerMerged = false;
  for (const g of groups) {
    if (g.kind === "user") {
      if (!ledgerMerged && priorLedgerText && g.messages.length > 0) {
        out.push(appendLedgerToUserMessage(g.messages[0]!, priorLedgerText), ...g.messages.slice(1));
        ledgerMerged = true;
      } else {
        out.push(...g.messages);
      }
    } else if (g.kind === "final_assistant") {
      out.push(...stripGroupReasoning(g));
    } else if (g === lastTool) {
      if (!ledgerMerged && priorLedgerText) {
        out.push({ role: "user", content: priorLedgerText });
        ledgerMerged = true;
      }
      out.push(...(hasFinal ? compactPriorLastToolGroup(g) : stripGroupReasoning(g)));
    }
  }
  return out;
}

export interface FoldOpenAIEpochOptions {
  readonly budgetTokens?: number;
  readonly hardCapTokens?: number;
  readonly backboneMaxTokens?: number;
  readonly keepRecentPairs?: number;
  readonly maxResultTokens?: number;
  readonly countTokens?: (text: string) => number;
}

export interface FoldOpenAIEpochResult {
  readonly request: Record<string, unknown>;
  readonly promptTokens: number;
  readonly epochCount: number;
  readonly folded: boolean;
}

export function foldEpochRequest(
  request: Record<string, unknown>,
  options: FoldOpenAIEpochOptions = {},
): FoldOpenAIEpochResult {
  const count = options.countTokens ?? estimateRwkvTokens;
  const rawMessages = Array.isArray(request.messages) ? (request.messages as OpenAIMessage[]) : [];
  const rawTools = Array.isArray(request.tools) ? request.tools : [];
  const budgetTokens = options.budgetTokens ?? DEFAULT_EPOCH_BUDGET_TOKENS;
  const hardCapTokens = options.hardCapTokens ?? (budgetTokens + DEFAULT_EPOCH_HARD_SLACK_TOKENS);
  const backboneMaxTokens = options.backboneMaxTokens ?? DEFAULT_EPOCH_BACKBONE_MAX_TOKENS;
  const keepRecentPairs = Math.max(1, options.keepRecentPairs ?? DEFAULT_EPOCH_KEEP_RECENT_PAIRS);
  const maxResultTokens = options.maxResultTokens ?? DEFAULT_EPOCH_MAX_RESULT_TOKENS;

  const systemMessages: OpenAIMessage[] = [];
  const conversation: OpenAIMessage[] = [];
  for (const msg of rawMessages) {
    if (msg.role === "system") systemMessages.push(msg);
    else conversation.push(msg);
  }

  const toolsHeaderTokens = rawTools.length > 0 ? count(`Tools:\n${JSON.stringify(rawTools)}\n`) + 45 : 0;
  const turnTokens = (msgs: readonly OpenAIMessage[]) => msgs.reduce((s, m) => s + count(renderSingleMessageEstimate(m)), 0);
  const systemTokens = turnTokens(systemMessages) + toolsHeaderTokens;

  const userIndices: number[] = [];
  for (let i = 0; i < conversation.length; i++) {
    if (conversation[i]!.role === "user") userIndices.push(i);
  }
  if (userIndices.length === 0) {
    const tok = systemTokens + turnTokens(conversation);
    return { request, promptTokens: tok, epochCount: 0, folded: false };
  }

  const priorTurns: OpenAIMessage[][] = [];
  for (let t = 0; t < userIndices.length - 1; t++) {
    const from = t === 0 ? 0 : userIndices[t]!;
    const to = userIndices[t + 1]!;
    priorTurns.push(foldCompletedPriorOpenAITurn(conversation.slice(from, to), maxResultTokens, count));
  }
  let priorTokens = priorTurns.reduce((s, turn) => s + turnTokens(turn), 0);
  while (priorTurns.length > 0 && priorTokens > backboneMaxTokens) {
    const dropped = priorTurns.shift()!;
    priorTokens -= turnTokens(dropped);
  }
  const priorFlat = priorTurns.flat();

  const activeStart = userIndices.length === 1 ? 0 : userIndices[userIndices.length - 1]!;
  const activeSlice = conversation.slice(activeStart);
  const activeGroups = parseEpochGroups(activeSlice, maxResultTokens, count);
  const toolGroups = activeGroups.filter((g) => g.kind === "tool_group");

  const fullGroupTokens = new Map<EpochGroup, number>();
  const bareGroupTokens = new Map<EpochGroup, number>();
  for (const g of activeGroups) {
    fullGroupTokens.set(g, turnTokens(g.messages));
    bareGroupTokens.set(g, g.kind === "tool_group" ? turnTokens(stripGroupReasoning(g)) : fullGroupTokens.get(g)!);
  }

  // Fast-path: single-turn, no oversized tool message slimmed, and total tokens <= budgetTokens
  // -> return exact untouched `request` reference for 100% byte-for-byte training parity!
  const anySlimmedInActive = activeGroups.some((g, idx) =>
    g.kind === "tool_group" && g.messages.some((m, mIdx) => m !== activeSlice[activeGroups.slice(0, idx).reduce((s, x) => s + x.messages.length, 0) + mIdx]),
  );
  const rawActiveTokens = activeGroups.reduce((s, g) => s + (fullGroupTokens.get(g) ?? 0), 0);
  if (priorTurns.length === 0 && !anySlimmedInActive && systemTokens + rawActiveTokens <= budgetTokens) {
    return { request, promptTokens: systemTokens + rawActiveTokens, epochCount: 0, folded: false };
  }

  // Select a pinned Working Set read strictly at the moment `cutIdx` is chosen (`cutTriggerCount`),
  // so subsequent steps within the same Epoch never mutate the Epoch prefix (`100%` intra-epoch Append-Only).
  const selectFrozenPinnedRead = (cutIdx: number, cutTriggerCount: number): EpochGroup | undefined => {
    if (cutIdx <= 0) return undefined;
    const foldedTools = toolGroups.slice(0, cutIdx);
    const candidate = findLatestUneditedReadGroup(foldedTools, fullGroupTokens);
    if (!candidate) return undefined;
    const candidatePath = pathOfArgs(candidate.calls[0]?.args ?? {});
    const initialActive = toolGroups.slice(cutIdx, cutTriggerCount);
    const superseded = initialActive.some((g) =>
      g.ok && g.calls.some((c) =>
        (READ_NAMES.has(c.name) && (candidatePath === undefined || pathOfArgs(c.args) === candidatePath))
        || (WRITE_NAMES.has(c.name) && candidatePath !== undefined && pathOfArgs(c.args) === candidatePath),
      ),
    );
    return superseded ? undefined : candidate;
  };

  const buildWindowForCut = (
    cutIdx: number,
    frozenPinnedRead: EpochGroup | undefined,
    endToolCount: number,
    includeFinalAssistant: boolean,
    enforceHardCap: boolean,
  ): { messages: OpenAIMessage[]; tokens: number; folded: boolean } => {
    const subTools = toolGroups.slice(0, endToolCount);
    const foldedTools = subTools.slice(0, cutIdx);
    const activeTools = subTools.slice(cutIdx);
    const activeSet = new Set<EpochGroup>(activeTools);

    const pinnedRead = foldedTools.length > 0 ? frozenPinnedRead : undefined;
    const ledgerBase = pinnedRead ? foldedTools.filter((g) => g !== pinnedRead) : foldedTools;

    let keptGroups = activeGroups.filter((g) => {
      if (g.kind === "user") return true;
      if (g.kind === "final_assistant") return includeFinalAssistant;
      return g === pinnedRead || activeSet.has(g);
    });
    const keptSet = new Set<EpochGroup>(keptGroups);
    const clearedReasoning = new Set<EpochGroup>();
    if (pinnedRead) clearedReasoning.add(pinnedRead);
    const extraDropped: EpochGroup[] = [];

    const assemble = (): { messages: OpenAIMessage[]; tokens: number } => {
      const out: OpenAIMessage[] = [];
      let tok = systemTokens + priorTokens;
      let ledgerMerged = false;
      const allFolded = [...ledgerBase, ...extraDropped];
      const ledgerMsg = allFolded.length > 0 ? buildOpenAIEpochStateLedger(allFolded) : undefined;
      const ledgerText = ledgerMsg ? messageText(ledgerMsg.content) : undefined;

      for (const g of keptGroups) {
        if (g.kind === "user") {
          if (!ledgerMerged && ledgerText && g.messages.length > 0) {
            const mergedFirst = appendLedgerToUserMessage(g.messages[0]!, ledgerText);
            const rest = g.messages.slice(1);
            out.push(mergedFirst, ...rest);
            tok += turnTokens([mergedFirst, ...rest]);
            ledgerMerged = true;
          } else {
            out.push(...g.messages);
            tok += fullGroupTokens.get(g) ?? 0;
          }
          continue;
        }
        if (!ledgerMerged && ledgerMsg) {
          out.push(ledgerMsg);
          tok += count(renderSingleMessageEstimate(ledgerMsg));
          ledgerMerged = true;
        }
        if (g.kind === "tool_group" && clearedReasoning.has(g)) {
          out.push(...stripGroupReasoning(g));
          tok += bareGroupTokens.get(g) ?? 0;
        } else {
          out.push(...g.messages);
          tok += fullGroupTokens.get(g) ?? 0;
        }
      }
      return { messages: out, tokens: tok };
    };

    let built = assemble();

    if (enforceHardCap && built.tokens > hardCapTokens) {
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
      const droppable = activeTools.filter((g) => keptSet.has(g));
      while (droppable.length > 1 && built.tokens > hardCapTokens) {
        const oldest = droppable.shift()!;
        extraDropped.push(oldest);
        keptSet.delete(oldest);
        keptGroups = keptGroups.filter((g) => g !== oldest);
        built = assemble();
      }
    }

    return {
      messages: built.messages,
      tokens: built.tokens,
      folded: priorTurns.length > 0 || cutIdx > 0 || extraDropped.length > 0 || anySlimmedInActive,
    };
  };

  let cutIdx = 0;
  let frozenPinnedRead: EpochGroup | undefined;
  let epochCount = 0;
  for (let k = 0; k < toolGroups.length; k++) {
    let win = buildWindowForCut(cutIdx, frozenPinnedRead, k + 1, false, false);
    if (win.tokens > budgetTokens && (k + 1 - cutIdx) > 1) {
      epochCount++;
      cutIdx = Math.max(cutIdx + 1, (k + 1) - keepRecentPairs);
      frozenPinnedRead = selectFrozenPinnedRead(cutIdx, k + 1);
      win = buildWindowForCut(cutIdx, frozenPinnedRead, k + 1, false, false);
      if (frozenPinnedRead && win.tokens > budgetTokens) {
        frozenPinnedRead = undefined;
        win = buildWindowForCut(cutIdx, undefined, k + 1, false, false);
      }
      while (win.tokens > budgetTokens && cutIdx < k) {
        cutIdx++;
        frozenPinnedRead = selectFrozenPinnedRead(cutIdx, k + 1);
        win = buildWindowForCut(cutIdx, frozenPinnedRead, k + 1, false, false);
        if (frozenPinnedRead && win.tokens > budgetTokens) {
          frozenPinnedRead = undefined;
          win = buildWindowForCut(cutIdx, undefined, k + 1, false, false);
        }
      }
    }
  }

  const hasFinal = activeGroups.some((g) => g.kind === "final_assistant");
  const finalWin = buildWindowForCut(cutIdx, frozenPinnedRead, toolGroups.length, hasFinal, true);
  return {
    request: { ...request, messages: [...systemMessages, ...priorFlat, ...finalWin.messages] },
    promptTokens: finalWin.tokens,
    epochCount,
    folded: finalWin.folded,
  };
}
