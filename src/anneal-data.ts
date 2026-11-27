/**
 * Offline Annealing Dataset Converter & Off-Policy Recovery Synthesizer for Anchored Batch Line-Edit v2.
 *
 * Converts existing RWKV-Pi text-edit SFT datasets (either intermediate `openai-messages-with-last-
 * assistant-loss` rows or packed `rwkv-lightning-segments-eos-target-v2` trajectory spans) into:
 *   1. Neutral numbered `read` outputs (`=== <path> (lines X-Y of Z) ===\n1: ...`, capped at 200 lines),
 *   2. Atomic multi-hunk anchored `edit` calls (`{"path","edits":[{"start_line","end_line","start_text","new_text"}]}`),
 *   3. CoT-Action alignment: ensures supervised `<think>` blocks explicitly ground `(start_line, start_text)` before emitting `edit`,
 *   4. Upgraded `System:` guidelines + `Tools:` schemas + Epoch `State Ledger` wording, and
 *   5. Optional zero-mistake-loss (`train: false` on simulated error, `train: true` on recovery) augmentation
 *      for `unread_first`, `anchor_mismatch`, `boundary_dup`, and `noop_recovery`.
 */
import { createHash } from "node:crypto";
import {
  applyAnchoredLineEdits,
  DEFAULT_LINE_READ_LIMIT,
  formatBatchLineEditSummary,
  isNumberedFileOutput,
  LINE_EDIT_DESCRIPTION,
  LINE_EDIT_PARAMETERS,
  LINE_READ_DESCRIPTION,
  LINE_READ_PARAMETERS,
  LineViewTracker,
  MAX_LINE_READ_LIMIT,
  parseFileHeaderLine,
  renderNumberedWindow,
  splitFileLines,
  type LineEditInput,
} from "./line-edit.js";
import { renderPrompt, type ToolFormat } from "./prompt.js";
import { object } from "./tasks.js";
import { tolerateEdits } from "./tolerant-tools.js";

export interface LightningSegment {
  text: string;
  train: boolean;
  end_target?: number;
}

export interface PackedSpanRow {
  schemaVersion?: number;
  format?: string;
  id?: string;
  taskId?: string;
  family?: string;
  cohort?: string;
  sourceKind?: string;
  historyMode?: string;
  epochBudgetTokens?: number;
  segments: LightningSegment[];
  [key: string]: unknown;
}

type OpenAIMessage = Record<string, any>;

export interface ParsedTrajectory {
  readonly system: string;
  readonly tools: any[];
  readonly messages: OpenAIMessage[];
  /** Indices into `messages` (assistant role) that carried `train: true` supervision in the source span. */
  readonly supervisedIndices: Set<number>;
}

const XML_TOOL_INSTRUCTION =
  'When using tools, output one XML envelope per call: <tool_call>{"name":"read","arguments":{"path":"input.txt"}}</tool_call>. Use only the declared tool names and arguments. Tool results arrive inside <tool_response>. Otherwise answer normally.';

export function upgradeSystemTextToLineMode(system: string): string {
  return system
    .replace(
      /(?:Make precise file edits with exact text replacement, including multiple disjoint edits in one call|Replace, insert, or delete numbered line ranges in a file)/g,
      "Apply one or more anchored line-range edits atomically in a file",
    )
    .replace(
      /(?:Use read to examine files instead of cat or sed\.|Use read to inspect files with 1-based line numbers before calling edit\.)/g,
      `Use read to inspect files with 1-based line numbers (default ${DEFAULT_LINE_READ_LIMIT} lines, max ${MAX_LINE_READ_LIMIT}) before calling edit.`,
    )
    .replace(
      /(?:Use edit for targeted changes\. Batch changes to one file in a single call\.|Always read the target lines with read first so start_line and end_line match the current file\.)/g,
      "Always read the target lines with read first; batch multiple non-overlapping edits to the same file into one edits[] call against the original pre-edit line numbers.",
    )
    .replace(
      /(?:Each edits\[\]\.oldText must exactly and uniquely match the original file\. Keep matches short and non-overlapping; merge nearby changes without including large unchanged regions\.|Write new_text without line-number prefixes; check the returned post-edit window and line-shift notice before making another edit to the same file\.)/g,
      "Include start_text (the exact original line at start_line without line-number prefix) for each edit, and write new_text without line-number prefixes.",
    );
}

export function upgradeToolsToLineMode(tools: readonly any[]): any[] {
  return tools.map((entry) => {
    const fn = entry?.function ?? entry;
    if (!fn || typeof fn.name !== "string") return entry;
    if (fn.name === "read") {
      const updated = {
        name: "read",
        description: LINE_READ_DESCRIPTION,
        parameters: JSON.parse(JSON.stringify(LINE_READ_PARAMETERS)),
      };
      return entry.function ? { ...entry, function: updated } : updated;
    }
    if (fn.name === "edit") {
      const updated = {
        name: "edit",
        description: LINE_EDIT_DESCRIPTION,
        parameters: JSON.parse(JSON.stringify(LINE_EDIT_PARAMETERS)),
      };
      return entry.function ? { ...entry, function: updated } : updated;
    }
    return structuredClone(entry);
  });
}

function upgradeLedgerTextInMessage(text: string): string {
  return text.replace(
    /^- Inspected: /gm,
    "- Inspected (content folded; call read with offset/limit before editing): ",
  );
}

/**
 * Parses a packed `rwkv-lightning-segments-eos-target-v2` row back into structured OpenAI messages
 * while tracking which assistant turns carried `train: true` supervision.
 */
export function parseXmlContinuationSegments(segments: readonly LightningSegment[]): ParsedTrajectory {
  const usable = segments.filter((s) => !(s.text === "<EOD>" && !s.train));
  let fullText = "";
  const trainRanges: { start: number; end: number }[] = [];
  for (const seg of usable) {
    const start = fullText.length;
    fullText += seg.text;
    if (seg.train) trainRanges.push({ start, end: fullText.length });
  }

  let cursor = 0;
  let system = "";
  let tools: any[] = [];
  if (fullText.startsWith("System: ")) {
    const firstTurn = fullText.indexOf("\n\nUser: ");
    if (firstTurn < 0) throw new Error("Missing first User turn after System header.");
    const headerBlock = fullText.slice("System: ".length, firstTurn);
    const toolsMarker = "\nTools:\n";
    const tAt = headerBlock.indexOf(toolsMarker);
    if (tAt >= 0) {
      system = headerBlock.slice(0, tAt);
      const afterTools = headerBlock.slice(tAt + toolsMarker.length);
      const nl = afterTools.indexOf("\n");
      const toolsJson = nl >= 0 ? afterTools.slice(0, nl) : afterTools;
      tools = JSON.parse(toolsJson);
    } else if (headerBlock.startsWith("Tools:\n")) {
      const afterTools = headerBlock.slice("Tools:\n".length);
      const nl = afterTools.indexOf("\n");
      tools = JSON.parse(nl >= 0 ? afterTools.slice(0, nl) : afterTools);
    } else {
      system = headerBlock;
    }
    cursor = firstTurn + 2;
  }

  const messages: OpenAIMessage[] = [];
  if (system) messages.push({ role: "system", content: system });
  const supervisedIndices = new Set<number>();
  let callSeq = 0;
  let pendingCallIds: string[] = [];

  while (cursor < fullText.length) {
    if (fullText.startsWith("User: <tool_response>\n", cursor)) {
      const bodyStart = cursor + "User: <tool_response>\n".length;
      const closeAt = fullText.indexOf("\n</tool_response>\n\n", bodyStart);
      if (closeAt < 0) throw new Error("Unclosed <tool_response> block.");
      const content = fullText.slice(bodyStart, closeAt);
      const callId = pendingCallIds.shift() ?? `call_parsed_${++callSeq}`;
      messages.push({ role: "tool", tool_call_id: callId, content });
      cursor = closeAt + "\n</tool_response>\n\n".length;
      continue;
    }
    if (fullText.startsWith("User: ", cursor)) {
      const bodyStart = cursor + "User: ".length;
      const nextAssistant = fullText.indexOf("\n\nAssistant: <think", bodyStart);
      const nextUser = fullText.indexOf("\n\nUser: ", bodyStart);
      let endAt = fullText.length;
      if (nextAssistant >= 0 && nextUser >= 0) endAt = Math.min(nextAssistant, nextUser);
      else if (nextAssistant >= 0) endAt = nextAssistant;
      else if (nextUser >= 0) endAt = nextUser;
      const content = fullText.slice(bodyStart, endAt).replace(/\n+$/, "");
      messages.push({ role: "user", content });
      cursor = endAt + 2;
      continue;
    }
    if (fullText.startsWith("Assistant: <think>", cursor)) {
      const turnStart = cursor;
      const thinkStart = cursor + "Assistant: <think>".length;
      const thinkClose = fullText.indexOf("</think>\n", thinkStart);
      if (thinkClose < 0) throw new Error("Unclosed <think> block.");
      const reasoning = fullText.slice(thinkStart, thinkClose);
      const bodyStart = thinkClose + "</think>\n".length;
      const nextUser = fullText.indexOf("\n\nUser: ", bodyStart);
      const turnEnd = nextUser >= 0 ? nextUser : fullText.replace(/\n+$/, "").length;
      const rawBody = fullText.slice(bodyStart, turnEnd);

      const toolCalls: any[] = [];
      const textLines: string[] = [];
      let rem = rawBody.trim();
      if (rem.includes("<tool_call>")) {
        const callIdx = rem.indexOf("<tool_call>");
        const beforeCalls = rem.slice(0, callIdx).trim();
        if (beforeCalls) textLines.push(beforeCalls);
        let tail = rem.slice(callIdx).trim();
        while (tail.startsWith("<tool_call>")) {
          const innerStart = "<tool_call>".length;
          // Respect quotes when locating </tool_call>
          let quoted = false, esc = false, depth = 0, jsonEnd = -1;
          for (let i = innerStart; i < tail.length; i++) {
            const c = tail[i]!;
            if (quoted) {
              if (esc) esc = false;
              else if (c === "\\") esc = true;
              else if (c === '"') quoted = false;
            } else if (c === '"') quoted = true;
            else if (c === "{") depth++;
            else if (c === "}") {
              if (--depth === 0) { jsonEnd = i + 1; break; }
            }
          }
          if (jsonEnd < 0) throw new Error("Malformed JSON inside <tool_call>.");
          const parsedCall = JSON.parse(tail.slice(innerStart, jsonEnd));
          const afterJson = tail.slice(jsonEnd).trimStart();
          if (!afterJson.startsWith("</tool_call>")) throw new Error("Missing </tool_call>.");
          const id = `call_${++callSeq}`;
          toolCalls.push({
            id,
            type: "function",
            function: { name: parsedCall.name, arguments: JSON.stringify(parsedCall.arguments ?? {}) },
          });
          pendingCallIds.push(id);
          tail = afterJson.slice("</tool_call>".length).trimStart();
        }
      } else if (rem) {
        textLines.push(rem);
      }

      const msgIdx = messages.length;
      messages.push({
        role: "assistant",
        content: textLines.length ? textLines.join("\n") : null,
        reasoning_content: reasoning,
        ...(toolCalls.length ? { tool_calls: toolCalls } : {}),
      });
      const isSupervised = trainRanges.some((r) => r.start < turnEnd && r.end > turnStart);
      if (isSupervised) supervisedIndices.add(msgIdx);
      cursor = nextUser >= 0 ? nextUser + 2 : fullText.length;
      continue;
    }
    break;
  }
  void XML_TOOL_INSTRUCTION;
  return { system, tools, messages, supervisedIndices };
}

export interface VirtualFileState {
  content: string;
  /** Stable version token across disjoint `read`s of the same unedited file; changes only on `write` / `edit`. */
  versionId: string;
  /** True when `content` came from an un-truncated full `read` or `write`. */
  complete: boolean;
  /** 1-based starting line of `content` when loaded from a partial `read(offset, limit)`. */
  baseLine: number;
  /** 1-based sparse line map before any edits when assembled from disjoint partial `read` calls. */
  sparseLines?: Record<number, string>;
  totalLinesHint?: number;
}

export type TrajectoryHistory = Map<string, VirtualFileState[]>;

const UNREAD_SENTINEL_PREFIX = "<<RWKV_UNREAD_LINE_";
let versionSeq = 0;

function nextVersionId(): string {
  return `v_${++versionSeq}`;
}

function materializeSparseState(
  sparseLines: Record<number, string>,
  versionId: string,
  totalLinesHint?: number,
): VirtualFileState {
  const lineNums = Object.keys(sparseLines).map(Number).sort((a, b) => a - b);
  if (lineNums.length === 0) {
    return { content: "", versionId, complete: false, baseLine: 1, sparseLines: {}, totalLinesHint };
  }
  const minLine = lineNums[0]!;
  const maxObserved = lineNums[lineNums.length - 1]!;
  const baseLine = minLine <= 2000 ? 1 : minLine;
  const maxLine = Math.max(maxObserved, minLine <= 2000 && totalLinesHint && totalLinesHint <= 4000 ? totalLinesHint : maxObserved);
  const built: string[] = [];
  for (let ln = baseLine; ln <= maxLine; ln++) {
    built.push(ln in sparseLines ? sparseLines[ln]! : `${UNREAD_SENTINEL_PREFIX}${ln}>>`);
  }
  return {
    content: built.join("\n"),
    versionId,
    complete: false,
    baseLine,
    sparseLines,
    totalLinesHint: Math.max(totalLinesHint ?? 0, maxObserved),
  };
}

function recordTrajectorySnapshot(history: TrajectoryHistory | undefined, key: string, state: VirtualFileState): void {
  if (!history) return;
  const list = history.get(key) ?? [];
  const last = list[list.length - 1];
  if (last && last.versionId === state.versionId) {
    list[list.length - 1] = structuredClone(state);
    history.set(key, list);
    return;
  }
  if (!last || last.content !== state.content || last.baseLine !== state.baseLine) {
    list.push(structuredClone(state));
    history.set(key, list);
  }
}

function findMatchingHistoricalState(
  history: TrajectoryHistory | undefined,
  key: string,
  rawPath: string,
  firstHunk: { oldText: string; newText: string } | undefined,
): VirtualFileState | undefined {
  const candidates = history?.get(key);
  if (!candidates || candidates.length === 0) return undefined;
  if (!firstHunk || typeof firstHunk.oldText !== "string" || typeof firstHunk.newText !== "string") {
    return structuredClone(candidates[candidates.length - 1]!);
  }
  const exactOld = firstHunk.oldText.replaceAll("\r\n", "\n");
  const exactNew = firstHunk.newText.replaceAll("\r\n", "\n");
  for (let idx = candidates.length - 1; idx >= 0; idx--) {
    const snap = candidates[idx]!;
    const cur = snap.content.replaceAll("\r\n", "\n");
    if (cur.includes(exactOld) && cur.indexOf(exactOld, cur.indexOf(exactOld) + 1) < 0) {
      return structuredClone(snap);
    }
    try {
      tolerateEdits(cur, [{ oldText: exactOld, newText: exactNew }], rawPath);
      return structuredClone(snap);
    } catch { /* try older snapshot */ }
  }
  return undefined;
}

function normVirtualKey(rawPath: string): string {
  return rawPath.trim().replace(/^@/, "").replace(/^(?:\/workspace\/|\/app\/|\.\/)+/, "");
}

function stripReadTruncationFooter(rawOutput: string): {
  body: string;
  complete: boolean;
  totalLinesHint?: number;
  shownLinesHint?: number;
} {
  // Handle pi's native `[Showing lines X-Y of Z ...]` or `[... characters omitted ...]`
  if (rawOutput.includes("[... ") && rawOutput.includes("omitted to preserve context")) {
    return { body: rawOutput.split("\n\n[... ")[0]!, complete: false };
  }
  const footerMatch = /\n\[Showing lines (\d+)-(\d+) of (\d+)[^\]]*\]\s*$/.exec(rawOutput);
  if (footerMatch) {
    const first = Number(footerMatch[1]);
    const last = Number(footerMatch[2]);
    return {
      body: rawOutput.slice(0, footerMatch.index),
      complete: first === 1 && last === Number(footerMatch[3]),
      totalLinesHint: Number(footerMatch[3]),
      shownLinesHint: last >= first ? last - first + 1 : undefined,
    };
  }
  return { body: rawOutput, complete: true };
}

export interface ConvertTrajectoryOptions {
  /** Reject trajectories that call `edit` on a file never `read` or `write`n earlier in the trajectory (default true). */
  readonly rejectBlindEdits?: boolean;
  /** Optional cross-span file version history from earlier Epoch spans of the same trajectory. */
  readonly trajectoryHistory?: TrajectoryHistory;
}

export type ConvertTrajectoryResult =
  | {
      ok: true;
      messages: OpenAIMessage[];
      tools: any[];
      supervisedIndices: Set<number>;
      convertedEdits: number;
      convertedReads: number;
      synthesizedPostFoldReads: number;
    }
  | { ok: false; reason: string };

function formatAnchorGroundSentence(
  rawPath: string,
  hunks: readonly { start_line: number; end_line: number; start_text: string }[],
  existingReasoning: string,
): string {
  const first = hunks[0];
  if (!first) return existingReasoning;
  const lineCited = new RegExp(`\\b${first.start_line}\\b`).test(existingReasoning);
  const clip = (s: string) => {
    const clean = s.trim().replace(/<\/?think>/gi, "").replace(/`/g, "'");
    return clean.length > 48 ? `${clean.slice(0, 45)}...` : clean;
  };
  const firstClip = clip(first.start_text);
  if (lineCited && (!firstClip || existingReasoning.includes(firstClip.slice(0, Math.min(16, firstClip.length))))) {
    return existingReasoning;
  }
  const useZh = /[\u4e00-\u9fff]/.test(existingReasoning);
  let note: string;
  if (hunks.length === 1) {
    const spanZh = first.start_line === first.end_line
      ? `第 ${first.start_line} 行`
      : `第 ${first.start_line}-${first.end_line} 行`;
    const spanEn = first.start_line === first.end_line
      ? `line ${first.start_line}`
      : `lines ${first.start_line}-${first.end_line}`;
    note = useZh
      ? `确认目标位置：${rawPath} ${spanZh}${firstClip ? `（首行 \`${firstClip}\`）` : ""}，按行号锚点替换。`
      : `Target anchor in ${rawPath}: ${spanEn}${firstClip ? ` (starting with \`${firstClip}\`)` : ""}.`;
  } else {
    const partsZh = hunks.slice(0, 4).map((h) => {
      const sp = h.start_line === h.end_line ? `第 ${h.start_line} 行` : `第 ${h.start_line}-${h.end_line} 行`;
      const c = clip(h.start_text);
      return c ? `${sp} \`${c}\`` : sp;
    });
    const partsEn = hunks.slice(0, 4).map((h) => {
      const sp = h.start_line === h.end_line ? `line ${h.start_line}` : `lines ${h.start_line}-${h.end_line}`;
      const c = clip(h.start_text);
      return c ? `${sp} (\`${c}\`)` : sp;
    });
    note = useZh
      ? `确认目标位置：${rawPath} 共 ${hunks.length} 处（${partsZh.join("；")}${hunks.length > 4 ? " 等" : ""}），合并为单次原子 edits[] 替换。`
      : `Target anchors in ${rawPath} (${hunks.length} hunks: ${partsEn.join("; ")}${hunks.length > 4 ? "; ..." : ""}); applying atomically in one edits[] call.`;
  }
  const trimmed = existingReasoning.trim();
  return trimmed ? `${trimmed}\n${note}` : note;
}

export type RecoveryAugmentKind = "none" | "unread_first" | "anchor_mismatch" | "boundary_dup" | "noop_recovery";

/**
 * Converts a parsed trajectory (`messages` + `tools`) from substring `edit` into Anchored Batch
 * Line-Edit v2 (`read` with neutral numbered windows + atomic anchored `edit`), verifying every
 * edit on the reconstructed file lines, aligning supervised CoT with `(start_line, start_text)`,
 * and enforcing `LineViewTracker` active-context visibility on all supervised (`train: true`) edits.
 */
export function convertTrajectoryToLineEdit(
  parsed: ParsedTrajectory,
  options: ConvertTrajectoryOptions = {},
): ConvertTrajectoryResult {
  const rejectBlind = options.rejectBlindEdits !== false;
  const tools = upgradeToolsToLineMode(parsed.tools);
  const files = new Map<string, VirtualFileState>();
  const tracker = new LineViewTracker();
  const outMessages: OpenAIMessage[] = [];
  const outSupervised = new Set<number>();
  let convertedEdits = 0;
  let convertedReads = 0;
  let synthesizedPostFoldReads = 0;

  // Map tool_call_id -> tool message index in parsed.messages
  const toolReplyById = new Map<string, OpenAIMessage>();
  for (const m of parsed.messages) {
    if (m.role === "tool" && typeof m.tool_call_id === "string") {
      toolReplyById.set(m.tool_call_id, m);
    }
  }

  for (let i = 0; i < parsed.messages.length; i++) {
    const msg = parsed.messages[i]!;
    if (msg.role === "system") {
      outMessages.push({ ...msg, content: upgradeSystemTextToLineMode(String(msg.content ?? "")) });
      continue;
    }
    if (msg.role === "user") {
      outMessages.push({ ...msg, content: upgradeLedgerTextInMessage(String(msg.content ?? "")) });
      continue;
    }
    if (msg.role === "tool") {
      continue;
    }
    if (msg.role !== "assistant") {
      outMessages.push(structuredClone(msg));
      continue;
    }

    const isSup = parsed.supervisedIndices.has(i);
    const rawCalls = Array.isArray(msg.tool_calls) ? msg.tool_calls : [];
    if (rawCalls.length === 0) {
      const idx = outMessages.length;
      outMessages.push(structuredClone(msg));
      if (isSup) outSupervised.add(idx);
      continue;
    }

    const expandedSteps: { assistant: OpenAIMessage; toolResults: OpenAIMessage[] }[] = [];
    const currentBatchCalls: any[] = [];
    const currentBatchResults: OpenAIMessage[] = [];
    let batchReasoning = String(msg.reasoning_content ?? "");

    const flushCurrentBatch = (keepContentAndReasoning: boolean) => {
      if (currentBatchCalls.length === 0) return;
      expandedSteps.push({
        assistant: {
          role: "assistant",
          content: keepContentAndReasoning ? (msg.content ?? null) : null,
          reasoning_content: keepContentAndReasoning ? batchReasoning : "",
          tool_calls: [...currentBatchCalls],
        },
        toolResults: [...currentBatchResults],
      });
      currentBatchCalls.length = 0;
      currentBatchResults.length = 0;
    };

    for (const tc of rawCalls) {
      const fn = tc.function ?? tc;
      const name = String(fn.name ?? "");
      const args = typeof fn.arguments === "string" ? JSON.parse(fn.arguments) : (fn.arguments ?? {});
      const origToolMsg = toolReplyById.get(tc.id);
      const origToolText = String(origToolMsg?.content ?? "");
      const origFailed = origToolText.startsWith("Error:")
        || origToolText.startsWith("Could not ")
        || origToolText.startsWith("ENOENT")
        || origToolText.startsWith("Validation failed");

      if (name === "read") {
        const rawPath = String(args.path ?? "");
        const key = normVirtualKey(rawPath);
        const offset = typeof args.offset === "number" && args.offset >= 1 ? Math.trunc(args.offset) : 1;
        const limit = typeof args.limit === "number" && args.limit >= 1
          ? Math.min(MAX_LINE_READ_LIMIT, Math.trunc(args.limit))
          : DEFAULT_LINE_READ_LIMIT;
        const normTc = structuredClone(tc);
        const normArgs = { ...args };
        let argsChanged = false;
        if (typeof args.offset === "number" && args.offset < 1) {
          normArgs.offset = 1;
          argsChanged = true;
        }
        if (typeof args.limit === "number" && args.limit > MAX_LINE_READ_LIMIT) {
          normArgs.limit = MAX_LINE_READ_LIMIT;
          argsChanged = true;
        }
        if (argsChanged) {
          if (normTc.function) normTc.function.arguments = JSON.stringify(normArgs);
          else normTc.arguments = normArgs;
        }

        if (!origFailed && origToolMsg && isNumberedFileOutput(origToolText)) {
          // Existing v1/v2 numbered read (`[File: ...]` or `=== ... ===`): populate virtual file
          // state and normalize header/window into canonical neutral v2 format!
          const rawLines = origToolText.split("\n");
          const hdr = parseFileHeaderLine(rawLines[0] ?? "");
          const parsedEntries: { ln: number; body: string }[] = [];
          for (let j = 1; j < rawLines.length; j++) {
            const l = rawLines[j]!;
            if (/^\(\d+ lines? (?:above|below)/.test(l) || l === "(empty file)") continue;
            const m = /^(\d+): (.*)$/.exec(l);
            if (!m) break;
            parsedEntries.push({ ln: Number(m[1]), body: m[2]! });
          }
          if (hdr && parsedEntries.length > 0) {
            const existing = files.get(key);
            const mergedSparse: Record<number, string> = existing?.sparseLines ? { ...existing.sparseLines } : {};
            if (existing && !existing.sparseLines) {
              const prevLines = splitFileLines(existing.content).lines;
              for (let idx = 0; idx < prevLines.length; idx++) {
                if (!prevLines[idx]!.startsWith(UNREAD_SENTINEL_PREFIX)) {
                  mergedSparse[existing.baseLine + idx] = prevLines[idx]!;
                }
              }
            }
            let mutated = false;
            for (const { ln, body } of parsedEntries) {
              if (ln in mergedSparse && mergedSparse[ln] !== body) mutated = true;
              mergedSparse[ln] = body;
            }
            const firstLn = parsedEntries[0]!.ln;
            const lastLn = parsedEntries[parsedEntries.length - 1]!.ln;
            const totalHint = Math.max(hdr.totalLines, lastLn, existing?.totalLinesHint ?? 0);
            const isComplete = firstLn === 1 && lastLn === totalHint && parsedEntries.length === totalHint;
            const verId = !existing || mutated ? nextVersionId() : existing.versionId;
            const nextState: VirtualFileState = isComplete
              ? {
                  content: parsedEntries.map((e) => e.body).join("\n"),
                  versionId: verId,
                  complete: true,
                  baseLine: 1,
                  totalLinesHint: totalHint,
                }
              : materializeSparseState(mergedSparse, verId, totalHint);
            files.set(key, nextState);
            recordTrajectorySnapshot(options.trajectoryHistory, key, nextState);
            const fullArr = Array.from({ length: totalHint }, (_, idx) => {
              const ln = idx + 1;
              const hit = parsedEntries.find((e) => e.ln === ln);
              if (hit) return hit.body;
              const sLines = splitFileLines(nextState.content).lines;
              const lIdx = ln - nextState.baseLine;
              return lIdx >= 0 && lIdx < sLines.length ? sLines[lIdx]! : "";
            });
            const win = renderNumberedWindow(rawPath, fullArr, firstLn, Math.min(lastLn, firstLn + limit - 1), true);
            tracker.recordRead(key, rawPath, nextState.versionId, win.firstShown, win.lastShown, totalHint);
            convertedReads++;
            currentBatchCalls.push(normTc);
            currentBatchResults.push({ role: "tool", tool_call_id: tc.id, content: win.text });
          } else {
            currentBatchCalls.push(normTc);
            currentBatchResults.push(structuredClone(origToolMsg));
          }
        } else if (!origFailed && origToolMsg && !isNumberedFileOutput(origToolText)) {
          const stripped = stripReadTruncationFooter(origToolText);
          const { lines: observedLines } = splitFileLines(stripped.body);
          if (
            stripped.shownLinesHint !== undefined
            && observedLines.length === stripped.shownLinesHint - 1
            && stripped.body.endsWith("\n")
          ) {
            observedLines.push("");
          }
          let existing = files.get(key);
          if (!existing && options.trajectoryHistory?.get(key)?.length) {
            const latestHist = options.trajectoryHistory.get(key)!;
            existing = structuredClone(latestHist[latestHist.length - 1]!);
          }

          if (offset === 1 && stripped.complete) {
            const sameVer = existing && existing.complete && existing.baseLine === 1 && existing.content === stripped.body
              ? existing.versionId
              : nextVersionId();
            files.set(key, {
              content: stripped.body,
              versionId: sameVer,
              complete: true,
              baseLine: 1,
              totalLinesHint: observedLines.length,
            });
          } else if (existing?.complete) {
            const prevLines = splitFileLines(existing.content).lines;
            let changed = stripped.totalLinesHint !== undefined && stripped.totalLinesHint !== prevLines.length;
            if (!changed) {
              for (let idx = 0; idx < observedLines.length; idx++) {
                if (prevLines[offset - 1 + idx] !== observedLines[idx]) {
                  changed = true;
                  break;
                }
              }
            }
            if (!changed) {
              files.set(key, existing);
            } else if (stripped.totalLinesHint === undefined || stripped.totalLinesHint === prevLines.length) {
              const updated = [...prevLines];
              for (let idx = 0; idx < observedLines.length; idx++) {
                updated[offset - 1 + idx] = observedLines[idx]!;
              }
              files.set(key, {
                content: updated.join("\n"),
                versionId: nextVersionId(),
                complete: true,
                baseLine: 1,
                totalLinesHint: updated.length,
              });
            } else {
              const freshSparse: Record<number, string> = {};
              for (let idx = 0; idx < observedLines.length; idx++) {
                freshSparse[offset + idx] = observedLines[idx]!;
              }
              files.set(key, materializeSparseState(freshSparse, nextVersionId(), stripped.totalLinesHint));
            }
          } else {
            let lineMutated = false;
            const mergedSparse: Record<number, string> = existing?.sparseLines
              ? { ...existing.sparseLines }
              : {};
            if (existing && !existing.sparseLines) {
              const prevLines = splitFileLines(existing.content).lines;
              for (let idx = 0; idx < prevLines.length; idx++) {
                if (!prevLines[idx]!.startsWith(UNREAD_SENTINEL_PREFIX)) {
                  mergedSparse[existing.baseLine + idx] = prevLines[idx]!;
                }
              }
            }
            for (let idx = 0; idx < observedLines.length; idx++) {
              const ln = offset + idx;
              if (ln in mergedSparse && mergedSparse[ln] !== observedLines[idx]) {
                lineMutated = true;
              }
              mergedSparse[ln] = observedLines[idx]!;
            }
            const verId = !existing || lineMutated ? nextVersionId() : existing.versionId;
            const nextHint = Math.max(
              existing?.totalLinesHint ?? 0,
              stripped.totalLinesHint ?? 0,
              offset + observedLines.length - 1,
            );
            files.set(key, materializeSparseState(mergedSparse, verId, nextHint));
          }

          const state = files.get(key)!;
          recordTrajectorySnapshot(options.trajectoryHistory, key, state);
          const stateLines = splitFileLines(state.content).lines;
          const totalCount = state.complete
            ? stateLines.length
            : Math.max(state.totalLinesHint ?? 0, (state.baseLine - 1) + stateLines.length, offset + observedLines.length - 1);
          const renderArr = Array.from({ length: totalCount }, (_, idx) => {
            const ln = idx + 1;
            if (ln >= offset && ln < offset + observedLines.length) {
              return observedLines[ln - offset] ?? "";
            }
            const localIdx = ln - state.baseLine;
            return (localIdx >= 0 && localIdx < stateLines.length) ? (stateLines[localIdx] ?? "") : "";
          });
          const win = renderNumberedWindow(rawPath, renderArr, offset, offset + Math.min(limit, observedLines.length || limit) - 1, true);
          if (win.text.includes(UNREAD_SENTINEL_PREFIX)) {
            return { ok: false, reason: "unread_sentinel_in_read_window" };
          }
          tracker.recordRead(key, rawPath, state.versionId, win.firstShown, win.lastShown, totalCount);
          convertedReads++;
          currentBatchCalls.push(normTc);
          currentBatchResults.push({ role: "tool", tool_call_id: tc.id, content: win.text });
        } else {
          currentBatchCalls.push(normTc);
          if (origToolMsg) currentBatchResults.push(structuredClone(origToolMsg));
        }
        continue;
      }

      if (name === "write") {
        const rawPath = String(args.path ?? "");
        const key = normVirtualKey(rawPath);
        if (!origFailed && typeof args.content === "string") {
          const nextState: VirtualFileState = {
            content: args.content,
            versionId: nextVersionId(),
            complete: true,
            baseLine: 1,
          };
          files.set(key, nextState);
          recordTrajectorySnapshot(options.trajectoryHistory, key, nextState);
          tracker.recordWrite(key, rawPath, nextState.versionId, splitFileLines(args.content).lines.length);
        }
        currentBatchCalls.push(structuredClone(tc));
        if (origToolMsg) currentBatchResults.push(structuredClone(origToolMsg));
        continue;
      }

      if (name === "edit") {
        // Already in v2 anchored batch mode?
        if (
          Array.isArray(args.edits)
          && args.edits.length > 0
          && typeof args.edits[0]?.start_line === "number"
          && typeof args.edits[0]?.end_line === "number"
          && typeof args.edits[0]?.new_text === "string"
        ) {
          currentBatchCalls.push(structuredClone(tc));
          if (origToolMsg) currentBatchResults.push(structuredClone(origToolMsg));
          continue;
        }
        // Legacy v1 single-hunk line mode -> upgrade into canonical v2 `edits: [...]` + virtual state + CoT!
        if (typeof args.start_line === "number" && typeof args.end_line === "number" && typeof args.new_text === "string") {
          const rawPath = String(args.path ?? "");
          const key = normVirtualKey(rawPath);
          const curState = files.get(key) ?? findMatchingHistoricalState(options.trajectoryHistory, key, rawPath, undefined);
          const curLines = curState ? splitFileLines(curState.content).lines : [];
          const baseOff = (curState?.baseLine ?? 1) - 1;
          const anchorIdx = args.start_line - baseOff - 1;
          const startText = typeof args.start_text === "string"
            ? args.start_text
            : (anchorIdx >= 0 && anchorIdx < curLines.length && !curLines[anchorIdx]!.startsWith(UNREAD_SENTINEL_PREFIX)
                ? curLines[anchorIdx]!
                : "");
          const wireEdit = {
            start_line: args.start_line,
            end_line: args.end_line,
            start_text: startText,
            new_text: args.new_text,
          };
          let upgradedRes: OpenAIMessage | undefined = origToolMsg ? structuredClone(origToolMsg) : undefined;
          if (!origFailed && curState && anchorIdx >= 0 && anchorIdx < curLines.length) {
            const localStart = args.start_line - baseOff;
            const localEnd = args.end_line - baseOff;
            const applied = applyAnchoredLineEdits(
              curState.content,
              rawPath,
              [{ startLine: localStart, endLine: localEnd, startText, newText: args.new_text }],
              { selfHeal: false },
            );
            if (!("error" in applied)) {
              const delta = applied.totalAdded - applied.totalRemoved;
              const nextState: VirtualFileState = {
                content: applied.content,
                versionId: nextVersionId(),
                complete: curState.complete,
                baseLine: curState.baseLine,
                totalLinesHint: curState.totalLinesHint ? Math.max(1, curState.totalLinesHint + delta) : undefined,
              };
              files.set(key, nextState);
              recordTrajectorySnapshot(options.trajectoryHistory, key, nextState);
              const totalBefore = curState.complete
                ? curLines.length
                : Math.max(curState.totalLinesHint ?? 0, baseOff + curLines.length);
              const globalApplied = baseOff === 0 && curState.complete
                ? applied
                : {
                    ...applied,
                    before: Array.from({ length: totalBefore }, (_, idx) => curLines[(idx + 1) - curState.baseLine] ?? ""),
                    after: Array.from({ length: Math.max(1, totalBefore + delta) }, (_, idx) => applied.after[(idx + 1) - curState.baseLine] ?? ""),
                    hunks: applied.hunks.map((h) => ({ ...h, startLine: h.startLine + baseOff, endLine: h.endLine + baseOff, requestedStart: h.requestedStart + baseOff, requestedEnd: h.requestedEnd + baseOff })),
                  };
              const summary = formatBatchLineEditSummary(rawPath, globalApplied);
              if (!summary.text.includes(UNREAD_SENTINEL_PREFIX)) {
                tracker.recordBatchEdit(key, rawPath, nextState.versionId, globalApplied, summary.windows);
                upgradedRes = { role: "tool", tool_call_id: tc.id, content: summary.text };
              }
            }
          }
          if (isSup && !origFailed) {
            batchReasoning = formatAnchorGroundSentence(rawPath, [wireEdit], batchReasoning);
          }
          convertedEdits++;
          const upgradedCall = {
            id: tc.id,
            type: "function",
            function: {
              name: "edit",
              arguments: JSON.stringify({ path: rawPath, edits: [wireEdit] }),
            },
          };
          currentBatchCalls.push(upgradedCall);
          if (upgradedRes) currentBatchResults.push(upgradedRes);
          continue;
        }

        const rawPath = String(args.path ?? "");
        const key = normVirtualKey(rawPath);
        const rawHunks: { oldText: string; newText: string }[] = Array.isArray(args.edits)
          ? args.edits
          : (typeof args.oldText === "string" && typeof args.newText === "string" ? [{ oldText: args.oldText, newText: args.newText }] : []);
        if (origFailed) {
          if (!isSup) {
            const fallbackNew = rawHunks[0]?.newText?.replaceAll("\r\n", "\n") ?? "";
            const fallbackStartText = rawHunks[0]?.oldText?.replaceAll("\r\n", "\n").split("\n", 1)[0] ?? "";
            const known = files.get(key) ?? findMatchingHistoricalState(options.trajectoryHistory, key, rawPath, rawHunks[0]);
            const simFailCall = {
              id: tc.id,
              type: "function",
              function: {
                name: "edit",
                arguments: JSON.stringify({
                  path: rawPath,
                  edits: [{
                    start_line: 1,
                    end_line: 1,
                    start_text: fallbackStartText,
                    new_text: fallbackNew || "pass",
                  }],
                }),
              },
            };
            const simFailText = !known
              ? `Cannot edit ${rawPath} before reading its numbered lines in the current context. Call read({"path":${JSON.stringify(rawPath)},"offset":1,"limit":80}) first to confirm exact line numbers and start_text.`
              : `Your edit in ${rawPath} was NOT applied: start_text did not match line 1. Call read({"path":${JSON.stringify(rawPath)},"offset":1,"limit":80}) first to verify exact 1-based line numbers and start_text before retrying.`;
            currentBatchCalls.push(simFailCall);
            currentBatchResults.push({ role: "tool", tool_call_id: tc.id, content: simFailText });
            continue;
          }
          return { ok: false, reason: "unconverted_failed_substring_edit" };
        }
        if (!rawHunks.length) return { ok: false, reason: "empty_edit_hunks" };

        let state = files.get(key);
        if (!state) {
          state = findMatchingHistoricalState(options.trajectoryHistory, key, rawPath, rawHunks[0]);
          if (state) files.set(key, state);
        }
        if (!state) {
          if (rejectBlind) return { ok: false, reason: "blind_edit_without_prior_read" };
          return { ok: false, reason: "missing_virtual_file_for_edit" };
        }

        const curContent = state.content.replaceAll("\r\n", "\n");
        const curLinesLocal = splitFileLines(curContent).lines;
        const baseOffset = (state.baseLine ?? 1) - 1;
        const totalBefore = state.complete
          ? curLinesLocal.length
          : Math.max(state.totalLinesHint ?? 0, baseOffset + curLinesLocal.length);
        const beforeVer = state.versionId;

        interface PlannedHunk {
          startLineLocal: number;
          endLineLocal: number;
          startLine: number;
          endLine: number;
          startText: string;
          newText: string;
        }
        const planned: PlannedHunk[] = [];

        for (const hunk of rawHunks) {
          if (!hunk || typeof hunk.oldText !== "string" || typeof hunk.newText !== "string") {
            return { ok: false, reason: "malformed_edit_hunk_fields" };
          }
          let exactOld = hunk.oldText.replaceAll("\r\n", "\n");
          let exactNew = hunk.newText.replaceAll("\r\n", "\n");
          if (!curContent.includes(exactOld) || curContent.indexOf(exactOld, curContent.indexOf(exactOld) + 1) >= 0) {
            try {
              const tol = tolerateEdits(curContent, [{ oldText: exactOld, newText: exactNew }], rawPath);
              exactOld = tol[0]!.oldText;
              exactNew = tol[0]!.newText;
            } catch {
              return { ok: false, reason: "edit_hunk_not_uniquely_locatable" };
            }
          }
          const at = curContent.indexOf(exactOld);
          if (at < 0) return { ok: false, reason: "edit_hunk_index_missing" };

          const lineStartChar = curContent.lastIndexOf("\n", at - 1) + 1;
          const endAnchor = at + exactOld.length;
          const rawLineEnd = exactOld.endsWith("\n")
            ? endAnchor - 1
            : curContent.indexOf("\n", endAnchor);
          const lineEndChar = rawLineEnd < 0 ? curContent.length : rawLineEnd;

          const prefixBefore = curContent.slice(0, lineStartChar);
          const startLineLocal = prefixBefore === "" ? 1 : prefixBefore.split("\n").length;
          const fullOldBlock = curContent.slice(lineStartChar, lineEndChar);
          if (fullOldBlock.includes(UNREAD_SENTINEL_PREFIX)) {
            return { ok: false, reason: "edit_touches_unread_lines" };
          }
          const oldBlockLineCount = fullOldBlock === "" ? 0 : fullOldBlock.split("\n").length;
          const endLineLocal = startLineLocal + oldBlockLineCount - 1;
          const startText = curLinesLocal[startLineLocal - 1] ?? "";

          const leadingKeep = curContent.slice(lineStartChar, at);
          const trailingKeep = exactOld.endsWith("\n") ? "" : curContent.slice(endAnchor, lineEndChar);
          const fullNewBlock = leadingKeep + (exactOld.endsWith("\n") && exactNew.endsWith("\n") ? exactNew.slice(0, -1) : exactNew) + trailingKeep;

          planned.push({
            startLineLocal,
            endLineLocal,
            startLine: startLineLocal + baseOffset,
            endLine: endLineLocal + baseOffset,
            startText,
            newText: fullNewBlock,
          });
        }

        // If supervised (`train: true`) and any target line is not visible in the current span (e.g.
        // folded into `[State Ledger]`), synthesize a compact supervised `read` first so training
        // teaches clean read-before-edit discipline.
        if (isSup) {
          const checkInputs: LineEditInput[] = planned.map((p) => ({
            startLine: p.startLine,
            endLine: p.endLine,
            startText: p.startText,
            newText: p.newText,
          }));
          const viewViolation = tracker.checkEditAllowed(key, rawPath, beforeVer, checkInputs, totalBefore);
          if (viewViolation) {
            const minTargetLocal = Math.min(...planned.map((p) => p.startLineLocal));
            const maxTargetLocal = Math.max(...planned.map((p) => p.endLineLocal));
            let minKnownLocal = minTargetLocal;
            while (minKnownLocal > 1 && !curLinesLocal[minKnownLocal - 2]!.startsWith(UNREAD_SENTINEL_PREFIX)) {
              minKnownLocal--;
            }
            let maxKnownLocal = maxTargetLocal;
            while (maxKnownLocal < curLinesLocal.length && !curLinesLocal[maxKnownLocal]!.startsWith(UNREAD_SENTINEL_PREFIX)) {
              maxKnownLocal++;
            }
            const desiredLocalFrom = Math.max(1, minTargetLocal - 10);
            const localFrom = Math.max(minKnownLocal, desiredLocalFrom);
            const desiredSpan = Math.min(DEFAULT_LINE_READ_LIMIT, Math.max(40, maxTargetLocal - localFrom + 16));
            const localTo = Math.min(maxKnownLocal, curLinesLocal.length, localFrom + desiredSpan - 1);
            if (localFrom < 1 || localTo < maxTargetLocal) {
              return { ok: false, reason: "post_fold_read_window_unavailable" };
            }
            const synthOffset = baseOffset + localFrom;
            const synthLimit = localTo - localFrom + 1;
            const fullArr = Array.from({ length: totalBefore }, (_, idx) => {
              const lIdx = (idx + 1) - state!.baseLine;
              return lIdx >= 0 && lIdx < curLinesLocal.length ? curLinesLocal[lIdx]! : "";
            });
            const synthWin = renderNumberedWindow(rawPath, fullArr, synthOffset, baseOffset + localTo, true);
            if (synthWin.text.includes(UNREAD_SENTINEL_PREFIX)) {
              return { ok: false, reason: "post_fold_read_sentinel_leak" };
            }
            flushCurrentBatch(expandedSteps.length === 0);
            const synthCallId = `${tc.id}_pfread`;
            const useZh = /[\u4e00-\u9fff]/.test(String(msg.reasoning_content ?? ""));
            expandedSteps.push({
              assistant: {
                role: "assistant",
                content: null,
                reasoning_content: useZh
                  ? `当前上下文尚未展示 ${rawPath} 对应区间的行号内容，先调用 read 从第 ${synthOffset} 行读取以核对准确行号与首行锚点后再修改。`
                  : `The numbered lines of ${rawPath} are not in the active context window. Let me read ${rawPath} starting at line ${synthOffset} first to confirm exact line numbers and start_text before editing.`,
                tool_calls: [{
                  id: synthCallId,
                  type: "function",
                  function: {
                    name: "read",
                    arguments: JSON.stringify({ path: rawPath, offset: synthOffset, limit: synthLimit }),
                  },
                }],
              },
              toolResults: [{ role: "tool", tool_call_id: synthCallId, content: synthWin.text }],
            });
            tracker.recordRead(key, rawPath, beforeVer, synthWin.firstShown, synthWin.lastShown, totalBefore);
            synthesizedPostFoldReads++;
          }
        }

        // Apply all hunks atomically against `curContent`!
        const localInputs: LineEditInput[] = planned.map((p) => ({
          startLine: p.startLineLocal,
          endLine: p.endLineLocal,
          startText: p.startText,
          newText: p.newText,
        }));
        const applied = applyAnchoredLineEdits(curContent, rawPath, localInputs, { selfHeal: false });
        if ("error" in applied) {
          return { ok: false, reason: `line_edit_apply_rejected:${applied.error.slice(0, 60)}` };
        }

        const totalDelta = applied.totalAdded - applied.totalRemoved;
        const nextTotalHint = state.totalLinesHint ? Math.max(1, state.totalLinesHint + totalDelta) : undefined;
        const nextState: VirtualFileState = {
          content: applied.content,
          versionId: nextVersionId(),
          complete: state.complete,
          baseLine: state.baseLine,
          totalLinesHint: nextTotalHint,
        };
        files.set(key, nextState);
        recordTrajectorySnapshot(options.trajectoryHistory, key, nextState);

        // Render summary using global 1-based line numbers
        const globalApplied = baseOffset === 0 && state.complete
          ? applied
          : (() => {
              const paddedBefore = Array.from({ length: totalBefore }, (_, idx) => {
                const lIdx = (idx + 1) - state!.baseLine;
                return lIdx >= 0 && lIdx < applied.before.length ? applied.before[lIdx]! : "";
              });
              const totalAfter = Math.max(1, totalBefore + totalDelta);
              const paddedAfter = Array.from({ length: totalAfter }, (_, idx) => {
                const lIdx = (idx + 1) - state!.baseLine;
                return lIdx >= 0 && lIdx < applied.after.length ? applied.after[lIdx]! : "";
              });
              return {
                ...applied,
                before: paddedBefore,
                after: paddedAfter,
                hunks: applied.hunks.map((h) => ({
                  ...h,
                  requestedStart: h.requestedStart + baseOffset,
                  requestedEnd: h.requestedEnd + baseOffset,
                  startLine: h.startLine + baseOffset,
                  endLine: h.endLine + baseOffset,
                })),
              };
            })();
        const summary = formatBatchLineEditSummary(rawPath, globalApplied);
        if (summary.text.includes(UNREAD_SENTINEL_PREFIX)) {
          return { ok: false, reason: "unread_sentinel_in_edit_summary" };
        }
        tracker.recordBatchEdit(key, rawPath, nextState.versionId, globalApplied, summary.windows);
        convertedEdits += planned.length;

        const wireEdits = [...planned]
          .sort((a, b) => a.startLine - b.startLine)
          .map((p) => ({
            start_line: p.startLine,
            end_line: p.endLine,
            start_text: p.startText,
            new_text: p.newText,
          }));

        // CoT-Action alignment on supervised turns!
        if (isSup) {
          batchReasoning = formatAnchorGroundSentence(rawPath, wireEdits, batchReasoning);
        }

        const newCall = {
          id: tc.id,
          type: "function",
          function: {
            name: "edit",
            arguments: JSON.stringify({ path: rawPath, edits: wireEdits }),
          },
        };
        const newRes: OpenAIMessage = { role: "tool", tool_call_id: tc.id, content: summary.text };
        currentBatchCalls.push(newCall);
        currentBatchResults.push(newRes);
        continue;
      }

      currentBatchCalls.push(structuredClone(tc));
      if (origToolMsg) currentBatchResults.push(structuredClone(origToolMsg));
    }

    if (currentBatchCalls.length > 0) {
      expandedSteps.push({
        assistant: {
          role: "assistant",
          content: expandedSteps.length === 0 ? (msg.content ?? null) : null,
          reasoning_content: expandedSteps.length === 0 ? batchReasoning : batchReasoning,
          tool_calls: currentBatchCalls,
        },
        toolResults: currentBatchResults,
      });
    }

    for (const step of expandedSteps) {
      const aIdx = outMessages.length;
      outMessages.push(step.assistant);
      if (isSup) outSupervised.add(aIdx);
      for (const tr of step.toolResults) outMessages.push(tr);
    }
  }

  return {
    ok: true,
    messages: outMessages,
    tools,
    supervisedIndices: outSupervised,
    convertedEdits,
    convertedReads,
    synthesizedPostFoldReads,
  };
}

function findPriorNumberedWindow(messages: readonly OpenAIMessage[], beforeIdx: number, targetPath: string): string | undefined {
  for (let j = beforeIdx - 1; j >= 0; j--) {
    const m = messages[j]!;
    if (m.role !== "tool" || typeof m.content !== "string") continue;
    for (const line of m.content.split("\n")) {
      const hdr = parseFileHeaderLine(line);
      if (hdr && hdr.displayPath === targetPath) return m.content;
    }
  }
  return undefined;
}

/**
 * Injects an off-policy error-recovery demonstration (`train: false` on the simulated mistake,
 * `train: true` ONLY on the recovery action!) before the first supervised `edit` in a converted
 * trajectory:
 *   - `"unread_first"`: simulates attempting `edit` without active-context read -> `read` -> `edit`.
 *   - `"anchor_mismatch"`: simulates an off-by-N / wrong `start_text` rejection with numbered window -> immediate corrected `edit`.
 *   - `"boundary_dup"`: simulates off-by-1 boundary duplication -> immediate corrected `edit`.
 *   - `"noop_recovery"`: simulates identical `new_text` no-op rejection -> immediate real `edit`.
 */
export function synthesizeRecoveryAugmentation(
  converted: Extract<ConvertTrajectoryResult, { ok: true }>,
  kind: Exclude<RecoveryAugmentKind, "none">,
): { messages: OpenAIMessage[]; supervisedIndices: Set<number> } | undefined {
  const { messages, supervisedIndices } = converted;
  const editIdx = messages.findIndex(
    (m, idx) =>
      supervisedIndices.has(idx)
      && m.role === "assistant"
      && Array.isArray(m.tool_calls)
      && m.tool_calls.length === 1
      && (m.tool_calls[0]?.function?.name ?? m.tool_calls[0]?.name) === "edit",
  );
  if (editIdx < 0) return undefined;

  const targetMsg = messages[editIdx]!;
  const call = targetMsg.tool_calls[0]!;
  const parsedArgs = JSON.parse(call.function.arguments) as {
    path: string;
    edits?: { start_line: number; end_line: number; start_text: string; new_text: string }[];
  };
  const hunks = parsedArgs.edits;
  if (!Array.isArray(hunks) || hunks.length === 0) return undefined;
  const first = hunks[0]!;
  const path = parsedArgs.path;
  const priorWindow = findPriorNumberedWindow(messages, editIdx, path);

  const splice = (injectedUnsup: OpenAIMessage[], injectedSup: OpenAIMessage[]): { messages: OpenAIMessage[]; supervisedIndices: Set<number> } => {
    const outMsgs: OpenAIMessage[] = [];
    const outSup = new Set<number>();
    for (let i = 0; i < messages.length; i++) {
      if (i === editIdx) {
        outMsgs.push(...injectedUnsup);
        for (const m of injectedSup) {
          if (m.role === "assistant") outSup.add(outMsgs.length);
          outMsgs.push(m);
        }
        continue;
      }
      const mappedIdx = outMsgs.length;
      outMsgs.push(structuredClone(messages[i]!));
      if (supervisedIndices.has(i)) outSup.add(mappedIdx);
    }
    return { messages: outMsgs, supervisedIndices: outSup };
  };

  if (kind === "unread_first") {
    if (!priorWindow) return undefined;
    const suggestOffset = Math.max(1, first.start_line - 10);
    const suggestLimit = Math.min(DEFAULT_LINE_READ_LIMIT, Math.max(40, first.end_line - first.start_line + 25));
    const badCallId = `call_sim_unread_${createHash("sha256").update(call.id).digest("hex").slice(0, 8)}`;
    const readCallId = `call_rec_read_${createHash("sha256").update(call.id).digest("hex").slice(0, 8)}`;
    const simAssistant: OpenAIMessage = {
      role: "assistant",
      content: null,
      reasoning_content: "",
      tool_calls: [{ id: badCallId, type: "function", function: { name: "edit", arguments: JSON.stringify({ path, edits: hunks.map((h) => ({ ...h, start_text: "" })) }) } }],
    };
    const simErrorTool: OpenAIMessage = {
      role: "tool",
      tool_call_id: badCallId,
      content: `Cannot edit ${path} before reading its numbered lines in the current context. Call read({"path":${JSON.stringify(path)},"offset":${suggestOffset},"limit":${suggestLimit}}) first to confirm exact line numbers and start_text.`,
    };
    const recReadAssistant: OpenAIMessage = {
      role: "assistant",
      content: null,
      reasoning_content: `The earlier file view for ${path} is not in the active context. Let me call read on ${path} around line ${suggestOffset} first to verify the exact line numbers and start_text before editing.`,
      tool_calls: [{
        id: readCallId,
        type: "function",
        function: { name: "read", arguments: JSON.stringify({ path, offset: suggestOffset, limit: suggestLimit }) },
      }],
    };
    const recReadTool: OpenAIMessage = { role: "tool", tool_call_id: readCallId, content: priorWindow };
    return splice([simAssistant, simErrorTool], [recReadAssistant, recReadTool, structuredClone(targetMsg)]);
  }

  if (kind === "anchor_mismatch") {
    if (!priorWindow || !first.start_text.trim()) return undefined;
    const wrongLine = Math.max(1, first.start_line + (first.start_line > 3 ? -3 : 3));
    const badCallId = `call_sim_anch_${createHash("sha256").update(call.id).digest("hex").slice(0, 8)}`;
    const windowSnippet = priorWindow.split("\n").slice(0, 14).join("\n");
    const simAssistant: OpenAIMessage = {
      role: "assistant",
      content: null,
      reasoning_content: "",
      tool_calls: [{
        id: badCallId,
        type: "function",
        function: {
          name: "edit",
          arguments: JSON.stringify({
            path,
            edits: [{ ...first, start_line: wrongLine, end_line: Math.max(wrongLine, wrongLine + (first.end_line - first.start_line)), start_text: "# wrong anchor" }, ...hunks.slice(1)],
          }),
        },
      }],
    };
    const simErrorTool: OpenAIMessage = {
      role: "tool",
      tool_call_id: badCallId,
      content: `Your edit in ${path} was NOT applied: start_text ("# wrong anchor") did not match line ${wrongLine}.\n\nCurrent lines around line ${wrongLine}:\n${windowSnippet}`,
    };
    const fixedAssistant: OpenAIMessage = {
      ...structuredClone(targetMsg),
      reasoning_content: `The returned numbered window shows the target actually starts at line ${first.start_line} (\`${first.start_text.trim().slice(0, 48)}\`). Let me retry with the exact start_line=${first.start_line}, end_line=${first.end_line}, and start_text.`,
    };
    return splice([simAssistant, simErrorTool], [fixedAssistant]);
  }

  if (kind === "noop_recovery") {
    if (!priorWindow || first.start_line !== first.end_line || !first.start_text.trim()) return undefined;
    const badCallId = `call_sim_noop_${createHash("sha256").update(call.id).digest("hex").slice(0, 8)}`;
    const simAssistant: OpenAIMessage = {
      role: "assistant",
      content: null,
      reasoning_content: "",
      tool_calls: [{
        id: badCallId,
        type: "function",
        function: {
          name: "edit",
          arguments: JSON.stringify({
            path,
            edits: [{ ...first, new_text: first.start_text }, ...hunks.slice(1)],
          }),
        },
      }],
    };
    const simErrorTool: OpenAIMessage = {
      role: "tool",
      tool_call_id: badCallId,
      content: `No edit needed in ${path}: new_text is identical to line ${first.start_line}. Provide the actual code change, or answer directly if all edits are complete.`,
    };
    const fixedAssistant: OpenAIMessage = {
      ...structuredClone(targetMsg),
      reasoning_content: `My previous edit accidentally repeated the unchanged line ${first.start_line}. Let me provide the actual updated replacement in new_text for line ${first.start_line}.`,
    };
    return splice([simAssistant, simErrorTool], [fixedAssistant]);
  }

  // `boundary_dup`: simulate accidental inclusion of line `start_line - 1` when `start_line > 1`
  if (first.start_line <= 1 || !priorWindow) return undefined;
  const linePattern = new RegExp(`^${first.start_line - 1}: (.*)$`, "m");
  const hit = linePattern.exec(priorWindow);
  if (!hit || hit[1]!.trim().length < 6) return undefined;
  const aboveLineText = hit[1]!;
  const windowSnippet = priorWindow.split("\n").slice(0, 12).join("\n");

  const badCallId = `call_sim_dup_${createHash("sha256").update(call.id).digest("hex").slice(0, 8)}`;
  const dupNewText = `${aboveLineText}\n${first.new_text}`;
  const simAssistant: OpenAIMessage = {
    role: "assistant",
    content: null,
    reasoning_content: "",
    tool_calls: [{
      id: badCallId,
      type: "function",
      function: {
        name: "edit",
        arguments: JSON.stringify({
          path,
          edits: [{ ...first, new_text: dupNewText }, ...hunks.slice(1)],
        }),
      },
    }],
  };
  const simErrorTool: OpenAIMessage = {
    role: "tool",
    tool_call_id: badCallId,
    content: `Your edit in ${path} was NOT applied: the first line of new_text duplicates line ${first.start_line - 1} (${JSON.stringify(aboveLineText.trim())}) immediately above start_line=${first.start_line}. Either set start_line=${first.start_line - 1} (with its start_text) or omit that unchanged line from new_text.\n${windowSnippet}`,
  };
  const fixedAssistant: OpenAIMessage = {
    ...structuredClone(targetMsg),
    reasoning_content: `Line ${first.start_line - 1} (${aboveLineText.trim()}) is just above start_line=${first.start_line}, so including it in new_text would duplicate it. Let me omit line ${first.start_line - 1} from new_text and replace only lines ${first.start_line}-${first.end_line}.`,
  };
  return splice([simAssistant, simErrorTool], [fixedAssistant]);
}

/**
 * Packs a structured `(messages, tools, supervisedIndices)` trajectory into a single continuous
 * `rwkv-lightning-segments-eos-target-v2` span (`train: true, end_target: 0` on supervised assistant
 * continuations; `train: false` everywhere else; trailing `<EOD>`).
 */
export function packSupervisedMessagesToSegments(
  messages: readonly OpenAIMessage[],
  tools: readonly any[],
  supervisedIndices: ReadonlySet<number>,
  format: ToolFormat = "xml",
): LightningSegment[] {
  const segments: LightningSegment[] = [];
  let cursorText = "";

  for (let i = 0; i < messages.length; i++) {
    if (messages[i]!.role !== "assistant" || !supervisedIndices.has(i)) continue;
    const prefix = renderPrompt(messages.slice(0, i) as any[], tools as any[], true, true, format);
    const full = renderPrompt(messages.slice(0, i + 1) as any[], tools as any[], true, false, format);
    if (!full.startsWith(prefix) || full.length === prefix.length) {
      throw new Error(`Non-continuation assistant turn at message index ${i}.`);
    }
    if (!prefix.startsWith(cursorText)) {
      throw new Error(`Prefix diverged within single span at message index ${i}.`);
    }
    const untrainedGap = prefix.slice(cursorText.length);
    if (!untrainedGap) {
      throw new Error(`Empty untrained boundary before message index ${i}.`);
    }
    const supervisedCont = full.slice(prefix.length);
    segments.push(
      { text: untrainedGap, train: false },
      { text: supervisedCont, train: true, end_target: 0 },
    );
    cursorText = full;
  }

  if (segments.length === 0) {
    throw new Error("No supervised assistant segments produced.");
  }
  // If there are trailing tool observations after the last supervised decision in a partial epoch span,
  // only emit up to the last supervised assistant decision + `<EOD>`.
  segments.push({ text: "<EOD>", train: false });
  return segments;
}

export function computeTokenLayout(
  segments: readonly LightningSegment[],
  encoded: readonly (readonly number[])[],
): { tokens: number[]; labels: number[]; train: number[] } {
  const tokens: number[] = [];
  const labels: number[] = [];
  const train: number[] = [];
  const overrides: number[] = [];
  for (let i = 0; i < segments.length; i++) {
    const s = segments[i]!;
    const ids = encoded[i];
    if (!ids || ids.length === 0) throw new Error("Empty token segment");
    for (let k = 0; k < ids.length; k++) {
      tokens.push(ids[k]!);
      labels.push(ids[k]!);
      train.push(s.train ? 1 : 0);
    }
    if (s.end_target !== undefined) {
      if (!s.train || s.end_target !== 0) throw new Error("Invalid EOS override");
      overrides.push(tokens.length);
    }
  }
  for (const idx of overrides) {
    if (idx >= tokens.length || train[idx]) throw new Error("EOS override collides with assistant supervision");
    labels[idx] = 0;
    train[idx] = 1;
  }
  return { tokens, labels, train };
}

export function convertPackedSpanRow(
  row: PackedSpanRow,
  options: ConvertTrajectoryOptions & { augmentRecovery?: RecoveryAugmentKind } = {},
): {
  ok: true;
  row: PackedSpanRow;
  convertedEdits: number;
  convertedReads: number;
  synthesizedPostFoldReads: number;
  augmented?: PackedSpanRow;
} | { ok: false; reason: string } {
  if (!object(row) || !Array.isArray(row.segments)) {
    return { ok: false, reason: "invalid_row_schema" };
  }
  let parsed: ParsedTrajectory;
  try {
    parsed = parseXmlContinuationSegments(row.segments);
  } catch (err) {
    return { ok: false, reason: `parse_error:${err instanceof Error ? err.message : String(err)}` };
  }
  const conv = convertTrajectoryToLineEdit(parsed, options);
  if (!conv.ok) return conv;

  let segments: LightningSegment[];
  try {
    segments = packSupervisedMessagesToSegments(conv.messages, conv.tools, conv.supervisedIndices, "xml");
  } catch (err) {
    return { ok: false, reason: `pack_error:${err instanceof Error ? err.message : String(err)}` };
  }

  const baseOut: PackedSpanRow = {
    ...row,
    editMode: "line",
    segments,
  };

  let augmented: PackedSpanRow | undefined;
  if (options.augmentRecovery && options.augmentRecovery !== "none" && conv.convertedEdits > 0) {
    const aug = synthesizeRecoveryAugmentation(conv, options.augmentRecovery);
    if (aug) {
      try {
        const augSegs = packSupervisedMessagesToSegments(aug.messages, conv.tools, aug.supervisedIndices, "xml");
        const augChars = augSegs.reduce((acc, s) => acc + s.text.length, 0);
        if (augChars <= 48_000) {
          augmented = {
            ...row,
            id: `${row.id ?? "span"}_rec_${options.augmentRecovery}`,
            sourceKind: `recovery_${options.augmentRecovery}`,
            editMode: "line",
            segments: augSegs,
          };
        }
      } catch { /* skip augmentation if span exceeds invariant */ }
    }
  }

  return {
    ok: true,
    row: baseOut,
    convertedEdits: conv.convertedEdits,
    convertedReads: conv.convertedReads,
    synthesizedPostFoldReads: conv.synthesizedPostFoldReads,
    ...(augmented ? { augmented } : {}),
  };
}
