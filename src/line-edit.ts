/**
 * Anchored Batch Line-Edit ("v2") for RWKV-7 linear-state agents.
 *
 * Calibrated directly against RWKV-7-G1L-7.2B state-recall measurements:
 *   - Neutral numbered header (`=== <path> (lines X-Y of Z) ===`) avoids triggering SWE-agent
 *     `str_replace_editor` priors baked into base weights.
 *   - Read windows default to 150 lines (hard cap 200 lines) to stay inside RWKV-7's empirically
 *     measured 200-line global line-number fidelity zone.
 *   - Atomic multi-hunk `edits: [{ start_line, end_line, start_text, end_text?, new_text }]`
 *     evaluates all coordinates against the pre-edit file and applies bottom-up in one write,
 *     eliminating mental line-shift arithmetic and redundant round-trips.
 *   - 3-tier self-healing semantic anchor resolution (`start_text` + optional `end_text`):
 *       L1: Exact line verified at `start_line` (whitespace-tolerant / Dice >= 0.90),
 *       L2: Local `±15`-line unique relocation (absorbs RNN ±1..5 function-header offset),
 *       L3: Whole-file unique fallback for non-trivial anchors (absorbs >200-line drift).
 *     Every mismatch is refused BEFORE touching disk and returns the actual numbered window so
 *     recovery happens in a single distance-0 turn.
 *   - Guarded by context-synchronized `LineViewTracker` (with verified-anchor bypass after Epoch
 *     folds), automatic stripping of copied `N: ` prefixes, indent alignment, elision/no-op/
 *     boundary-duplication checks, and pre-write AST syntax validation.
 */
import { createHash } from "node:crypto";
import { posix } from "node:path";
import type { ToolDefinition } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import {
  hostFs,
  introducedEditSyntaxError,
  reindent,
  repairPath,
  similarity,
  withTolerance,
  type WorkspaceFs,
} from "./tolerant-tools.js";

export type EditMode = "line" | "text";

/** Sweet spot inside RWKV-7's 200-line global line-number state fidelity boundary. */
export const DEFAULT_LINE_READ_LIMIT = 150;
export const MAX_LINE_READ_LIMIT = 200;
/** Stays comfortably below `epochMaxResultTokens` (2000 tok) so `slimOversizedToolMessage` never cuts middle lines. */
export const MAX_LINE_READ_CHARS = 6_000;
const RESULT_CONTEXT_LINES = 3;
const ERROR_WINDOW_RADIUS = 8;
const LOCAL_RELOC_RADIUS = 15;

const COPIED_LINE_PREFIX = /^[ \t]*(\d+)(?:\s*[→│|]\s?|\t|: ?)(.*)$/;
const ELISION_LINE = /^\s*(?:(?:#|\/\/|--)\s*(?:\.{3}|…)\s*(?:existing|rest|unchanged|omitted|remaining|other|previous|more|keep)\b.*|\/\*+\s*(?:\.{3}|…).*?\*+\/)\s*$/i;
const TRIVIAL_BOUNDARY = /^(?:[{}()[\];,]+|end|fi|done|else:?|try:?|except\b.*:|finally:?|return|pass|break|continue)$/i;

const sha256 = (text: string): string => createHash("sha256").update(text, "utf8").digest("hex");
const plural = (count: number, word = "line") => `${count} ${word}${count === 1 ? "" : "s"}`;
const indentOf = (line: string): string => /^[ \t]*/.exec(line)?.[0] ?? "";
const spaceKey = (line: string): string => line.replace(/\s+/g, " ").trim();

/** Matches both v2 neutral headers (`=== path (lines X-Y of Z) ===` / `=== path (empty file, 0 lines) ===`) and legacy `[File: path (N lines total)]`. */
export const LINE_FILE_HEADER_RE = /^(?:===\s+(.+?)\s+\((?:lines\s+\d+-\d+\s+of\s+(\d+)|empty file,\s+0 lines)\)\s+===|\[File:\s+(.+?)\s+\((\d+)\s+lines?\s+total\)\])$/gm;

export function parseFileHeaderLine(line: string): { displayPath: string; totalLines: number } | undefined {
  const mNew = /^===\s+(.+?)\s+\(lines\s+\d+-\d+\s+of\s+(\d+)\)\s+===$/.exec(line);
  if (mNew) return { displayPath: mNew[1]!.trim(), totalLines: Number(mNew[2]) };
  const mEmpty = /^===\s+(.+?)\s+\(empty file,\s+0 lines\)\s+===$/.exec(line);
  if (mEmpty) return { displayPath: mEmpty[1]!.trim(), totalLines: 0 };
  const mLegacy = /^\[File:\s+(.+?)\s+\((\d+)\s+lines?\s+total\)\]$/.exec(line);
  if (mLegacy) return { displayPath: mLegacy[1]!.trim(), totalLines: Number(mLegacy[2]) };
  return undefined;
}

export function isNumberedFileOutput(text: string): boolean {
  const firstLine = text.split("\n", 1)[0] ?? "";
  return parseFileHeaderLine(firstLine) !== undefined;
}

export interface SplitLinesResult {
  readonly eol: "\r\n" | "\n";
  readonly trailingNewline: boolean;
  readonly lines: string[];
}

export function splitFileLines(rawText: string): SplitLinesResult {
  const clean = rawText.startsWith("\uFEFF") ? rawText.slice(1) : rawText;
  const eol: "\r\n" | "\n" = clean.includes("\r\n") ? "\r\n" : "\n";
  const normalized = eol === "\r\n" ? clean.replaceAll("\r\n", "\n") : clean;
  const trailingNewline = normalized.endsWith("\n");
  const lines = normalized === ""
    ? []
    : (trailingNewline ? normalized.slice(0, -1) : normalized).split("\n");
  return { eol, trailingNewline, lines };
}

function parseReplacementLines(newText: string): string[] {
  if (newText === "") return [];
  const normalized = newText.replaceAll("\r\n", "\n");
  return (normalized.endsWith("\n") ? normalized.slice(0, -1) : normalized).split("\n");
}

function cleanSingleAnchorLine(raw: string): string {
  const first = raw.replaceAll("\r\n", "\n").split("\n", 1)[0] ?? "";
  const m = COPIED_LINE_PREFIX.exec(first);
  return m ? m[2]! : first;
}

function lineAnchorScore(anchor: string, fileLine: string): number {
  const aTrim = anchor.trim();
  const fTrim = fileLine.trim();
  if (aTrim === "" && fTrim === "") return 1.0;
  if (aTrim === "" || fTrim === "") return 0.0;
  if (aTrim === fTrim) return 1.0;
  const aSpace = spaceKey(anchor);
  const fSpace = spaceKey(fileLine);
  if (aSpace === fSpace) return 0.98;
  return similarity(aSpace, fSpace);
}

export function isNonTrivialAnchor(anchor: string | undefined, minLen = 8): boolean {
  if (!anchor) return false;
  const trimmed = cleanSingleAnchorLine(anchor).trim();
  return trimmed.length >= minLen && !TRIVIAL_BOUNDARY.test(trimmed);
}

function pickUniqueHit<T extends { score: number }>(hits: readonly T[]): T | undefined {
  if (hits.length === 0) return undefined;
  if (hits.length === 1) return hits[0];
  const exact = hits.filter((h) => h.score >= 0.98);
  if (exact.length === 1) return exact[0];
  if (exact.length > 1) return undefined;
  return hits[0]!.score >= hits[1]!.score + 0.05 ? hits[0] : undefined;
}

/** Strips copied `20: ...` prefixes from `new_text` when sequential and absent from original file lines. */
export function stripNumberedReplacement(
  newLines: readonly string[],
  expectedStartLine: number,
  originalLines: readonly string[],
): { lines: string[]; stripped: boolean } {
  if (newLines.length === 0) return { lines: [], stripped: false };
  if (originalLines.some((l) => COPIED_LINE_PREFIX.test(l))) {
    return { lines: [...newLines], stripped: false };
  }
  let expected = -1;
  let numberedCount = 0;
  const out: string[] = [];
  for (const line of newLines) {
    const m = COPIED_LINE_PREFIX.exec(line);
    if (!m) {
      if (line.trim() !== "") return { lines: [...newLines], stripped: false };
      out.push(line);
      if (expected >= 0) expected++;
      continue;
    }
    const num = Number(m[1]);
    if (!Number.isSafeInteger(num) || num < 1) return { lines: [...newLines], stripped: false };
    if (expected < 0) {
      if (newLines.length === 1 && Math.abs(num - expectedStartLine) > 2) {
        return { lines: [...newLines], stripped: false };
      }
      expected = num + 1;
    } else {
      if (num !== expected) return { lines: [...newLines], stripped: false };
      expected++;
    }
    numberedCount++;
    out.push(m[2]!);
  }
  if (numberedCount === 0 || (numberedCount === 1 && newLines.length > 1)) {
    return { lines: [...newLines], stripped: false };
  }
  return { lines: out, stripped: true };
}

export function renderNumberedWindow(
  displayPath: string,
  lines: readonly string[],
  fromLine: number,
  toLine: number,
  truncatedByChars = false,
): { text: string; firstShown: number; lastShown: number } {
  if (lines.length === 0) {
    return { text: `=== ${displayPath} (empty file, 0 lines) ===`, firstShown: 0, lastShown: 0 };
  }
  const first = Math.max(1, Math.min(lines.length, fromLine));
  const targetLast = Math.max(first, Math.min(lines.length, toLine));
  const bodyLines: string[] = [];
  let chars = displayPath.length + 40;
  let actualLast = first - 1;
  for (let n = first; n <= targetLast; n++) {
    const rendered = `${n}: ${lines[n - 1]!}`;
    if (truncatedByChars && actualLast >= first && chars + rendered.length + 1 > MAX_LINE_READ_CHARS) {
      break;
    }
    bodyLines.push(rendered);
    chars += rendered.length + 1;
    actualLast = n;
  }
  const header = `=== ${displayPath} (lines ${first}-${actualLast} of ${lines.length}) ===`;
  const out: string[] = [header];
  if (first > 1) out.push(`(${plural(first - 1)} above)`);
  out.push(...bodyLines);
  if (actualLast < lines.length) {
    const remaining = lines.length - actualLast;
    out.push(`(${plural(remaining)} below; call read with offset=${actualLast + 1} to continue)`);
  }
  return { text: out.join("\n"), firstShown: first, lastShown: actualLast };
}

export interface LineEditInput {
  readonly startLine: number;
  readonly endLine: number;
  readonly startText?: string;
  readonly endText?: string;
  readonly newText: string;
}

export interface ResolvedLineEdit {
  readonly index: number;
  readonly requestedStart: number;
  readonly requestedEnd: number;
  readonly startLine: number;
  readonly endLine: number;
  readonly newLines: readonly string[];
  readonly removed: number;
  readonly added: number;
  readonly strippedLineNumbers: boolean;
  readonly reindented: boolean;
  readonly alignedNote?: string;
  readonly anchorVerified: boolean;
}

export interface AppliedLineEdit {
  readonly content: string;
  readonly before: readonly string[];
  readonly after: readonly string[];
  readonly removed: number;
  readonly added: number;
  readonly strippedLineNumbers: boolean;
  readonly reindented: boolean;
  readonly resolvedStart: number;
  readonly resolvedEnd: number;
  readonly alignedNote?: string;
  readonly anchorVerified: boolean;
}

export interface AppliedBatchLineEdit {
  readonly content: string;
  readonly before: readonly string[];
  readonly after: readonly string[];
  readonly hunks: readonly ResolvedLineEdit[];
  readonly totalRemoved: number;
  readonly totalAdded: number;
}

export interface ApplyLineEditsOptions {
  /** Allow ±15 local and whole-file unique `start_text` self-healing + indent alignment (default true). */
  readonly selfHeal?: boolean;
}

function refineEndLineWithEndText(
  before: readonly string[],
  startLine: number,
  candidateEnd: number,
  endText: string | undefined,
  allowSelfHeal: boolean,
): { endLine: number; ok: boolean; score: number } {
  if (!endText || !endText.trim() || candidateEnd < startLine) {
    return { endLine: candidateEnd, ok: true, score: 1.0 };
  }
  const cleanEnd = cleanSingleAnchorLine(endText);
  const total = before.length;
  if (candidateEnd >= startLine && candidateEnd <= total) {
    const directScore = lineAnchorScore(cleanEnd, before[candidateEnd - 1]!);
    if (directScore >= 0.90) return { endLine: candidateEnd, ok: true, score: directScore };
    if (!allowSelfHeal) return { endLine: candidateEnd, ok: false, score: directScore };
  } else if (!allowSelfHeal) {
    return { endLine: candidateEnd, ok: false, score: 0.0 };
  }
  const lo = Math.max(startLine, candidateEnd - 5);
  const hi = Math.min(total, candidateEnd + 5);
  const hits: { ln: number; score: number; dist: number }[] = [];
  for (let ln = lo; ln <= hi; ln++) {
    const sc = lineAnchorScore(cleanEnd, before[ln - 1]!);
    if (sc >= 0.85) hits.push({ ln, score: sc, dist: Math.abs(ln - candidateEnd) });
  }
  hits.sort((a, b) => (b.score - a.score) || (a.dist - b.dist));
  if (hits.length === 1 || (hits.length > 1 && hits[0]!.score >= hits[1]!.score + 0.05)) {
    return { endLine: hits[0]!.ln, ok: true, score: hits[0]!.score };
  }
  return { endLine: candidateEnd, ok: false, score: hits[0]?.score ?? 0 };
}

function resolveSingleHunk(
  before: readonly string[],
  displayPath: string,
  edit: LineEditInput,
  index: number,
  totalHunks: number,
  allowSelfHeal: boolean,
): ResolvedLineEdit | { error: string } {
  const label = totalHunks === 1 ? "edit" : `edits[${index}]`;
  const { startLine: start, endLine: end } = edit;
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end)) {
    return { error: `${label}: start_line and end_line must be integers.` };
  }
  const total = before.length;
  // Empty file convenience: allow replacing line 1..1 or inserting at 1..0 on a 0-line file.
  const effectiveEnd = total === 0 && start === 1 && end === 1 ? 0 : end;

  if (start < 1) return { error: `${label}: start_line must be >= 1 (got ${start}).` };
  if (start > total + 1) {
    const win = total > 0 ? `\n${renderNumberedWindow(displayPath, before, Math.max(1, total - 10), total).text}` : "";
    return { error: `${label}: start_line ${start} is past the end of ${displayPath} (${plural(total)} total).${win}` };
  }
  if (effectiveEnd > total) {
    const win = total > 0 ? `\n${renderNumberedWindow(displayPath, before, Math.max(1, start - 4), total).text}` : "";
    return { error: `${label}: end_line ${end} is past the end of ${displayPath} (${plural(total)} total).${win}` };
  }
  if (effectiveEnd < start - 1) {
    return {
      error: `${label}: end_line (${end}) is smaller than start_line - 1 (${start - 1}). Use end_line = start_line - 1 to insert before start_line without deleting lines.`,
    };
  }

  const isInsert = effectiveEnd === start - 1;
  const span = effectiveEnd - start; // >= 0 for replace/delete, -1 for insert
  const rawAnchor = typeof edit.startText === "string" ? cleanSingleAnchorLine(edit.startText) : "";
  const hasAnchor = rawAnchor.trim().length > 0;

  let resolvedStart = start;
  let resolvedEnd = effectiveEnd;
  let alignedNote: string | undefined;
  let anchorVerified = false;
  let matchedAnchorLine: string | undefined;

  if (total === 0) {
    anchorVerified = true;
  } else if (hasAnchor) {
    const anchorTargetLine = isInsert ? (start <= total ? start : total) : start;
    const directScore = lineAnchorScore(rawAnchor, before[anchorTargetLine - 1]!);
    if (directScore >= 0.90) {
      matchedAnchorLine = before[anchorTargetLine - 1]!;
      if (!isInsert) {
        const endCheck = refineEndLineWithEndText(before, resolvedStart, resolvedEnd, edit.endText, allowSelfHeal);
        if (!endCheck.ok) {
          const win = renderNumberedWindow(displayPath, before, Math.max(1, resolvedStart - 3), Math.min(total, resolvedEnd + ERROR_WINDOW_RADIUS));
          return {
            error: `Your ${label} in ${displayPath} was NOT applied: end_text (${JSON.stringify(cleanSingleAnchorLine(edit.endText!).trim())}) did not match line ${resolvedEnd} (${JSON.stringify(before[resolvedEnd - 1]?.trim() ?? "")}). Verify exact end_line below:\n${win.text}`,
          };
        }
        if (endCheck.endLine !== resolvedEnd) {
          alignedNote = `Auto-aligned ${label} end_line ${resolvedEnd} -> ${endCheck.endLine} via end_text.`;
          resolvedEnd = endCheck.endLine;
        }
      }
      anchorVerified = true;
    } else if (allowSelfHeal) {
      // Tier 2: Local ±15-line unique self-healing relocation (absorbs RWKV ±1..5 function-header offset).
      const lo = Math.max(1, anchorTargetLine - LOCAL_RELOC_RADIUS);
      const hi = Math.min(total, anchorTargetLine + LOCAL_RELOC_RADIUS);
      const localHits: { ln: number; endLn: number; score: number; dist: number }[] = [];
      for (let ln = lo; ln <= hi; ln++) {
        const sc = lineAnchorScore(rawAnchor, before[ln - 1]!);
        if (sc < 0.85) continue;
        if (isInsert) {
          const candStart = start === total + 1 ? ln + 1 : ln;
          localHits.push({ ln: candStart, endLn: candStart - 1, score: sc, dist: Math.abs(ln - anchorTargetLine) });
        } else {
          const candEnd = ln + span;
          if (candEnd > total) continue;
          const endCheck = refineEndLineWithEndText(before, ln, candEnd, edit.endText, true);
          if (!endCheck.ok) continue;
          localHits.push({ ln, endLn: endCheck.endLine, score: (sc + endCheck.score) / 2, dist: Math.abs(ln - anchorTargetLine) });
        }
      }
      localHits.sort((a, b) => (b.score - a.score) || (a.dist - b.dist));
      const localBest = pickUniqueHit(localHits);
      const localExactCount = localHits.filter((h) => h.score >= 0.98).length;

      if (localBest) {
        resolvedStart = localBest.ln;
        resolvedEnd = localBest.endLn;
        matchedAnchorLine = before[(isInsert && resolvedStart > total ? total : resolvedStart) - 1]!;
        anchorVerified = true;
        alignedNote = `Auto-aligned ${label} lines ${start}-${effectiveEnd} -> ${resolvedStart}-${resolvedEnd} via start_text.`;
      } else if (localExactCount > 1) {
        const win = renderNumberedWindow(displayPath, before, Math.max(1, start - ERROR_WINDOW_RADIUS), Math.min(total, Math.max(effectiveEnd, start) + ERROR_WINDOW_RADIUS));
        const locs = localHits.filter((h) => h.score >= 0.98).slice(0, 5).map((h) => `line ${h.ln}`).join(", ");
        return {
          error: `Your ${label} in ${displayPath} was NOT applied: start_text (${JSON.stringify(rawAnchor.trim())}) did not match line ${anchorTargetLine} and matched multiple nearby lines (${locs}). Check exact line numbers below and retry:\n${win.text}`,
        };
      } else {
        // Tier 3: Whole-file unique fallback for non-trivial anchors (absorbs >200-line global drift).
        const fullHits: { ln: number; endLn: number; score: number }[] = [];
        if (isNonTrivialAnchor(rawAnchor, 10)) {
          for (let ln = 1; ln <= total; ln++) {
            const sc = lineAnchorScore(rawAnchor, before[ln - 1]!);
            if (sc < 0.90) continue;
            if (isInsert) {
              const candStart = start === total + 1 ? ln + 1 : ln;
              fullHits.push({ ln: candStart, endLn: candStart - 1, score: sc });
            } else {
              const candEnd = ln + span;
              if (candEnd > total) continue;
              const endCheck = refineEndLineWithEndText(before, ln, candEnd, edit.endText, true);
              if (!endCheck.ok) continue;
              fullHits.push({ ln, endLn: endCheck.endLine, score: (sc + endCheck.score) / 2 });
            }
          }
          fullHits.sort((a, b) => b.score - a.score);
        }
        const fullBest = pickUniqueHit(fullHits);
        if (fullBest) {
          resolvedStart = fullBest.ln;
          resolvedEnd = fullBest.endLn;
          matchedAnchorLine = before[(isInsert && resolvedStart > total ? total : resolvedStart) - 1]!;
          anchorVerified = true;
          alignedNote = `Auto-aligned ${label} lines ${start}-${effectiveEnd} -> ${resolvedStart}-${resolvedEnd} via unique start_text in file.`;
        } else {
          const win = renderNumberedWindow(displayPath, before, Math.max(1, start - ERROR_WINDOW_RADIUS), Math.min(total, Math.max(effectiveEnd, start) + ERROR_WINDOW_RADIUS));
          const extraMatches: string[] = [];
          for (let ln = 1; ln <= total; ln++) {
            if (lineAnchorScore(rawAnchor, before[ln - 1]!) >= 0.80) {
              extraMatches.push(`  ${ln}: ${before[ln - 1]!}`);
              if (extraMatches.length >= 4) break;
            }
          }
          const candHint = extraMatches.length > 0
            ? `\nMatching lines elsewhere in ${displayPath}:\n${extraMatches.join("\n")}`
            : "";
          return {
            error: `Your ${label} in ${displayPath} was NOT applied: start_text (${JSON.stringify(rawAnchor.trim())}) did not match line ${anchorTargetLine} (${JSON.stringify(before[anchorTargetLine - 1]?.trim() ?? "")}).${candHint}\n\nCurrent lines around line ${anchorTargetLine}:\n${win.text}`,
          };
        }
      }
    } else {
      const win = renderNumberedWindow(displayPath, before, Math.max(1, start - ERROR_WINDOW_RADIUS), Math.min(total, Math.max(effectiveEnd, start) + ERROR_WINDOW_RADIUS));
      return {
        error: `Your ${label} in ${displayPath} was NOT applied: start_text (${JSON.stringify(rawAnchor.trim())}) did not match line ${anchorTargetLine} (${JSON.stringify(before[anchorTargetLine - 1]?.trim() ?? "")}).\n${win.text}`,
      };
    }
  } else if (!isInsert && edit.endText && edit.endText.trim()) {
    const endCheck = refineEndLineWithEndText(before, resolvedStart, resolvedEnd, edit.endText, allowSelfHeal);
    if (!endCheck.ok) {
      const win = renderNumberedWindow(displayPath, before, Math.max(1, resolvedStart - 3), Math.min(total, resolvedEnd + ERROR_WINDOW_RADIUS));
      return {
        error: `Your ${label} in ${displayPath} was NOT applied: end_text did not match line ${resolvedEnd}.\n${win.text}`,
      };
    }
    resolvedEnd = endCheck.endLine;
  }

  const rawNewLines = parseReplacementLines(edit.newText);
  const { lines: strippedLines, stripped: strippedLineNumbers } = stripNumberedReplacement(rawNewLines, resolvedStart, before);
  let newLines = strippedLines;
  let reindented = false;

  // Auto-align base indentation if `start_text` drifted in leading indentation relative to the matched file line.
  if (allowSelfHeal && matchedAnchorLine !== undefined && newLines.length > 0 && hasAnchor) {
    const oldIndent = indentOf(rawAnchor);
    const fileIndent = indentOf(matchedAnchorLine);
    const firstNonBlank = newLines.find((l) => l.trim() !== "");
    if (
      oldIndent !== fileIndent
      && firstNonBlank !== undefined
      && ((oldIndent.length > 0 && firstNonBlank.startsWith(oldIndent)) || (oldIndent.length === 0 && indentOf(firstNonBlank) === ""))
    ) {
      const moved = reindent(newLines.join("\n"), [[oldIndent, fileIndent]]).split("\n");
      if (moved.join("\n") !== newLines.join("\n")) {
        newLines = moved;
        reindented = true;
      }
    }
  }

  const removedSlice = before.slice(resolvedStart - 1, resolvedEnd);
  const oldTrimmedSet = new Set(removedSlice.map((l) => l.trim()));
  for (const nl of newLines) {
    if (ELISION_LINE.test(nl) && !oldTrimmedSet.has(nl.trim())) {
      return {
        error: `Your ${label} in ${displayPath} was NOT applied: new_text contains an elision placeholder (${JSON.stringify(nl.trim())}), which would delete code. Narrow start_line..end_line to only the lines changing and write complete replacement code.`,
      };
    }
  }

  if (
    removedSlice.length === newLines.length
    && removedSlice.every((line, idx) => line === newLines[idx])
  ) {
    const spanStr = resolvedStart === resolvedEnd ? `line ${resolvedStart}` : `lines ${resolvedStart}-${resolvedEnd}`;
    return {
      error: `No edit needed in ${displayPath}: new_text is identical to ${spanStr}. Provide the actual code change, or answer directly if all edits are complete.`,
    };
  }

  // Off-by-one boundary duplication guard: catches accidental inclusion of unchanged line start-1 or end+1.
  if (newLines.length >= 2 && resolvedStart > 1) {
    const above = before[resolvedStart - 2]!.trim();
    const firstReplaced = before[resolvedStart - 1]?.trim() ?? "";
    if (above.length >= 6 && !TRIVIAL_BOUNDARY.test(above) && newLines[0]!.trim() === above && firstReplaced !== above) {
      const win = renderNumberedWindow(displayPath, before, resolvedStart - 3, Math.min(total, resolvedEnd + 3));
      return {
        error: `Your ${label} in ${displayPath} was NOT applied: the first line of new_text duplicates line ${resolvedStart - 1} (${JSON.stringify(above)}) immediately above start_line=${resolvedStart}. Either set start_line=${resolvedStart - 1} (with its start_text) or omit that unchanged line from new_text.\n${win.text}`,
      };
    }
  }
  if (newLines.length >= 2 && resolvedEnd < total) {
    const below = before[resolvedEnd]!.trim();
    const lastReplaced = resolvedEnd >= resolvedStart ? (before[resolvedEnd - 1]?.trim() ?? "") : "";
    if (below.length >= 6 && !TRIVIAL_BOUNDARY.test(below) && newLines.at(-1)!.trim() === below && lastReplaced !== below) {
      const win = renderNumberedWindow(displayPath, before, Math.max(1, resolvedStart - 3), Math.min(total, resolvedEnd + 3));
      return {
        error: `Your ${label} in ${displayPath} was NOT applied: the last line of new_text duplicates line ${resolvedEnd + 1} (${JSON.stringify(below)}) immediately below end_line=${resolvedEnd}. Either set end_line=${resolvedEnd + 1} or omit that unchanged line from new_text.\n${win.text}`,
      };
    }
  }

  return {
    index,
    requestedStart: start,
    requestedEnd: effectiveEnd,
    startLine: resolvedStart,
    endLine: resolvedEnd,
    newLines,
    removed: removedSlice.length,
    added: newLines.length,
    strippedLineNumbers,
    reindented,
    ...(alignedNote ? { alignedNote } : {}),
    anchorVerified,
  };
}

/**
 * Resolves and applies one or more non-overlapping anchored line edits atomically (evaluated
 * against the original pre-edit file and applied bottom-up).
 */
export function applyAnchoredLineEdits(
  rawText: string,
  displayPath: string,
  edits: readonly LineEditInput[],
  options: ApplyLineEditsOptions = {},
): AppliedBatchLineEdit | { error: string } {
  if (!edits.length) {
    return { error: `edit requires at least one entry in edits[] for ${displayPath}.` };
  }
  const allowSelfHeal = options.selfHeal !== false;
  const { eol, trailingNewline, lines: before } = splitFileLines(rawText);
  const resolved: ResolvedLineEdit[] = [];

  for (let i = 0; i < edits.length; i++) {
    const r = resolveSingleHunk(before, displayPath, edits[i]!, i, edits.length, allowSelfHeal);
    if ("error" in r) return r;
    resolved.push(r);
  }

  // Check for overlapping ranges on the original pre-edit coordinates.
  const asc = [...resolved].sort((a, b) => (a.startLine - b.startLine) || (a.endLine - b.endLine));
  for (let i = 1; i < asc.length; i++) {
    const prev = asc[i - 1]!;
    const cur = asc[i]!;
    const prevIsInsert = prev.endLine === prev.startLine - 1;
    const curIsInsert = cur.endLine === cur.startLine - 1;
    const overlaps = prevIsInsert
      ? (curIsInsert && cur.startLine === prev.startLine)
      : cur.startLine <= prev.endLine;
    if (overlaps) {
      return {
        error: `Your edits in ${displayPath} were NOT applied: edits[${prev.index}] (lines ${prev.startLine}-${prev.endLine}) and edits[${cur.index}] (lines ${cur.startLine}-${cur.endLine}) overlap. Merge overlapping edits into a single range.`,
      };
    }
  }

  // Apply bottom-up so earlier hunks' indices remain invariant.
  const after = [...before];
  for (const h of [...asc].reverse()) {
    after.splice(h.startLine - 1, h.removed, ...h.newLines);
  }

  const keepTrailingNewline = trailingNewline || (before.length === 0 && after.length > 0);
  const joined = after.join("\n") + (keepTrailingNewline && after.length > 0 ? "\n" : "");
  const content = eol === "\r\n" ? joined.replaceAll("\n", "\r\n") : joined;

  return {
    content,
    before,
    after,
    hunks: asc,
    totalRemoved: asc.reduce((s, h) => s + h.removed, 0),
    totalAdded: asc.reduce((s, h) => s + h.added, 0),
  };
}

/** Single-hunk convenience wrapper around `applyAnchoredLineEdits`. */
export function applyLineEdit(
  rawText: string,
  displayPath: string,
  edit: LineEditInput,
  options: ApplyLineEditsOptions = {},
): AppliedLineEdit | { error: string } {
  const batch = applyAnchoredLineEdits(rawText, displayPath, [edit], options);
  if ("error" in batch) return batch;
  const h = batch.hunks[0]!;
  return {
    content: batch.content,
    before: batch.before,
    after: batch.after,
    removed: h.removed,
    added: h.added,
    strippedLineNumbers: h.strippedLineNumbers,
    reindented: h.reindented,
    resolvedStart: h.startLine,
    resolvedEnd: h.endLine,
    ...(h.alignedNote ? { alignedNote: h.alignedNote } : {}),
    anchorVerified: h.anchorVerified,
  };
}

export function formatBatchLineEditSummary(
  displayPath: string,
  result: AppliedBatchLineEdit,
): { text: string; windows: { firstShown: number; lastShown: number }[]; windowFirst: number; windowLast: number } {
  const notes: string[] = [];
  const postRanges: { from: number; to: number }[] = [];
  let cumulativeDelta = 0;

  for (const h of result.hunks) {
    const postStart = h.startLine + cumulativeDelta;
    let action: string;
    if (h.removed === 0) {
      action = h.startLine > 1
        ? `Inserted ${plural(h.added)} after line ${h.startLine - 1}`
        : `Inserted ${plural(h.added)} at the start`;
    } else if (h.added === 0) {
      action = h.startLine === h.endLine ? `Deleted line ${h.startLine}` : `Deleted lines ${h.startLine}-${h.endLine}`;
    } else {
      const span = h.startLine === h.endLine ? `line ${h.startLine}` : `lines ${h.startLine}-${h.endLine}`;
      action = `Replaced ${span} with ${plural(h.added)}`;
    }
    notes.push(action + ".");
    if (h.alignedNote) notes.push(`(${h.alignedNote})`);
    if (h.strippedLineNumbers) notes.push("(Stripped copied line-number prefixes from new_text.)");
    if (h.reindented) notes.push("(Re-indented replacement to match file.)");

    const winFrom = Math.max(1, postStart - RESULT_CONTEXT_LINES);
    const winTo = Math.min(
      Math.max(1, result.after.length),
      postStart + Math.max(h.added, 1) - 1 + RESULT_CONTEXT_LINES,
    );
    postRanges.push({ from: winFrom, to: winTo });
    cumulativeDelta += h.added - h.removed;
  }

  const lastHunk = result.hunks[result.hunks.length - 1]!;
  if (cumulativeDelta !== 0 && lastHunk.endLine < result.before.length) {
    notes.push(
      `Lines after line ${lastHunk.endLine} moved ${cumulativeDelta > 0 ? "down" : "up"} by ${Math.abs(cumulativeDelta)}: old line ${lastHunk.endLine + 1} is now line ${lastHunk.endLine + 1 + cumulativeDelta}.`,
    );
  }

  // Merge adjacent/overlapping post-edit context windows.
  const merged: { from: number; to: number }[] = [];
  for (const r of postRanges) {
    const prev = merged[merged.length - 1];
    if (prev && r.from <= prev.to + 2) prev.to = Math.max(prev.to, r.to);
    else merged.push({ ...r });
  }

  const windows: { firstShown: number; lastShown: number }[] = [];
  const renderedBlocks: string[] = [];
  for (const m of merged.slice(0, 4)) {
    const w = renderNumberedWindow(displayPath, result.after, m.from, m.to);
    windows.push({ firstShown: w.firstShown, lastShown: w.lastShown });
    renderedBlocks.push(w.text);
  }
  if (merged.length > 4) {
    renderedBlocks.push(`(${merged.length - 4} more edited region(s) omitted)`);
  }

  return {
    text: `File updated. ${notes.join(" ")}\n${renderedBlocks.join("\n\n")}`,
    windows,
    windowFirst: windows[0]?.firstShown ?? 0,
    windowLast: windows[windows.length - 1]?.lastShown ?? 0,
  };
}

export function formatLineEditSummary(
  displayPath: string,
  edit: LineEditInput,
  result: AppliedLineEdit,
): { text: string; windowFirst: number; windowLast: number } {
  const h: ResolvedLineEdit = {
    index: 0,
    requestedStart: edit.startLine,
    requestedEnd: edit.endLine,
    startLine: result.resolvedStart,
    endLine: result.resolvedEnd,
    newLines: [],
    removed: result.removed,
    added: result.added,
    strippedLineNumbers: result.strippedLineNumbers,
    reindented: result.reindented,
    ...(result.alignedNote ? { alignedNote: result.alignedNote } : {}),
    anchorVerified: result.anchorVerified,
  };
  const s = formatBatchLineEditSummary(displayPath, {
    content: result.content,
    before: result.before,
    after: result.after,
    hunks: [h],
    totalRemoved: result.removed,
    totalAdded: result.added,
  });
  return { text: s.text, windowFirst: s.windowFirst, windowLast: s.windowLast };
}

export function formatBatchLineEditSyntaxRejection(
  displayPath: string,
  result: AppliedBatchLineEdit,
  syntaxError: string,
): string {
  const first = result.hunks[0]!;
  const last = result.hunks[result.hunks.length - 1]!;
  const totalDelta = result.totalAdded - result.totalRemoved;
  const previewFrom = Math.max(1, first.startLine - RESULT_CONTEXT_LINES);
  const previewTo = Math.min(
    Math.max(1, result.after.length),
    Math.max(last.startLine + totalDelta, first.startLine) + Math.max(last.added, 1) - 1 + RESULT_CONTEXT_LINES,
  );
  const origTo = Math.min(
    Math.max(1, result.before.length),
    Math.max(last.endLine, first.startLine) + RESULT_CONTEXT_LINES,
  );
  const wouldLook = renderNumberedWindow(displayPath, result.after, previewFrom, Math.min(previewTo, previewFrom + 40)).text;
  const original = renderNumberedWindow(displayPath, result.before, previewFrom, Math.min(origTo, previewFrom + 40)).text;
  return [
    `Your edit in ${displayPath} was NOT applied: it introduces a syntax error (${syntaxError}).`,
    "",
    "How the file would have looked:",
    wouldLook,
    "",
    "Current file lines before your edit:",
    original,
    "",
    "Fix start_line/end_line, start_text, or new_text (check indentation and brackets) and try again.",
  ].join("\n");
}

export function formatLineEditSyntaxRejection(
  displayPath: string,
  edit: LineEditInput,
  result: AppliedLineEdit,
  syntaxError: string,
): string {
  return formatBatchLineEditSyntaxRejection(
    displayPath,
    {
      content: result.content,
      before: result.before,
      after: result.after,
      hunks: [{
        index: 0,
        requestedStart: edit.startLine,
        requestedEnd: edit.endLine,
        startLine: result.resolvedStart,
        endLine: result.resolvedEnd,
        newLines: [],
        removed: result.removed,
        added: result.added,
        strippedLineNumbers: result.strippedLineNumbers,
        reindented: result.reindented,
        anchorVerified: result.anchorVerified,
      }],
      totalRemoved: result.removed,
      totalAdded: result.added,
    },
    syntaxError,
  );
}

// ---------------------------------------------------------------------------------------------
// Context-Synchronized Seen-Lines Tracker (Exact Line-Number Synchronization)

interface TrackedFileView {
  sha256: string;
  seen: Set<number> | "all";
  displayPath: string;
  totalLines: number;
}

/**
 * Tracks which 1-based line numbers of each file's *current* content hash have been shown to the
 * model, and accurately recomputes visible lines from the post-fold active prompt window.
 */
export class LineViewTracker {
  private readonly files = new Map<string, TrackedFileView>();

  recordRead(canonicalPath: string, displayPath: string, contentSha: string, first: number, last: number, total: number): void {
    if (total === 0 || (first <= 1 && last >= total)) {
      this.files.set(canonicalPath, { sha256: contentSha, seen: "all", displayPath, totalLines: total });
      return;
    }
    const existing = this.files.get(canonicalPath);
    const base = existing && existing.sha256 === contentSha
      ? (existing.seen === "all" ? ("all" as const) : new Set(existing.seen))
      : new Set<number>();
    if (base !== "all") {
      for (let n = Math.max(1, first); n <= last; n++) base.add(n);
    }
    this.files.set(canonicalPath, { sha256: contentSha, seen: base, displayPath, totalLines: total });
  }

  recordWrite(canonicalPath: string, displayPath: string, contentSha: string, totalLines = 0): void {
    this.files.set(canonicalPath, { sha256: contentSha, seen: "all", displayPath, totalLines });
  }

  recordBatchEdit(
    canonicalPath: string,
    displayPath: string,
    newContentSha: string,
    result: AppliedBatchLineEdit,
    windows: readonly { firstShown: number; lastShown: number }[],
  ): void {
    const prev = this.files.get(canonicalPath);
    if (prev?.seen === "all") {
      this.files.set(canonicalPath, { sha256: newContentSha, seen: "all", displayPath, totalLines: result.after.length });
      return;
    }
    const next = new Set<number>();
    if (prev) {
      for (const origLine of prev.seen) {
        let delta = 0;
        let deleted = false;
        for (const h of result.hunks) {
          if (origLine >= h.startLine && origLine <= h.endLine) {
            deleted = true;
            break;
          }
          if (origLine > h.endLine) {
            delta += h.added - h.removed;
          }
        }
        if (!deleted && origLine + delta >= 1) next.add(origLine + delta);
      }
    }
    for (const w of windows) {
      for (let n = Math.max(1, w.firstShown); n <= w.lastShown; n++) next.add(n);
    }
    this.files.set(canonicalPath, { sha256: newContentSha, seen: next, displayPath, totalLines: result.after.length });
  }

  recordEdit(
    canonicalPath: string,
    displayPath: string,
    newContentSha: string,
    edit: LineEditInput,
    result: AppliedLineEdit,
    windowFirst: number,
    windowLast: number,
  ): void {
    this.recordBatchEdit(
      canonicalPath,
      displayPath,
      newContentSha,
      {
        content: result.content,
        before: result.before,
        after: result.after,
        hunks: [{
          index: 0,
          requestedStart: edit.startLine,
          requestedEnd: edit.endLine,
          startLine: result.resolvedStart,
          endLine: result.resolvedEnd,
          newLines: [],
          removed: result.removed,
          added: result.added,
          strippedLineNumbers: result.strippedLineNumbers,
          reindented: result.reindented,
          anchorVerified: result.anchorVerified,
        }],
        totalRemoved: result.removed,
        totalAdded: result.added,
      },
      [{ firstShown: windowFirst, lastShown: windowLast }],
    );
  }

  checkEditAllowed(
    canonicalPath: string,
    displayPath: string,
    currentSha: string,
    edits: LineEditInput | readonly LineEditInput[],
    totalLines: number,
    options?: { readonly allAnchorsVerified?: boolean },
  ): string | undefined {
    if (totalLines === 0) return undefined;
    const list = Array.isArray(edits) ? edits : [edits];
    const firstEdit = list[0];
    if (!firstEdit) return undefined;

    // When EVERY hunk provides a non-trivial `start_text` that has already been uniquely verified
    // against the current file on disk, allow the edit even if Epoch folded the earlier read window.
    if (options?.allAnchorsVerified && list.every((e) => isNonTrivialAnchor(e.startText))) {
      return undefined;
    }

    const entry = this.files.get(canonicalPath);
    const minStart = Math.min(...list.map((e) => e.startLine));
    const maxEnd = Math.max(...list.map((e) => e.endLine));
    const suggestOffset = Math.max(1, minStart - 10);
    const suggestLimit = Math.min(DEFAULT_LINE_READ_LIMIT, Math.max(40, maxEnd - minStart + 25));

    if (!entry) {
      return `Cannot edit ${displayPath} before reading its numbered lines in the current context. Call read({"path":${JSON.stringify(displayPath)},"offset":${suggestOffset},"limit":${suggestLimit}}) first to confirm exact line numbers and start_text.`;
    }
    if (entry.sha256 !== currentSha) {
      this.files.delete(canonicalPath);
      return `Cannot edit ${displayPath}: the file was modified since you last read it, so line numbers may have shifted. Call read({"path":${JSON.stringify(displayPath)},"offset":${suggestOffset},"limit":${suggestLimit}}) first to see current line numbers.`;
    }
    if (entry.seen === "all") return undefined;

    for (const edit of list) {
      const { startLine: start, endLine: end } = edit;
      if (end >= start) {
        for (let n = start; n <= end; n++) {
          if (!entry.seen.has(n)) {
            const off = Math.max(1, start - 10);
            const lim = Math.min(DEFAULT_LINE_READ_LIMIT, Math.max(40, end - start + 25));
            return `Cannot edit ${displayPath} lines ${start}-${end}: line ${n} has not been read yet. Call read({"path":${JSON.stringify(displayPath)},"offset":${off},"limit":${lim}}) first to inspect lines ${start}-${end}.`;
          }
        }
      } else {
        const neighbors = [start - 1, start].filter((n) => n >= 1 && n <= totalLines);
        if (neighbors.length > 0 && !neighbors.some((n) => entry.seen === "all" || entry.seen.has(n))) {
          const off = Math.max(1, start - 10);
          return `Cannot insert at line ${start} of ${displayPath} without reading the surrounding lines first. Call read({"path":${JSON.stringify(displayPath)},"offset":${off},"limit":40}) first.`;
        }
      }
    }
    return undefined;
  }

  /**
   * Synchronizes tracked views with the post-fold OpenAI `messages` array by parsing the exact
   * visible line numbers still present in active tool messages (not just the file header), so a
   * folded 200-line read never falsely leaves `seen = "all"` when only a `±3` edit snippet remains.
   */
  syncWithActiveMessages(messages: readonly Record<string, any>[]): void {
    if (this.files.size === 0) return;
    const visibleByPath = new Map<string, Set<number> | "all">();
    const markAll = (raw: string) => {
      const clean = raw.trim().replace(/^\.\//, "");
      if (clean) visibleByPath.set(clean, "all");
    };
    const markLine = (raw: string, ln: number, total: number) => {
      const clean = raw.trim().replace(/^\.\//, "");
      if (!clean) return;
      const cur = visibleByPath.get(clean);
      if (cur === "all") return;
      const set = cur ?? new Set<number>();
      set.add(ln);
      if (total > 0 && set.size >= total) visibleByPath.set(clean, "all");
      else visibleByPath.set(clean, set);
    };

    for (const msg of messages) {
      if (msg?.role === "assistant" && Array.isArray(msg.tool_calls)) {
        for (const tc of msg.tool_calls) {
          const fn = tc?.function ?? tc;
          if (fn?.name === "write") {
            try {
              const args = typeof fn.arguments === "string" ? JSON.parse(fn.arguments) : fn.arguments;
              if (typeof args?.path === "string") markAll(args.path);
            } catch { /* ignore */ }
          }
        }
      }
      if (msg?.role !== "tool") continue;
      const text = typeof msg.content === "string"
        ? msg.content
        : Array.isArray(msg.content) ? msg.content.map((p: any) => (typeof p?.text === "string" ? p.text : "")).join("\n") : "";
      let activePath: string | undefined;
      let activeTotal = 0;
      for (const line of text.split("\n")) {
        const hdr = parseFileHeaderLine(line);
        if (hdr) {
          activePath = hdr.displayPath;
          activeTotal = hdr.totalLines;
          if (activeTotal === 0) markAll(activePath);
          continue;
        }
        if (!activePath) continue;
        const mLine = /^(\d+): /.exec(line);
        if (mLine) {
          markLine(activePath, Number(mLine[1]), activeTotal);
        } else if (line.trim() === "") {
          activePath = undefined;
        }
      }
    }

    for (const [canonicalPath, info] of [...this.files.entries()]) {
      const disp = info.displayPath.replace(/^\.\//, "");
      let matchedView: Set<number> | "all" | undefined;
      for (const [visPath, visLines] of visibleByPath.entries()) {
        if (visPath === disp || canonicalPath === visPath || canonicalPath.endsWith(`/${visPath}`) || disp.endsWith(`/${visPath}`)) {
          if (visLines === "all" || matchedView === "all") {
            matchedView = "all";
          } else {
            const merged = matchedView ? new Set(matchedView) : new Set<number>();
            for (const n of visLines) merged.add(n);
            matchedView = merged;
          }
        }
      }
      if (!matchedView || (matchedView !== "all" && matchedView.size === 0)) {
        this.files.delete(canonicalPath);
      } else {
        info.seen = matchedView === "all" || (info.totalLines > 0 && matchedView.size >= info.totalLines)
          ? "all"
          : matchedView;
      }
    }
  }

  /**
   * Seeds tracked views from pre-existing Pi native or OpenAI messages (e.g. when a teacher session
   * resumes from an RWKV student's workspace and message prefix in DAgger rollout).
   */
  async seedFromMessages(
    messages: readonly Record<string, any>[],
    cwd: string,
    readText: (absPath: string) => Promise<string>,
  ): Promise<void> {
    const sha = (s: string) => createHash("sha256").update(s).digest("hex");
    const resolveAbs = (raw: string) => {
      const clean = raw.trim().replace(/^@/, "");
      return clean.startsWith("/") ? clean : posix.resolve(cwd, clean);
    };
    for (const msg of messages) {
      if (msg?.role === "assistant") {
        const blocks = Array.isArray(msg.content) ? msg.content : [];
        const tcs = Array.isArray(msg.tool_calls) ? msg.tool_calls : [];
        for (const b of blocks) {
          if (b?.type === "toolCall" && b.name === "write" && typeof b.arguments?.path === "string") {
            const disp = b.arguments.path;
            const abs = resolveAbs(disp);
            try {
              const disk = await readText(abs);
              if (typeof b.arguments.content === "string" && disk === b.arguments.content) {
                this.recordWrite(abs, disp, sha(disk), splitFileLines(disk).lines.length);
              }
            } catch { /* file missing or unreadable */ }
          }
        }
        for (const tc of tcs) {
          const fn = tc?.function ?? tc;
          if (fn?.name === "write") {
            try {
              const args = typeof fn.arguments === "string" ? JSON.parse(fn.arguments) : fn.arguments;
              if (typeof args?.path === "string") {
                const abs = resolveAbs(args.path);
                const disk = await readText(abs);
                if (typeof args.content === "string" && disk === args.content) {
                  this.recordWrite(abs, args.path, sha(disk), splitFileLines(disk).lines.length);
                }
              }
            } catch { /* ignore */ }
          }
        }
      }
      if (msg?.role !== "toolResult" && msg?.role !== "tool") continue;
      if (msg?.isError) continue;
      const text = typeof msg.content === "string"
        ? msg.content
        : Array.isArray(msg.content) ? msg.content.map((p: any) => (typeof p?.text === "string" ? p.text : "")).join("\n") : "";
      const lines = text.split("\n");
      for (let i = 0; i < lines.length; i++) {
        const hdr = parseFileHeaderLine(lines[i]!);
        if (!hdr) continue;
        const disp = hdr.displayPath;
        const abs = resolveAbs(disp);
        try {
          const disk = await readText(abs);
          const diskLines = splitFileLines(disk).lines;
          const numbered: { ln: number; body: string }[] = [];
          for (let j = i + 1; j < lines.length; j++) {
            const l = lines[j]!;
            if (parseFileHeaderLine(l)) break;
            if (/^\(\d+ lines? (?:above|below)/.test(l)) continue;
            const m = /^(\d+): (.*)$/.exec(l);
            if (!m) break;
            numbered.push({ ln: Number(m[1]), body: m[2]! });
          }
          if (numbered.length > 0 && numbered.every(({ ln, body }) => diskLines[ln - 1] === body)) {
            this.recordRead(
              abs,
              disp,
              sha(disk),
              numbered[0]!.ln,
              numbered[numbered.length - 1]!.ln,
              diskLines.length,
            );
          }
        } catch { /* file missing */ }
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Tool Definitions & Canonical Schemas

export const LINE_READ_DESCRIPTION =
  `Read a UTF-8 text file (or image) in the workspace with 1-based line numbers (\`1: ...\`). Default limit is ${DEFAULT_LINE_READ_LIMIT} lines (max ${MAX_LINE_READ_LIMIT}); use offset and limit to inspect specific sections before editing.`;

export const LINE_EDIT_DESCRIPTION =
  "Atomically apply one or more non-overlapping anchored line-range edits to a workspace text file. Each entry in `edits[]` specifies `start_line`, `end_line` (1-based, inclusive against the original pre-edit file; use `end_line = start_line - 1` to insert before `start_line`), `start_text` (the original line at `start_line` without line-number prefix, used to verify and auto-align the target), optional `end_text`, and `new_text` (replacement lines without line-number prefixes; empty string deletes the range). Batch multiple changes to one file in a single call.";

export const LINE_READ_PARAMETERS = Type.Object({
  path: Type.String({ description: "Path to the file to read (relative or absolute)" }),
  offset: Type.Optional(Type.Number({ description: "Line number to start reading from (1-indexed)" })),
  limit: Type.Optional(Type.Number({ description: `Maximum number of lines to read (default ${DEFAULT_LINE_READ_LIMIT}, max ${MAX_LINE_READ_LIMIT})` })),
});

export const LINE_EDIT_HUNK_PARAMETERS = Type.Object({
  start_line: Type.Integer({ minimum: 1, description: "1-based first line number to replace (from read output)" }),
  end_line: Type.Integer({ minimum: 0, description: "1-based last line number to replace (inclusive; use start_line - 1 to insert before start_line)" }),
  start_text: Type.String({ description: "Original text of line start_line without line-number prefix (empty string only for an empty file)" }),
  end_text: Type.Optional(Type.String({ description: "Optional original text of line end_line for multi-line spans" })),
  new_text: Type.String({ description: "Replacement lines without line-number prefixes (empty string deletes the range)" }),
});

export const LINE_EDIT_PARAMETERS = Type.Object({
  path: Type.String({ description: "Path to the file to edit (relative or absolute)" }),
  edits: Type.Array(LINE_EDIT_HUNK_PARAMETERS, {
    minItems: 1,
    description: "One or more non-overlapping anchored line edits evaluated against the original pre-edit file.",
  }),
});

function normalizeSingleHunkObject(raw: any): Record<string, unknown> | undefined {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return undefined;
  const sl = raw.start_line ?? raw.startLine;
  const el = raw.end_line ?? raw.endLine;
  const nt = typeof raw.new_text === "string" ? raw.new_text : (typeof raw.newText === "string" ? raw.newText : undefined);
  if (sl === undefined || el === undefined || nt === undefined) return undefined;
  const st = typeof raw.start_text === "string"
    ? raw.start_text
    : (typeof raw.startText === "string" ? raw.startText : "");
  const et = typeof raw.end_text === "string"
    ? raw.end_text
    : (typeof raw.endText === "string" ? raw.endText : undefined);
  return {
    start_line: Number(sl),
    end_line: Number(el),
    start_text: st,
    ...(et !== undefined ? { end_text: et } : {}),
    new_text: nt,
  };
}

/**
 * Normalizes single-hunk shorthand (`{path, start_line, end_line, start_text?, new_text}`),
 * JSON-stringified `edits`, single-object `edits`, and camelCase aliases into canonical `{path, edits[]}`.
 */
export function prepareLineEditArguments(input: unknown): unknown {
  if (!input || typeof input !== "object" || Array.isArray(input)) return input;
  const args = { ...(input as Record<string, unknown>) };
  if (typeof args.path !== "string" && typeof args.file_path === "string") {
    args.path = args.file_path;
    delete args.file_path;
  }

  let editsList: unknown[] | undefined;
  if (typeof args.edits === "string") {
    try {
      const parsed = JSON.parse(args.edits);
      if (Array.isArray(parsed)) editsList = parsed;
      else if (parsed && typeof parsed === "object") editsList = [parsed];
    } catch { /* let validator report */ }
  } else if (Array.isArray(args.edits)) {
    editsList = args.edits;
  } else if (args.edits && typeof args.edits === "object") {
    editsList = [args.edits];
  }

  if (editsList) {
    args.edits = editsList.map((item) => normalizeSingleHunkObject(item) ?? item);
    return args;
  }

  const single = normalizeSingleHunkObject(args);
  if (single) {
    const out: Record<string, unknown> = { path: args.path, edits: [single] };
    return out;
  }
  return args;
}

function normalizePathArg(cwd: string, rawPath: string): string {
  const cleaned = rawPath.trim().replace(/^@/, "");
  if (cwd !== "/workspace" && (cleaned === "/workspace" || cleaned.startsWith("/workspace/"))) {
    return cwd + cleaned.slice("/workspace".length);
  }
  return cleaned;
}

function toDisplayPath(cwd: string, absPath: string): string {
  const rel = posix.relative(cwd, absPath);
  return rel && !rel.startsWith("..") && !posix.isAbsolute(rel) ? rel : absPath;
}

async function resolveExistingFile(
  rawPath: string,
  cwd: string,
  fs: WorkspaceFs,
  tolerance: boolean,
): Promise<{ absPath: string; displayPath: string; note?: string }> {
  const direct = posix.resolve(cwd, normalizePathArg(cwd, rawPath));
  if (await fs.exists(direct)) {
    return { absPath: direct, displayPath: toDisplayPath(cwd, direct) };
  }
  if (tolerance) {
    const repair = await repairPath(rawPath, cwd, fs, "file");
    if (repair.path) {
      const repairedAbs = posix.resolve(cwd, repair.path);
      return {
        absPath: repairedAbs,
        displayPath: repair.path,
        note: `[${rawPath} does not exist; used ${repair.path} instead.]`,
      };
    }
    const hint = repair.suggestions.length
      ? ` Nearby workspace files: ${repair.suggestions.slice(0, 8).join(", ")}.`
      : "";
    throw new Error(`ENOENT: no such file ${rawPath}.${hint}`);
  }
  throw new Error(`ENOENT: no such file ${rawPath}`);
}

const IMAGE_EXTENSIONS = new Set([".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"]);

export interface CreateLineModeToolsOptions {
  readonly tolerance?: boolean;
  readonly tracker?: LineViewTracker;
}

/**
 * Transforms a standard pi tool set (`read`, `write`, `edit`, `bash`, ...) into Anchored Batch
 * Line-Edit v2 (`read` with neutral 1-based line numbers + atomic anchored `edit`) while keeping
 * `write`/`ls` tolerance and `bash` untouched.
 */
export function createLineModeTools(
  rawTools: readonly ToolDefinition<any, any, any>[],
  baseCwd: string,
  fs: WorkspaceFs = hostFs,
  options: CreateLineModeToolsOptions = {},
): ToolDefinition<any, any, any>[] {
  const tolerance = options.tolerance !== false;
  const tracker = options.tracker ?? new LineViewTracker();

  return rawTools.map((tool) => {
    if (tool.name === "read") {
      const fallbackExec = tool.execute.bind(tool) as (...a: any[]) => Promise<any>;
      return {
        ...tool,
        description: LINE_READ_DESCRIPTION,
        promptSnippet: "Read file contents with 1-based line numbers",
        promptGuidelines: [
          `Use read to inspect files with 1-based line numbers (default ${DEFAULT_LINE_READ_LIMIT} lines, max ${MAX_LINE_READ_LIMIT}) before calling edit.`,
        ],
        parameters: LINE_READ_PARAMETERS,
        async execute(id: string, args: any, signal?: AbortSignal, onUpdate?: unknown, ctx?: any) {
          const cwd = ctx?.cwd || baseCwd;
          const rawPath = String(args?.path ?? (tolerance ? args?.file_path : "") ?? "").trim();
          if (!rawPath) throw new Error("read requires a non-empty path.");
          const ext = posix.extname(rawPath).toLowerCase();
          if (IMAGE_EXTENSIONS.has(ext)) {
            return fallbackExec(id, { path: rawPath }, signal, onUpdate, ctx);
          }
          const { absPath, displayPath, note } = await resolveExistingFile(rawPath, cwd, fs, tolerance);
          const buf = await fs.readFile(absPath);
          const text = buf.toString("utf8");
          const fileSha = sha256(text);
          const { lines } = splitFileLines(text);

          const rawOffset = args?.offset ?? (tolerance ? args?.start_line : undefined);
          const offset = typeof rawOffset === "number" && Number.isFinite(rawOffset) ? Math.max(1, Math.trunc(rawOffset)) : 1;
          let limit = DEFAULT_LINE_READ_LIMIT;
          if (typeof args?.limit === "number" && Number.isFinite(args.limit)) {
            limit = Math.min(MAX_LINE_READ_LIMIT, Math.max(1, Math.trunc(args.limit)));
          } else if (tolerance && typeof args?.end_line === "number" && Number.isFinite(args.end_line)) {
            limit = Math.min(MAX_LINE_READ_LIMIT, Math.max(1, Math.trunc(args.end_line) - offset + 1));
          }

          if (lines.length > 0 && offset > lines.length) {
            throw new Error(`offset ${offset} is past the end of ${displayPath} (${plural(lines.length)} total). Use offset=1 to read from the beginning.`);
          }
          const win = renderNumberedWindow(displayPath, lines, offset, offset + limit - 1, true);
          tracker.recordRead(absPath, displayPath, fileSha, win.firstShown, win.lastShown, lines.length);
          const fullOut = note ? `${note}\n${win.text}` : win.text;
          return {
            content: [{ type: "text" as const, text: fullOut }],
            details: { path: displayPath, firstLine: win.firstShown, lastLine: win.lastShown, totalLines: lines.length },
          };
        },
      } as ToolDefinition<any, any, any>;
    }

    if (tool.name === "edit") {
      return {
        ...tool,
        description: LINE_EDIT_DESCRIPTION,
        promptSnippet: "Apply one or more anchored line-range edits atomically in a file",
        promptGuidelines: [
          "Always read the target lines with read first; batch multiple non-overlapping edits to the same file into one edits[] call against the original pre-edit line numbers.",
          "Include start_text (the exact original line at start_line without line-number prefix) for each edit, and write new_text without line-number prefixes.",
        ],
        parameters: LINE_EDIT_PARAMETERS,
        prepareArguments: prepareLineEditArguments,
        async execute(_id: string, rawArgs: any, _signal?: AbortSignal, _onUpdate?: unknown, ctx?: any) {
          const args = (tolerance ? prepareLineEditArguments(rawArgs) : rawArgs) as any;
          const cwd = ctx?.cwd || baseCwd;
          const rawPath = String(args?.path ?? "").trim();
          if (!rawPath) throw new Error("edit requires a non-empty path.");
          if (!Array.isArray(args?.edits) || args.edits.length === 0) {
            throw new Error('edit requires path and non-empty edits[] (e.g. {"path":"a.py","edits":[{"start_line":10,"end_line":12,"start_text":"def f():","new_text":"..."}]}).');
          }
          const editInputs: LineEditInput[] = [];
          for (let i = 0; i < args.edits.length; i++) {
            const h = args.edits[i];
            const startLine = Number(h?.start_line);
            const endLine = Number(h?.end_line);
            const newText = typeof h?.new_text === "string" ? h.new_text : undefined;
            const startText = typeof h?.start_text === "string" ? h.start_text : undefined;
            const endText = typeof h?.end_text === "string" ? h.end_text : undefined;
            if (!Number.isSafeInteger(startLine) || !Number.isSafeInteger(endLine) || newText === undefined) {
              throw new Error(`edits[${i}] requires integer start_line, integer end_line, string start_text, and string new_text.`);
            }
            editInputs.push({
              startLine,
              endLine,
              ...(startText !== undefined ? { startText } : {}),
              ...(endText !== undefined ? { endText } : {}),
              newText,
            });
          }
          if (!fs.writeFile) {
            throw new Error("WorkspaceFs.writeFile is required for line-mode edit.");
          }

          const { absPath, displayPath, note } = await resolveExistingFile(rawPath, cwd, fs, tolerance);
          const beforeRaw = (await fs.readFile(absPath)).toString("utf8");
          const beforeSha = sha256(beforeRaw);
          const { lines: beforeLines } = splitFileLines(beforeRaw);

          const applied = applyAnchoredLineEdits(beforeRaw, displayPath, editInputs, { selfHeal: tolerance });
          if ("error" in applied) {
            // If file wasn't read yet and anchor failed, give the even clearer read-first hint if no window was shown.
            if (tolerance && !applied.error.includes("=== ")) {
              const unseenErr = tracker.checkEditAllowed(absPath, displayPath, beforeSha, editInputs, beforeLines.length);
              if (unseenErr) throw new Error(unseenErr);
            }
            throw new Error(applied.error);
          }

          if (tolerance) {
            const allAnchorsVerified = applied.hunks.every((h) => h.anchorVerified);
            const unseenErr = tracker.checkEditAllowed(absPath, displayPath, beforeSha, editInputs, beforeLines.length, {
              allAnchorsVerified,
            });
            if (unseenErr) throw new Error(unseenErr);

            const beforeLF = beforeRaw.replaceAll("\r\n", "\n");
            const afterLF = applied.content.replaceAll("\r\n", "\n");
            const syntaxErr = await introducedEditSyntaxError(beforeLF, afterLF, displayPath);
            if (syntaxErr) {
              throw new Error(formatBatchLineEditSyntaxRejection(displayPath, applied, syntaxErr));
            }
          }

          await fs.writeFile(absPath, applied.content);
          const afterSha = sha256(applied.content);
          const summary = formatBatchLineEditSummary(displayPath, applied);
          tracker.recordBatchEdit(absPath, displayPath, afterSha, applied, summary.windows);
          const fullOut = note ? `${note}\n${summary.text}` : summary.text;
          return {
            content: [{ type: "text" as const, text: fullOut }],
            details: { path: displayPath, removed: applied.totalRemoved, added: applied.totalAdded, hunks: applied.hunks.length },
          };
        },
      } as ToolDefinition<any, any, any>;
    }

    if (tool.name === "write") {
      const wrappedWrite = tolerance ? withTolerance(tool, baseCwd, fs) : tool;
      const origExec = wrappedWrite.execute.bind(wrappedWrite) as (...a: any[]) => Promise<any>;
      return {
        ...wrappedWrite,
        async execute(id: string, args: any, signal?: AbortSignal, onUpdate?: unknown, ctx?: any) {
          const res = await origExec(id, args, signal, onUpdate, ctx);
          const cwd = ctx?.cwd || baseCwd;
          const rawPath = typeof args?.path === "string" ? args.path.trim() : "";
          if (rawPath && typeof args?.content === "string") {
            const absPath = posix.resolve(cwd, normalizePathArg(cwd, rawPath));
            tracker.recordWrite(absPath, toDisplayPath(cwd, absPath), sha256(args.content), splitFileLines(args.content).lines.length);
          }
          return res;
        },
      } as ToolDefinition<any, any, any>;
    }

    return tolerance && tool.name === "ls" ? withTolerance(tool, baseCwd, fs) : tool;
  });
}
