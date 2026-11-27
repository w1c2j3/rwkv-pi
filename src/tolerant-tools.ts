/**
 * Recall-tolerant execution for pi's read, edit, write and ls tools.
 *
 * RWKV reproduces earlier text from a compressed state instead of copying it, so paths drift
 * by a directory level and edit oldText drifts in indentation, blank lines or single tokens.
 * These wrappers keep pi's own tool definition (name, schema, description and prompt guidelines
 * are unchanged, so prompts stay byte-identical) and change execution only:
 *
 * - read/edit/ls: a path that does not exist is resolved to the one workspace file it most
 *   plausibly meant; the result says so. Otherwise the error lists real nearby paths.
 * - edit: when pi's exact/fuzzy match fails, a stricter-to-looser ladder (indentation, blank
 *   lines, whitespace, then a unique close match) finds the region; the replacement is
 *   re-indented to the file, and lines the edit leaves unchanged keep the file's exact text.
 *   A one-line oldText that pi would match inside a line's indentation is widened to the whole
 *   line first. A failure returns the closest region's exact text to copy from.
 * - write: the result reports created directories and what an overwrite removed.
 *
 * Every tolerated repair is stated in the tool result. Successful exact calls are untouched.
 */
import { spawn } from "node:child_process";
import { mkdir, readdir, readFile, stat, writeFile } from "node:fs/promises";
import * as nodeModule from "node:module";
import { posix } from "node:path";
import type { ToolDefinition } from "@earendil-works/pi-coding-agent";

/** File access in the agent's view of the workspace (the host, or inside a sandbox). */
export interface WorkspaceFs {
  readFile(path: string): Promise<Buffer>;
  writeFile?(path: string, content: string | Buffer): Promise<void>;
  exists(path: string): Promise<boolean>;
  /** Regular files below `root` as sorted relative POSIX paths, bounded and without dependency/cache directories. */
  listFiles(root: string): Promise<string[]>;
}

export const SKIPPED_DIRECTORIES = [".git", "node_modules", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache"];
export const MAX_LISTED_FILES = 20_000;
const byCodePoint = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0);

export const hostFs: WorkspaceFs = {
  readFile: (path) => readFile(path),
  writeFile: async (path, content) => {
    await mkdir(posix.dirname(path), { recursive: true });
    await writeFile(path, content);
  },
  exists: async (path) => { try { await stat(path); return true; } catch { return false; } },
  async listFiles(root) {
    const skip = new Set(SKIPPED_DIRECTORIES), out: string[] = [];
    const walk = async (dir: string, rel: string): Promise<void> => {
      let entries;
      try { entries = await readdir(dir, { withFileTypes: true }); } catch { return; }
      for (const entry of entries.sort((a, b) => byCodePoint(a.name, b.name))) {
        if (out.length >= MAX_LISTED_FILES) return;
        if (skip.has(entry.name)) continue;
        const path = rel ? `${rel}/${entry.name}` : entry.name;
        if (entry.isDirectory()) await walk(posix.join(dir, entry.name), path);
        else if (entry.isFile()) out.push(path);
      }
    };
    await walk(root, "");
    return out;
  },
};

/** The same listing as a script for `node -e` inside a sandbox: argv = [root, skipJson, max]. */
export const LIST_FILES_SCRIPT = `const fs=require('node:fs'),p=require('node:path');const skip=new Set(JSON.parse(process.argv[2])),max=+process.argv[3],out=[];
const cmp=(a,b)=>a.name<b.name?-1:a.name>b.name?1:0;
(function walk(d,r){let es;try{es=fs.readdirSync(d,{withFileTypes:true})}catch{return}es.sort(cmp);
for(const e of es){if(out.length>=max)return;if(skip.has(e.name))continue;const x=r?r+'/'+e.name:e.name;
if(e.isDirectory())walk(p.join(d,e.name),x);else if(e.isFile())out.push(x)}})(process.argv[1],'');
process.stdout.write(JSON.stringify(out))`;

// ---------------------------------------------------------------------------------------------
// Paths

const normalizeWorkspaceAlias = (cwd: string, rawPath: string): string => {
  const cleaned = rawPath.replace(/^@/, "");
  if (cwd !== "/workspace" && (cleaned === "/workspace" || cleaned.startsWith("/workspace/"))) {
    return cwd + cleaned.slice("/workspace".length);
  }
  return cleaned;
};
const components = (path: string, cwd = "/workspace") => normalizeWorkspaceAlias(cwd, path).split("/").filter((part) => part && part !== ".");
const resolveIn = (cwd: string, path: string) => posix.resolve(cwd, normalizeWorkspaceAlias(cwd, path));

/** Drop an immediately repeated run of components: inputs/src/inputs/src/app.py -> inputs/src/app.py. */
export function collapseRepeats(parts: readonly string[]): string[] {
  const out = [...parts];
  for (let changed = true; changed;) {
    changed = false;
    for (let length = Math.floor(out.length / 2); length >= 1 && !changed; length--) {
      for (let i = 0; i + 2 * length <= out.length; i++) {
        if (out.slice(i, i + length).join("/") === out.slice(i + length, i + 2 * length).join("/")) {
          out.splice(i + length, length);
          changed = true;
          break;
        }
      }
    }
  }
  return out;
}

function sharedTail(a: readonly string[], b: readonly string[]): number {
  let n = 0;
  while (n < a.length && n < b.length && a[a.length - 1 - n] === b[b.length - 1 - n]) n++;
  return n;
}

export interface PathRepair {
  /** Workspace-relative path the request most plausibly meant. */
  path?: string;
  /** Real paths to offer when no single repair is safe. */
  suggestions: string[];
}

/**
 * The workspace path a non-existent `requested` path most plausibly meant. Only paths inside
 * the workspace are candidates, and a candidate must be unique: the longest existing suffix
 * of at least two components (a mis-copied absolute prefix), the same after collapsing a
 * repeated directory run, or the only file with that name (best shared tail when several).
 */
export async function repairPath(requested: string, cwd: string, fs: WorkspaceFs, kind: "file" | "directory"): Promise<PathRepair> {
  const files = await fs.listFiles(cwd);
  const pool = new Set<string>();
  if (kind === "file") for (const file of files) pool.add(file);
  else for (const file of files) { const parts = file.split("/"); for (let i = 1; i < parts.length; i++) pool.add(parts.slice(0, i).join("/")); }
  const original = components(requested);
  const variants = [original, collapseRepeats(original)];
  for (const parts of variants) {
    for (let k = parts.length; k >= 2; k--) {
      const suffix = parts.slice(-k).join("/");
      if (pool.has(suffix)) return { path: suffix, suggestions: [] };
    }
  }
  const name = original.at(-1);
  if (!name) return { suggestions: [] };
  const same = [...pool].filter((path) => posix.basename(path) === name);
  if (same.length === 1) return { path: same[0], suggestions: [] };
  if (same.length > 1) {
    const scored = same.map((path) => ({ path, score: Math.max(...variants.map((parts) => sharedTail(parts, path.split("/")))) }))
      .sort((a, b) => b.score - a.score);
    if (scored[0]!.score >= 2 && scored[1]!.score < scored[0]!.score) return { path: scored[0]!.path, suggestions: [] };
    return { suggestions: scored.slice(0, 6).map((entry) => entry.path) };
  }
  // Nothing by that name: list the deepest existing directory on the requested path.
  const absolute = resolveIn(cwd, requested);
  const relative = posix.relative(cwd, absolute);
  const inside = relative && !relative.startsWith("..") ? relative.split("/") : [];
  const directories = new Set<string>([""]);
  for (const file of files) { const parts = file.split("/"); for (let i = 1; i < parts.length; i++) directories.add(parts.slice(0, i).join("/")); }
  let base = "";
  for (let i = inside.length - 1; i >= 1; i--) { const candidate = inside.slice(0, i).join("/"); if (directories.has(candidate)) { base = candidate; break; } }
  const entries = new Set<string>();
  for (const file of files) {
    if (base && !file.startsWith(base + "/")) continue;
    const rest = (base ? file.slice(base.length + 1) : file).split("/");
    entries.add((base ? base + "/" : "") + rest[0] + (rest.length > 1 ? "/" : ""));
    if (entries.size >= 20) break;
  }
  return { suggestions: [...entries] };
}

function notFoundHint(requested: string, repair: PathRepair): string {
  if (!repair.suggestions.length) return `${requested} does not exist in the workspace.`;
  return `${requested} does not exist. Existing paths that may be meant (relative to the workspace):\n${repair.suggestions.join("\n")}`;
}

// ---------------------------------------------------------------------------------------------
// Edit matching

export type MatchLevel = "indentation" | "blank-lines" | "whitespace" | "similar";
const LEVEL_TEXT: Record<MatchLevel, string> = {
  "indentation": "ignoring indentation",
  "blank-lines": "ignoring blank lines",
  "whitespace": "ignoring whitespace differences",
  "similar": "as the closest match",
};

export interface RegionMatch {
  level: MatchLevel;
  /** Character offsets in the LF-normalized file. */
  start: number;
  end: number;
  /** 1-based inclusive line range. */
  firstLine: number;
  lastLine: number;
  /** Aligned [oldText line, file line] pairs of the matched lines (blank lines omitted where the level skips them). */
  pairs: (readonly [string, string])[];
  /** gaps[t]: the file's blank lines between pair t-1 and pair t that the old text left out. */
  gaps?: string[][];
  /** The region starts / ends inside its first / last line, so that line is only partly replaced. */
  midStart?: boolean;
  midEnd?: boolean;
  similarity?: number;
}

export type RegionResult =
  | { kind: "match"; match: RegionMatch }
  | { kind: "ambiguous"; level: MatchLevel; lines: number[] }
  | { kind: "none"; closest?: { firstLine: number; lastLine: number; similarity: number } };

/** pi's own fuzzy normalization (edit-diff.ts normalizeForFuzzyMatch): what pi already tolerates. */
function piFuzzy(text: string): string {
  return text.normalize("NFKC").split("\n").map((line) => line.trimEnd()).join("\n")
    .replace(/[‘’‚‛]/g, "'").replace(/[“”„‟]/g, '"')
    .replace(/[‐‑‒–—―−]/g, "-").replace(/[  -   　]/g, " ");
}
export const toLF = (text: string) => text.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
const piFinds = (content: string, old: string) => content.includes(old) || piFuzzy(content).includes(piFuzzy(old));
const indentOf = (line: string) => /^[ \t]*/.exec(line)![0];
const isBlank = (line: string) => line.trim() === "";
const trimKey = (line: string) => line.trim();
const spaceKey = (line: string) => line.replace(/\s+/g, " ").trim();

function bigrams(text: string): Map<string, number> {
  const map = new Map<string, number>();
  for (let i = 0; i < text.length - 1; i++) { const gram = text.slice(i, i + 2); map.set(gram, (map.get(gram) ?? 0) + 1); }
  return map;
}
/** Dice coefficient over character bigrams. */
export function similarity(a: string, b: string): number {
  if (a === b) return 1;
  if (a.length < 2 || b.length < 2) return 0;
  const x = bigrams(a), y = bigrams(b);
  let shared = 0;
  for (const [gram, count] of x) shared += Math.min(count, y.get(gram) ?? 0);
  return (2 * shared) / (a.length - 1 + b.length - 1);
}

interface Lines { text: string[]; starts: number[] }
function splitLines(content: string): Lines {
  const text = content.split("\n"), starts: number[] = [];
  let offset = 0;
  for (const line of text) { starts.push(offset); offset += line.length + 1; }
  return { text, starts };
}

/**
 * Where `oldText` occurs in `content` (LF, no BOM) under the first tolerance level that finds
 * anything. A level that finds several places stops the search as ambiguous rather than
 * falling through to a looser level.
 */
export function findRegion(content: string, oldText: string): RegionResult {
  const file = splitLines(content);
  const oldLF = toLF(oldText);
  const endsWithNewline = oldLF.endsWith("\n");
  const old = (endsWithNewline ? oldLF.slice(0, -1) : oldLF).split("\n");
  const levels: { level: MatchLevel; key: (line: string) => string; skipBlank: boolean; partial: boolean }[] = [
    { level: "indentation", key: trimKey, skipBlank: false, partial: true },
    { level: "blank-lines", key: trimKey, skipBlank: true, partial: true },
    { level: "whitespace", key: spaceKey, skipBlank: true, partial: false },
  ];
  for (const { level, key, skipBlank, partial } of levels) {
    const oldIndex = old.map((_, i) => i).filter((i) => !skipBlank || !isBlank(old[i]!));
    if (!oldIndex.length || oldIndex.every((i) => isBlank(old[i]!))) return { kind: "none" };
    const pool = file.text.map((_, i) => i).filter((i) => !skipBlank || !isBlank(file.text[i]!));
    const oldKeys = oldIndex.map((i) => key(old[i]!));
    const found: RegionMatch[] = [];
    for (let p = 0; p + oldIndex.length <= pool.length; p++) {
      const window = pool.slice(p, p + oldIndex.length);
      if (!skipBlank && window.at(-1)! - window[0]! !== oldIndex.length - 1) continue;
      const fileKeys = window.map((i) => key(file.text[i]!));
      const single = oldKeys.length === 1;
      let ok = true;
      for (let t = 0; t < oldKeys.length && ok; t++) {
        const fileKey = fileKeys[t]!, oldKey = oldKeys[t]!;
        if (single) ok = partial ? oldKey.length > 0 && fileKey.includes(oldKey) : fileKey === oldKey;
        else if (t === 0 && partial) ok = fileKey.endsWith(oldKey) && oldKey.length > 0;
        else if (t === oldKeys.length - 1 && partial) ok = fileKey.startsWith(oldKey) && oldKey.length > 0;
        else ok = fileKey === oldKey;
      }
      if (!ok) continue;
      const first = window[0]!, last = window.at(-1)!;
      const firstLine = file.text[first]!, lastLine = file.text[last]!;
      const oldFirst = old[oldIndex[0]!]!, oldLast = old[oldIndex.at(-1)!]!;
      const pairs = oldIndex.map((i, t) => [old[i]!, file.text[window[t]!]!] as const);
      const gaps = skipBlank ? blankGaps(file, window) : undefined;
      if (single && key(firstLine) !== key(oldFirst)) {
        // Part of one line: every occurrence on the line is a separate candidate.
        const needle = oldFirst.trim();
        for (let at = firstLine.indexOf(needle); at >= 0; at = firstLine.indexOf(needle, at + 1)) {
          found.push({ level, start: file.starts[first]! + at, end: file.starts[first]! + at + needle.length, firstLine: first + 1, lastLine: first + 1,
            pairs, midStart: true, midEnd: true });
        }
        continue;
      }
      const midStart = partial && key(firstLine) !== key(oldFirst);
      const start = midStart ? file.starts[first]! + firstLine.trimEnd().length - oldFirst.trim().length : file.starts[first]!;
      const midEnd = partial && key(lastLine) !== key(oldLast);
      let end = midEnd ? file.starts[last]! + indentOf(lastLine).length + oldLast.trim().length : file.starts[last]! + lastLine.length;
      if (!midEnd && endsWithNewline && last < file.text.length - 1) end += 1;
      found.push({ level, start, end, firstLine: first + 1, lastLine: last + 1, pairs, ...(gaps ? { gaps } : {}),
        ...(midStart ? { midStart } : {}), ...(midEnd ? { midEnd } : {}) });
    }
    if (found.length === 1) return { kind: "match", match: found[0]! };
    if (found.length > 1) return { kind: "ambiguous", level, lines: found.map((match) => match.firstLine) };
  }
  return similarRegion(file, old);
}

/** A unique run of non-blank lines whose average per-line similarity is at least 0.9, anchored at both ends. */
function similarRegion(file: Lines, old: string[]): RegionResult {
  const oldLines = old.filter((line) => !isBlank(line));
  const pool = file.text.map((_, i) => i).filter((i) => !isBlank(file.text[i]!));
  const n = oldLines.length;
  if (!n || pool.length < n) return { kind: "none" };
  const oldKeys = oldLines.map(spaceKey), fileKeys = pool.map((i) => spaceKey(file.text[i]!));
  const firstScores = fileKeys.map((line) => similarity(oldKeys[0]!, line));
  const lastScores = fileKeys.map((line) => similarity(oldKeys[n - 1]!, line));
  const scored: { p: number; score: number }[] = [];
  for (let p = 0; p + n <= pool.length; p++) {
    if (firstScores[p]! < 0.5 && lastScores[p + n - 1]! < 0.5) continue;
    let total = 0;
    for (let t = 0; t < n; t++) total += t === 0 ? firstScores[p]! : t === n - 1 ? lastScores[p + t]! : similarity(oldKeys[t]!, fileKeys[p + t]!);
    scored.push({ p, score: total / n });
  }
  scored.sort((a, b) => b.score - a.score);
  const best = scored[0];
  if (!best) return { kind: "none" };
  const runner = scored.find((entry) => Math.abs(entry.p - best.p) >= n);
  let first = pool[best.p]!, last = pool[best.p + n - 1]!;
  // Detect omitted-middle sliding window: if oldText's first line matches strictly better earlier (or last line later)
  // while the opposite end anchors `best`, lines in the middle of the block were dropped.
  let splitAnchor = false;
  if (firstScores[best.p]! < 1) {
    for (let q = Math.max(0, best.p - n); q < best.p; q++) {
      if (firstScores[q]! >= 0.95 && firstScores[q]! > firstScores[best.p]! + 0.03) {
        first = pool[q]!;
        splitAnchor = true;
        break;
      }
    }
  }
  if (lastScores[best.p + n - 1]! < 1) {
    for (let r = Math.min(pool.length - 1, best.p + 2 * n - 1); r > best.p + n - 1; r--) {
      if (lastScores[r]! >= 0.95 && lastScores[r]! > lastScores[best.p + n - 1]! + 0.03) {
        last = pool[r]!;
        splitAnchor = true;
        break;
      }
    }
  }
  const anchorsOk = !splitAnchor && firstScores[best.p]! >= 0.88 && lastScores[best.p + n - 1]! >= 0.88;
  const closest = { firstLine: first + 1, lastLine: last + 1, similarity: best.score };
  const singleOk = n === 1 && oldKeys[0]!.length >= 20 && (!runner || runner.score <= best.score - 0.12);
  if ((n < 2 && !singleOk) || best.score < 0.9 || !anchorsOk || (runner && runner.score > best.score - 0.1)) {
    return { kind: "none", ...(best.score >= 0.5 ? { closest } : {}) };
  }
  return { kind: "match", match: { level: "similar", start: file.starts[first]!, end: file.starts[last]! + file.text[last]!.length,
    firstLine: first + 1, lastLine: last + 1, pairs: oldLines.map((line, t) => [line, file.text[pool[best.p + t]!]!] as const),
    gaps: blankGaps(file, pool.slice(best.p, best.p + n)), similarity: best.score } };
}

/** The file lines between consecutive matched lines (only blank lines are ever skipped). */
function blankGaps(file: Lines, window: readonly number[]): string[][] {
  return window.map((line, t) => (t === 0 ? [] : file.text.slice(window[t - 1]! + 1, line)));
}

/**
 * Move `text` from the model's indentation to the file's, using the [old, file] indentation pairs
 * of the matched lines. An indentation seen in the match maps directly; a deeper one keeps the
 * longest mapped prefix and converts the rest between spaces and tabs at the width the pairs show.
 */
export function reindent(text: string, indents: readonly (readonly [string, string])[]): string {
  if (indents.every(([from, to]) => from === to)) return text;
  const map = new Map<string, string>();
  for (const [from, to] of indents) if (!map.has(from)) map.set(from, to);
  let spacesPerTab = 0, tabToSpaces = 0;
  for (const [from, to] of indents) {
    if (!spacesPerTab && /^ +$/.test(from) && /^\t+$/.test(to) && from.length % to.length === 0) spacesPerTab = from.length / to.length;
    if (!tabToSpaces && /^\t+$/.test(from) && /^ +$/.test(to) && to.length % from.length === 0) tabToSpaces = to.length / from.length;
  }
  return text.split("\n").map((line) => {
    if (isBlank(line)) return line;
    const ws = indentOf(line);
    const mapped = map.get(ws);
    if (mapped !== undefined) return mapped + line.slice(ws.length);
    const from = [...map.keys()].filter((key) => ws.startsWith(key)).sort((a, b) => b.length - a.length)[0];
    if (from === undefined) return line;
    let rest = ws.slice(from.length);
    if (spacesPerTab) rest = rest.replace(new RegExp(` {${spacesPerTab}}`, "g"), "\t");
    if (tabToSpaces) rest = rest.replace(/\t/g, " ".repeat(tabToSpaces));
    return map.get(from)! + rest + line.slice(ws.length);
  }).join("\n");
}

const lineAt = (content: string, offset: number) => content.slice(0, offset).split("\n").length;
function visibleLineCount(content: string): number {
  const lines = content.split("\n");
  return lines.length > 1 && lines.at(-1) === "" ? lines.length - 1 : lines.length;
}
function excerpt(content: string, first: number, last: number): string {
  return content.split("\n").slice(first - 1, last).join("\n");
}

const COPIED_LINE_NUMBER = /^[ \t]*(\d+)(?:: ?|\t|→ ?)(.*)$/;
const ELISION_PLACEHOLDER_COMMENT = /^[ \t]*(?:\/\/+|#+|\/\*+|<!--)[ \t]*(?:\.{3}|…)[ \t]*(?:existing|rest|unchanged|remaining|omitted|other|previous|keep|same|code|lines|body|impl)\b/i;
const STANDALONE_ELISION_LINE = /^[ \t]*(?:\/\/+[ \t]*|#+[ \t]*|\/\*+[ \t]*)?(?:\.{3}|…)[ \t]*(?:\*+\/)?$/;

function stripSequentialNumbers(text: string, allowSingleLine: boolean): { stripped: string; firstNum: number; count: number } | undefined {
  const lf = toLF(text);
  const trailing = lf.endsWith("\n");
  const lines = (trailing ? lf.slice(0, -1) : lf).split("\n");
  let firstNum = -1, expectedNext = -1, numbered = 0;
  const out: string[] = [];
  for (const line of lines) {
    const m = COPIED_LINE_NUMBER.exec(line);
    if (!m) {
      if (!isBlank(line)) return undefined;
      out.push(line);
      if (expectedNext >= 0) expectedNext++;
      continue;
    }
    const num = Number(m[1]);
    if (!Number.isSafeInteger(num) || num < 1) return undefined;
    if (expectedNext >= 0 && num !== expectedNext) return undefined;
    if (firstNum < 0) firstNum = num;
    expectedNext = num + 1;
    numbered++;
    out.push(m[2]!);
  }
  if (numbered === 0 || (numbered === 1 && (!allowSingleLine || out.every(isBlank)))) return undefined;
  return { stripped: out.join("\n") + (trailing ? "\n" : ""), firstNum, count: numbered };
}

/**
 * Strips line-number prefixes (`12: ...`, `  12\t...`, `12→...`) copied from numbered read/grep/nl output,
 * only when the raw oldText does not already match the file and the stripped oldText uniquely does.
 */
export function stripCopiedLineNumbers(content: string, oldText: string, newText: string): { oldText: string; newText: string; stripped: boolean } {
  const oldParsed = stripSequentialNumbers(oldText, true);
  if (!oldParsed || !oldParsed.stripped.trim()) return { oldText, newText, stripped: false };
  const strippedOld = oldParsed.stripped;
  const exactUnique = content.includes(strippedOld) && content.indexOf(strippedOld, content.indexOf(strippedOld) + 1) < 0;
  if (!exactUnique && findRegion(content, strippedOld).kind !== "match") {
    return { oldText, newText, stripped: false };
  }
  const newParsed = stripSequentialNumbers(newText, oldParsed.count === 1);
  const shouldStripNew = newParsed !== undefined
    && (newParsed.count >= 2 || Math.abs(newParsed.firstNum - oldParsed.firstNum) <= 1);
  return {
    oldText: strippedOld,
    newText: shouldStripNew ? newParsed.stripped : newText,
    stripped: true,
  };
}

function findIntroducedElision(oldLF: string, newLF: string): string | undefined {
  const oldLines = new Set(oldLF.split("\n").map((l) => l.trim()));
  for (const line of newLF.split("\n")) {
    if (ELISION_PLACEHOLDER_COMMENT.test(line) && !oldLines.has(line.trim())) return line.trim();
  }
  return undefined;
}

function elisionBracketSpan(content: string, oldLF: string): { firstLine: number; lastLine: number } | undefined {
  const lines = (oldLF.endsWith("\n") ? oldLF.slice(0, -1) : oldLF).split("\n");
  const idx = lines.findIndex((line) => STANDALONE_ELISION_LINE.test(line));
  if (idx <= 0 || idx >= lines.length - 1) return undefined;
  const head = lines.slice(0, idx).filter((l) => !isBlank(l) && !STANDALONE_ELISION_LINE.test(l));
  const tail = lines.slice(idx + 1).filter((l) => !isBlank(l) && !STANDALONE_ELISION_LINE.test(l));
  if (!head.length || !tail.length) return undefined;
  const headRes = findRegion(content, head.join("\n"));
  const tailRes = findRegion(content, tail.join("\n"));
  if (headRes.kind !== "match" || tailRes.kind !== "match") return undefined;
  if (tailRes.match.lastLine <= headRes.match.firstLine || tailRes.match.lastLine - headRes.match.firstLine > 80) return undefined;
  return { firstLine: headRes.match.firstLine, lastLine: tailRes.match.lastLine };
}

interface EditInput { oldText: string; newText: string }
interface ToleratedEdit extends EditInput { note?: string }
const labelOf = (count: number, index: number) => (count === 1 ? "the edit" : `edits[${index}]`);

/**
 * Moves the model's replacement onto a matched region. Whitespace the match trimmed from the ends of
 * the old text is trimmed from the replacement too; whole lines are re-indented to the file; and lines
 * the edit leaves unchanged (identical in old and new text) are written exactly as the file has them,
 * so recall drift in context lines never reaches the file.
 */
export function fitReplacement(newText: string, match: Pick<RegionMatch, "pairs" | "gaps" | "midStart" | "midEnd">): { text: string; reindented: boolean; kept: number } {
  const { pairs, gaps, midStart = false, midEnd = false } = match;
  const lines = toLF(newText).split("\n");
  const last = lines.length - 1;
  if (midStart) {
    const lead = indentOf(pairs[0]![0]);
    if (lines[0]!.startsWith(lead)) lines[0] = lines[0]!.slice(lead.length);
  }
  if (midEnd) {
    const trail = /[ \t]*$/.exec(pairs.at(-1)![0])![0];
    if (trail && lines[last]!.endsWith(trail)) lines[last] = lines[last]!.slice(0, -trail.length);
  }
  const from = midStart ? 1 : 0;
  // Context lines: a whole old line that occurs once in the old text, with its pair index.
  const seen = new Map<string, number>();
  for (const [oldLine] of pairs) seen.set(oldLine, (seen.get(oldLine) ?? 0) + 1);
  const context = new Map<string, number>();
  pairs.forEach(([oldLine], t) => {
    if (t >= from && !(midEnd && t === pairs.length - 1) && !isBlank(oldLine) && seen.get(oldLine) === 1) context.set(oldLine, t);
  });
  const indents = pairs.slice(from).map(([oldLine, fileLine]) => [indentOf(oldLine), indentOf(fileLine)] as const);
  const reindented = indents.some(([a, b]) => a !== b);
  const body = lines.slice(from);
  const moved = [...lines.slice(0, from), ...(reindented ? reindent(body.join("\n"), indents).split("\n") : body)];
  let kept = 0, lastContext = -1;
  const out: string[] = [];
  moved.forEach((line, i) => {
    const t = i >= from && !(midEnd && i === last) ? context.get(lines[i]!) : undefined;
    if (t === undefined) { out.push(line); return; }
    // The file's blank lines the old text left out before this line stay in front of it.
    if (out.length && !isBlank(out.at(-1)!) && t > lastContext) out.push(...(gaps?.[t] ?? []));
    const fileLine = pairs[t]![1];
    if (fileLine.trim() !== lines[i]!.trim()) kept++;
    out.push(fileLine);
    lastContext = t;
  });
  return { text: out.join("\n"), reindented, kept };
}

const isSingleIndentedLine = (edit: EditInput) => {
  const old = toLF(edit.oldText), body = old.endsWith("\n") ? old.slice(0, -1) : old;
  return /^[ \t]/.test(body) && !body.includes("\n") && (old !== body || toLF(edit.newText).includes("\n"));
};

/**
 * pi matches oldText as a plain substring, so a one-line oldText whose indentation drifted short
 * matches inside the line's indentation, and the replacement's further lines land under-indented.
 * Such edits are widened to the whole line and re-indented; all other edits are returned unchanged.
 */
export function alignToLines(content: string, edits: readonly EditInput[], path: string): ToleratedEdit[] {
  return edits.map((edit, index) => {
    if (!isSingleIndentedLine(edit)) return { ...edit };
    const old = toLF(edit.oldText), at = content.indexOf(old);
    if (at < 0 || content.indexOf(old, at + 1) >= 0) return { ...edit };
    const prefix = content.slice(content.lastIndexOf("\n", at - 1) + 1, at);
    if (!/^[ \t]+$/.test(prefix)) return { ...edit };
    const line = old.endsWith("\n") ? old.slice(0, -1) : old;
    const { text } = fitReplacement(edit.newText, { pairs: [[line, prefix + line]] });
    return { oldText: prefix + old, newText: text,
      note: `Matched ${labelOf(edits.length, index)} at line ${lineAt(content, at)} of ${path} ignoring indentation; re-indented the replacement to the file.` };
  });
}

/**
 * Rewrites edits pi could not match into exact file text. Edits pi can match are returned
 * unchanged. Throws a model-facing error for an edit no level can place uniquely.
 */
export function tolerateEdits(content: string, edits: readonly EditInput[], path: string): ToleratedEdit[] {
  const maxLine = visibleLineCount(content);
  return edits.map((edit, index) => {
    const label = labelOf(edits.length, index);
    let old = toLF(edit.oldText);
    let candidateNew = toLF(edit.newText);
    if (!old || piFinds(content, old)) return { ...edit };
    let strippedNumbers = false;
    let result = findRegion(content, old);
    if (result.kind !== "match" || (result.match.level === "similar" && !result.match.pairs.some(([, fileLine]) => COPIED_LINE_NUMBER.test(fileLine)))) {
      const deNumbered = stripCopiedLineNumbers(content, old, candidateNew);
      if (deNumbered.stripped) {
        old = deNumbered.oldText;
        candidateNew = deNumbered.newText;
        strippedNumbers = true;
        if (old === candidateNew) {
          throw new Error(`No edit needed in ${path}: oldText and newText are identical after stripping copied line numbers. Provide the actual target text to change, or answer directly if all edits are complete.`);
        }
        const at = content.indexOf(old);
        if (at >= 0 && content.indexOf(old, at + 1) < 0) {
          if (isSingleIndentedLine({ oldText: old, newText: candidateNew })) {
            const prefix = content.slice(content.lastIndexOf("\n", at - 1) + 1, at);
            if (/^[ \t]+$/.test(prefix)) {
              const line = old.endsWith("\n") ? old.slice(0, -1) : old;
              const { text } = fitReplacement(candidateNew, { pairs: [[line, prefix + line]] });
              return { oldText: prefix + old, newText: text,
                note: `Matched ${label} at line ${lineAt(content, at)} of ${path} after stripping copied line numbers and ignoring indentation; re-indented the replacement to the file.` };
            }
          }
          return { oldText: old, newText: candidateNew,
            note: `Matched ${label} at line ${lineAt(content, at)} of ${path} after stripping copied line numbers.` };
        }
        result = findRegion(content, old);
      }
    }
    if (result.kind === "match") {
      const { match } = result;
      const exact = content.slice(match.start, match.end);
      const fit = fitReplacement(candidateNew, match);
      const range = match.firstLine === match.lastLine ? `line ${match.firstLine}` : `lines ${match.firstLine}-${match.lastLine}`;
      let note = `Matched ${label} at ${range} of ${path} ${LEVEL_TEXT[match.level]}`;
      if (strippedNumbers) note += " (stripped copied line numbers)";
      if (match.similarity !== undefined) note += ` (${Math.round(match.similarity * 100)}% similar)`;
      if (fit.reindented) note += "; re-indented the replacement to the file";
      if (fit.kept) note += `; kept ${fit.kept} unchanged line(s) as the file has them`;
      note += ".";
      if (match.level === "similar") note += ` Replaced text:\n${exact}`;
      return { oldText: exact, newText: fit.text, note };
    }
    const bracketed = elisionBracketSpan(content, old);
    if (bracketed !== undefined) {
      const padFirst = Math.max(1, bracketed.firstLine - 1), padLast = Math.min(maxLine, bracketed.lastLine + 1);
      throw new Error(`Could not apply ${label} in ${path}: oldText contains an elision marker ("..."), which cannot be matched safely. `
        + `Target only the specific lines changing, or copy oldText verbatim from lines ${bracketed.firstLine}-${bracketed.lastLine}:\n`
        + excerpt(content, padFirst, padLast));
    }
    // "Already applied" only when the new text sits (unique, not merely similar) where the old text best resembles.
    const newLF = candidateNew;
    if (result.kind === "none" && result.closest && newLF.trim().length >= 8) {
      const present = findRegion(content, newLF), { firstLine, lastLine } = result.closest;
      if (present.kind === "match" && present.match.level !== "similar" && present.match.firstLine <= lastLine && present.match.lastLine >= firstLine) {
        throw new Error(`Could not apply ${label} in ${path}: its old text is not in the file, but its new text already is (line ${present.match.firstLine}). `
          + "The change appears to be applied already; do not repeat this edit. Continue with any remaining changes, or answer directly if all work is done.");
      }
    }
    if (result.kind === "ambiguous") {
      const snippets = result.lines.slice(0, 3).map((ln) => {
        const a = Math.max(1, ln - 2), b = Math.min(maxLine, ln + 2);
        return `[lines ${a}-${b}]\n${excerpt(content, a, b)}`;
      }).join("\n\n");
      throw new Error(`Could not apply ${label} in ${path}: when ${LEVEL_TEXT[result.level]}, the old text matches ${result.lines.length} places `
        + `(lines ${result.lines.slice(0, 8).join(", ")}). Include more surrounding lines so it matches exactly one place.\n\n${snippets}`);
    }
    let message = `Could not find ${label} in ${path}. The old text must match the file exactly.`;
    if (result.closest) {
      const { firstLine, lastLine, similarity: score } = result.closest;
      const padFirst = Math.max(1, firstLine - 2), padLast = Math.min(maxLine, lastLine + 2);
      message += ` The closest text is lines ${firstLine}-${lastLine} (${Math.round(score * 100)}% similar). Copy oldText exactly from these lines:\n`
        + excerpt(content, padFirst, padLast);
    } else message += ` Nothing similar was found; read ${path} again to see its current content.`;
    throw new Error(message);
  });
}

// ---------------------------------------------------------------------------------------------
// Tool wrappers

type Execute = (toolCallId: string, input: any, signal?: AbortSignal, onUpdate?: unknown, ctx?: any) => Promise<any>;

/** Adds `note` to the result's text, before or after pi's own text. */
function annotate(result: any, note: string, place: "before" | "after"): any {
  const content = Array.isArray(result?.content) ? [...result.content] : [];
  const index = content.findIndex((block: any) => block?.type === "text");
  const join = (text: string) => (place === "before" ? `${note}\n${text}` : `${text}\n${note}`);
  if (index >= 0) content[index] = { ...content[index], text: join(content[index].text) };
  else content.unshift({ type: "text", text: note });
  const structured = typeof result?.structuredContent === "string" ? { structuredContent: join(result.structuredContent) } : {};
  return { ...result, content, ...structured };
}

const messageOf = (error: unknown) => (error instanceof Error ? error.message : String(error));

/** Runs `attempt(path)`; a missing path is retried once at its repaired workspace path. */
async function withPathRepair(execute: Execute, args: any, cwd: string, fs: WorkspaceFs, kind: "file" | "directory",
  call: [string, AbortSignal | undefined, unknown, any]): Promise<{ result: any; repairedFrom?: string; repaired?: string }> {
  const [id, signal, onUpdate, ctx] = call;
  try {
    return { result: await execute(id, args, signal, onUpdate, ctx) };
  } catch (error) {
    const requested = typeof args?.path === "string" ? args.path : undefined;
    if (!requested || signal?.aborted || await fs.exists(resolveIn(cwd, requested))) throw error;
    const repair = await repairPath(requested, cwd, fs, kind);
    if (!repair.path) throw new Error(`${messageOf(error)}\n${notFoundHint(requested, repair)}`);
    return { result: await execute(id, { ...args, path: repair.path }, signal, onUpdate, ctx), repairedFrom: requested, repaired: repair.path };
  }
}

const repairNote = (from: string, to: string) => `[${from} does not exist; used ${to} instead.]`;

function tolerantRead(execute: Execute, baseCwd: string, fs: WorkspaceFs): Execute {
  return async (id, args, signal, onUpdate, ctx) => {
    const { result, repairedFrom, repaired } = await withPathRepair(execute, args, ctx?.cwd || baseCwd, fs, "file", [id, signal, onUpdate, ctx]);
    return repaired ? annotate(result, repairNote(repairedFrom!, repaired), "before") : result;
  };
}

function tolerantLs(execute: Execute, baseCwd: string, fs: WorkspaceFs): Execute {
  return async (id, args, signal, onUpdate, ctx) => {
    const { result, repairedFrom, repaired } = await withPathRepair(execute, args, ctx?.cwd || baseCwd, fs, "directory", [id, signal, onUpdate, ctx]);
    return repaired ? annotate(result, repairNote(repairedFrom!, repaired), "before") : result;
  };
}

const SYNTAX_CHECK_EXTS = new Set([".json", ".js", ".mjs", ".cjs", ".ts", ".mts", ".cts", ".py", ".sh", ".bash"]);
const stripTsTypes = (nodeModule as { stripTypeScriptTypes?: (code: string, options?: { mode?: string }) => string }).stripTypeScriptTypes;
const PY_AST_CHECK = "import ast,sys\ntry:\n ast.parse(sys.stdin.read(),sys.argv[1])\nexcept SyntaxError as e:\n print(f'{type(e).__name__}: {e.msg} (line {e.lineno})');sys.exit(1)\n";

function spawnSyntaxCheck(cmd: string, args: readonly string[], text: string): Promise<string | undefined> {
  return new Promise((resolve) => {
    let out = "", settled = false;
    const finish = (v: string | undefined) => { if (!settled) { settled = true; resolve(v); } };
    let child;
    try {
      child = spawn(cmd, args, {
        stdio: ["pipe", "pipe", "pipe"], timeout: 5_000, env: { PATH: process.env.PATH ?? "/usr/bin:/bin", LC_ALL: "C", LANG: "C" },
      });
    } catch { finish(undefined); return; }
    child.stdout.on("data", (c) => { out += String(c); });
    child.stderr.on("data", (c) => { out += String(c); });
    child.on("error", () => finish(undefined));
    child.on("close", (code, sig) => finish(sig !== null ? undefined : code === 0 ? "" : out.trim().split("\n")[0] || `exit ${code}`));
    child.stdin.on("error", () => {});
    child.stdin.end(text);
  });
}

async function checkFileSyntax(text: string, path: string): Promise<string | undefined> {
  const ext = posix.extname(path).toLowerCase();
  if (!SYNTAX_CHECK_EXTS.has(ext)) return undefined;
  if (ext === ".json") {
    if (text.trim() === "") return "";
    try { JSON.parse(text); return ""; }
    catch (error) { return `JSON SyntaxError: ${messageOf(error).split("\n")[0]}`; }
  }
  if (ext === ".py") return spawnSyntaxCheck("python3", ["-I", "-c", PY_AST_CHECK, path], text);
  if (ext === ".sh" || ext === ".bash") return spawnSyntaxCheck("bash", ["-n"], text);
  if (!stripTsTypes) return undefined;
  try { stripTsTypes(text, { mode: "strip" }); return ""; }
  catch (error) { return `SyntaxError: ${messageOf(error).split("\n")[0]}`; }
}

/** Non-empty error message only when `before` parsed cleanly and `after` introduces a syntax error. */
export async function introducedEditSyntaxError(before: string, after: string, path: string): Promise<string | undefined> {
  const ext = posix.extname(path).toLowerCase();
  if (!SYNTAX_CHECK_EXTS.has(ext)) return undefined;
  const afterErr = await checkFileSyntax(after, path);
  if (!afterErr) return undefined;
  return (await checkFileSyntax(before, path)) === "" ? afterErr : undefined;
}

export interface AppliedEditRange {
  /** 1-based inclusive start and end line in the post-edit file (`endLine < startLine` when lines were deleted). */
  readonly startLine: number;
  readonly endLine: number;
}

const POST_EDIT_CONTEXT = 2;
const MAX_POST_EDIT_WINDOW_LINES = 14;

export function previewEditsWithRanges(before: string, edits: readonly EditInput[]): { after: string; ranges: AppliedEditRange[] } | undefined {
  const located: { start: number; end: number; newLF: string }[] = [];
  for (const edit of edits) {
    const oldLF = toLF(edit.oldText);
    if (!oldLF) return undefined;
    const at = before.indexOf(oldLF);
    if (at < 0 || before.indexOf(oldLF, at + 1) >= 0) return undefined;
    located.push({ start: at, end: at + oldLF.length, newLF: toLF(edit.newText) });
  }
  located.sort((a, b) => a.start - b.start);
  for (let i = 1; i < located.length; i++) {
    if (located[i]!.start < located[i - 1]!.end) return undefined;
  }
  const countNewlines = (s: string) => {
    let n = 0;
    for (let i = 0; i < s.length; i++) if (s.charCodeAt(i) === 10) n++;
    return n;
  };
  let out = "", cursor = 0, currentLine = 1;
  const ranges: AppliedEditRange[] = [];
  for (const h of located) {
    const pre = before.slice(cursor, h.start);
    out += pre;
    currentLine += countNewlines(pre);
    const startLine = currentLine;
    out += h.newLF;
    const nl = countNewlines(h.newLF);
    const span = h.newLF === "" ? 0 : (h.newLF.endsWith("\n") ? nl : nl + 1);
    ranges.push({ startLine, endLine: startLine + span - 1 });
    currentLine += nl;
    cursor = h.end;
  }
  out += before.slice(cursor);
  return { after: out, ranges };
}

function diffChangedRange(before: string, after: string): AppliedEditRange[] {
  const a = before.split("\n"), b = after.split("\n");
  let pre = 0;
  while (pre < a.length && pre < b.length && a[pre] === b[pre]) pre++;
  let suf = 0;
  while (suf < a.length - pre && suf < b.length - pre && a[a.length - 1 - suf] === b[b.length - 1 - suf]) suf++;
  const startLine = pre + 1;
  const endLine = b.length - suf;
  return [{ startLine, endLine: Math.max(startLine - 1, endLine) }];
}

/** Compact verbatim post-edit micro-window (`±2` context lines, no line-number prefixes) for distance-0 follow-up edits. */
export function formatPostEditWindow(after: string, ranges: readonly AppliedEditRange[], displayPath: string): string {
  const maxLine = visibleLineCount(after);
  if (after === "" || maxLine === 0) return `[${displayPath} is now empty]`;
  if (!ranges.length) return "";
  const windows = ranges.map(({ startLine, endLine }) => {
    const first = Math.max(1, Math.min(maxLine, startLine) - POST_EDIT_CONTEXT);
    const last = endLine < startLine
      ? Math.min(maxLine, Math.max(1, startLine - 1) + POST_EDIT_CONTEXT)
      : Math.min(maxLine, endLine + POST_EDIT_CONTEXT);
    return { first, last };
  });
  const merged: { first: number; last: number }[] = [];
  for (const win of windows) {
    const prev = merged.at(-1);
    if (prev && win.first <= prev.last + 1) prev.last = Math.max(prev.last, win.last);
    else merged.push({ ...win });
  }
  const lines = after.split("\n");
  const blocks = merged.slice(0, 3).map(({ first, last }) => {
    const header = first === last ? `[${displayPath} line ${first}]` : `[${displayPath} lines ${first}-${last}]`;
    const span = last - first + 1;
    if (span <= MAX_POST_EDIT_WINDOW_LINES) return `${header}\n${lines.slice(first - 1, last).join("\n")}`;
    const head = lines.slice(first - 1, first + 5).join("\n");
    const tail = lines.slice(last - 6, last).join("\n");
    return `${header}\n${head}\n... (${span - 12} lines omitted) ...\n${tail}`;
  });
  if (merged.length > 3) blocks.push(`... (${merged.length - 3} more edited region(s) omitted)`);
  return blocks.join("\n\n");
}

function previewExactEdits(content: string, edits: readonly EditInput[]): string | undefined {
  return previewEditsWithRanges(content, edits)?.after;
}

function tolerantEdit(execute: Execute, baseCwd: string, fs: WorkspaceFs): Execute {
  const matched: Execute = async (id, args, signal, onUpdate, ctx) => {
    const edits: EditInput[] | undefined = Array.isArray(args?.edits) ? args.edits : undefined;
    const ext = typeof args?.path === "string" ? posix.extname(args.path).toLowerCase() : "";
    let cachedContent: string | undefined;
    const readRaw = async () => {
      const raw = (await fs.readFile(resolveIn(ctx?.cwd || baseCwd, args.path))).toString("utf8");
      return toLF(raw.startsWith("﻿") ? raw.slice(1) : raw);
    };
    const read = async () => {
      if (cachedContent !== undefined) return cachedContent;
      cachedContent = await readRaw();
      return cachedContent;
    };
    const guardSyntax = async (candidateEdits: readonly EditInput[] | undefined) => {
      if (!candidateEdits?.length || !SYNTAX_CHECK_EXTS.has(ext)) return;
      const before = await read().catch(() => undefined);
      if (before === undefined) return;
      const after = previewExactEdits(before, candidateEdits);
      if (after === undefined) return;
      const broken = await introducedEditSyntaxError(before, after, args.path);
      if (broken !== undefined) {
        throw new Error(`Could not apply edit in ${args.path}: it introduces a syntax error (${broken}). Fix newText and try again.`);
      }
    };
    const notesOf = (list: ToleratedEdit[]) => list.flatMap((edit) => (edit.note ? [edit.note] : []));
    const strip = (list: ToleratedEdit[]) => list.map(({ oldText, newText }) => ({ oldText, newText }));
    let input = args, notes: string[] = [];
    if (edits?.some((e) => typeof e?.oldText === "string" && typeof e?.newText === "string" && e.oldText !== "" && toLF(e.oldText) === toLF(e.newText))) {
      throw new Error(`No edit needed in ${args.path}: oldText and newText are identical, so the file would not change. Provide the actual target text to change, or answer directly if all edits are complete.`);
    }
    if (edits) {
      for (const e of edits) {
        if (typeof e?.oldText !== "string" || typeof e?.newText !== "string") continue;
        const placeholder = findIntroducedElision(toLF(e.oldText), toLF(e.newText));
        if (placeholder !== undefined) {
          throw new Error(`Could not apply edit in ${args.path}: newText contains an elision placeholder ("${placeholder}"), which would delete code. Write complete replacement lines or narrow oldText to only the lines changing.`);
        }
      }
    }
    const beforeContent = edits?.length ? await read().catch(() => undefined) : undefined;
    if (edits?.some(isSingleIndentedLine) && beforeContent !== undefined) {
      const aligned = alignToLines(beforeContent, edits, args.path);
      notes = notesOf(aligned);
      if (notes.length) input = { ...args, edits: strip(aligned) };
    }
    await guardSyntax(input.edits);
    let appliedEdits: readonly EditInput[] | undefined = input.edits;
    let result;
    try {
      result = await execute(id, input, signal, onUpdate, ctx);
    } catch (error) {
      if (signal?.aborted || !/Could not find/.test(messageOf(error)) || !edits) throw error;
      const tolerated = tolerateEdits(await read(), input.edits, args.path);
      const more = notesOf(tolerated);
      if (!more.length) throw error;
      const stripped = strip(tolerated);
      await guardSyntax(stripped);
      result = await execute(id, { ...args, edits: stripped }, signal, onUpdate, ctx);
      appliedEdits = stripped;
      notes = [...notes, ...more];
    }
    if (beforeContent !== undefined && appliedEdits?.length && typeof args?.path === "string") {
      const preview = previewEditsWithRanges(beforeContent, appliedEdits);
      const resolved = preview ?? await readRaw().then((after) => ({ after, ranges: diffChangedRange(beforeContent, after) })).catch(() => undefined);
      if (resolved !== undefined) {
        const snippet = formatPostEditWindow(resolved.after, resolved.ranges, args.path);
        if (snippet) notes = [...notes, snippet];
      }
    }
    return notes.length ? annotate(result, notes.join("\n"), "after") : result;
  };
  return async (id, args, signal, onUpdate, ctx) => {
    const { result, repairedFrom, repaired } = await withPathRepair(matched, args, ctx?.cwd || baseCwd, fs, "file", [id, signal, onUpdate, ctx]);
    return repaired ? annotate(result, repairNote(repairedFrom!, repaired), "before") : result;
  };
}

function tolerantWrite(execute: Execute, baseCwd: string, fs: WorkspaceFs): Execute {
  return async (id, args, signal, onUpdate, ctx) => {
    const cwd = ctx?.cwd || baseCwd;
    const path = typeof args?.path === "string" ? args.path : undefined;
    if (!path) return execute(id, args, signal, onUpdate, ctx);
    const absolute = resolveIn(cwd, path);
    let previous: string | undefined;
    if (await fs.exists(absolute)) { try { previous = (await fs.readFile(absolute)).toString("utf8"); } catch { /* A directory: pi reports it. */ } }
    let missing = "";
    for (let dir = posix.dirname(absolute); dir !== posix.dirname(dir) && !(await fs.exists(dir)); dir = posix.dirname(dir)) missing = dir;
    const result = await execute(id, args, signal, onUpdate, ctx);
    const notes: string[] = [];
    if (missing) notes.push(`Created new directory ${posix.relative(cwd, missing) || missing}; it did not exist before.`);
    if (previous !== undefined) notes.push(overwriteSummary(previous, String(args.content ?? "")));
    return notes.length ? annotate(result, notes.join("\n"), "after") : result;
  };
}

/** What an overwrite changed, so a rewrite that silently dropped or altered lines is visible. */
export function overwriteSummary(previous: string, next: string): string {
  const before = toLF(previous), after = toLF(next);
  if (before === after) return "The new content is identical to the previous file.";
  const count = (text: string) => (text === "" ? 0 : text.replace(/\n$/, "").split("\n").length);
  const remaining = new Map<string, number>();
  for (const line of after.split("\n")) remaining.set(line, (remaining.get(line) ?? 0) + 1);
  const gone: string[] = [];
  for (const line of before.split("\n")) {
    if (isBlank(line)) continue;
    const left = remaining.get(line) ?? 0;
    if (left > 0) remaining.set(line, left - 1); else gone.push(line);
  }
  let summary = `Overwrote the existing file (${count(before)} -> ${count(after)} lines).`;
  if (gone.length) summary += ` ${gone.length} previous line(s) are no longer present, for example:\n${gone.slice(0, 3).map((line) => `- ${line.trim()}`).join("\n")}`;
  return summary;
}

const WRAPPERS: Record<string, (execute: Execute, cwd: string, fs: WorkspaceFs) => Execute> = {
  read: tolerantRead, edit: tolerantEdit, write: tolerantWrite, ls: tolerantLs,
};
export const TOLERANT_TOOL_NAMES = Object.keys(WRAPPERS);

/** pi's definition with recall-tolerant execution; every model-visible field is kept as it is. */
export function withTolerance<T extends ToolDefinition<any, any, any>>(definition: T, cwd: string, fs: WorkspaceFs = hostFs): T {
  const wrap = WRAPPERS[definition.name];
  if (!wrap) return definition;
  return { ...definition, execute: wrap((definition.execute as Execute).bind(definition), cwd, fs) as T["execute"] };
}
