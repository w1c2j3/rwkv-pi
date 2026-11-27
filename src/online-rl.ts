import { randomUUID } from "node:crypto";
import { appendFileSync, existsSync, mkdirSync, readFileSync, rmSync } from "node:fs";
import { mkdir, readFile, realpath, writeFile } from "node:fs/promises";
import { execFile } from "node:child_process";
import { dirname, join, resolve, sep } from "node:path";
import { promisify } from "node:util";
import { gunzipSync } from "node:zlib";
import type { Message } from "@earendil-works/pi-ai";
import {
  createBashToolDefinition,
  createEditToolDefinition,
  createFindToolDefinition,
  createGrepToolDefinition,
  createLsToolDefinition,
  createReadToolDefinition,
  createWriteToolDefinition,
  type ToolDefinition,
} from "@earendil-works/pi-coding-agent";
import { createLineModeTools, DEFAULT_LINE_READ_LIMIT, LineViewTracker, MAX_LINE_READ_LIMIT } from "./line-edit.js";
import { repeatedIdenticalCall, repeatedTargetFailure } from "./loop-guard.js";
import { materializeEnvironment } from "./environment.js";
import { renderPrompt, STOP_TEXTS } from "./prompt.js";
import { WorkspaceSandbox } from "./sandbox.js";
import { object, parseTask, taskPath, type Task } from "./tasks.js";
import { hostFs, TOLERANT_TOOL_NAMES, withTolerance } from "./tolerant-tools.js";
import { parseToolList, PI_TOOL_NAMES, type PiToolName } from "./tools.js";
import { ContinuationParser } from "./xml.js";
import {
  computeLearningRate,
  OnlineRlCluster,
  type OnlineRlRankClient,
  type TrainSample,
  type TrainTurn,
} from "./online-rl-client.js";

const execFileAsync = promisify(execFile);

export type RlAlgorithm = "dr_grpo" | "grpo";

export interface OnlineRlOptions {
  /** Task manifest path (.jsonl or .jsonl.gz; supports raw Task rows or { task, cohort } bundles). */
  tasksPath: string;
  /** Optional holdout eval manifest path. */
  evalTasksPath?: string;
  /** Output directory for metrics.jsonl, rollout summaries, and scratch workspaces. */
  outputDir: string;
  /** Total optimizer steps to run (counts successful optimizer updates, not skipped batches). */
  steps: number;
  /** Number of parallel rollouts sampled per task (G). */
  groupSize: number;
  /** Tasks assigned per rank per optimizer step (total groups per step = dpWorldSize * tasksPerRank). */
  tasksPerRank: number;
  /** Maximum assistant turns per rollout episode. */
  maxTurns: number;
  /** Per-turn generation safety cap (tokens). */
  maxTokensPerTurn: number;
  /** Sampling and training temperature (must match between /v1/completions and /v1/train). */
  temperature: number;
  /** Optional rollout nucleus sampling threshold (0 < topP <= 1; omitted disables). */
  topP?: number;
  /** Optional rollout top-k filter (0 disables). */
  topK?: number;
  /** Optional RWKV presence penalty applied to generated tokens during rollout (0 disables). */
  presencePenalty?: number;
  /** Optional RWKV repetition/frequency penalty applied on repeat hits during rollout (0 disables). */
  repetitionPenalty?: number;
  /** Optional exponential decay factor for RWKV rollout penalties (default 0.996). */
  penaltyDecay?: number;
  /** Require mixed pass/fail variance (0 < passRate < 1) instead of accepting all-fail penalty-only groups. */
  requirePassVariance?: boolean;
  /** Advantage algorithm: `dr_grpo` (default, unbiased constant denominator) or `grpo`. */
  algorithm: RlAlgorithm;
  /** Optional explicit per-group token budget divisor for Dr.GRPO (default: groupSize * maxTokensPerTurn * maxTurns). */
  drGrpoGroupBudget?: number;
  /** Peak learning rate. */
  lr: number;
  /** Minimum cosine learning rate. */
  lrMin: number;
  /** Linear warmup steps. */
  warmupSteps: number;
  /** LR schedule shape. */
  lrSchedule: "cosine" | "constant";
  /** Per-token KL penalty weight against startup reference_version (0 disables /v1/logprobs). */
  klBeta: number;
  /** Clamp per-token (lp - ref_lp) to [-klClip, +klClip] before applying klBeta. */
  klClip: number;
  /** Dynamic sampling retries per slot when a task yields zero reward variance (all-pass or all-fail). */
  maxResampleAttempts: number;
  /** Isolation mode for tool execution and checks. */
  sandbox: "bubblewrap" | "none";
  /** Edit mode (`line` = Anchored Batch Line-Edit v2; `text` = legacy). */
  editMode: "line" | "text";
  /** Maximum characters kept per tool observation before middle-elision. */
  maxToolResultChars: number;
  /** Per-tool execution timeout in ms. */
  toolTimeoutMs: number;
  /** Minimum free context tokens required to launch a task or continue another turn. */
  minGenHeadroom: number;
  /** Loop-guard identical call limit (0 disables). */
  identicalCallLimit: number;
  /** Loop-guard repeated target failure limit (0 disables). */
  targetFailureLimit: number;
  /** Save adapter/state every N optimizer updates (0 = only on final step). */
  saveEvery: number;
  /** Save full Adam checkpoint every N optimizer updates (0 = disabled). */
  checkpointEvery: number;
  /** Run holdout eval every N optimizer updates (0 = disabled). */
  evalEvery: number;
  /** Max tasks evaluated per holdout eval sweep (0 = all). */
  evalLimit: number;
  /** Deterministic PRNG seed. */
  seed: number;
  /** Keep rollout temp workspaces on disk for debugging (default false). */
  keepWorkspaces?: boolean;
  /** Optional extra system prompt appended to pi's standard system prompt. */
  appendSystemPrompt?: string;
}

export interface ParsedToolCall {
  id: string;
  name: string;
  argumentsText: string;
  argumentsObj?: Record<string, unknown>;
}

export interface EpisodeTurnRecord {
  /** Token IDs of this turn (either prompt/observation with lossMask=0 or assistant generation with lossMask=1). */
  tokens: number[];
  lossMask: 0 | 1;
  /** Aligned sampling logprobs (present iff lossMask === 1). */
  logprobs?: number[];
  /** Start index of this turn in the concatenated episode token sequence. */
  offset: number;
}

export interface EpisodeOutcome {
  taskId: string;
  cohort?: string;
  rolloutIndex: number;
  turns: EpisodeTurnRecord[];
  fullTokens: number[];
  completedAssistantTurns: number;
  generatedTokens: number;
  cachedTokens: number;
  prefillTokens: number;
  finalStopReason: "stop" | "length" | "context_length" | "parse_error" | "loop_guard" | "max_turns";
  finalAnswerText: string;
  passed: boolean;
  checks: { passed: boolean; detail: string }[];
  editCalls: number;
  editFailures: number;
  truncated: boolean;
  loopGuardTriggered: boolean;
  parseError?: string;
  languageMismatch: boolean;
  reward: number;
  rewardBreakdown: {
    task: number;
    truncationPenalty: number;
    loopPenalty: number;
    languagePenalty: number;
    cleanEditBonus: number;
  };
}

export interface GroupRolloutResult {
  task: Task;
  cohort?: string;
  episodes: EpisodeOutcome[];
  rewardMean: number;
  rewardStd: number;
  passRate: number;
  rolloutSeconds: number;
  cachedTokens: number;
  prefillTokens: number;
  generatedTokens: number;
}

export interface LoadedTaskEntry {
  task: Task;
  cohort?: string;
  /** Optional sequential user follow-up prompts sharing the same workspace & token history. */
  followups: string[];
}

/** Deterministic Mulberry32 PRNG for reproducible task shuffling and seeds. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Load raw Task JSONL or `{ task, cohort }` bundle JSONL (supports .gz). */
export function loadTaskEntries(path: string): LoadedTaskEntry[] {
  const rawBuf = readFileSync(path);
  const text = path.endsWith(".gz") ? gunzipSync(rawBuf).toString("utf8") : rawBuf.toString("utf8");
  const entries: LoadedTaskEntry[] = [];
  const seen = new Set<string>();
  for (const line of text.split(/\r?\n/)) {
    if (!line.trim()) continue;
    const obj = JSON.parse(line) as Record<string, unknown>;
    const rawTask = object(obj.task) ? obj.task : obj;
    const task = parseTask(rawTask);
    // Filter out tasks that require web tools when running offline sandbox RL.
    if (task.tools.some((t) => !(PI_TOOL_NAMES as readonly string[]).includes(t))) continue;
    if (seen.has(task.id)) continue;
    seen.add(task.id);
    const rawFollowups = Array.isArray(obj.followups) ? obj.followups : [];
    const followups = rawFollowups
      .map((f) => (typeof f === "string" ? f.trim() : object(f) && typeof f.prompt === "string" ? f.prompt.trim() : ""))
      .filter(Boolean);
    entries.push({
      task,
      cohort: typeof obj.cohort === "string" ? obj.cohort : undefined,
      followups,
    });
  }
  if (!entries.length) throw new Error(`No runnable sandbox tasks found in ${path}.`);
  return entries;
}

/**
 * Builds the exact pi coding-agent system prompt aligned with Stage-2 Anchored Batch Line-Edit v2
 * (matching `datasets/stage2-anneal-v4-anchored-line-9k6-20261123`).
 */
export function buildRlSystemPrompt(tools: readonly ToolDefinition[], cwd: string, editMode: "line" | "text", extra?: string): string {
  const toolLines = tools.map((t) => {
    const snippet = typeof (t as { promptSnippet?: unknown }).promptSnippet === "string"
      ? (t as { promptSnippet: string }).promptSnippet.trim()
      : "";
    const firstLine = snippet || (t.description ?? "").split("\n")[0]?.trim() || t.name;
    return `- ${t.name}: ${firstLine}`;
  });
  const rules = editMode === "line"
    ? [
        `- Use read to inspect files with 1-based line numbers (default ${DEFAULT_LINE_READ_LIMIT} lines, max ${MAX_LINE_READ_LIMIT}) before calling edit.`,
        "- Use write only for new files or complete rewrites.",
        "- Always read the target lines with read first; batch multiple non-overlapping edits to the same file into one edits[] call against the original pre-edit line numbers.",
        "- Include start_text (the exact original line at start_line without line-number prefix) for each edit, and write new_text without line-number prefixes.",
        "- Be concise in your responses",
        "- Show file paths clearly when working with files",
        "- Match the language of your final answer to the user's request.",
      ]
    : [
        "- Use read to examine files before editing.",
        "- Use write only for new files or complete rewrites.",
        "- Be concise in your responses",
        "- Show file paths clearly when working with files",
        "- Match the language of your final answer to the user's request.",
      ];
  const parts = [
    "You are a coding assistant running inside pi, a coding agent harness. Help users inspect files, run commands, and modify code.",
    `<tools>\n${toolLines.join("\n")}\n\nAdditional tools may be available.\n</tools>`,
    `<rules>\n${rules.join("\n")}\n</rules>`,
    `Current working directory: ${cwd}`,
  ];
  if (extra?.trim()) parts.push(extra.trim());
  return parts.join("\n\n");
}

function toOpenAiTools(tools: readonly ToolDefinition[]): Record<string, unknown>[] {
  return tools.map((t) => ({
    type: "function",
    function: {
      name: t.name,
      description: t.description ?? "",
      parameters: t.parameters ?? { type: "object", properties: {} },
    },
  }));
}

const WORKSPACE_ALIAS_RE = /(?<![\w.-])\/workspace(?=\/|$|[\s"'`),;:])/g;

function rewriteWorkspaceInText(text: string, cwd: string): string {
  return cwd !== "/workspace" ? text.replace(WORKSPACE_ALIAS_RE, cwd) : text;
}

function rewriteWorkspacePathArg(rawPath: unknown, cwd: string): unknown {
  if (typeof rawPath !== "string" || cwd === "/workspace") return rawPath;
  const cleaned = rawPath.trim().replace(/^@/, "");
  if (cleaned === "/workspace" || cleaned.startsWith("/workspace/")) {
    return cwd + cleaned.slice("/workspace".length);
  }
  return rawPath;
}

function maskHostCwdInResult(result: unknown, cwd: string): unknown {
  if (cwd === "/workspace") return result;
  if (typeof result === "string") return result.replaceAll(cwd, "/workspace");
  if (object(result) && Array.isArray(result.content)) {
    return {
      ...result,
      content: result.content.map((part) =>
        object(part) && typeof part.text === "string"
          ? { ...part, text: part.text.replaceAll(cwd, "/workspace") }
          : part,
      ),
    };
  }
  return result;
}

function wrapHostToolWithWorkspaceAlias(tool: ToolDefinition<any, any, any>, cwd: string): ToolDefinition<any, any, any> {
  if (cwd === "/workspace") return tool;
  const origExec = tool.execute.bind(tool) as (...a: any[]) => Promise<any>;
  return {
    ...tool,
    async execute(id: string, args: any, signal?: AbortSignal, onUpdate?: unknown, ctx?: any) {
      let nextArgs = args;
      if (args && typeof args === "object" && !Array.isArray(args)) {
        if (tool.name === "bash" && typeof args.command === "string") {
          nextArgs = { ...args, command: rewriteWorkspaceInText(args.command, cwd) };
        } else if ("path" in args || "file_path" in args) {
          nextArgs = {
            ...args,
            ...("path" in args ? { path: rewriteWorkspacePathArg(args.path, cwd) } : {}),
            ...("file_path" in args ? { file_path: rewriteWorkspacePathArg(args.file_path, cwd) } : {}),
          };
        }
      }
      const res = await origExec(id, nextArgs, signal, onUpdate, ctx);
      return maskHostCwdInResult(res, cwd) as any;
    },
  } as ToolDefinition<any, any, any>;
}

function hostTools(cwd: string, names: readonly PiToolName[]): ToolDefinition[] {
  const make: Record<PiToolName, () => ToolDefinition<any, any, any>> = {
    read: () => wrapHostToolWithWorkspaceAlias(createReadToolDefinition(cwd) as ToolDefinition<any, any, any>, cwd),
    write: () => wrapHostToolWithWorkspaceAlias(createWriteToolDefinition(cwd) as ToolDefinition<any, any, any>, cwd),
    edit: () => wrapHostToolWithWorkspaceAlias(createEditToolDefinition(cwd) as ToolDefinition<any, any, any>, cwd),
    bash: () => wrapHostToolWithWorkspaceAlias(createBashToolDefinition(cwd, { exposeSessionEnvironment: false }) as ToolDefinition<any, any, any>, cwd),
    grep: () => wrapHostToolWithWorkspaceAlias(createGrepToolDefinition(cwd) as ToolDefinition<any, any, any>, cwd),
    find: () => wrapHostToolWithWorkspaceAlias(createFindToolDefinition(cwd) as ToolDefinition<any, any, any>, cwd),
    ls: () => wrapHostToolWithWorkspaceAlias(createLsToolDefinition(cwd) as ToolDefinition<any, any, any>, cwd),
  };
  return PI_TOOL_NAMES.filter((n) => names.includes(n)).map((n) => make[n]()) as ToolDefinition[];
}

function slimToolResult(text: string, maxChars: number): string {
  if (maxChars <= 0 || text.length <= maxChars) return text;
  const head = Math.floor(maxChars * 0.72);
  const tail = Math.max(200, maxChars - head);
  const omitted = text.length - head - tail;
  return `${text.slice(0, head)}\n\n[... ${omitted} characters omitted; use read with offset/limit or grep for targeted lines ...]\n\n${text.slice(text.length - tail)}`;
}

/**
 * Renders ONLY the incremental observation tokens to append after an assistant turn's raw
 * generated tokens, matching `prompt.ts` (`rwkv-pi-xml-continuation-v3`) while respecting whatever
 * stop string ended the previous completion so token boundaries stay exact.
 */
export function buildToolObservationText(
  toolResults: readonly string[],
  rawGeneratedText: string,
  matchedStop?: string | number | number[],
): string {
  const stopStr = typeof matchedStop === "string" ? matchedStop : "";
  const bodyBeforeStop = stopStr && rawGeneratedText.endsWith(stopStr)
    ? rawGeneratedText.slice(0, -stopStr.length)
    : rawGeneratedText;

  let out = "";
  for (let i = 0; i < toolResults.length; i++) {
    const content = toolResults[i]!;
    if (i === 0 && stopStr === "\nUser:") {
      out += ` <tool_response>\n${content}\n</tool_response>\n\n`;
      continue;
    }
    if (i === 0) {
      if (bodyBeforeStop.endsWith("\n\n") || stopStr === "<EOD>") {
        // After `\n\n` or `<EOD>`, start User turn cleanly without duplicating blank lines.
        const needNewlines = stopStr === "<EOD>" && !bodyBeforeStop.endsWith("\n") ? "\n\n"
          : stopStr === "<EOD>" && !bodyBeforeStop.endsWith("\n\n") ? "\n"
          : "";
        out += `${needNewlines}User: <tool_response>\n${content}\n</tool_response>\n\n`;
      } else if (bodyBeforeStop.endsWith("\n")) {
        out += `\nUser: <tool_response>\n${content}\n</tool_response>\n\n`;
      } else {
        out += `\n\nUser: <tool_response>\n${content}\n</tool_response>\n\n`;
      }
    } else {
      out += `User: <tool_response>\n${content}\n</tool_response>\n\n`;
    }
  }
  out += "Assistant: <think";
  return out;
}

/** Renders the incremental next user follow-up prompt after an assistant answer turn. */
export function buildFollowupUserText(
  followupPrompt: string,
  rawGeneratedText: string,
  matchedStop?: string | number | number[],
): string {
  const stopStr = typeof matchedStop === "string" ? matchedStop : "";
  const bodyBeforeStop = stopStr && rawGeneratedText.endsWith(stopStr)
    ? rawGeneratedText.slice(0, -stopStr.length)
    : rawGeneratedText;
  const trimmed = followupPrompt.trim();
  if (stopStr === "\nUser:") {
    return ` ${trimmed}\n\nAssistant: <think`;
  }
  const sep = (bodyBeforeStop.endsWith("\n\n") || (stopStr === "<EOD>" && bodyBeforeStop.endsWith("\n\n")))
    ? ""
    : bodyBeforeStop.endsWith("\n")
      ? "\n"
      : "\n\n";
  return `${sep}User: ${trimmed}\n\nAssistant: <think`;
}

/**
 * Checks ONLY the final user-facing answer language against the user's prompt language.
 * Never inspects `<think>` / CoT text.
 */
export function hasFinalAnswerLanguageMismatch(userPrompt: string, finalAnswer: string): boolean {
  const promptTrim = userPrompt.trim();
  const ansTrim = finalAnswer.trim();
  if (!promptTrim || ansTrim.length < 24) return false;
  // Strip fenced code blocks and inline code so code identifiers don't skew language detection.
  const cleanPrompt = promptTrim.replace(/```[\s\S]*?```/g, " ").replace(/`[^`]*`/g, " ");
  const cleanAns = ansTrim.replace(/```[\s\S]*?```/g, " ").replace(/`[^`]*`/g, " ");
  const countCjk = (s: string) => (s.match(/[\u4e00-\u9fff]/g) ?? []).length;
  const countLatin = (s: string) => (s.match(/[A-Za-z]/g) ?? []).length;
  const pCjk = countCjk(cleanPrompt);
  const pLatin = countLatin(cleanPrompt);
  const aCjk = countCjk(cleanAns);
  const aLatin = countLatin(cleanAns);
  if (aCjk + aLatin < 16) return false;
  const promptClearlyChinese = pCjk >= 6 && pCjk * 3 >= pLatin;
  const promptClearlyEnglish = pCjk === 0 && pLatin >= 24;
  if (promptClearlyChinese && aCjk === 0 && aLatin >= 24) return true;
  if (promptClearlyEnglish && aCjk >= 12 && aCjk * 2 > aLatin) return true;
  return false;
}

export function computeEpisodeReward(params: {
  passed: boolean;
  truncated: boolean;
  parseError?: string;
  loopGuardTriggered: boolean;
  languageMismatch: boolean;
  editCalls: number;
  editFailures: number;
  completedAssistantTurns: number;
  maxTurns: number;
}): { reward: number; breakdown: EpisodeOutcome["rewardBreakdown"] } {
  const taskScore = params.passed ? 1.0 : 0.0;
  const truncationPenalty = params.truncated || Boolean(params.parseError) ? -0.1 : 0.0;
  const loopPenalty = params.loopGuardTriggered ? -0.1 : 0.0;
  const languagePenalty = params.languageMismatch ? -0.05 : 0.0;
  const cleanEditBonus = params.passed
    && params.editCalls > 0
    && params.editFailures === 0
    && params.completedAssistantTurns <= Math.ceil(params.maxTurns * 0.75)
    ? 0.05
    : 0.0;
  const reward = Number((taskScore + truncationPenalty + loopPenalty + languagePenalty + cleanEditBonus).toFixed(6));
  return {
    reward,
    breakdown: {
      task: taskScore,
      truncationPenalty,
      loopPenalty,
      languagePenalty,
      cleanEditBonus,
    },
  };
}

/** Run the task's independent acceptance checks inside the episode's sandbox. */
async function evaluateChecks(
  task: Task,
  cwd: string,
  sandbox?: WorkspaceSandbox,
): Promise<{ passed: boolean; detail: string }[]> {
  const results: { passed: boolean; detail: string }[] = [];
  const agentCwd = sandbox?.cwd ?? cwd;
  for (const check of task.checks) {
    try {
      if (check.type === "file_equals") {
        const hostTarget = taskPath(cwd, check.path);
        if (sandbox) {
          const actual = (await sandbox.readFile(taskPath(agentCwd, check.path))).toString("utf8");
          results.push({ passed: actual === check.expected, detail: `file_equals:${check.path}` });
          continue;
        }
        const resolved = await realpath(hostTarget);
        if (!resolved.startsWith((await realpath(cwd)) + sep)) {
          throw new Error("Check path follows a symlink outside workspace.");
        }
        const actual = await readFile(resolved, "utf8");
        results.push({ passed: actual === check.expected, detail: `file_equals:${check.path}` });
      } else {
        if (sandbox) {
          const res = await sandbox.exec(check.argv, { timeoutMs: check.timeoutMs });
          const ok = res.exitCode === 0 && (check.expectedStdout === undefined || res.stdout.toString() === check.expectedStdout);
          results.push({ passed: ok, detail: (res.stdout.toString() + res.stderr.toString()).slice(0, 8192) });
          continue;
        }
        const argv = check.argv.map((arg) => rewriteWorkspaceInText(arg, cwd));
        const res = await execFileAsync(argv[0]!, argv.slice(1), {
          cwd,
          timeout: check.timeoutMs,
          maxBuffer: 256 * 1024,
        });
        const ok = check.expectedStdout === undefined || res.stdout === check.expectedStdout;
        results.push({ passed: ok, detail: (res.stdout + res.stderr).slice(0, 8192) });
      }
    } catch (error) {
      results.push({ passed: false, detail: error instanceof Error ? error.message : String(error) });
    }
  }
  return results;
}

interface ActiveRolloutEnv {
  rolloutIndex: number;
  workDir: string;
  cwd: string;
  sandbox?: WorkspaceSandbox;
  toolsByName: Map<string, ToolDefinition>;
  openaiTools: Record<string, unknown>[];
  historyIds: number[];
  turns: EpisodeTurnRecord[];
  /** Pi-compatible message log used exclusively by loop-guard. */
  guardMessages: Message[];
  /** Structured record of tool calls executed during the episode (written to `.task-evidence/final.json` when `task.answerEvidence` is true). */
  toolCallsLog: { name: string; arguments: Record<string, unknown> }[];
  active: boolean;
  userTurnIndex: number;
  completedAssistantTurns: number;
  generatedTokens: number;
  cachedTokens: number;
  prefillTokens: number;
  finalStopReason: EpisodeOutcome["finalStopReason"];
  finalAnswerText: string;
  editCalls: number;
  editFailures: number;
  truncated: boolean;
  loopGuardTriggered: boolean;
  parseError?: string;
}

async function createRolloutEnv(
  task: Task,
  rolloutIndex: number,
  scratchRoot: string,
  options: OnlineRlOptions,
): Promise<{ env: ActiveRolloutEnv; initialPromptText: string }> {
  const piTools = parseToolList(task.tools.filter((t): t is PiToolName => (PI_TOOL_NAMES as readonly string[]).includes(t)));
  const workDir = join(scratchRoot, `${task.id.replace(/[^a-zA-Z0-9._-]/g, "_")}-${rolloutIndex}-${randomUUID().slice(0, 8)}`);
  const cwd = join(workDir, "workspace");
  await mkdir(cwd, { recursive: true });
  if (task.environment && options.sandbox === "none") {
    throw new Error("Container tasks require sandbox isolation.");
  }
  const sandbox = options.sandbox === "none" ? undefined : new WorkspaceSandbox(cwd, task.environment);
  const agentCwd = sandbox?.cwd ?? cwd;
  const promptCwd = sandbox?.cwd ?? "/workspace";
  await sandbox?.probe();
  await sandbox?.probeSearch(piTools);
  if (task.environment) await materializeEnvironment(task.environment, cwd);
  for (const [relPath, content] of Object.entries(task.files)) {
    const target = taskPath(cwd, relPath);
    await mkdir(dirname(target), { recursive: true });
    const mode = content.startsWith("#!") ? 0o755 : 0o644;
    await writeFile(target, content, { flag: "wx", mode });
  }
  if (Object.prototype.hasOwnProperty.call(task.files, ".task-setup.sh")) {
    if (sandbox) {
      await sandbox.checked(["/bin/bash", ".task-setup.sh"]);
    } else {
      await execFileAsync("/bin/bash", [".task-setup.sh"], { cwd, timeout: 30_000, maxBuffer: 1024 * 1024 });
    }
  }
  const rawTools = sandbox?.tools(piTools) ?? hostTools(cwd, piTools);
  const wsFs = sandbox?.fs ?? hostFs;
  const lineTracker = options.editMode === "line" ? new LineViewTracker() : undefined;
  const customTools = options.editMode === "line"
    ? createLineModeTools(rawTools, agentCwd, wsFs, { tolerance: true, tracker: lineTracker })
    : rawTools.map((t) => (TOLERANT_TOOL_NAMES.includes(t.name) ? withTolerance(t, agentCwd, wsFs) : t));
  const toolsByName = new Map(customTools.map((t) => [t.name, t]));
  const openaiTools = toOpenAiTools(customTools);
  const systemText = buildRlSystemPrompt(customTools, promptCwd, options.editMode, options.appendSystemPrompt);
  const initialPromptText = renderPrompt(
    [
      { role: "system", content: systemText },
      { role: "user", content: task.prompt },
    ],
    openaiTools,
    true,
    true,
    "xml",
  );
  return {
    initialPromptText,
    env: {
      rolloutIndex,
      workDir,
      cwd,
      sandbox,
      toolsByName,
      openaiTools,
      historyIds: [],
      turns: [],
      guardMessages: [],
      toolCallsLog: [],
      active: true,
      userTurnIndex: 0,
      completedAssistantTurns: 0,
      generatedTokens: 0,
      cachedTokens: 0,
      prefillTokens: 0,
      finalStopReason: "max_turns",
      finalAnswerText: "",
      editCalls: 0,
      editFailures: 0,
      truncated: false,
      loopGuardTriggered: false,
    },
  };
}

function extractToolText(result: unknown): string {
  if (typeof result === "string") return result;
  if (object(result) && Array.isArray(result.content)) {
    return result.content
      .map((part) => (object(part) && typeof part.text === "string" ? part.text : ""))
      .filter(Boolean)
      .join("\n");
  }
  return JSON.stringify(result ?? "");
}

async function executeParsedCall(
  env: ActiveRolloutEnv,
  call: ParsedToolCall,
  options: OnlineRlOptions,
): Promise<{ text: string; isError: boolean; loopBlocked: boolean }> {
  const tool = env.toolsByName.get(call.name);
  if (!tool) {
    const declared = [...env.toolsByName.keys()].join(", ");
    return {
      text: `Error: Tool ${call.name} not found. Use only the declared tools: ${declared}.`,
      isError: true,
      loopBlocked: false,
    };
  }
  let parsedArgs: Record<string, unknown>;
  try {
    const value = JSON.parse(call.argumentsText);
    if (!object(value)) throw new Error("Arguments must be a JSON object.");
    parsedArgs = value;
  } catch (error) {
    return {
      text: `Error: Invalid arguments for ${call.name}: ${error instanceof Error ? error.message : String(error)}`,
      isError: true,
      loopBlocked: false,
    };
  }
  const guardReason = (options.identicalCallLimit > 0
    ? repeatedIdenticalCall(env.guardMessages, call.name, parsedArgs, options.identicalCallLimit)
    : undefined)
    ?? (options.targetFailureLimit > 0
      ? repeatedTargetFailure(env.guardMessages, call.name, parsedArgs, options.targetFailureLimit, options.editMode)
      : undefined);
  if (guardReason) {
    return { text: guardReason, isError: true, loopBlocked: true };
  }
  if (call.name === "edit") env.editCalls++;
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), options.toolTimeoutMs);
    let output: unknown;
    try {
      output = await (tool.execute as (...args: unknown[]) => Promise<unknown>)(
        call.id,
        parsedArgs,
        controller.signal,
        undefined,
        undefined,
      );
    } finally {
      clearTimeout(timer);
    }
    const rawText = slimToolResult(extractToolText(output), options.maxToolResultChars);
    const text = env.cwd !== "/workspace" ? rawText.replaceAll(env.cwd, "/workspace") : rawText;
    const isError = Boolean(object(output) && output.isError === true)
      || /^(?:Error:|Command exited with code [1-9]\d*)/m.test(text);
    if (call.name === "edit" && isError) env.editFailures++;
    return { text, isError, loopBlocked: false };
  } catch (error) {
    if (call.name === "edit") env.editFailures++;
    const msg = error instanceof Error ? error.message : String(error);
    return {
      text: `Error: ${env.cwd !== "/workspace" ? msg.replaceAll(env.cwd, "/workspace") : msg}`,
      isError: true,
      loopBlocked: false,
    };
  }
}

/**
 * SFT supervises `<EOD>` (token 0) only as `end_target` label at turn boundaries and NEVER feeds
 * token 0 into the RWKV state between turns (token 0 is RWKV-7's document-reset boundary!).
 * Strip any trailing token 0 from a non-final assistant turn before appending the next
 * `<tool_response>` or user follow-up so rollout & `/v1/train` stay 100% aligned without
 * resetting recurrent memory mid-episode.
 */
function stripTrailingEodBeforeContinuation(env: ActiveRolloutEnv): void {
  const lastTurn = env.turns[env.turns.length - 1];
  if (!lastTurn || lastTurn.lossMask !== 1 || lastTurn.tokens.length <= 1) return;
  if (lastTurn.tokens[lastTurn.tokens.length - 1] !== 0) return;
  lastTurn.tokens.pop();
  lastTurn.logprobs?.pop();
  if (env.historyIds[env.historyIds.length - 1] === 0) env.historyIds.pop();
  env.generatedTokens = Math.max(0, env.generatedTokens - 1);
}

/**
 * Rolls out a full group of `G` parallel isolated episodes for one task on a single server rank.
 * All active rollouts decode together in one batched `/v1/completions` call per turn.
 */
export async function rolloutTaskGroup(
  rank: OnlineRlRankClient,
  entry: LoadedTaskEntry,
  options: OnlineRlOptions,
  seedBase: number,
  scratchRoot: string,
): Promise<GroupRolloutResult | null> {
  const started = performance.now();
  const envs: ActiveRolloutEnv[] = [];
  let initialPromptText = "";
  try {
    for (let i = 0; i < options.groupSize; i++) {
      let created: { env: ActiveRolloutEnv; initialPromptText: string };
      try {
        created = await createRolloutEnv(entry.task, i, scratchRoot, options);
      } catch (error) {
        const msg = error instanceof Error ? error.message : String(error);
        console.warn(`[online-rl] skipping task ${entry.task.id} due to setup error: ${msg.split("\n")[0]}`);
        return null;
      }
      envs.push(created.env);
      if (i === 0) initialPromptText = created.initialPromptText;
    }
    const promptIds = await rank.tokenize(initialPromptText);
    if (promptIds.length + options.minGenHeadroom >= rank.health.ctx) {
      return null;
    }
    for (const env of envs) {
      env.historyIds = [...promptIds];
      env.turns.push({ tokens: [...promptIds], lossMask: 0, offset: 0 });
    }

    let totalCached = 0;
    let totalPrefill = 0;
    let totalGenerated = 0;

    for (let turnIdx = 0; turnIdx < options.maxTurns; turnIdx++) {
      const activeEnvs = envs.filter((e) => e.active);
      if (!activeEnvs.length) break;

      const comp = await rank.completions({
        prompts: activeEnvs.map((e) => e.historyIds),
        max_tokens: options.maxTokensPerTurn,
        temperature: options.temperature,
        ...(options.topP !== undefined ? { top_p: options.topP } : {}),
        ...(options.topK !== undefined ? { top_k: options.topK } : {}),
        ...(options.presencePenalty !== undefined ? { presence_penalty: options.presencePenalty } : {}),
        ...(options.repetitionPenalty !== undefined ? { repetition_penalty: options.repetitionPenalty } : {}),
        ...(options.penaltyDecay !== undefined ? { penalty_decay: options.penaltyDecay } : {}),
        seed: seedBase + turnIdx * 1009,
        stop: STOP_TEXTS,
        use_cache: true,
      });
      totalCached += comp.cached_prompt_tokens;
      totalPrefill += comp.prefill_tokens;
      totalGenerated += comp.generated_tokens;

      // Parse completions and execute sandbox tool calls concurrently across the group!
      await Promise.all(
        activeEnvs.map(async (env, idx) => {
          const item = comp.completions[idx]!;
          const turnOffset = env.historyIds.length;
          if (!item.tokens.length || item.tokens.length !== item.logprobs.length) {
            env.active = false;
            env.finalStopReason = "parse_error";
            env.parseError = "Empty completion or misaligned logprobs.";
            return;
          }
          env.historyIds.push(...item.tokens);
          env.turns.push({
            tokens: [...item.tokens],
            lossMask: 1,
            logprobs: [...item.logprobs],
            offset: turnOffset,
          });
          env.completedAssistantTurns++;
          env.generatedTokens += item.tokens.length;
          env.cachedTokens += item.cached_prompt_tokens;
          env.prefillTokens += item.prefill_tokens;

          if (item.truncated || item.finish_reason === "length") {
            env.active = false;
            env.truncated = true;
            env.finalStopReason = item.truncation_reason === "context_length" ? "context_length" : "length";
            return;
          }

          const parser = new ContinuationParser(true, env.openaiTools, "auto", "xml", true);
          let parsedDeltas: Record<string, unknown>[];
          try {
            const pushDeltas = parser.push(item.text);
            const matchedStop = typeof item.matched_stop === "string" && STOP_TEXTS.includes(item.matched_stop)
              ? item.matched_stop
              : undefined;
            const fin = parser.finish(item.finish_reason, matchedStop);
            parsedDeltas = [...pushDeltas, ...fin.deltas];
          } catch (error) {
            env.active = false;
            env.finalStopReason = "parse_error";
            env.parseError = error instanceof Error ? error.message : String(error);
            return;
          }

          const textParts: string[] = [];
          const toolCalls: ParsedToolCall[] = [];
          for (const d of parsedDeltas) {
            if (typeof d.content === "string") textParts.push(d.content);
            if (Array.isArray(d.tool_calls)) {
              for (const c of d.tool_calls as any[]) {
                toolCalls.push({
                  id: String(c.id ?? `call_${randomUUID().slice(0, 8)}`),
                  name: String(c.function?.name ?? ""),
                  argumentsText: String(c.function?.arguments ?? "{}"),
                });
              }
            }
          }

          if (!toolCalls.length) {
            const answerText = textParts.join("").trim();
            if (env.userTurnIndex < entry.followups.length) {
              const nextFollowup = entry.followups[env.userTurnIndex]!;
              env.userTurnIndex++;
              if (turnIdx + 1 >= options.maxTurns) {
                env.active = false;
                env.finalStopReason = "max_turns";
                return;
              }
              const followupText = buildFollowupUserText(nextFollowup, item.text, item.matched_stop);
              const followupIds = await rank.tokenize(followupText);
              if (env.historyIds.length + followupIds.length + options.minGenHeadroom >= rank.health.ctx) {
                env.active = false;
                env.truncated = true;
                env.finalStopReason = "context_length";
                return;
              }
              stripTrailingEodBeforeContinuation(env);
              const fuOffset = env.historyIds.length;
              env.historyIds.push(...followupIds);
              env.turns.push({ tokens: followupIds, lossMask: 0, offset: fuOffset });
              return;
            }
            env.active = false;
            env.finalStopReason = "stop";
            env.finalAnswerText = answerText;
            return;
          }

          // Record assistant message for loop-guard before executing calls.
          const assistantContent: any[] = [];
          for (const tc of toolCalls) {
            let argsObj: Record<string, unknown> = {};
            try {
              const v = JSON.parse(tc.argumentsText);
              if (object(v)) argsObj = v;
            } catch {}
            assistantContent.push({ type: "toolCall", id: tc.id, name: tc.name, arguments: argsObj });
            env.toolCallsLog.push({ name: tc.name, arguments: argsObj });
          }
          env.guardMessages.push({
            role: "assistant",
            content: assistantContent,
            api: "openai-completions",
            provider: "rwkv",
            model: "rwkv-rl",
            usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
            stopReason: "toolUse",
            timestamp: Date.now(),
          } as Message);

          const toolResultTexts: string[] = [];
          let hitLoopGuard = false;
          for (const tc of toolCalls) {
            const execRes = await executeParsedCall(env, tc, options);
            toolResultTexts.push(execRes.text);
            if (execRes.loopBlocked) hitLoopGuard = true;
            env.guardMessages.push({
              role: "toolResult",
              toolCallId: tc.id,
              toolName: tc.name,
              content: [{ type: "text", text: execRes.text }],
              isError: execRes.isError,
              timestamp: Date.now(),
            } as Message);
          }

          if (hitLoopGuard) {
            env.active = false;
            env.loopGuardTriggered = true;
            env.finalStopReason = "loop_guard";
            return;
          }

          // Do not append another observation if this was the last allowed turn.
          if (turnIdx + 1 >= options.maxTurns) {
            env.active = false;
            env.finalStopReason = "max_turns";
            return;
          }

          const obsText = buildToolObservationText(toolResultTexts, item.text, item.matched_stop);
          const obsIds = await rank.tokenize(obsText);
          if (env.historyIds.length + obsIds.length + options.minGenHeadroom >= rank.health.ctx) {
            env.active = false;
            env.truncated = true;
            env.finalStopReason = "context_length";
            return;
          }
          stripTrailingEodBeforeContinuation(env);
          const obsOffset = env.historyIds.length;
          env.historyIds.push(...obsIds);
          env.turns.push({ tokens: obsIds, lossMask: 0, offset: obsOffset });
        }),
      );
    }

    // Independently verify all workspaces in parallel!
    const episodes: EpisodeOutcome[] = await Promise.all(
      envs.map(async (env) => {
        const canPass = env.finalStopReason === "stop" && !env.truncated && !env.parseError && !env.loopGuardTriggered;
        if (canPass && entry.task.answerEvidence) {
          const evDir = join(env.cwd, ".task-evidence");
          await mkdir(evDir, { recursive: true });
          await writeFile(
            join(evDir, "final.json"),
            JSON.stringify({ finalAnswer: env.finalAnswerText, toolCalls: env.toolCallsLog }, null, 2) + "\n",
            "utf8",
          );
        }
        const checks = canPass ? await evaluateChecks(entry.task, env.cwd, env.sandbox) : [];
        const passed = canPass && checks.length === entry.task.checks.length && checks.every((c) => c.passed);
        const languageMismatch = canPass && hasFinalAnswerLanguageMismatch(entry.task.prompt, env.finalAnswerText);
        const { reward, breakdown } = computeEpisodeReward({
          passed,
          truncated: env.truncated,
          parseError: env.parseError,
          loopGuardTriggered: env.loopGuardTriggered,
          languageMismatch,
          editCalls: env.editCalls,
          editFailures: env.editFailures,
          completedAssistantTurns: env.completedAssistantTurns,
          maxTurns: options.maxTurns,
        });
        return {
          taskId: entry.task.id,
          cohort: entry.cohort,
          rolloutIndex: env.rolloutIndex,
          turns: env.turns,
          fullTokens: env.historyIds,
          completedAssistantTurns: env.completedAssistantTurns,
          generatedTokens: env.generatedTokens,
          cachedTokens: env.cachedTokens,
          prefillTokens: env.prefillTokens,
          finalStopReason: env.finalStopReason,
          finalAnswerText: env.finalAnswerText,
          passed,
          checks,
          editCalls: env.editCalls,
          editFailures: env.editFailures,
          truncated: env.truncated,
          loopGuardTriggered: env.loopGuardTriggered,
          parseError: env.parseError,
          languageMismatch,
          reward,
          rewardBreakdown: breakdown,
        };
      }),
    );

    const rewards = episodes.map((e) => e.reward);
    const mean = rewards.reduce((a, b) => a + b, 0) / rewards.length;
    const variance = rewards.reduce((a, r) => a + (r - mean) ** 2, 0) / rewards.length;
    const std = Math.sqrt(variance);
    const passRate = episodes.filter((e) => e.passed).length / episodes.length;

    return {
      task: entry.task,
      cohort: entry.cohort,
      episodes,
      rewardMean: mean,
      rewardStd: std,
      passRate,
      rolloutSeconds: (performance.now() - started) / 1000,
      cachedTokens: totalCached,
      prefillTokens: totalPrefill,
      generatedTokens: totalGenerated,
    };
  } finally {
    if (!options.keepWorkspaces) {
      for (const env of envs) {
        try {
          rmSync(env.workDir, { recursive: true, force: true });
        } catch {}
      }
    }
  }
}

export interface PreparedTrainingStep {
  samplesByRank: TrainSample[][];
  globalTrainedTokens: number;
  expectedRolloutLoss: number;
  meanKl: number;
  maxAbsKl: number;
  klSeconds: number;
}

/**
 * Converts active groups across all DP ranks into `/v1/train` samples with exact Dr.GRPO
 * constant-denominator scaling and optional per-token reference KL penalty via `/v1/logprobs`.
 */
export async function buildTrainingBatch(
  cluster: OnlineRlCluster,
  groupsByRank: readonly (readonly GroupRolloutResult[])[],
  options: OnlineRlOptions,
): Promise<PreparedTrainingStep> {
  let globalTrainedTokens = 0;
  let totalGroups = 0;
  for (const rankGroups of groupsByRank) {
    for (const g of rankGroups) {
      totalGroups++;
      for (const ep of g.episodes) {
        globalTrainedTokens += ep.generatedTokens;
      }
    }
  }
  if (globalTrainedTokens <= 0 || totalGroups <= 0) {
    throw new Error("No informative groups available for training.");
  }

  const groupBudget = options.drGrpoGroupBudget ?? (options.groupSize * options.maxTokensPerTurn * options.maxTurns);
  const stepBudget = totalGroups * groupBudget;
  const drGrpoScale = globalTrainedTokens / stepBudget;

  // Score reference logprobs in parallel across all ranks if klBeta > 0.
  const klStart = performance.now();
  const refLogprobsByRank: (number[][][] | null)[] = await Promise.all(
    cluster.ranks.map(async (rank, rIdx) => {
      if (options.klBeta <= 0) return null;
      const rankGroups = groupsByRank[rIdx]!;
      const flatSeqs: number[][] = [];
      for (const g of rankGroups) {
        for (const ep of g.episodes) flatSeqs.push(ep.fullTokens);
      }
      if (!flatSeqs.length) return [];
      const flatRef: number[][] = [];
      for (let off = 0; off < flatSeqs.length; off += rank.health.max_batch) {
        const slice = flatSeqs.slice(off, off + rank.health.max_batch);
        const res = await rank.logprobs(slice, options.temperature, rank.referenceVersion);
        flatRef.push(...res.logprobs);
      }
      const nested: number[][][] = [];
      let cursor = 0;
      for (const g of rankGroups) {
        const gRefs: number[][] = [];
        for (let i = 0; i < g.episodes.length; i++) gRefs.push(flatRef[cursor++]!);
        nested.push(gRefs);
      }
      return nested;
    }),
  );
  const klSeconds = options.klBeta > 0 ? (performance.now() - klStart) / 1000 : 0;

  let weightedLogprobSum = 0;
  let klSum = 0;
  let klCount = 0;
  let maxAbsKl = 0;

  const samplesByRank: TrainSample[][] = groupsByRank.map((rankGroups, rIdx) => {
    const out: TrainSample[] = [];
    const rankRefs = refLogprobsByRank[rIdx];
    for (let gIdx = 0; gIdx < rankGroups.length; gIdx++) {
      const group = rankGroups[gIdx]!;
      const mean = group.rewardMean;
      const std = Math.max(group.rewardStd, 1e-8);
      for (let eIdx = 0; eIdx < group.episodes.length; eIdx++) {
        const ep = group.episodes[eIdx]!;
        const baseAdv = options.algorithm === "dr_grpo"
          ? (ep.reward - mean) * drGrpoScale
          : (ep.reward - mean) / std;
        const refSeq = rankRefs?.[gIdx]?.[eIdx];
        const trainTurns: TrainTurn[] = [];
        for (const turn of ep.turns) {
          if (turn.lossMask === 0) {
            trainTurns.push({
              tokens: turn.tokens,
              loss_mask: new Array<0>(turn.tokens.length).fill(0),
              advantages: 0,
            });
            continue;
          }
          const lps = turn.logprobs!;
          if (options.klBeta > 0 && refSeq) {
            const perTokAdv = new Array<number>(turn.tokens.length);
            for (let k = 0; k < turn.tokens.length; k++) {
              const predPos = turn.offset + k - 1;
              const lp = lps[k]!;
              const refLp = refSeq[predPos]!;
              const rawKl = lp - refLp;
              const clippedKl = Math.max(-options.klClip, Math.min(options.klClip, rawKl));
              klSum += clippedKl;
              klCount++;
              if (Math.abs(rawKl) > maxAbsKl) maxAbsKl = Math.abs(rawKl);
              // Scale KL term by the same Dr.GRPO factor when algorithm === "dr_grpo" so KL weight stays invariant to budget.
              const advT = baseAdv - options.klBeta * clippedKl * (options.algorithm === "dr_grpo" ? drGrpoScale : 1.0);
              perTokAdv[k] = advT;
              weightedLogprobSum += advT * lp;
            }
            trainTurns.push({
              tokens: turn.tokens,
              loss_mask: new Array<1>(turn.tokens.length).fill(1),
              advantages: perTokAdv,
            });
          } else {
            for (let k = 0; k < lps.length; k++) weightedLogprobSum += baseAdv * lps[k]!;
            trainTurns.push({
              tokens: turn.tokens,
              loss_mask: new Array<1>(turn.tokens.length).fill(1),
              advantages: baseAdv,
            });
          }
        }
        out.push({
          turns: trainTurns,
          advantages: baseAdv,
          temperature: options.temperature,
        });
      }
    }
    return out;
  });

  return {
    samplesByRank,
    globalTrainedTokens,
    expectedRolloutLoss: -weightedLogprobSum / globalTrainedTokens,
    meanKl: klCount > 0 ? klSum / klCount : 0,
    maxAbsKl,
    klSeconds,
  };
}

export interface OnlineRlStepMetric {
  update: number;
  serverStep: number;
  algorithm: RlAlgorithm;
  activeGroups: number;
  zeroVarSkipped: number;
  ctxSkipped: number;
  rewardMean: number;
  rewardStdMean: number;
  passRate: number;
  loss: number;
  rolloutLoss: number;
  lossAbsError: number;
  gradNorm: number;
  learningRate: number;
  meanKl: number;
  maxAbsKl: number;
  trainedTokens: number;
  totalTokens: number;
  generatedTokens: number;
  cacheHitRate: number;
  rolloutSeconds: number;
  klSeconds: number;
  trainSeconds: number;
  publishSeconds: number;
  stepSeconds: number;
  directory?: string;
  checkpoint?: string;
}

export interface OnlineRlRunSummary {
  updatesCompleted: number;
  finalServerStep: number;
  metricsPath: string;
  lastCheckpoint?: string;
  lastDirectory?: string;
}

/** Main Agentic Dr.GRPO + KL online RL training loop across 1..N NCCL DP ranks. */
export async function runOnlineRl(cluster: OnlineRlCluster, options: OnlineRlOptions): Promise<OnlineRlRunSummary> {
  await cluster.connect();
  if (options.groupSize < 2) throw new Error("--group-size must be >= 2.");
  if (options.groupSize > cluster.maxBatch) {
    throw new Error(`--group-size (${options.groupSize}) exceeds server max_batch (${cluster.maxBatch}).`);
  }
  const maxSamplesPerRank = options.tasksPerRank * options.groupSize;
  if (maxSamplesPerRank > cluster.maxBatch) {
    throw new Error(
      `tasksPerRank * groupSize (${maxSamplesPerRank}) exceeds server max_batch (${cluster.maxBatch}); raise --max-batch on the server or lower --tasks-per-rank.`,
    );
  }

  mkdirSync(options.outputDir, { recursive: true, mode: 0o700 });
  const scratchRoot = join(options.outputDir, "scratch");
  mkdirSync(scratchRoot, { recursive: true, mode: 0o700 });
  const metricsPath = join(options.outputDir, "metrics.jsonl");
  const evalMetricsPath = join(options.outputDir, "eval.jsonl");

  const entries = loadTaskEntries(options.tasksPath);
  const evalEntries = options.evalTasksPath ? loadTaskEntries(options.evalTasksPath) : [];
  const rng = mulberry32(options.seed);
  const shuffled = [...entries];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j]!, shuffled[i]!];
  }
  let taskCursor = 0;
  const nextTask = (): LoadedTaskEntry => {
    const item = shuffled[taskCursor % shuffled.length]!;
    taskCursor++;
    return item;
  };

  let updatesCompleted = 0;
  let lastDirectory: string | undefined;
  let lastCheckpoint: string | undefined;

  const runEval = async (serverStep: number) => {
    if (!evalEntries.length) return;
    const subset = options.evalLimit > 0 ? evalEntries.slice(0, options.evalLimit) : evalEntries;
    const evalOpts: OnlineRlOptions = { ...options, groupSize: 1 };
    const buckets: LoadedTaskEntry[][] = cluster.ranks.map(() => []);
    subset.forEach((e, idx) => buckets[idx % cluster.ranks.length]!.push(e));
    const results = await Promise.all(
      cluster.ranks.map(async (rank, rIdx) => {
        let passed = 0;
        let total = 0;
        let rewardSum = 0;
        for (let k = 0; k < buckets[rIdx]!.length; k++) {
          const g = await rolloutTaskGroup(rank, buckets[rIdx]![k]!, evalOpts, options.seed + 900_000 + serverStep * 100 + rIdx * 1000 + k, scratchRoot);
          if (!g) continue;
          total++;
          if (g.episodes[0]!.passed) passed++;
          rewardSum += g.episodes[0]!.reward;
        }
        return { passed, total, rewardSum };
      }),
    );
    const passed = results.reduce((a, b) => a + b.passed, 0);
    const total = results.reduce((a, b) => a + b.total, 0);
    const rewardMean = total > 0 ? results.reduce((a, b) => a + b.rewardSum, 0) / total : 0;
    const row = {
      type: "eval",
      serverStep,
      passed,
      total,
      passRate: total > 0 ? passed / total : 0,
      rewardMean,
      timestamp: new Date().toISOString(),
    };
    appendFileSync(evalMetricsPath, JSON.stringify(row) + "\n", { mode: 0o600 });
    console.log(`[online-rl][eval] step=${serverStep} pass=${passed}/${total} (${(row.passRate * 100).toFixed(1)}%) reward=${rewardMean.toFixed(3)}`);
  };

  for (let update = 1; update <= options.steps; update++) {
    const stepStart = performance.now();
    let zeroVarSkipped = 0;
    let ctxSkipped = 0;

    // Parallel rollout + dynamic sampling across all DP ranks!
    const rolloutStart = performance.now();
    const groupsByRank: GroupRolloutResult[][] = await Promise.all(
      cluster.ranks.map(async (rank, rIdx) => {
        const kept: GroupRolloutResult[] = [];
        for (let slot = 0; slot < options.tasksPerRank; slot++) {
          for (let attempt = 0; attempt <= options.maxResampleAttempts; attempt++) {
            const entry = nextTask();
            const seedBase = options.seed + update * 100_003 + rIdx * 10_007 + slot * 503 + attempt * 71;
            const group = await rolloutTaskGroup(rank, entry, options, seedBase, scratchRoot);
            if (!group) {
              ctxSkipped++;
              continue;
            }
            if (group.rewardStd < 1e-6 || (options.requirePassVariance && (group.passRate <= 0 || group.passRate >= 1))) {
              zeroVarSkipped++;
              continue;
            }
            kept.push(group);
            break;
          }
        }
        return kept;
      }),
    );
    const rolloutSeconds = (performance.now() - rolloutStart) / 1000;

    const flatGroups = groupsByRank.flat();
    if (!flatGroups.length) {
      console.warn(`[online-rl] update=${update}: all ${cluster.dpWorldSize * options.tasksPerRank} slots had zero reward variance after resampling; skipping optimizer step.`);
      continue;
    }

    const prepared = await buildTrainingBatch(cluster, groupsByRank, options);
    const nextOptStep = cluster.step + 1;
    const lr = computeLearningRate(nextOptStep, {
      lr: options.lr,
      lrMin: options.lrMin,
      warmupSteps: options.warmupSteps,
      totalSteps: options.steps,
      schedule: options.lrSchedule,
    });

    const trainRes = await cluster.trainStep(prepared.samplesByRank, lr, options.temperature);
    let publishSeconds = 0;
    let directory: string | undefined;
    let checkpoint: string | undefined;

    if (trainRes.updated) {
      const pubStart = performance.now();
      const shouldCheckpoint = options.checkpointEvery > 0 && (update % options.checkpointEvery === 0 || update === options.steps);
      const shouldSave = shouldCheckpoint || (options.saveEvery > 0 && update % options.saveEvery === 0) || update === options.steps;
      const pub = await cluster.publishStep(trainRes.step, { save: shouldSave, checkpoint: shouldCheckpoint });
      publishSeconds = (performance.now() - pubStart) / 1000;
      directory = pub.directory;
      checkpoint = pub.checkpoint;
      if (directory) lastDirectory = directory;
      if (checkpoint) lastCheckpoint = checkpoint;
      updatesCompleted++;
    }

    const allEps = flatGroups.flatMap((g) => g.episodes);
    const rewardMean = allEps.reduce((a, e) => a + e.reward, 0) / allEps.length;
    const rewardStdMean = flatGroups.reduce((a, g) => a + g.rewardStd, 0) / flatGroups.length;
    const passRate = allEps.filter((e) => e.passed).length / allEps.length;
    const cachedTokens = flatGroups.reduce((a, g) => a + g.cachedTokens, 0);
    const prefillTokens = flatGroups.reduce((a, g) => a + g.prefillTokens, 0);
    const generatedTokens = flatGroups.reduce((a, g) => a + g.generatedTokens, 0);
    const metric: OnlineRlStepMetric = {
      update,
      serverStep: trainRes.step,
      algorithm: options.algorithm,
      activeGroups: flatGroups.length,
      zeroVarSkipped,
      ctxSkipped,
      rewardMean,
      rewardStdMean,
      passRate,
      loss: trainRes.loss,
      rolloutLoss: prepared.expectedRolloutLoss,
      lossAbsError: Math.abs(trainRes.loss - prepared.expectedRolloutLoss),
      gradNorm: trainRes.gradNorm,
      learningRate: trainRes.learningRate,
      meanKl: prepared.meanKl,
      maxAbsKl: prepared.maxAbsKl,
      trainedTokens: trainRes.trainedTokens,
      totalTokens: trainRes.totalTokens,
      generatedTokens,
      cacheHitRate: cachedTokens / Math.max(1, cachedTokens + prefillTokens),
      rolloutSeconds,
      klSeconds: prepared.klSeconds,
      trainSeconds: trainRes.seconds,
      publishSeconds,
      stepSeconds: (performance.now() - stepStart) / 1000,
      ...(directory ? { directory } : {}),
      ...(checkpoint ? { checkpoint } : {}),
    };
    appendFileSync(metricsPath, JSON.stringify(metric) + "\n", { mode: 0o600 });
    console.log(
      `[online-rl] step=${metric.serverStep} (${update}/${options.steps}) ` +
      `groups=${metric.activeGroups} skip0=${metric.zeroVarSkipped} pass=${(metric.passRate * 100).toFixed(1)}% ` +
      `R=${metric.rewardMean.toFixed(3)}±${metric.rewardStdMean.toFixed(3)} ` +
      `loss=${metric.loss.toFixed(4)} err=${metric.lossAbsError.toExponential(2)} ` +
      `kl=${metric.meanKl.toFixed(4)} grad=${metric.gradNorm.toFixed(3)} lr=${metric.learningRate.toExponential(2)} ` +
      `cache=${(metric.cacheHitRate * 100).toFixed(0)}% t=${metric.stepSeconds.toFixed(1)}s`,
    );

    if (options.evalEvery > 0 && evalEntries.length > 0 && (update % options.evalEvery === 0 || update === options.steps)) {
      await runEval(trainRes.step);
    }
  }

  return {
    updatesCompleted,
    finalServerStep: cluster.step,
    metricsPath,
    lastCheckpoint,
    lastDirectory,
  };
}
