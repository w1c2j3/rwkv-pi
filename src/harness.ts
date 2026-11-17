import { execFile } from "node:child_process";
import { randomUUID } from "node:crypto";
import { appendFileSync } from "node:fs";
import { mkdir, readFile, realpath, rename, writeFile } from "node:fs/promises";
import { dirname, join, resolve, sep } from "node:path";
import { promisify } from "node:util";
import { InMemoryCredentialStore, InMemoryModelsStore, type AssistantMessage, type Message, type Provider } from "@earendil-works/pi-ai";
import {
  createAgentSession, createBashToolDefinition, createEditToolDefinition, createExtensionRuntime, createLsToolDefinition,
  createReadToolDefinition, createWriteToolDefinition, ModelRuntime, SessionManager, SettingsManager, type ResourceLoader, type ToolDefinition,
} from "@earendil-works/pi-coding-agent";
import { digest, runtimeProvenance } from "./provenance.js";
import { object, parseTask, taskPath, WEB_TOOL_NAMES, type Task } from "./tasks.js";
import { WorkspaceSandbox } from "./sandbox.js";
import { materializeEnvironment } from "./environment.js";
import { parseToolList, PI_TOOL_NAMES, type PiToolName } from "./tools.js";
import type { RwkvConfig } from "./config.js";
import { foldEpochRequest, type FoldOpenAIEpochOptions } from "./epoch.js";
import { budgetRwkvRequest, type TokenBudget } from "./token-budget.js";
import { promptVersion } from "./prompt.js";
import { repeatedIdenticalCall, repeatedTargetFailure, repeatedWorklistSpin } from "./loop-guard.js";
import { hostFs, TOLERANT_TOOL_NAMES, withTolerance } from "./tolerant-tools.js";
import { createWorklistTools, foldWorklistRequest, type FoldOpenAIWorklistOptions, isWorklistTool, PiWorklistState, WORKLIST_TOOL_NAMES } from "./worklist.js";
import type { createRwkvProvider } from "./provider.js";

const runCommand = promisify(execFile);
export interface Step {
  request: Record<string, unknown> & { messages: Record<string, unknown>[]; tools: unknown[] };
  tokenBudget?: TokenBudget;
  response?: AssistantMessage;
}
export interface RunRecord {
  schemaVersion: 1;
  runId: string;
  task: Task;
  taskSha256: string;
  role: "teacher" | "student" | "smoke" | "question-author";
  model: { provider: string; id: string };
  provenance: ReturnType<typeof runtimeProvenance>;
  isolation: "bubblewrap" | "docker" | "none";
  workspacePath?: string;
  responseAdapter: "none" | "rwkv-json-v1" | "rwkv-xml-v1" | "rwkv-xml-continuation-v2" | "rwkv-json-continuation-v1";
  promptVersion?: string;
  loopGuard?: { repeatFailureLimit: number; identicalCallLimit?: number; targetFailureLimit?: number };
  /** Recall-tolerant read/edit/write/ls execution (src/tolerant-tools.ts). */
  toolTolerance?: "rwkv-tolerant-tools-v1";
  toolCallRecovery?: "pi-native-validation";
  historyMode?: "full" | "epoch" | "worklist";
  appendSystemPrompt?: string;
  promptSuffix?: string;
  conversation?: { allTurnsCompleted: boolean; turns: { index: number; userPrompt: string; completed: boolean }[] };
  inference?: { deploymentSha256: string; model: string; engineRevision: string; metadataSha256: Record<string, string> };
  startedAt: string;
  finishedAt: string;
  budget: { maxRequests: number; timeoutMs: number };
  generation?: { contextWindow: number; maxTokens: number; thinkingLevel: "high" | "off" };
  status: "passed" | "failed" | "error" | "budget_exhausted" | "timeout";
  error?: string;
  checks: { passed: boolean; detail: string }[];
  steps: Step[];
  messages: unknown[];
}
export interface HarnessOptions {
  provider: Provider;
  role: RunRecord["role"];
  outputDir: string;
  maxRequests: number;
  timeoutMs: number;
  sandbox?: "bubblewrap" | "none";
  /** Optional unseen mount for path-generalization tests; no path aliases are created. */
  workspaceMount?: string;
  inference?: RunRecord["inference"];
  contextBudget?: RwkvConfig;
  thinkingLevel?: "high" | "off";
  /** Optional sequential user follow-up prompts sharing the same session and workspace. */
  followups?: readonly (string | { prompt: string })[];
  appendSystemPrompt?: string;
  promptSuffix?: string;
  webTools?: ToolDefinition[];
  /** Optional explicit history compaction mode for teacher/collector runs (student runs inherit from provider). */
  historyMode?: "full" | "epoch" | "worklist";
  epoch?: FoldOpenAIEpochOptions;
  worklist?: FoldOpenAIWorklistOptions;
}

export async function runTask(input: Task, options: HarnessOptions): Promise<{ record: RunRecord; path: string }> {
  const task = parseTask(input);
  for (const number of [options.maxRequests, options.timeoutMs]) {
    if (!Number.isSafeInteger(number) || number <= 0) throw new Error("Harness budgets must be positive integers.");
  }
  const model = options.provider.getModels()[0];
  if (!model || !["openai-completions", "openai-responses"].includes(model.api)) {
    throw new Error("Harness v1 requires an OpenAI Chat Completions or Responses model.");
  }
  const piTools = task.tools.filter((t): t is PiToolName => (PI_TOOL_NAMES as readonly string[]).includes(t));
  const requestedWebTools = task.tools.filter((t) => (WEB_TOOL_NAMES as readonly string[]).includes(t));
  const extraWebTools = options.webTools ?? [];
  const providedWebNames = new Set(extraWebTools.map((t) => t.name));
  if (requestedWebTools.some((name) => !providedWebNames.has(name))) {
    throw new Error(`Task requires web tools (${requestedWebTools.join(", ")}), but they were not configured.`);
  }
  const followups = (options.followups ?? []).map((f) => (typeof f === "string" ? f : f?.prompt));
  if (followups.some((f) => typeof f !== "string" || !f.trim())) throw new Error("Invalid follow-up prompt.");
  const runId = randomUUID();
  const directory = resolve(options.outputDir, runId);
  const cwd = join(directory, "workspace");
  await mkdir(cwd, { recursive: true });
  if (task.environment && options.sandbox === "none") throw new Error("Environment tasks always run in their container.");
  if (options.workspaceMount && (options.sandbox === "none" || task.environment)) {
    throw new Error("A custom workspace mount requires the bubblewrap sandbox without a container environment.");
  }
  const sandbox = options.sandbox === "none" ? undefined : new WorkspaceSandbox(cwd, task.environment, options.workspaceMount);
  // The agent and the checks see the workspace at the sandbox's path (the image workdir for containers).
  const agentCwd = sandbox?.cwd ?? cwd;
  await sandbox?.probe();
  await sandbox?.probeSearch(piTools);
  if (task.environment) await materializeEnvironment(task.environment, cwd);
  for (const [path, content] of Object.entries(task.files)) {
    const target = taskPath(cwd, path);
    await mkdir(dirname(target), { recursive: true });
    await writeFile(target, content, { flag: "wx" });
  }
  const { id: _id, split: _split, ...definition } = task;
  // Tags createRwkvProvider / createTeacherProvider add; options override for standalone collectors.
  const rwkv = options.provider as Partial<ReturnType<typeof createRwkvProvider>> & {
    rwkvPiEpochConfig?: FoldOpenAIEpochOptions;
    rwkvPiWorklistConfig?: FoldOpenAIWorklistOptions;
  };
  const identicalCallLimit = rwkv.rwkvPiIdenticalCallLimit ?? 0;
  const targetFailureLimit = rwkv.rwkvPiTargetFailureLimit ?? 0;
  const tolerant = rwkv.rwkvPiToolTolerance === true;
  const effectiveHistoryMode = options.historyMode ?? rwkv.rwkvPiHistoryMode ?? "full";
  const effectiveEpochConfig = options.epoch ?? rwkv.rwkvPiEpochConfig;
  const effectiveWorklistConfig = options.worklist ?? rwkv.rwkvPiWorklistConfig;
  const worklistMode = effectiveHistoryMode === "worklist";
  const worklistState = worklistMode ? new PiWorklistState() : undefined;
  const record: RunRecord = {
    schemaVersion: 1, runId, task, taskSha256: digest(definition), role: options.role,
    model: { provider: model.provider, id: model.id }, provenance: runtimeProvenance(),
    isolation: sandbox?.kind ?? "none",
    workspacePath: agentCwd,
    responseAdapter: rwkv.rwkvPiToolFormat === "json" ? "rwkv-json-continuation-v1" : rwkv.rwkvPiToolFormat === "xml" ? "rwkv-xml-continuation-v2" : "none",
    ...(rwkv.rwkvPiToolFormat === "xml" || rwkv.rwkvPiToolFormat === "json" ? { promptVersion: promptVersion(rwkv.rwkvPiToolFormat) } : {}),
    ...(rwkv.rwkvPiRepeatFailureLimit !== undefined
      ? { loopGuard: { repeatFailureLimit: rwkv.rwkvPiRepeatFailureLimit, ...(identicalCallLimit ? { identicalCallLimit } : {}),
        ...(targetFailureLimit ? { targetFailureLimit } : {}) } } : {}),
    ...(tolerant ? { toolTolerance: "rwkv-tolerant-tools-v1" as const } : {}),
    ...(rwkv.rwkvPiToolCallRecovery ? { toolCallRecovery: "pi-native-validation" as const } : {}),
    ...(options.historyMode !== undefined || effectiveHistoryMode === "epoch" || effectiveHistoryMode === "worklist" ? { historyMode: effectiveHistoryMode } : {}),
    ...(options.appendSystemPrompt !== undefined ? { appendSystemPrompt: options.appendSystemPrompt } : {}),
    ...(options.promptSuffix !== undefined ? { promptSuffix: options.promptSuffix } : {}),
    ...(options.inference ? { inference: options.inference } : {}),
    startedAt: new Date().toISOString(), finishedAt: "", budget: { maxRequests: options.maxRequests, timeoutMs: options.timeoutMs },
    generation: { contextWindow: model.contextWindow, maxTokens: model.maxTokens, thinkingLevel: options.thinkingLevel ?? "high" },
    status: "error", checks: [], steps: [], messages: [],
  };
  const eventsPath = join(directory, "events.jsonl");
  const log = (event: unknown) => appendFileSync(eventsPath, JSON.stringify(event) + "\n", { mode: 0o600 });
  log({ type: "run_start", runId, taskSha256: record.taskSha256, provenance: record.provenance, budget: record.budget, generation: record.generation });
  const settingsManager = SettingsManager.inMemory({
    compaction: { enabled: false }, retry: { enabled: false, provider: { maxRetries: 0 } },
    cacheWarming: "off",
  });
  // Empty resource discovery leaves pi's own system prompt and native tools intact.
  const resourceLoader: ResourceLoader = {
    getExtensions: () => ({ extensions: [], errors: [], runtime: createExtensionRuntime() }),
    getSkills: () => ({ skills: [], diagnostics: [] }),
    getPrompts: () => ({ prompts: [], diagnostics: [] }),
    getThemes: () => ({ themes: [], diagnostics: [] }),
    getAgentsFiles: () => ({ agentsFiles: [] }),
    getSystemPrompt: () => undefined, getSystemPromptSource: () => undefined,
    getAppendSystemPrompt: () => (options.appendSystemPrompt ? [options.appendSystemPrompt] : []), getAppendSystemPromptSources: () => [],
    extendResources: () => {}, reload: async () => {},
  };
  const modelRuntime = await ModelRuntime.create({
    credentials: new InMemoryCredentialStore(), modelsStore: new InMemoryModelsStore(),
    modelsPath: null, refreshOnCreate: false, allowModelNetwork: false,
  });
  modelRuntime.registerNativeProvider(options.provider);
  const rawTools = sandbox?.tools(piTools) ?? hostTools(cwd, piTools);
  const baseCustomTools = tolerant ? tolerantTools(rawTools, agentCwd, sandbox?.fs ?? hostFs) : rawTools;
  const customTools = [...baseCustomTools, ...extraWebTools, ...(worklistState ? createWorklistTools(worklistState) : [])];
  const orderedPiTools = piTools.length ? parseToolList(piTools) : [];
  const webToolNames = extraWebTools.map((t) => t.name);
  const sessionToolNames = [...orderedPiTools, ...webToolNames, ...(worklistMode ? WORKLIST_TOOL_NAMES : [])];
  const { session } = await createAgentSession({
    cwd: agentCwd, agentDir: join(directory, "agent"), modelRuntime, model, thinkingLevel: options.thinkingLevel ?? "high",
    settingsManager, resourceLoader, sessionManager: SessionManager.inMemory(agentCwd), tools: sessionToolNames,
    customTools,
  });
  if (identicalCallLimit || targetFailureLimit || worklistMode) {
    const previousBefore = session.agent.beforeToolCall;
    session.agent.beforeToolCall = async (context, signal) => {
      const messages = context.context.messages as readonly Message[];
      const { name, arguments: args } = context.toolCall;
      const reason = (worklistMode ? repeatedWorklistSpin(messages, name) : undefined)
        ?? (identicalCallLimit ? repeatedIdenticalCall(messages, name, args, identicalCallLimit) : undefined)
        ?? (targetFailureLimit ? repeatedTargetFailure(messages, name, args, targetFailureLimit) : undefined);
      if (reason) return { block: true, reason };
      if (worklistState && !isWorklistTool(name)) worklistState.recordWorkspaceTool();
      return previousBefore?.(context, signal);
    };
  }
  let interrupted: "budget_exhausted" | "timeout" | undefined;
  let incompleteGeneration: string | undefined;
  let lastAssistant: AssistantMessage | undefined;
  const budgetController = new AbortController();
  const previousPayload = session.agent.onPayload;
  session.agent.onPayload = async (payload, requestModel) => {
    if (record.steps.length >= options.maxRequests) {
      interrupted = "budget_exhausted";
      throw new Error("Harness request budget exhausted.");
    }
    const updated = await previousPayload?.(payload, requestModel);
    const initial = updated ?? payload;
    if (!object(initial) || !Array.isArray(initial.messages) || !initial.messages.every(object)) {
      throw new Error("Expected OpenAI messages in the final pi request.");
    }
    let request: Record<string, unknown> = initial;
    // Apply identical Epoch / Worklist compaction for non-RWKV providers (e.g. Teacher data collection),
    // so captured teacher requests in `run.json` match RWKV student inference byte-for-byte.
    if (!rwkv.rwkvPiTransportFolded) {
      if (effectiveHistoryMode === "epoch") {
        request = foldEpochRequest(request, effectiveEpochConfig).request;
      } else if (effectiveHistoryMode === "worklist") {
        request = foldWorklistRequest(request, effectiveWorklistConfig).request;
      }
    }
    let tokenBudget: TokenBudget | undefined;
    if (options.contextBudget) {
      const prepared = await budgetRwkvRequest(request, options.contextBudget, budgetController.signal);
      request = prepared.request;
      tokenBudget = prepared.budget;
    }
    if (!object(request) || !Array.isArray(request.messages) || !request.messages.every(object)) throw new Error("Invalid budgeted chat request.");
    const captured = structuredClone({ ...request, messages: request.messages, tools: request.tools ?? [] });
    if (!Array.isArray(captured.tools)) throw new Error("Expected OpenAI tool definitions.");
    const step: Step = { request: { ...captured, tools: captured.tools }, ...(tokenBudget ? { tokenBudget } : {}) };
    record.steps.push(step);
    log({ type: "request", index: record.steps.length - 1, request: step.request, tokenBudget });
    return request;
  };
  const unsubscribe = session.subscribe((event) => {
    if (event.type !== "message_end") return;
    const message = structuredClone(event.message);
    record.messages.push(message);
    if (message.role === "assistant") {
      lastAssistant = message;
      const step = record.steps.at(-1);
      // A budget refusal has no outgoing request and must not replace the preceding response.
      if (step && !step.response) step.response = message;
      if (message.stopReason === "length") {
        incompleteGeneration = "Generation ended with length; truncated trajectories are not accepted.";
        void session.abort();
      }
    }
    log({ type: "message_end", message });
  });
  const timer = setTimeout(() => { interrupted = "timeout"; budgetController.abort(); void session.abort(); }, options.timeoutMs);
  const withPromptSuffix = (text: string) => {
    const suffix = options.promptSuffix?.trim();
    return suffix ? `${text.trimEnd()}\n\n${suffix}` : text;
  };
  try {
    const prompts = [task.prompt, ...followups];
    const turns: { index: number; userPrompt: string; completed: boolean }[] = [];
    for (let i = 0; i < prompts.length; i++) {
      lastAssistant = undefined;
      await session.prompt(withPromptSuffix(prompts[i]!));
      const last = lastAssistant ?? record.steps.at(-1)?.response;
      if (interrupted || incompleteGeneration || last?.stopReason !== "stop") break;
      turns.push({ index: i + 1, userPrompt: prompts[i]!, completed: true });
      if (followups.length > 0) {
        record.conversation = { allTurnsCompleted: turns.length === prompts.length, turns: [...turns] };
      }
    }
    if (followups.length > 0 && !record.conversation) {
      record.conversation = { allTurnsCompleted: false, turns };
    }
    // A client-side refusal has no outgoing step, but its error is still the final message.
    const last = lastAssistant ?? record.steps.at(-1)?.response;
    const completedAllTurns = followups.length === 0 || Boolean(record.conversation?.allTurnsCompleted);
    record.status = interrupted ?? (incompleteGeneration ? "error" : last?.stopReason === "stop" && completedAllTurns ? "failed" : "error");
    if (record.status === "error") record.error = incompleteGeneration ?? last?.errorMessage ?? `Generation ended with ${last?.stopReason ?? "no response"}.`;
  } catch (error) {
    record.status = interrupted ?? "error";
    record.error = error instanceof Error ? error.message : String(error);
  } finally {
    clearTimeout(timer);
    budgetController.abort();
    unsubscribe();
    session.dispose();
  }
  // Checks run after the agent stops; expected outputs are never added to its prompt.
  if (record.status === "failed") {
    for (const check of task.checks) {
      try {
        if (check.type === "file_equals") {
          const path = taskPath(cwd, check.path);
          if (sandbox) {
            const actual = (await sandbox.readFile(taskPath(agentCwd, check.path))).toString("utf8");
            record.checks.push({ passed: actual === check.expected, detail: `file_equals:${check.path}` });
            continue;
          }
          const resolved = await realpath(path);
          if (!resolved.startsWith((await realpath(cwd)) + sep)) throw new Error("Check path follows a symlink outside the workspace.");
          const actual = await readFile(resolved, "utf8");
          record.checks.push({ passed: actual === check.expected, detail: `file_equals:${check.path}` });
        } else {
          if (sandbox) {
            const result = await sandbox.exec(check.argv, { timeoutMs: check.timeoutMs });
            record.checks.push({ passed: result.exitCode === 0 && (check.expectedStdout === undefined || result.stdout.toString() === check.expectedStdout), detail: (result.stdout.toString() + result.stderr.toString()).slice(0, 256 * 1024) });
            continue;
          }
          const result = await runCommand(check.argv[0]!, check.argv.slice(1), {
            cwd, timeout: check.timeoutMs, maxBuffer: 256 * 1024,
          });
          record.checks.push({ passed: check.expectedStdout === undefined || result.stdout === check.expectedStdout, detail: result.stdout + result.stderr });
        }
      } catch (error) {
        record.checks.push({ passed: false, detail: error instanceof Error ? error.message : String(error) });
      }
    }
    if (record.checks.every((check) => check.passed)) record.status = "passed";
  }
  record.finishedAt = new Date().toISOString();
  log({ type: "run_end", status: record.status, checks: record.checks });
  const path = join(directory, "run.json");
  await writeFile(path + ".tmp", JSON.stringify(record, null, 2) + "\n", { mode: 0o600 });
  await rename(path + ".tmp", path);
  return { record, path };
}

/** Unsandboxed runs: pi's host definitions with identical prompt rules (`exposeSessionEnvironment: false`). */
function hostTools(cwd: string, names: readonly string[]): ToolDefinition[] {
  const make: Record<string, () => unknown> = {
    read: () => createReadToolDefinition(cwd), write: () => createWriteToolDefinition(cwd),
    edit: () => createEditToolDefinition(cwd), ls: () => createLsToolDefinition(cwd),
    bash: () => createBashToolDefinition(cwd, { exposeSessionEnvironment: false }),
  };
  return names.filter((name) => make[name]).map((name) => make[name]!()) as ToolDefinition[];
}

function tolerantTools(tools: ToolDefinition[], cwd: string, fs: typeof hostFs): ToolDefinition[] {
  return tools.map((tool) => (TOLERANT_TOOL_NAMES.includes(tool.name) ? withTolerance(tool, cwd, fs) : tool));
}
