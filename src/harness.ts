import { execFile } from "node:child_process";
import { randomUUID } from "node:crypto";
import { appendFileSync } from "node:fs";
import { mkdir, readFile, realpath, rename, writeFile } from "node:fs/promises";
import { dirname, join, resolve, sep } from "node:path";
import { promisify } from "node:util";
import { InMemoryCredentialStore, InMemoryModelsStore, type AssistantMessage, type Provider } from "@earendil-works/pi-ai";
import { createAgentSession, createExtensionRuntime, ModelRuntime, SessionManager, SettingsManager, type ResourceLoader } from "@earendil-works/pi-coding-agent";
import { digest, runtimeProvenance } from "./provenance.js";
import { object, parseTask, taskPath, type Task } from "./tasks.js";
import { WorkspaceSandbox } from "./sandbox.js";
import { parseToolList } from "./tools.js";
import type { RwkvConfig } from "./config.js";
import { budgetRwkvRequest, type TokenBudget } from "./token-budget.js";
import { promptVersion } from "./prompt.js";

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
  role: "teacher" | "student" | "smoke";
  model: { provider: string; id: string };
  provenance: ReturnType<typeof runtimeProvenance>;
  isolation: "bubblewrap" | "none";
  responseAdapter: "none" | "rwkv-json-v1" | "rwkv-xml-v1" | "rwkv-xml-continuation-v2" | "rwkv-json-continuation-v1";
  promptVersion?: string;
  loopGuard?: { repeatFailureLimit: number };
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
  inference?: RunRecord["inference"];
  contextBudget?: RwkvConfig;
  thinkingLevel?: "high" | "off";
}

export async function runTask(input: Task, options: HarnessOptions): Promise<{ record: RunRecord; path: string }> {
  const task = parseTask(input);
  for (const number of [options.maxRequests, options.timeoutMs]) {
    if (!Number.isSafeInteger(number) || number <= 0) throw new Error("Harness budgets must be positive integers.");
  }
  const model = options.provider.getModels()[0];
  if (!model || model.api !== "openai-completions") throw new Error("Harness v1 requires an OpenAI Chat Completions model.");
  const runId = randomUUID();
  const directory = resolve(options.outputDir, runId);
  const cwd = join(directory, "workspace");
  await mkdir(cwd, { recursive: true });
  const sandbox = options.sandbox === "none" ? undefined : new WorkspaceSandbox(cwd);
  await sandbox?.probe();
  await sandbox?.probeSearch(task.tools);
  for (const [path, content] of Object.entries(task.files)) {
    const target = taskPath(cwd, path);
    await mkdir(dirname(target), { recursive: true });
    await writeFile(target, content, { flag: "wx" });
  }
  const { id: _id, split: _split, ...definition } = task;
  const record: RunRecord = {
    schemaVersion: 1, runId, task, taskSha256: digest(definition), role: options.role,
    model: { provider: model.provider, id: model.id }, provenance: runtimeProvenance(),
    isolation: sandbox?.kind ?? "none",
    responseAdapter: "rwkvPiToolFormat" in options.provider ? options.provider.rwkvPiToolFormat === "json" ? "rwkv-json-continuation-v1" : options.provider.rwkvPiToolFormat === "xml" ? "rwkv-xml-continuation-v2" : "none" : "none",
    ...("rwkvPiToolFormat" in options.provider && ["xml", "json"].includes(String(options.provider.rwkvPiToolFormat)) ? { promptVersion: promptVersion(options.provider.rwkvPiToolFormat === "json" ? "json" : "xml") } : {}),
    ...("rwkvPiRepeatFailureLimit" in options.provider && typeof options.provider.rwkvPiRepeatFailureLimit === "number"
      ? { loopGuard: { repeatFailureLimit: options.provider.rwkvPiRepeatFailureLimit } } : {}),
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
    getAppendSystemPrompt: () => [], getAppendSystemPromptSources: () => [],
    extendResources: () => {}, reload: async () => {},
  };
  const modelRuntime = await ModelRuntime.create({
    credentials: new InMemoryCredentialStore(), modelsStore: new InMemoryModelsStore(),
    modelsPath: null, refreshOnCreate: false, allowModelNetwork: false,
  });
  modelRuntime.registerNativeProvider(options.provider);
  const { session } = await createAgentSession({
    cwd, agentDir: join(directory, "agent"), modelRuntime, model, thinkingLevel: options.thinkingLevel ?? "high",
    settingsManager, resourceLoader, sessionManager: SessionManager.inMemory(cwd), tools: parseToolList(task.tools),
    customTools: sandbox?.tools(task.tools),
  });
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
    let request = updated ?? payload;
    if (!object(request) || !Array.isArray(request.messages) || !request.messages.every(object)) {
      throw new Error("Expected OpenAI messages in the final pi request.");
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
  try {
    await session.prompt(task.prompt);
    // A client-side refusal has no outgoing step, but its error is still the final message.
    const last = lastAssistant ?? record.steps.at(-1)?.response;
    record.status = interrupted ?? (incompleteGeneration ? "error" : last?.stopReason === "stop" ? "failed" : "error");
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
            const actual = (await sandbox.readFile(path)).toString("utf8");
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
