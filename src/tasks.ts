import { isAbsolute, resolve, sep } from "node:path";
import { PI_TOOL_NAMES, type PiToolName } from "./tools.js";
import { parseEnvironment, type TaskEnvironment } from "./environment.js";

export const WEB_TOOL_NAMES = ["web_search", "web_fetch"] as const;
export type WebToolName = (typeof WEB_TOOL_NAMES)[number];
export const TASK_TOOL_NAMES = [...PI_TOOL_NAMES, ...WEB_TOOL_NAMES] as const;
export type Split = "train" | "validation" | "test" | "smoke";
export type ToolName = PiToolName | WebToolName;
export type Check = { type: "file_equals"; path: string; expected: string }
  | { type: "command"; argv: string[]; timeoutMs: number; expectedStdout?: string };
export interface Task {
  schemaVersion: 1;
  id: string;
  family: string;
  split: Split;
  source: string;
  prompt: string;
  files: Record<string, string>;
  tools: ToolName[];
  checks: Check[];
  /** Optional pinned container environment; see environment.ts. */
  environment?: TaskEnvironment;
  /** When true, acceptance checks read `.task-evidence/final.json` (`finalAnswer`, `toolCalls`). */
  answerEvidence?: boolean;
}

export function object(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function taskPath(cwd: string, path: string): string {
  if (!path || isAbsolute(path) || path.includes("\\") || path.includes(":")) throw new Error(`Invalid task path: ${path}`);
  const target = resolve(cwd, path);
  if (!target.startsWith(resolve(cwd) + sep)) throw new Error(`Task path escapes workspace: ${path}`);
  return target;
}

export function parseTask(value: unknown): Task {
  if (!object(value) || value.schemaVersion !== 1) throw new Error("Task requires schemaVersion=1.");
  for (const field of ["id", "family", "source", "prompt"] as const) {
    if (typeof value[field] !== "string" || !value[field].trim()) throw new Error(`Task requires ${field}.`);
  }
  if (!["train", "validation", "test", "smoke"].includes(String(value.split))) throw new Error("Invalid task split.");
  if (!object(value.files)) throw new Error("Task files must map relative paths to text.");
  for (const [path, content] of Object.entries(value.files)) {
    taskPath("/workspace", path);
    if (typeof content !== "string") throw new Error(`Task file ${path} must contain text.`);
  }
  if (!Array.isArray(value.tools) || !value.tools.length || new Set(value.tools).size !== value.tools.length
    || value.tools.some((tool) => !(TASK_TOOL_NAMES as readonly unknown[]).includes(tool))) {
    throw new Error(`Task tools must select distinct tools from ${TASK_TOOL_NAMES.join(", ")}.`);
  }
  if (value.environment !== undefined) parseEnvironment(value.environment);
  if (!Array.isArray(value.checks) || !value.checks.length) throw new Error("Task requires independent acceptance checks.");
  for (const check of value.checks) {
    if (!object(check)) throw new Error("Invalid acceptance check.");
    if (check.type === "file_equals" && typeof check.path === "string" && typeof check.expected === "string") {
      taskPath("/workspace", check.path);
    } else if (check.type === "command" && Array.isArray(check.argv) && check.argv.length && check.argv.every((arg) => typeof arg === "string")
      && Number.isSafeInteger(check.timeoutMs) && Number(check.timeoutMs) > 0) {
      if (!check.argv[0]) throw new Error("Check command must have an executable.");
      if (check.expectedStdout !== undefined && typeof check.expectedStdout !== "string") throw new Error("expectedStdout must be text.");
    } else throw new Error("Expected file_equals or command acceptance check.");
  }
  return value as unknown as Task;
}

export function parseTasks(text: string): Task[] {
  const tasks = text.split(/\r?\n/).filter((line) => line.trim()).map((line) => parseTask(JSON.parse(line)));
  if (!tasks.length) throw new Error("Task manifest is empty.");
  const ids = new Set<string>();
  const families = new Map<string, Split>();
  for (const task of tasks) {
    if (ids.has(task.id)) throw new Error(`Duplicate task ID: ${task.id}`);
    if (families.has(task.family) && families.get(task.family) !== task.split) throw new Error(`Task family crosses splits: ${task.family}`);
    ids.add(task.id);
    families.set(task.family, task.split);
  }
  return tasks;
}
