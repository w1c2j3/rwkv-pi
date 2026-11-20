import type { AssistantMessage } from "@earendil-works/pi-ai";
import type { RunRecord } from "./harness.js";
import { digest } from "./provenance.js";
import { object, parseTask } from "./tasks.js";

function targetMessage(message: AssistantMessage): Record<string, unknown> {
  const text = message.content.filter((block) => block.type === "text").map((block) => block.text).join("");
  const thinking = message.content.filter((block) => block.type === "thinking").map((block) => block.thinking).join("");
  const calls = message.content.filter((block) => block.type === "toolCall").map((block) => ({
    id: block.id, type: "function", function: { name: block.name, arguments: JSON.stringify(block.arguments) },
  }));
  if (!text && !calls.length) throw new Error("Empty assistant target.");
  return { role: "assistant", content: text || null, ...(thinking ? { reasoning_content: thinking } : {}), ...(calls.length ? { tool_calls: calls } : {}) };
}

function validateMessages(messages: Record<string, unknown>[], tools: unknown[]): void {
  const names = new Set(tools.map((tool) => object(tool) && object(tool.function) ? tool.function.name : undefined));
  const pending = new Set<string>();
  for (const [index, message] of messages.entries()) {
    if (message.role === "tool") {
      if (typeof message.tool_call_id !== "string" || !pending.delete(message.tool_call_id)) throw new Error("Orphan tool result.");
    } else {
      if (pending.size) throw new Error("Tool call missing a result before the next conversation turn.");
      if (message.tool_calls !== undefined) {
        if (!Array.isArray(message.tool_calls)) throw new Error("Invalid tool_calls.");
        for (const call of message.tool_calls) {
          if (!object(call) || typeof call.id !== "string" || !call.id || pending.has(call.id) || !object(call.function)
            || !names.has(call.function.name) || typeof call.function.arguments !== "string" || !object(JSON.parse(call.function.arguments))) {
            throw new Error("Malformed or undeclared tool call.");
          }
          pending.add(call.id);
        }
      }
    }
    if (index === messages.length - 2 && pending.size) throw new Error("Training prompt has unresolved tool calls.");
  }
}

export function buildDataset(runs: RunRecord[], includeSmoke = false, approvedQuestionAuthors: ReadonlySet<string> = new Set()) {
  const families = new Map<string, string>();
  const taskSplits = new Map<string, string>();
  const taskIds = new Map<string, string>();
  const seen = new Set<string>();
  const samples: Record<string, unknown>[] = [];
  const rejected: { runId: string; reason: string }[] = [];
  for (const run of runs) {
    if (!object(run) || run.schemaVersion !== 1 || typeof run.runId !== "string"
      || !["teacher", "student", "smoke", "question-author"].includes(run.role) || !Array.isArray(run.steps) || !Array.isArray(run.checks)
      || !object(run.model) || !object(run.provenance)) throw new Error("Invalid or unknown run schema.");
    const task = parseTask(run.task);
    const { id: _id, split: _split, ...definition } = task;
    if (run.taskSha256 !== digest(definition)) throw new Error(`Task fingerprint mismatch: ${run.runId}`);
    const contentFingerprint = digest([task.prompt, task.files, task.tools, task.checks]);
    for (const [registry, key] of [[families, task.family], [taskSplits, contentFingerprint]] as const) {
      if (registry.has(key) && registry.get(key) !== task.split) throw new Error(`Dataset leakage across splits: ${key}`);
      registry.set(key, task.split);
    }
    if (taskIds.has(task.id) && taskIds.get(task.id) !== run.taskSha256) throw new Error(`Task ID has conflicting definitions: ${task.id}`);
    taskIds.set(task.id, run.taskSha256);
    const smoke = run.role === "smoke" || task.split === "smoke";
    if (run.role === "question-author" && !approvedQuestionAuthors.has(run.runId)) {
      rejected.push({ runId: run.runId, reason: "question_author_needs_independent_acceptance" });
      continue;
    }
    if ((smoke && !includeSmoke) || run.status !== "passed" || run.checks.length !== task.checks.length || !run.checks.every((check) => check.passed)) {
      rejected.push({ runId: run.runId, reason: smoke && !includeSmoke ? "smoke_fixture" : "not_verified" });
      continue;
    }
    const candidate: Record<string, unknown>[] = [];
    try {
      if (!run.steps.length || run.steps.at(-1)?.response?.stopReason !== "stop") throw new Error("Incomplete final answer.");
      for (const [index, step] of run.steps.entries()) {
        const meta = step as { supervised?: boolean; excludedFromTraining?: string };
        if (meta.supervised === false || meta.excludedFromTraining) continue;
        if (!step.response || !["stop", "toolUse"].includes(step.response.stopReason)) throw new Error("Incomplete generation.");
        const messages = [...step.request.messages, targetMessage(step.response)];
        validateMessages(messages, step.request.tools);
        candidate.push({
          schemaVersion: 1, id: digest([run.runId, index]), split: smoke ? "smoke" : task.split,
          family: task.family, taskId: task.id, taskSha256: run.taskSha256, source: task.source,
          runId: run.runId, step: index, role: run.role, model: run.model, provenance: run.provenance,
          isolation: run.isolation, responseAdapter: run.responseAdapter ?? "none", ...(run.inference ? { inference: run.inference } : {}),
          ...(step.response.diagnostics?.some(diagnostic => diagnostic.type === "rwkv_pi_tool_adapter")
            ? { adaptation: step.response.diagnostics.filter(diagnostic => diagnostic.type === "rwkv_pi_tool_adapter") } : {}),
          messages, tools: step.request.tools, loss: { scope: "last_assistant", messageIndex: messages.length - 1 },
        });
      }
      if (!candidate.length) throw new Error("No supervised steps in run.");
    } catch (error) {
      rejected.push({ runId: run.runId, reason: error instanceof Error ? error.message : String(error) });
      continue;
    }
    for (const sample of candidate) {
      // Preserve distinct trajectories; remove repeated input/target pairs, not arbitrary retries.
      const fingerprint = digest([sample.messages, sample.tools]);
      if (!seen.has(fingerprint)) { seen.add(fingerprint); samples.push(sample); }
    }
  }
  return { samples, rejected };
}
