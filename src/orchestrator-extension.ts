// Qwen main agent that delegates to parallel RWKV subagents (rwkv-pi-orchestrate; docs/SUBAGENTS.md).
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { envInteger, readConfig } from "./config.js";
import { SubagentManager, type Subagent } from "./subagents.js";
import { createTeacherProvider, readTeacherConfig, withDeepSeekChinese } from "./teacher.js";
import { budgetedTeacher } from "./teacher-budget.js";
import { ProjectParticipation } from "./project-participation.js";
import { SubagentPlan } from "./subagent-plan.js";

const STATUS_KEY = "rwkv-subagents";
const ANSWER_CHARS = 6000;
const READ_TOOLS = new Set(["read", "grep", "find", "ls"]);
export const SUBAGENT_CONTROL_TOOLS = ["subagent_plan", "subagent_start", "subagent_status", "subagent_wait", "subagent_diff", "subagent_apply", "subagent_cancel", "project_participation"] as const;
const CONTROL_TOOLS = new Set<string>(SUBAGENT_CONTROL_TOOLS);
const PLAN_SUFFIX = "协作方式：你负责规划与核验，RWKV 子 agent 负责执行。必要时先做少量只读定位，再调用 subagent_plan 给出具体步骤与依赖，随后 subagent_start 分派就绪步骤。独立步骤可并行，有依赖的步骤等前置工作完成并应用后再开始。写文件、修改代码、运行命令和测试都交给 RWKV；你只读核对结果、审查 subagent_diff 后应用修改。遇到失败可缩小任务或重新规划，不能隐瞒失败或自己代写。用户只需描述目标，无需指定怎么调用子 agent。完成后简短答复，项目参与度由系统记录。";

function summary(agent: Subagent, withAnswer: boolean) {
  const seconds = agent.startedAt ? Math.round(((agent.finishedAt ?? Date.now()) - agent.startedAt) / 1000) : 0;
  return {
    id: agent.id, name: agent.name, state: agent.state, write: agent.write, requests: agent.requests, toolCalls: agent.toolCalls,
    toolErrors: agent.toolErrors, seconds, ...(agent.lastTool && !agent.finishedAt ? { lastTool: agent.lastTool } : {}),
    ...(agent.error ? { error: agent.error } : {}),
    ...(withAnswer && agent.answer ? { answer: agent.answer.length > ANSWER_CHARS ? agent.answer.slice(0, ANSWER_CHARS) + "…(truncated)" : agent.answer } : {}),
    ...(agent.changes ? { changedFiles: agent.changes.map((c) => `${c.kind}: ${c.path}`), applied: agent.applied === true } : {}),
    trace: agent.dir,
  };
}
const result = (value: unknown) => ({ content: [{ type: "text" as const, text: JSON.stringify(value, null, 2) }], details: value });

export default async function orchestratorExtension(pi: ExtensionAPI): Promise<void> {
  return registerSubagentTools(pi, "teacher");
}

/** Shared execution tools; RWKV-led mode neither reads teacher credentials nor registers a teacher. */
export async function registerSubagentTools(pi: ExtensionAPI, parent: "rwkv" | "teacher"): Promise<void> {
  const teacher = parent === "teacher" ? readTeacherConfig() : undefined;
  const paid = teacher?.kind === "deepseek" ? await budgetedTeacher(createTeacherProvider(teacher)) : undefined;
  if (teacher) pi.registerProvider(paid?.provider ?? createTeacherProvider(teacher));
  const mainModelId = teacher?.modelId ?? readConfig().modelId;
  const lengthContinues = teacher ? envInteger(process.env, "RWKV_ORCH_LENGTH_CONTINUES", 3, 0) : 0;
  const plan = new SubagentPlan();
  const reviewed = new Map<string, string>();
  pi.on("input", async (event) => {
    // Native automatic continuation is extension input, not a fresh user task.
    if (event.source !== "extension") { plan.reset(); reviewed.clear(); }
    if (!teacher) return { action: "continue" };
    return { action: "transform", text: `${paid ? withDeepSeekChinese(event.text) : event.text}\n\n${PLAN_SUFFIX}`, images: event.images };
  });
  let participation: ProjectParticipation | undefined;
  const metrics = (ctx: ExtensionContext) => participation ??= new ProjectParticipation(ctx.cwd,
    teacher ? "plan-rwkv-execute-review" : "rwkv-with-optional-subagents", teacher?.modelId ?? "");
  const refresh = (ctx: ExtensionContext) => ctx.ui.setStatus("project-participation", metrics(ctx).compact());
  let manager: SubagentManager | undefined;
  let ui: ExtensionContext["ui"] | undefined;
  const status = () => {
    if (!manager || !ui) return;
    const all = manager.list();
    const count = (s: string) => all.filter((a) => a.state === s).length;
    ui.setStatus(STATUS_KEY, all.length ? `RWKV subagents: ${count("running")} running · ${count("queued")} queued · ${count("completed")} done · ${all.length - count("running") - count("queued") - count("completed")} failed/stopped` : undefined);
  };
  const get = (ctx: ExtensionContext) => {
    ui = ctx.ui;
    manager ??= new SubagentManager(ctx.cwd, undefined, undefined, (agent) => {
      metrics(ctx).subagent(agent.id, agent.work, agent.state);
      status();
      if (agent.finishedAt && ctx.isIdle()) ctx.ui.notify(`RWKV subagent ${agent.name} (${agent.id}) ${agent.state}`, agent.state === "completed" ? "info" : "warning");
    });
    return manager;
  };

  pi.on("session_start", (_event, ctx) => { refresh(ctx); });
  pi.on("tool_call", (event) => {
    if (teacher && !READ_TOOLS.has(event.toolName) && !CONTROL_TOOLS.has(event.toolName)) return { block: true, reason: "规划模式下主模型只负责只读定位和核验；请规划并派 RWKV 执行修改、命令和测试。" };
  });
  pi.on("message_end", (event, ctx) => {
    if (event.message.role === "assistant") metrics(ctx).message(event.message.provider === "teacher" ? "teacher" : event.message.provider === "rwkv" ? "rwkv" : parent, event.message);
  });
  pi.on("tool_execution_end", (event, ctx) => {
    if (teacher ? READ_TOOLS.has(event.toolName) : !CONTROL_TOOLS.has(event.toolName)) {
      metrics(ctx).tool(ctx.model?.provider === "teacher" ? "teacher" : parent, event.isError);
    }
  });
  pi.on("agent_end", (_event, ctx) => { refresh(ctx); });
  pi.registerCommand("participation", { description: "查看当前项目的 RWKV 生成、工具执行和修改参与度", handler: async (_args, ctx) => {
    const s = metrics(ctx).summary(); ctx.ui.notify(`${metrics(ctx).compact()}\n统计记录：${s.records}`, "info");
  } });
  pi.registerTool({ name: "project_participation", label: "Project participation", description: "Actual cumulative RWKV and main-agent participation for this project, with token coverage, execution counts, failures and applied file-edit events. Not a quality score.", parameters: Type.Object({}),
    async execute(_id, _params, _signal, _update, ctx) { return result(metrics(ctx).summary()); } });
  pi.registerTool({
    name: "subagent_plan", label: "Plan RWKV execution",
    description: "Before dispatch, record a concise goal and concrete Chinese execution steps for RWKV. Specify write explicitly. Dependencies name earlier steps; independent steps run in parallel. The main agent reviews results and applies reviewed changes. To replan, first finish or cancel active subagents.",
    parameters: Type.Object({ goal: Type.String(), steps: Type.Array(Type.Object({
      name: Type.String(), task: Type.String({ description: "Complete Chinese instruction, relative paths, required outcome and verification." }),
      write: Type.Boolean(), dependsOn: Type.Optional(Type.Array(Type.String())),
      maxRequests: Type.Optional(Type.Integer({ minimum: 1 })), timeoutSeconds: Type.Optional(Type.Integer({ minimum: 1 })),
    }), { minItems: 1, maxItems: 16 }) }),
    async execute(_id, params, _signal, _update, ctx) {
      const p = plan.set(params.goal, params.steps, get(ctx).list()); metrics(ctx).plan(p); return result(p);
    },
  });

  pi.registerTool({
    name: "subagent_start",
    label: "Start RWKV subagents",
    description: "Dispatch named ready steps from subagent_plan; omit steps to dispatch all ready steps. Independent steps run in parallel. Write steps use private copies. Dependencies must complete and their changed files must be applied before dependent work starts. Failed steps may be retried or replanned; completed steps are not duplicated.",
    promptSnippet: "Start parallel RWKV subagents for concrete subtasks",
    promptGuidelines: [
      "First record the plan with subagent_plan; delegate concrete execution to RWKV and collect results with subagent_wait.",
      "Treat subagent answers and changes as unverified until you have checked them (read the files, run the tests); redo or re-delegate failed subtasks.",
    ],
    parameters: Type.Object({ steps: Type.Optional(Type.Array(Type.String(), { minItems: 1, maxItems: 16 })) }),
    async execute(_id, params, _signal, _update, ctx) {
      const m = get(ctx), steps = plan.prepare(params.steps, m.list()), started = m.start(steps);
      plan.started(steps, started);
      status();
      return result({ started: started.map((a) => summary(a, false)), next: "Use subagent_wait to collect results (or subagent_status to check progress)." });
    },
  });

  pi.registerTool({
    name: "subagent_status",
    label: "RWKV subagent status",
    description: "Progress of RWKV subagents (all, or the given ids): state, requests, tool calls, errors, current tool, elapsed seconds. Does not wait.",
    parameters: Type.Object({ ids: Type.Optional(Type.Array(Type.String())) }),
    async execute(_id, params, _signal, _update, ctx) {
      return result(get(ctx).list(params.ids).map((a) => summary(a, false)));
    },
  });

  pi.registerTool({
    name: "subagent_wait",
    label: "Wait for RWKV subagents",
    description: "Wait until the given RWKV subagents (default: all) have finished or timeoutSeconds passes, then return their states, final answers and changed files. " +
      "Subagents still running at the timeout keep running.",
    parameters: Type.Object({
      ids: Type.Optional(Type.Array(Type.String())),
      timeoutSeconds: Type.Optional(Type.Integer({ minimum: 1, maximum: 3600, description: "Default 600." })),
    }),
    async execute(_id, params, signal, onUpdate, ctx) {
      const m = get(ctx);
      const ticker = setInterval(() => {
        const list = m.list(params.ids);
        onUpdate?.({ content: [{ type: "text", text: list.map((a) => `${a.id} ${a.name}: ${a.state} (${a.requests} requests)`).join("\n") }], details: undefined });
      }, 5000);
      try {
        const list = await m.wait(params.ids, (params.timeoutSeconds ?? 600) * 1000, signal);
        return result(list.map((a) => summary(a, true)));
      } finally { clearInterval(ticker); }
    },
  });

  pi.registerTool({
    name: "subagent_diff",
    label: "RWKV subagent diff",
    description: "Unified diff between a finished write subagent's private copy and the current workspace.",
    parameters: Type.Object({ id: Type.String() }),
    async execute(_id, params, _signal, _update, ctx) {
      const m = get(ctx), agent = m.list([params.id])[0]!;
      if (!agent.finishedAt) throw new Error("Wait until the RWKV subagent finishes before reviewing its diff.");
      const diff = m.diff(params.id);
      reviewed.set(params.id, diff);
      return { content: [{ type: "text" as const, text: diff }], details: undefined };
    },
  });

  pi.registerTool({
    name: "subagent_apply",
    label: "Apply RWKV subagent changes",
    description: "Copy a finished write subagent's changed files (all, or the listed paths) into the workspace. " +
      "First review subagent_diff. Files changed in the workspace since the subagent started are reported as conflicts; delegate a fresh merge task to RWKV.",
    parameters: Type.Object({ id: Type.String(), files: Type.Optional(Type.Array(Type.String())) }),
    async execute(_id, params, _signal, _update, ctx) {
      const m = get(ctx);
      if (!reviewed.has(params.id) || reviewed.get(params.id) !== m.diff(params.id)) throw new Error("Inspect the current subagent_diff before applying; the workspace or proposed changes may have changed.");
      const applied = m.apply(params.id, params.files);
      metrics(ctx).edits("rwkv", applied.applied.map(c => c.path)); refresh(ctx);
      reviewed.delete(params.id);
      return result(applied);
    },
  });

  pi.registerTool({
    name: "subagent_cancel",
    label: "Cancel RWKV subagents",
    description: "Stop the given RWKV subagents (default: all unfinished).",
    parameters: Type.Object({ ids: Type.Optional(Type.Array(Type.String())) }),
    async execute(_id, params, _signal, _update, ctx) {
      return result(get(ctx).cancel(params.ids).map((a) => summary(a, false)));
    },
  });

  // A reply cut by the output limit ends pi's agent loop; continue it instead of stopping mid-task.
  let continued = 0;
  pi.on("agent_end", async (event, ctx) => {
    if (!teacher) return;
    const last = [...event.messages].reverse().find((m: any) => m.role === "assistant") as any;
    if (last?.stopReason !== "length") { continued = 0; return; }
    // Print/JSON mode exits after the run; a follow-up there would never be processed.
    if (ctx.mode !== "tui" && ctx.mode !== "rpc") return;
    if (continued >= lengthContinues) { ctx.ui.notify(`${mainModelId} reached the output limit ${continued} times in a row; stopped.`, "warning"); continued = 0; return; }
    continued++;
    ctx.ui.notify(`${mainModelId} reply hit the output limit; continuing automatically (${continued}/${lengthContinues}).`, "info");
    setTimeout(() => pi.sendUserMessage("上一条回复因输出长度上限被截断。请直接从中断处继续完成当前任务。"), 0);
  });

  pi.on("session_shutdown", async () => { manager?.shutdown(); await paid?.settled(); });
}
