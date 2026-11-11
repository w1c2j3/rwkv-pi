// The CLI loads automatic RWKV -> Qwen takeover by default (--no-collab opts out).
// Native tools, working directory, conversation and permission checks are retained.
import type { Model } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { CollaborationTrace, collaborationDataRoot } from "./collab-trace.js";
import { StallDetector, prose, repeatedProse, stallLimits, type StallReason } from "./stall-detector.js";
import { createTeacherProvider, readTeacherConfig } from "./teacher.js";
import { budgetedTeacher } from "./teacher-budget.js";
import { ProjectParticipation } from "./project-participation.js";

const REPORT = "rwkv-collaboration";
const HANDOFF = "rwkv-qwen-handoff";

export default async function collaborationExtension(pi: ExtensionAPI): Promise<void> {
  const config = readTeacherConfig();
  const original = createTeacherProvider(config);
  const paid = config.kind === "deepseek" ? await budgetedTeacher(original) : undefined;
  const teacher = paid?.provider ?? original;
  const teacherLabel = config.kind === "deepseek" ? "DeepSeek" : "Qwen";
  const teacherModel = teacher.getModels()[0]!;
  const limits = stallLimits(), root = collaborationDataRoot();
  pi.registerProvider(teacher);
  let enabled = true, monitoring = false, phase: "rwkv" | "teacher" | "idle" = "idle";
  let originalModel: Model<any> | undefined, originalThinking = "off" as ReturnType<typeof pi.getThinkingLevel>;
  let detector = new StallDetector(limits), trace: CollaborationTrace | undefined;
  let pending: StallReason | undefined, stopped: StallReason | undefined, cancelled = false;
  let timer: ReturnType<typeof setInterval> | undefined, unsubscribeInput: (() => void) | undefined;
  let phaseStarted = 0, lastActivity = 0, activeTools = new Set<string>(), lastProseCheck = 0;
  let uiPausedAt = 0;
  let lastReport: string | undefined;
  let participation: ProjectParticipation | undefined;
  const metrics = (ctx: ExtensionContext) => participation ??= new ProjectParticipation(ctx.cwd, "rwkv-first-repair", teacherModel.id);
  const refresh = (ctx: ExtensionContext) => { if (paid) ctx.ui.setStatus("project-participation", metrics(ctx).compact()); };
  const clearTimer = () => { if (timer) clearInterval(timer); timer = undefined; };

  const stop = (reason: StallReason, ctx: ExtensionContext) => {
    if (!monitoring || cancelled || pending || stopped || phase === "idle") return;
    if (phase === "rwkv") pending = reason;
    else stopped = reason;
    // stop() also runs from the watchdog timer, where an uncaught throw (e.g. F: unavailable) would
    // exit interactive pi and skip the abort below; record the write failure as the stop reason instead.
    try { trace?.event("stall_detected", { phase, reason }); }
    catch (error) { pending = undefined; stopped = { code: "trace_write_failed", detail: String(error) }; }
    clearTimer();
    ctx.ui.notify(`${phase === "rwkv" ? `检测到停滞，准备让 ${teacherLabel} 接管` : `${teacherLabel} 也未能继续，停止本任务`}：${reason.detail}`, "warning");
    // Switch only once the cancelled native loop and pending tool results have settled.
    ctx.abort();
  };
  const watch = (ctx: ExtensionContext) => {
    clearTimer(); phaseStarted = lastActivity = Date.now(); activeTools.clear(); lastProseCheck = 0;
    timer = setInterval(() => {
      if (phase === "idle" || cancelled || uiPausedAt || ctx.isIdle()) return;
      if (Date.now() - phaseStarted >= limits.phaseMs) stop({ code: "time_budget", detail: `当前模型执行超过 ${Math.round(limits.phaseMs / 1000)} 秒。` }, ctx);
      else if (!activeTools.size && Date.now() - lastActivity >= limits.silenceMs) stop({ code: "generation_silence", detail: `模型连续 ${Math.round(limits.silenceMs / 1000)} 秒没有输出。` }, ctx);
    }, Math.min(1000, limits.silenceMs));
    timer.unref();
  };
  const restore = async () => {
    if (originalModel) { await pi.setModel(originalModel); pi.setThinkingLevel(originalThinking); }
  };
  const finish = (ctx: ExtensionContext, result: string) => {
    clearTimer();
    try {
      const report = trace?.finish(result, stopped);
      if (trace?.lastRecord) {
        metrics(ctx).edits("rwkv", trace.lastRecord.files.filter(f => f.rwkv.existence !== "same").map(f => f.path));
        metrics(ctx).edits("teacher", trace.lastRecord.files.filter(f => f.qwen.existence !== "same").map(f => f.path));
      } else if (trace) {
        metrics(ctx).edits("rwkv", trace.rwkvOnlyChangedFiles);
      }
      if (report) {
        lastReport = report;
        // Text print mode prints only a final assistant message; a custom report would hide it.
        pi.appendEntry(REPORT, { report, directory: trace?.directory });
        if (ctx.mode === "tui" || ctx.mode === "rpc") ctx.ui.notify(`${teacherLabel} 接管结束：${result}。详细记录可用 /collab report 查看。`, "info");
      }
    } catch (error) { ctx.ui.notify(`协作报告写入失败：${String(error)}`, "error"); }
    trace = undefined; pending = undefined; stopped = undefined; monitoring = false; phase = "idle";
    refresh(ctx);
  };

  pi.on("session_start", (_event, ctx) => {
    unsubscribeInput?.();
    if (ctx.mode === "tui") unsubscribeInput = ctx.ui.onTerminalInput(data => {
      if (data === "\x1b" || data === "\x03") { cancelled = true; pending = undefined; clearTimer(); }
      return undefined;
    });
    // Clear an older extension's footer on reload. Model transitions are recorded in the log.
    ctx.ui.setStatus("collab", undefined);
    if (paid) refresh(ctx);
  });
  pi.on("context", event => ({ messages: event.messages.filter(m => (m as { customType?: string }).customType !== REPORT) }));
  pi.on("before_agent_start", async (event, ctx) => {
    if (event.messageType === "custom" && event.message.customType === HANDOFF) { watch(ctx); return; }
    if (!enabled || event.messageType !== "user") return;
    if (ctx.model?.provider !== "rwkv") return;
    originalModel = ctx.model; originalThinking = pi.getThinkingLevel();
    cancelled = false; pending = undefined; stopped = undefined; monitoring = true; phase = "rwkv";
    detector = new StallDetector(limits); lastReport = undefined;
    try {
      trace = new CollaborationTrace(ctx.cwd, event.prompt, ctx.model.id,
        ctx.sessionManager.getBranch().flatMap(e => e.type === "message" ? [e.message] : []), root, teacherLabel);
      trace.save("runtime.json", { systemPrompt: event.systemPrompt,
        tools: pi.getAllTools().filter(t => pi.getActiveTools().includes(t.name)),
        models: { rwkv: originalModel, teacher: teacherModel }, thinking: originalThinking, limits,
        sessionId: ctx.sessionManager.getSessionId(), cwd: ctx.cwd });
    } catch (error) { phase = "idle"; ctx.ui.notify(`无法保存协作轨迹：${String(error)}`, "error"); ctx.abort(); return; }
    watch(ctx);
  });
  pi.on("message_start", () => { lastActivity = Date.now(); lastProseCheck = 0; });
  pi.on("message_update", (event, ctx) => {
    lastActivity = Date.now();
    if (!enabled || !monitoring || phase === "idle" || event.message.role !== "assistant") return;
    const text = prose(event.message), thinking = prose(event.message, true);
    if (text.length + thinking.length - lastProseCheck < 256) return;
    lastProseCheck = text.length + thinking.length;
    const reason = repeatedProse(text) ?? repeatedProse(thinking);
    if (reason) stop(reason, ctx);
  });
  pi.on("message_end", (event, ctx) => {
    trace?.message(event.message); lastActivity = Date.now();
    if (event.message.role === "assistant") metrics(ctx).message(event.message.provider === "rwkv" ? "rwkv" : "teacher", event.message);
  });
  pi.on("tool_execution_start", event => { activeTools.add(event.toolCallId); lastActivity = Date.now(); });
  pi.on("tool_execution_update", () => { lastActivity = Date.now(); });
  pi.on("tool_execution_end", (event, ctx) => {
    activeTools.delete(event.toolCallId); lastActivity = Date.now();
    metrics(ctx).tool(phase === "teacher" ? "teacher" : "rwkv", event.isError);
  });
  // Waiting for the user's existing pi permission/input dialog is not a model stall.
  pi.on("ui_prompt_start", () => { uiPausedAt ||= Date.now(); });
  pi.on("ui_prompt_end", () => { if (uiPausedAt) phaseStarted += Date.now() - uiPausedAt; uiPausedAt = 0; lastActivity = Date.now(); });
  pi.on("turn_end", (event, ctx) => {
    if (!enabled || !monitoring || phase === "idle" || cancelled || pending || stopped) return;
    const reason = detector.turn(event.message, event.toolResults);
    if (reason) stop(reason, ctx);
  });
  pi.on("agent_settled", async (event, ctx) => {
    clearTimer();
    if (phase === "idle") return;
    if (enabled && !cancelled && pending && phase === "rwkv") {
      const reason = pending; pending = undefined;
      try {
        trace?.takeover(reason, teacherModel.id);
        if (!await pi.setModel(teacherModel)) throw new Error(`${teacherLabel} authentication is not configured`);
        phase = "teacher"; detector = new StallDetector(limits, limits.teacherTurns);
        // Native agent_settled defers the new run until the old one has fully unwound.
        pi.sendMessage({ customType: HANDOFF, display: true,
          content: `${teacherLabel} 自动接管。原因：${reason.detail}\n继续完成用户当前任务，工作目录仍是 ${ctx.cwd}。保留已正确完成的部分，以真实文件和工具结果为准，修正错误并实际验证。不要只解释前一个助手为什么失败；不要扩展用户授权范围。原始需求：\n${trace?.prompt ?? "见上文用户消息"}` +
            (paid ? "\n\n请用中文分析工具结果、推理并给出最终答复，代码、命令、路径和必要原文保持原样。完成必要验证后简短回答并结束。" : ""),
          details: { reason, rwkv: originalModel?.id, teacher: teacherModel.id } }, { triggerTurn: true });
        return;
      } catch (error) { stopped = { code: "handoff_failed", detail: String(error) }; ctx.ui.notify(`${teacherLabel} 接管失败：${String(error)}`, "error"); }
    }
    const result = cancelled ? "cancelled_by_user" : stopped ? "stalled_or_failed" : event.aborted ? "cancelled" : "completed_unverified";
    finish(ctx, result);
    await restore();
  });
  pi.on("session_shutdown", async (_event, ctx) => { cancelled = true; pending = undefined; clearTimer(); unsubscribeInput?.(); if (trace) finish(ctx, "session_closed"); await paid?.settled(); });

  pi.registerCommand("collab", {
    description: `自动 RWKV→${teacherLabel} 接管：on | off | status | report`,
    handler: async (args, ctx) => {
      switch (args.trim() || "status") {
        case "off": enabled = false; monitoring = false; pending = undefined; clearTimer(); ctx.ui.notify("自动接管已关闭；当前运行的工具仍由 pi 管理。", "info"); break;
        case "on": enabled = true; ctx.ui.notify("自动接管已开启，从下一条用户任务开始生效。", "info"); break;
        case "report": if (lastReport) pi.sendMessage({ customType: REPORT, content: lastReport, display: true }); else ctx.ui.notify(`尚无 ${teacherLabel} 接管报告。`, "info"); break;
        default: ctx.ui.notify(`自动接管：${enabled ? "开启" : "关闭"}；当前：${phase === "teacher" ? teacherLabel : phase}；连续错误阈值 ${limits.consecutiveErrors}；轮数预算 RWKV ${limits.rwkvTurns} / ${teacherLabel} ${limits.teacherTurns}。`, "info");
      }
    },
  });
  pi.registerCommand("participation", { description: "查看本项目 RWKV 与接管模型的实际参与度", handler: async (_args, ctx) => {
    ctx.ui.notify(`${metrics(ctx).compact()}\n统计记录：${metrics(ctx).directory}`, "info");
  } });
}
