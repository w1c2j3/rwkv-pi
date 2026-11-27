import { existsSync, mkdirSync, readFileSync, realpathSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import { createProvider, type Model } from "@earendil-works/pi-ai";
import { openAICompletionsApi } from "@earendil-works/pi-ai/api/openai-completions.lazy";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { readConfig } from "./config.js";
import { createRwkvModel } from "./provider.js";
import { SubagentPlan } from "./subagent-plan.js";
import { SubagentManager, type Subagent } from "./subagents.js";

const WIDGET_KEY = "rwkv-subagent-hud";
const STATUS_KEY = "rwkv-subagents";
const ANSWER_CHARS = 6000;
const READ_TOOLS = new Set(["read", "grep", "find", "ls", "web_search", "web_fetch"]);
export const SUBAGENT_CMD_TOOLS = [
  "subagent_plan",
  "subagent_start",
  "subagent_status",
  "subagent_wait",
  "subagent_diff",
  "subagent_apply",
  "subagent_cancel",
] as const;
const CONTROL_TOOLS = new Set<string>(SUBAGENT_CMD_TOOLS);

const DELEGATION_PROMPT_SUFFIX =
  "【/rwkv-subagent 协同编排模式已开启】\n" +
  "分工约束：你（DeepSeek）负责架构分析、只读定位与质量审查；所有具体文件编写、代码修改、命令运行与单元测试执行全部派发给 RWKV-7 7.2B 子 Agent 并发完成。\n" +
  "1. 先用极少量 `read`/`grep` 精确定位真正存在缺陷的文件与准确 1-based 行号（无缺陷文件严禁派发写任务；禁止给出“约 X-Y 行”等模糊多行区间以免误删邻行）。\n" +
  "2. 调用 `subagent_plan` 将各独立文件的修改拆分为**无依赖并行步骤**（每个 `write: true` 步骤设 `maxRequests: 16`，要求子 Agent 先 `read` 目标行、精确修改后仅运行本模块单测如 `tests.test_xxx`，避免并行沙盒因其他文件尚未合入而误判失败），立即单次调用 `subagent_start` 全量并发拉起并用 `subagent_wait` 收集。\n" +
  "3. 对已完成的写子任务依次 `subagent_diff` 审查后 `subagent_apply` 合入主工作区；全部合入后派发 1 个只读验证子任务（`write: false, maxRequests: 16`）仅运行单条 `python3 -m unittest discover -s tests -v` 返回退出码与统计即可收尾（严禁编写多行 python -c 内联测试脚本，严禁主模型越权代写）。\n" +
  "4. 全程使用简明简体中文说明与总结。";

// Official public API pricing (CNY per 1M tokens) for transparent cost & savings comparison:
// DeepSeek V3.2 official RMB pricing: Cache Hit ¥0.20/M, Cache Miss ¥2.00/M, Output ¥3.00/M
const DS_IN_MISS_CNY_PER_M = 2.0;
const DS_IN_HIT_CNY_PER_M = 0.2;
// Multi-step coding loops average ~65% prefix KV cache hit + 35% miss => ¥0.83/M blended input
const DS_BLENDED_IN_CNY_PER_M = 0.65 * DS_IN_HIT_CNY_PER_M + 0.35 * DS_IN_MISS_CNY_PER_M;
const DS_OUT_CNY_PER_M = 3.0;

// Flagship commercial coding API reference (Claude Sonnet 4: $3/M in, $15/M out @ 7.2 CNY/USD, ~65% cache read)
const FLAGSHIP_BLENDED_IN_CNY_PER_M = 8.64;
const FLAGSHIP_OUT_CNY_PER_M = 108.0;

interface DeepSeekUsage {
  responses: number;
  inputMissTokens: number;
  cacheHitTokens: number;
  outputTokens: number;
  readToolCalls: number;
}

interface DirectRwkvUsage {
  responses: number;
  inputTokens: number;
  outputTokens: number;
  toolCalls: number;
}

function resolveDeepSeekApiKey(env: NodeJS.ProcessEnv = process.env): string {
  const direct = env.DEEPSEEK_API_KEY?.trim() || env.TEACHER_API_KEY?.trim() || env.RWKV_WEB_API_KEY?.trim();
  if (direct) return direct;
  const candidates = [
    env.TEACHER_API_KEY_FILE?.trim(),
    join(homedir(), ".config/rwkv-pi/deepseek.key"),
    join(homedir(), ".config/rwkv-pi-paid-pilot/deepseek.key"),
  ].filter((p): p is string => Boolean(p));
  for (const file of candidates) {
    if (!existsSync(file)) continue;
    const raw = readFileSync(file, "utf8");
    if (file.endsWith(".env.local") || file.includes(".env")) {
      const m = /^DEEPSEEK_API_KEY=(.+)$/m.exec(raw);
      if (m?.[1]) return m[1].trim().replace(/^['"]|['"]$/g, "");
    } else if (raw.trim()) {
      return raw.trim();
    }
  }
  return "";
}

function resolveSubagentDataDir(env: NodeJS.ProcessEnv = process.env): string {
  const configured = env.RWKV_PI_DATA_ROOT?.trim();
  if (configured && existsSync(configured)) {
    const dir = join(realpathSync(configured), "runs", "subagents");
    mkdirSync(dir, { recursive: true });
    return realpathSync(dir);
  }
  const fallback = join(homedir(), ".local", "share", "rwkv-pi", "runs", "subagents");
  mkdirSync(fallback, { recursive: true });
  return realpathSync(fallback);
}

const DS_PROVIDER_ID = "rwkv-deepseek";

function createDeepSeekModels(baseUrl: string): { chat: Model<"openai-completions">; reasoner: Model<"openai-completions"> } {
  const common = {
    provider: DS_PROVIDER_ID,
    api: "openai-completions" as const,
    baseUrl,
    input: ["text"] as ("text")[],
    contextWindow: 131072,
    maxTokens: 8192,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  };
  return {
    chat: {
      ...common,
      id: "deepseek-chat",
      name: "DeepSeek V3.2 Chat (主规划)",
      reasoning: false,
      compat: { supportsStore: false, supportsDeveloperRole: false, supportsReasoningEffort: false, supportsStrictMode: false, maxTokensField: "max_tokens" },
    },
    reasoner: {
      ...common,
      id: "deepseek-reasoner",
      name: "DeepSeek R1/V3.2 Reasoner (深度主规划)",
      reasoning: true,
      maxTokens: 16384,
      compat: {
        supportsStore: false,
        supportsDeveloperRole: false,
        supportsReasoningEffort: false,
        supportsStrictMode: false,
        maxTokensField: "max_tokens",
        requiresReasoningContentOnAssistantMessages: true,
      },
    },
  };
}

function fmtTok(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

function fmtCny(cny: number): string {
  if (cny < 0.01 && cny > 0) return `¥${cny.toFixed(4)}`;
  if (cny < 1) return `¥${cny.toFixed(4)}`;
  return `¥${cny.toFixed(2)}`;
}

function pct(part: number, total: number): string {
  if (total <= 0) return "0.0%";
  return `${((part / total) * 100).toFixed(1)}%`;
}

function summaryItem(agent: Subagent, withAnswer: boolean) {
  const seconds = agent.startedAt ? Math.round(((agent.finishedAt ?? Date.now()) - agent.startedAt) / 1000) : 0;
  return {
    id: agent.id,
    name: agent.name,
    state: agent.state,
    write: agent.write,
    requests: agent.requests,
    toolCalls: agent.toolCalls,
    toolErrors: agent.toolErrors,
    inputTokens: agent.work.inputTokens,
    outputTokens: agent.work.outputTokens,
    seconds,
    ...(agent.lastTool && !agent.finishedAt ? { lastTool: agent.lastTool } : {}),
    ...(agent.error ? { error: agent.error } : {}),
    ...(withAnswer && agent.answer ? { answer: agent.answer.length > ANSWER_CHARS ? agent.answer.slice(0, ANSWER_CHARS) + "…(truncated)" : agent.answer } : {}),
    ...(agent.changes ? {
      changedFiles: agent.changes.map((c) => `${c.kind}: ${c.path}`),
      diffLines: `+${agent.addedLines ?? 0}/-${agent.removedLines ?? 0}`,
      applied: agent.applied === true,
    } : {}),
  };
}

const jsonResult = (value: unknown) => ({
  content: [{ type: "text" as const, text: JSON.stringify(value, null, 2) }],
  details: value,
});

export function installRwkvSubagentCommand(pi: ExtensionAPI, hostCwd: string): void {
  if (
    process.env.RWKV_IS_CHILD_SUBAGENT === "1" ||
    process.argv.some((arg) => /(?:^|[/\\])(subagent|orchestrator)-extension\.[jt]s$/.test(arg))
  ) {
    return;
  }

  const rwkvConfig = readConfig();
  const rwkvModel = createRwkvModel(rwkvConfig);
  const deepseekBaseUrl = (process.env.DEEPSEEK_BASE_URL?.trim() || "https://api.deepseek.com/v1").replace(/\/+$/, "");
  const dsModels = createDeepSeekModels(deepseekBaseUrl);
  const initialApiKey = resolveDeepSeekApiKey();

  let enabled = false;
  let activeDsVariant: "chat" | "reasoner" = "chat";
  let providerRegistered = false;
  let toolsRegistered = false;

  // Pre-register passive provider metadata at load time when key is present so pi's async
  // credential snapshot (`hasConfiguredAuth`) is already ready when `/rwkv-subagent` is invoked.
  // Default model, tools, and prompts remain 100% pure RWKV with zero DeepSeek traffic until enabled.
  if (initialApiKey) {
    providerRegistered = true;
    pi.registerProvider(createProvider({
      id: DS_PROVIDER_ID,
      name: "DeepSeek (Orchestrator)",
      baseUrl: deepseekBaseUrl,
      models: [dsModels.chat, dsModels.reasoner],
      api: openAICompletionsApi(),
      auth: {
        apiKey: {
          name: "DeepSeek API Key",
          resolve: async () => ({ auth: { apiKey: resolveDeepSeekApiKey() || initialApiKey } }),
        },
      },
    }));
  }
  let baselineRwkvTools: string[] | undefined;
  let manager: SubagentManager | undefined;
  let lastCtx: ExtensionContext | undefined;
  let subagentsCountAtTurnStart = 0;

  const plan = new SubagentPlan();
  const reviewed = new Map<string, string>();
  const dsUsage: DeepSeekUsage = { responses: 0, inputMissTokens: 0, cacheHitTokens: 0, outputTokens: 0, readToolCalls: 0 };
  const directRwkv: DirectRwkvUsage = { responses: 0, inputTokens: 0, outputTokens: 0, toolCalls: 0 };

  function computeMetrics() {
    const subs = manager?.list() ?? [];
    let subResponses = 0, subIn = 0, subOut = 0, subTools = 0;
    let appliedFiles = 0, addedLines = 0, removedLines = 0;
    for (const a of subs) {
      subResponses += a.work.responses;
      subIn += a.work.inputTokens;
      subOut += a.work.outputTokens;
      subTools += a.work.toolCalls;
      if (a.applied) {
        appliedFiles += a.appliedFiles?.length ?? a.changes?.length ?? 0;
        addedLines += a.addedLines ?? 0;
        removedLines += a.removedLines ?? 0;
      }
    }
    const totalRwkvResponses = directRwkv.responses + subResponses;
    const totalRwkvIn = directRwkv.inputTokens + subIn;
    const totalRwkvOut = directRwkv.outputTokens + subOut;
    const totalRwkvTools = directRwkv.toolCalls + subTools;

    const totalTools = totalRwkvTools + dsUsage.readToolCalls;
    const totalOut = totalRwkvOut + dsUsage.outputTokens;

    const actualDsCny =
      (dsUsage.inputMissTokens / 1_000_000) * DS_IN_MISS_CNY_PER_M +
      (dsUsage.cacheHitTokens / 1_000_000) * DS_IN_HIT_CNY_PER_M +
      (dsUsage.outputTokens / 1_000_000) * DS_OUT_CNY_PER_M;

    const rwkvEquivalentDsCny =
      (totalRwkvIn / 1_000_000) * DS_BLENDED_IN_CNY_PER_M +
      (totalRwkvOut / 1_000_000) * DS_OUT_CNY_PER_M;

    const soloDsCny = actualDsCny + rwkvEquivalentDsCny;
    const savedDsCny = soloDsCny - actualDsCny;
    const savedDsPct = soloDsCny > 0 ? (savedDsCny / soloDsCny) * 100 : 100;

    const dsTotalIn = dsUsage.inputMissTokens + dsUsage.cacheHitTokens;
    const soloFlagshipCny =
      ((dsTotalIn + totalRwkvIn) / 1_000_000) * FLAGSHIP_BLENDED_IN_CNY_PER_M +
      ((dsUsage.outputTokens + totalRwkvOut) / 1_000_000) * FLAGSHIP_OUT_CNY_PER_M;
    const savedFlagshipCny = Math.max(0, soloFlagshipCny - actualDsCny);
    const savedFlagshipPct = soloFlagshipCny > 0 ? (savedFlagshipCny / soloFlagshipCny) * 100 : 100;

    return {
      subs,
      subResponses, subIn, subOut, subTools,
      totalRwkvResponses, totalRwkvIn, totalRwkvOut, totalRwkvTools,
      totalTools, totalOut,
      appliedFiles, addedLines, removedLines,
      actualDsCny, soloDsCny, savedDsCny, savedDsPct,
      soloFlagshipCny, savedFlagshipCny, savedFlagshipPct,
    };
  }

  function refreshHud(ctx?: ExtensionContext) {
    const ui = (ctx ?? lastCtx)?.ui;
    if (!ui) return;
    if (!enabled) {
      ui.setWidget(WIDGET_KEY, undefined);
      ui.setStatus(STATUS_KEY, undefined);
      return;
    }
    const m = computeMetrics();
    const running = m.subs.filter((a) => a.state === "running");
    const done = m.subs.filter((a) => a.state === "completed").length;
    const failed = m.subs.filter((a) => a.finishedAt && a.state !== "completed").length;
    const activeAction = running[0]?.lastTool
      ? `正在执行: ${running[0].name} → ${running[0].lastTool.slice(0, 58)}`
      : m.subs.length
        ? `子Agent累计: ${done} 完成${failed ? ` · ${failed} 异常` : ""}`
        : "等待主Agent拆解规划并派发 RWKV 子任务…";

    const dsName = activeDsVariant === "reasoner" ? "deepseek-reasoner" : "deepseek-chat";
    const lines = [
      `┌─ ◆ /rwkv-subagent 已启用 │ 主规划: ${dsName} ⇄ 子执行: RWKV-7 7.2B (${rwkvConfig.modelId}) │ /rwkv-subagent off 切回纯RWKV`,
      `│ 并发状态: ${running.length} 运行中 · ${done}/${m.subs.length} 完成 │ ${activeAction}`,
      `│ 工作分担: 工具执行 RWKV ${m.totalRwkvTools}/${m.totalTools} (${pct(m.totalRwkvTools, m.totalTools)}) │ 代码写入 RWKV +${m.addedLines}/-${m.removedLines} 行 (${m.appliedFiles} 文件, 100%) │ 生成Token RWKV ${fmtTok(m.totalRwkvOut)} (${pct(m.totalRwkvOut, m.totalOut)})`,
      `└─ 省钱账单: 实付 ${fmtCny(m.actualDsCny)} vs 纯DeepSeek独跑 ${fmtCny(m.soloDsCny)} → 省 ${fmtCny(m.savedDsCny)} (${m.savedDsPct.toFixed(1)}% · 千次省 ¥${(m.savedDsCny * 1000).toFixed(1)}) │ 对标Claude旗舰省 ${m.savedFlagshipPct.toFixed(1)}% (千次省 ¥${(m.savedFlagshipCny * 1000).toFixed(0)})`,
    ];
    ui.setWidget(WIDGET_KEY, lines, { placement: "aboveEditor" });
    ui.setStatus(STATUS_KEY, `RWKV干活 ${pct(m.totalRwkvTools, m.totalTools)} · 已省 ${m.savedDsPct.toFixed(0)}%`);
  }

  function buildMarkdownReport(): string {
    const m = computeMetrics();
    const modeLabel = enabled
      ? `DeepSeek (${activeDsVariant === "reasoner" ? "deepseek-reasoner" : "deepseek-chat"}) 主规划 + RWKV-7 7.2B (${rwkvConfig.modelId}) 并发子 Agent`
      : `纯 RWKV-7 7.2B (${rwkvConfig.modelId}) 独立完成全部工作（零外部 API 调用）`;

    const subLines = m.subs.length
      ? m.subs.map((a) => {
          const sec = a.startedAt ? Math.round(((a.finishedAt ?? Date.now()) - a.startedAt) / 1000) : 0;
          const editTag = a.write ? ` | 改动 +${a.addedLines ?? 0}/-${a.removedLines ?? 0} 行${a.applied ? " (已合入)" : ""}` : " | 只读核查";
          return `- \`${a.id}\` **${a.name}** [${a.state}]：${a.requests} 轮推理 · ${a.toolCalls} 次工具 · 输入 ${fmtTok(a.work.inputTokens)} / 输出 ${fmtTok(a.work.outputTokens)} tok · ${sec}s${editTag}`;
        }).join("\n")
      : "- （尚未派发子 Agent，所有任务均由主 RWKV 直接执行）";

    return [
      `### ◆ RWKV 工作量分担与成本节省报告`,
      `- **当前模式**：${modeLabel}`,
      ``,
      `| 维度 | RWKV-7 7.2B (自有卡 ¥0) | DeepSeek 主规划 | RWKV 承担占比 |`,
      `| :--- | :---: | :---: | :---: |`,
      `| **工具执行次数** (读写/编辑/单测) | **${m.totalRwkvTools} 次** | ${dsUsage.readToolCalls} 次 (仅只读) | **${pct(m.totalRwkvTools, m.totalTools)}** |`,
      `| **模型推理轮数** | **${m.totalRwkvResponses} 轮** | ${dsUsage.responses} 轮 | **${pct(m.totalRwkvResponses, m.totalRwkvResponses + dsUsage.responses)}** |`,
      `| **处理上下文输入 Token** | **${fmtTok(m.totalRwkvIn)}** | ${fmtTok(dsUsage.inputMissTokens + dsUsage.cacheHitTokens)} (缓存命中 ${fmtTok(dsUsage.cacheHitTokens)}) | **${pct(m.totalRwkvIn, m.totalRwkvIn + dsUsage.inputMissTokens + dsUsage.cacheHitTokens)}** |`,
      `| **生成输出 Token** (含思考与代码) | **${fmtTok(m.totalRwkvOut)}** | ${fmtTok(dsUsage.outputTokens)} | **${pct(m.totalRwkvOut, m.totalOut)}** |`,
      `| **落盘代码修改** | **${m.appliedFiles} 个文件 (+${m.addedLines} / -${m.removedLines} 行)** | 0 行 (严禁代写) | **100.0%** |`,
      ``,
      `#### ★ 真实 API 费用对比与节省测算`,
      `- **本会话实付 API 成本**：**${fmtCny(m.actualDsCny)}**（RWKV 跑在自有显卡，边际 API 费为 \`¥0\`）`,
      `- **对比 ① · 纯 DeepSeek V3.2 官方 API 独跑**：预估 **${fmtCny(m.soloDsCny)}** → 已省 **${fmtCny(m.savedDsCny)}（节省 ${m.savedDsPct.toFixed(1)}%，每 1,000 次同类任务省 ¥${(m.savedDsCny * 1000).toFixed(2)}）**`,
      `- **对比 ② · 海外旗舰 Claude Sonnet 4 API 独跑**：预估 **${fmtCny(m.soloFlagshipCny)}** → 已省 **${fmtCny(m.savedFlagshipCny)}（节省 ${m.savedFlagshipPct.toFixed(1)}%，每 1,000 次同类任务省 ¥${(m.savedFlagshipCny * 1000).toFixed(0)}）**`,
      ``,
      `#### ❖ RWKV 子 Agent 执行明细`,
      subLines,
    ].join("\n");
  }

  function getManager(ctx: ExtensionContext): SubagentManager {
    lastCtx = ctx;
    manager ??= new SubagentManager(hostCwd, undefined, resolveSubagentDataDir(), (agent) => {
      refreshHud(ctx);
      if (agent.finishedAt && ctx.isIdle()) {
        ctx.ui.notify(`RWKV 子 Agent ${agent.name} (${agent.id}) 已结束: ${agent.state}`, agent.state === "completed" ? "info" : "warning");
      }
    });
    return manager;
  }

  function ensureSubagentToolsRegistered() {
    if (toolsRegistered) return;
    toolsRegistered = true;

    pi.registerTool({
      name: "subagent_plan",
      label: "规划 RWKV 子任务",
      defaultActive: false,
      promptSnippet: "Record concrete parallel Chinese steps for RWKV-7 7.2B subagents",
      description: "制定简洁目标与交给 RWKV-7 7.2B 子 Agent 执行的具体中文步骤。明确标记 write（写代码/跑测试设为 true，纯分析设为 false）。无依赖步骤将并发启动。",
      parameters: Type.Object({
        goal: Type.String(),
        steps: Type.Array(Type.Object({
          name: Type.String(),
          task: Type.String({ description: "完整清晰的中文指令：指明相对路径、修复目标与单测验收命令。" }),
          write: Type.Boolean(),
          dependsOn: Type.Optional(Type.Array(Type.String())),
          maxRequests: Type.Optional(Type.Integer({ minimum: 1 })),
          timeoutSeconds: Type.Optional(Type.Integer({ minimum: 1 })),
        }), { minItems: 1, maxItems: 16 }),
      }),
      async execute(_id, params, _signal, _update, ctx) {
        const p = plan.set(params.goal, params.steps, getManager(ctx).list());
        refreshHud(ctx);
        return jsonResult(p);
      },
    });

    pi.registerTool({
      name: "subagent_start",
      label: "启动 RWKV 子 Agent",
      defaultActive: false,
      description: "并发派发 subagent_plan 中已就绪的步骤（省略 steps 则派发全部就绪步骤）。每个写步骤在独立 bwrap 沙盒副本中由 RWKV-7 7.2B 执行。",
      promptSnippet: "Dispatch parallel RWKV-7 7.2B subagents in isolated sandboxes",
      parameters: Type.Object({ steps: Type.Optional(Type.Array(Type.String(), { minItems: 1, maxItems: 16 })) }),
      async execute(_id, params, _signal, _update, ctx) {
        const m = getManager(ctx);
        const steps = plan.prepare(params.steps, m.list());
        const started = m.start(steps);
        plan.started(steps, started);
        refreshHud(ctx);
        return jsonResult({ started: started.map((a) => summaryItem(a, false)), next: "调用 subagent_wait 等待完成并获取结果。" });
      },
    });

    pi.registerTool({
      name: "subagent_status",
      label: "查询 RWKV 子 Agent 状态",
      defaultActive: false,
      promptSnippet: "Inspect live RWKV subagent status non-blockingly",
      description: "非阻塞查看各 RWKV 子 Agent 当前状态、轮数、工具调用与耗时。",
      parameters: Type.Object({ ids: Type.Optional(Type.Array(Type.String())) }),
      async execute(_id, params, _signal, _update, ctx) {
        return jsonResult(getManager(ctx).list(params.ids).map((a) => summaryItem(a, false)));
      },
    });

    pi.registerTool({
      name: "subagent_wait",
      label: "等待 RWKV 子 Agent 完成",
      defaultActive: false,
      promptSnippet: "Wait for running RWKV subagents and collect results",
      description: "等待指定（默认全部）RWKV 子 Agent 执行完毕，返回总结、Token 用量与改动文件清单。",
      parameters: Type.Object({
        ids: Type.Optional(Type.Array(Type.String())),
        timeoutSeconds: Type.Optional(Type.Integer({ minimum: 1, maximum: 3600 })),
      }),
      async execute(_id, params, signal, onUpdate, ctx) {
        const m = getManager(ctx);
        const ticker = setInterval(() => {
          const list = m.list(params.ids);
          refreshHud(ctx);
          onUpdate?.({
            content: [{ type: "text", text: list.map((a) => `${a.id} (${a.name}): ${a.state} · ${a.requests}轮 · ${a.toolCalls}次工具${a.lastTool ? ` · ${a.lastTool.slice(0, 60)}` : ""}`).join("\n") }],
            details: undefined,
          });
        }, 2500);
        try {
          const list = await m.wait(params.ids, (params.timeoutSeconds ?? 600) * 1000, signal);
          refreshHud(ctx);
          return jsonResult(list.map((a) => summaryItem(a, true)));
        } finally {
          clearInterval(ticker);
        }
      },
    });

    pi.registerTool({
      name: "subagent_diff",
      label: "审查 RWKV 子 Agent 代码改动",
      defaultActive: false,
      promptSnippet: "Review unified diff from a finished write subagent before applying",
      description: "查看已完成写子任务的 unified diff 补丁（应用修改前必须先审查 diff）。",
      parameters: Type.Object({ id: Type.String() }),
      async execute(_id, params, _signal, _update, ctx) {
        const m = getManager(ctx);
        const agent = m.list([params.id])[0]!;
        if (!agent.finishedAt) throw new Error("请先通过 subagent_wait 等待该子任务完成。");
        const diff = m.diff(params.id);
        reviewed.set(params.id, diff);
        return { content: [{ type: "text" as const, text: diff }], details: undefined };
      },
    });

    pi.registerTool({
      name: "subagent_apply",
      label: "应用 RWKV 子 Agent 修改",
      defaultActive: false,
      promptSnippet: "Merge reviewed subagent file changes into workspace",
      description: "将已通过 subagent_diff 审查的 RWKV 子任务文件修改安全合并回当前工作区。",
      parameters: Type.Object({ id: Type.String(), files: Type.Optional(Type.Array(Type.String())) }),
      async execute(_id, params, _signal, _update, ctx) {
        const m = getManager(ctx);
        if (!reviewed.has(params.id) || reviewed.get(params.id) !== m.diff(params.id)) {
          throw new Error("请先调用 subagent_diff 审查最新补丁后再执行 subagent_apply。");
        }
        const applied = m.apply(params.id, params.files);
        reviewed.delete(params.id);
        refreshHud(ctx);
        return jsonResult(applied);
      },
    });

    pi.registerTool({
      name: "subagent_cancel",
      label: "取消 RWKV 子 Agent",
      defaultActive: false,
      promptSnippet: "Cancel running or queued RWKV subagents",
      description: "终止未完成的 RWKV 子 Agent。",
      parameters: Type.Object({ ids: Type.Optional(Type.Array(Type.String())) }),
      async execute(_id, params, _signal, _update, ctx) {
        const cancelled = getManager(ctx).cancel(params.ids).map((a) => summaryItem(a, false));
        refreshHud(ctx);
        return jsonResult(cancelled);
      },
    });
  }

  // Register tool definitions at extension load time so pi's `_toolRegistry` indexes them;
  // `defaultActive: false` + `syncInactiveWhenDisabled()` keeps them 100% hidden in default pure-RWKV mode.
  ensureSubagentToolsRegistered();

  function syncInactiveWhenDisabled() {
    const active = pi.getActiveTools();
    if (!baselineRwkvTools) {
      baselineRwkvTools = active.filter((t) => !CONTROL_TOOLS.has(t));
    }
    if (!enabled && active.some((t) => CONTROL_TOOLS.has(t))) {
      pi.setActiveTools(baselineRwkvTools);
    }
  }

  async function enableMode(variant: "chat" | "reasoner", ctx: ExtensionContext): Promise<boolean> {
    lastCtx = ctx;
    const apiKey = resolveDeepSeekApiKey();
    if (!apiKey) {
      ctx.ui.notify("未找到 DEEPSEEK_API_KEY（请设置环境变量或在 RWKV-SearchReader/.env.local 中配置）。", "error");
      return false;
    }
    if (!baselineRwkvTools) {
      baselineRwkvTools = pi.getActiveTools().filter((t) => !CONTROL_TOOLS.has(t));
    }
    if (!providerRegistered) {
      providerRegistered = true;
      pi.registerProvider(createProvider({
        id: DS_PROVIDER_ID,
        name: "DeepSeek (Orchestrator)",
        baseUrl: deepseekBaseUrl,
        models: [dsModels.chat, dsModels.reasoner],
        api: openAICompletionsApi(),
        auth: {
          apiKey: {
            name: "DeepSeek API Key",
            resolve: async () => ({ auth: { apiKey } }),
          },
        },
      }));
    }
    ensureSubagentToolsRegistered();
    activeDsVariant = variant;
    const targetModel = variant === "reasoner" ? dsModels.reasoner : dsModels.chat;
    const ok = await pi.setModel(targetModel);
    if (!ok) {
      ctx.ui.notify(`切换到 ${targetModel.id} 失败，请检查 DeepSeek 密钥配置。`, "error");
      return false;
    }
    const readSubset = (baselineRwkvTools ?? ["read"]).filter((t) => READ_TOOLS.has(t));
    pi.setActiveTools([...readSubset, ...SUBAGENT_CMD_TOOLS]);
    enabled = true;
    refreshHud(ctx);
    ctx.ui.notify(
      `已开启 /rwkv-subagent 协同模式：主规划=${targetModel.id}，执行子Agent=RWKV-7 7.2B (${rwkvConfig.modelId})。输入 /rwkv-subagent off 随时切回纯 RWKV。`,
      "info",
    );
    return true;
  }

  async function disableMode(ctx: ExtensionContext): Promise<void> {
    lastCtx = ctx;
    manager?.cancel();
    enabled = false;
    await pi.setModel(rwkvModel);
    if (baselineRwkvTools?.length) {
      pi.setActiveTools(baselineRwkvTools);
    }
    refreshHud(ctx);
    ctx.ui.notify(`已关闭 /rwkv-subagent：恢复纯 RWKV-7 7.2B (${rwkvConfig.modelId}) 独立完成全部工作（零 DeepSeek 调用）。`, "info");
  }

  pi.registerCommand("rwkv-subagent", {
    description: "开启/关闭 DeepSeek 主规划 + RWKV-7 7.2B 并发子Agent模式，或查看工作量与省钱账单 (on|off|reasoner|cost)",
    handler: async (rawArgs, ctx) => {
      lastCtx = ctx;
      const arg = (rawArgs ?? "").trim().toLowerCase();
      if (arg === "off" || arg === "disable" || arg === "stop") {
        await disableMode(ctx);
        return;
      }
      if (arg === "cost" || arg === "status" || arg === "stats" || arg === "report") {
        pi.sendMessage({ customType: "rwkv-subagent-report", content: buildMarkdownReport(), display: true });
        return;
      }
      if (arg === "reasoner" || arg === "r1") {
        await enableMode("reasoner", ctx);
        return;
      }
      if (arg === "on" || arg === "enable" || arg === "chat") {
        await enableMode("chat", ctx);
        return;
      }
      // Bare `/rwkv-subagent`: toggle ON if currently OFF; if already ON, show report + toggle hint
      if (!enabled) {
        await enableMode("chat", ctx);
      } else {
        pi.sendMessage({ customType: "rwkv-subagent-report", content: buildMarkdownReport(), display: true });
        ctx.ui.notify("当前已处于 /rwkv-subagent 模式（输 /rwkv-subagent off 切回纯 RWKV，输 /rwkv-subagent reasoner 切换深度思考主模型）。", "info");
      }
    },
  });

  let autoApplied = false;
  async function maybeAutoEnable(ctx: ExtensionContext): Promise<void> {
    if (autoApplied) return;
    const auto = process.env.RWKV_SUBAGENT_AUTO?.trim().toLowerCase();
    if (auto === "1" || auto === "on" || auto === "chat" || auto === "reasoner") {
      autoApplied = true;
      await enableMode(auto === "reasoner" ? "reasoner" : "chat", ctx);
    }
  }

  pi.on("session_start", async (_event, ctx) => {
    lastCtx = ctx;
    syncInactiveWhenDisabled();
    await maybeAutoEnable(ctx);
  });

  pi.on("before_agent_start", async (event, ctx) => {
    lastCtx = ctx;
    syncInactiveWhenDisabled();
    await maybeAutoEnable(ctx);
    plan.reset();
    reviewed.clear();
    subagentsCountAtTurnStart = manager?.list().length ?? 0;
    const liveTools = pi.getActiveTools();
    if (event?.systemPromptOptions && Array.isArray(event.systemPromptOptions.selectedTools)) {
      event.systemPromptOptions.selectedTools = liveTools;
    }
    if (!enabled) return;
    const basePrompt = typeof event?.systemPrompt === "string" ? event.systemPrompt : "";
    return {
      systemPrompt: basePrompt ? `${basePrompt}\n\n${DELEGATION_PROMPT_SUFFIX}` : DELEGATION_PROMPT_SUFFIX,
    };
  });

  pi.on("tool_call", (event) => {
    if (!enabled) return;
    if (!READ_TOOLS.has(event.toolName) && !CONTROL_TOOLS.has(event.toolName)) {
      return { block: true, reason: "/rwkv-subagent 模式下主模型仅负责只读定位与审查；请调用 subagent_plan / subagent_start 派发给 RWKV-7 7.2B 执行修改与测试。" };
    }
  });

  pi.on("message_end", (event, ctx) => {
    lastCtx = ctx;
    const msg: any = event.message;
    if (msg?.role !== "assistant") return;
    const usage = msg.usage ?? {};
    if (msg.provider === DS_PROVIDER_ID) {
      dsUsage.responses++;
      if (Number.isSafeInteger(usage.input) && usage.input > 0) dsUsage.inputMissTokens += usage.input;
      if (Number.isSafeInteger(usage.cacheRead) && usage.cacheRead > 0) dsUsage.cacheHitTokens += usage.cacheRead;
      if (Number.isSafeInteger(usage.output) && usage.output > 0) dsUsage.outputTokens += usage.output;
    } else if (msg.provider === "rwkv") {
      directRwkv.responses++;
      for (const k of ["input", "cacheRead", "cacheWrite"]) {
        if (Number.isSafeInteger(usage[k]) && usage[k] > 0) directRwkv.inputTokens += usage[k];
      }
      if (Number.isSafeInteger(usage.output) && usage.output > 0) directRwkv.outputTokens += usage.output;
    }
    refreshHud(ctx);
  });

  pi.on("tool_execution_end", (event, ctx) => {
    lastCtx = ctx;
    if (CONTROL_TOOLS.has(event.toolName)) return;
    if (enabled || ctx.model?.provider === DS_PROVIDER_ID) {
      if (READ_TOOLS.has(event.toolName)) dsUsage.readToolCalls++;
    } else {
      directRwkv.toolCalls++;
    }
    refreshHud(ctx);
  });

  pi.on("agent_end", (_event, ctx) => {
    lastCtx = ctx;
    refreshHud(ctx);
    const currentSubs = manager?.list().length ?? 0;
    if (enabled && currentSubs > subagentsCountAtTurnStart) {
      subagentsCountAtTurnStart = currentSubs;
      const report = buildMarkdownReport();
      if ((ctx as { hasUI?: boolean }).hasUI) {
        pi.sendMessage({ customType: "rwkv-subagent-report", content: report, display: true });
      } else {
        process.stderr.write(`\n${report}\n`);
      }
    }
  });

  pi.on("session_shutdown", () => {
    manager?.shutdown();
  });
}
