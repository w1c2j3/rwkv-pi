import assert from "node:assert/strict";
import test from "node:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import rwkvExtension from "../dist/extension.js";

test("/rwkv-subagent stays 100% pure RWKV by default, toggles DeepSeek+RWKV on demand, and computes exact cost savings", async (t) => {
  const root = mkdtempSync(join(tmpdir(), "rwkv-subagent-cmd-"));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const prior = { ...process.env };
  t.after(() => {
    for (const k of Object.keys(process.env)) if (!(k in prior)) delete process.env[k];
    Object.assign(process.env, prior);
  });

  Object.assign(process.env, {
    RWKV_BASE_URL: "http://127.0.0.1:19580/v1",
    RWKV_MODEL: "rwkv7-g1l-7.2b-mastery-r2-merged",
    RWKV_API_KEY: "local-rwkv",
    RWKV_SANDBOX: "none",
    RWKV_SUBAGENT_SANDBOX: "none",
    RWKV_PI_DATA_ROOT: root,
    DEEPSEEK_API_KEY: "sk-test-fixture",
  });

  const providers = [];
  const tools = new Map();
  const commands = new Map();
  const hooks = new Map();
  const messages = [];
  let activeTools = ["read", "write", "edit", "bash", "grep", "find", "ls", "web_search", "web_fetch"];
  let currentModel = { provider: "rwkv", id: "rwkv7-g1l-7.2b-mastery-r2-merged" };
  let widgetContent;

  const pi = {
    registerProvider(p) { providers.push(p); },
    registerTool(tool) { tools.set(tool.name, tool); },
    registerCommand(name, opts) { commands.set(name, opts); },
    on(event, fn) { hooks.set(event, [...(hooks.get(event) ?? []), fn]); },
    getActiveTools: () => [...activeTools],
    setActiveTools(next) { activeTools = [...next]; },
    async setModel(m) { currentModel = m; return true; },
    sendMessage(m) { messages.push(m); },
  };

  rwkvExtension(pi);

  // 1. Default state: active model is 100% pure RWKV; subagent_* tools are defaultActive=false and inactive!
  assert.equal(currentModel.provider, "rwkv");
  assert.equal(tools.get("subagent_plan")?.defaultActive, false);
  assert.equal(activeTools.includes("subagent_plan"), false);
  assert.equal(commands.has("rwkv-subagent"), true);

  const ctx = {
    cwd: root,
    model: currentModel,
    isIdle: () => true,
    sessionManager: { getBranch: () => [] },
    ui: {
      notify() {},
      setStatus() {},
      setWidget(_k, lines) { widgetContent = lines; },
    },
  };
  const emit = async (name, ev) => Promise.all((hooks.get(name) ?? []).map((fn) => fn(ev, ctx)));

  // Even if CLI `--tools` initially activated subagent_plan before session_start, session_start strips it when disabled
  activeTools.push("subagent_plan", "subagent_start");
  await emit("session_start", {});
  assert.equal(activeTools.includes("subagent_plan"), false);
  assert.equal(activeTools.includes("edit"), true);

  // Write tools & system prompt untouched when /rwkv-subagent is NOT enabled
  const sysOptsOff = { selectedTools: [...activeTools] };
  const beforeOff = await emit("before_agent_start", { prompt: "修复 bug", systemPrompt: "BASE_RWKV_PROMPT", systemPromptOptions: sysOptsOff });
  assert.equal(beforeOff.every((r) => r === undefined), true);
  const writeCall = await emit("tool_call", { toolName: "edit" });
  assert.equal(writeCall.every((r) => r === undefined), true);

  // Record a pure RWKV turn
  await emit("message_end", { message: { role: "assistant", provider: "rwkv", usage: { input: 3000, output: 600 } } });
  await emit("tool_execution_end", { toolName: "edit", isError: false });

  // 2. Enable via `/rwkv-subagent`
  await commands.get("rwkv-subagent").handler("", ctx);
  ctx.model = currentModel;
  assert.equal(currentModel.provider, "rwkv-deepseek");
  assert.equal(currentModel.id, "deepseek-chat");
  assert.equal(tools.has("subagent_plan"), true);
  assert.equal(activeTools.includes("subagent_start"), true);
  assert.equal(activeTools.includes("edit"), false);
  assert.ok(Array.isArray(widgetContent) && widgetContent.length === 4);

  // Main DeepSeek agent is blocked from direct code edits; before_agent_start returns systemPrompt with delegation suffix
  const blocked = await emit("tool_call", { toolName: "edit" });
  assert.equal(blocked.some((r) => r?.block === true), true);
  const sysOptsOn = { selectedTools: [...activeTools] };
  const beforeOn = await emit("before_agent_start", { prompt: "重构模块", systemPrompt: "BASE_DS_PROMPT", systemPromptOptions: sysOptsOn });
  const promptResult = beforeOn.find((r) => r?.systemPrompt)?.systemPrompt ?? "";
  assert.match(promptResult, /BASE_DS_PROMPT/);
  assert.match(promptResult, /\/rwkv-subagent 协同编排模式已开启/);

  // Simulate DeepSeek light planning turn + cost report
  await emit("message_end", { message: { role: "assistant", provider: "rwkv-deepseek", usage: { input: 800, cacheRead: 1200, output: 150 } } });
  await commands.get("rwkv-subagent").handler("cost", ctx);
  assert.equal(messages.length, 1);
  assert.match(messages[0].content, /RWKV 工作量分担与成本节省报告/);
  assert.match(messages[0].content, /100\.0%/);

  // 3. Disable via `/rwkv-subagent off` -> back to pure RWKV solo!
  await commands.get("rwkv-subagent").handler("off", ctx);
  ctx.model = currentModel;
  assert.equal(currentModel.provider, "rwkv");
  assert.equal(activeTools.includes("edit"), true);
  assert.equal(activeTools.includes("subagent_start"), false);
  assert.equal(widgetContent, undefined);

  await emit("session_shutdown", {});
});
