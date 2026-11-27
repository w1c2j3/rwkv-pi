import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  createBashToolDefinition, createEditToolDefinition, createLsToolDefinition, createReadToolDefinition,
  createWriteToolDefinition, type ExtensionAPI,
} from "@earendil-works/pi-coding-agent";
import type { Message } from "@earendil-works/pi-ai";
import { readConfig } from "./config.js";
import { createLineModeTools, LineViewTracker } from "./line-edit.js";
import { repeatedIdenticalCall, repeatedTargetFailure, repeatedWorklistSpin } from "./loop-guard.js";
import { createRwkvProvider } from "./provider.js";
import { WORKSPACE_MOUNT, WorkspaceSandbox } from "./sandbox.js";
import { hostFs, withTolerance } from "./tolerant-tools.js";
import { PI_TOOL_NAMES } from "./tools.js";
import { createWebToolsController } from "./web-tools.js";
import { createWorklistTools, isWorklistTool, PiWorklistState } from "./worklist.js";
import { installRwkvSubagentCommand } from "./rwkv-subagent-command.js";

function selectsTool(name: string, argv: readonly string[] = process.argv, env: NodeJS.ProcessEnv = process.env): boolean {
  const idx = argv.findIndex((arg) => arg === "--tools" || arg === "-t");
  const fromArgv = idx >= 0 ? argv[idx + 1] : argv.find((arg) => arg.startsWith("--tools="))?.slice("--tools=".length);
  const raw = fromArgv ?? env.RWKV_TOOLS;
  return Boolean(raw?.split(",").map((s) => s.trim()).includes(name));
}

export default function rwkvExtension(pi: ExtensionAPI): void {
  const config = readConfig();
  const worklistMode = config.historyMode === "worklist";
  const worklistState = worklistMode ? new PiWorklistState() : undefined;
  const lineTracker = config.editMode === "line" ? new LineViewTracker() : undefined;
  const sandboxMode = process.env.RWKV_SANDBOX?.trim().toLowerCase();
  const useBwrap = sandboxMode === "bwrap" || sandboxMode === "bubblewrap";
  const hostCwd = process.cwd();
  let spillHostDir: string | undefined;
  let spillSeq = 0;
  if (useBwrap) {
    spillHostDir = mkdtempSync(join(tmpdir(), "rwkv-pi-spill-"));
    process.once("exit", () => { try { rmSync(spillHostDir!, { recursive: true, force: true }); } catch {} });
  }
  const sandbox = useBwrap ? new WorkspaceSandbox(hostCwd, undefined, WORKSPACE_MOUNT, spillHostDir) : undefined;
  const hostCwdPromptTag = `<cwd>\n${hostCwd.replace(/\\/g, "/")}\n</cwd>`;
  const sandboxCwdPromptTag = sandbox ? `<cwd>\n${sandbox.cwd}\n</cwd>` : "";
  let latestUserQuestion = "";

  pi.registerProvider(createRwkvProvider(config, {
    onPayloadMessages: (msgs) => {
      if (sandbox && msgs[0]?.role === "system" && typeof msgs[0].content === "string" && msgs[0].content.includes(hostCwdPromptTag)) {
        msgs[0].content = msgs[0].content.replace(hostCwdPromptTag, sandboxCwdPromptTag);
      }
      for (let i = msgs.length - 1; i >= 0; i--) {
        const m = msgs[i];
        if (m?.role === "user" && typeof m.content === "string" && !m.content.startsWith("<tool_response>") && !m.content.startsWith("[Epoch State Ledger")) {
          latestUserQuestion = m.content;
          break;
        }
      }
      lineTracker?.syncWithActiveMessages(msgs);
    },
  }));

  if (sandbox) {
    const selectedPi = PI_TOOL_NAMES.filter(
      (name) => name === "read" || name === "write" || name === "edit" || name === "bash" || selectsTool(name),
    );
    const rawTools = sandbox.tools(selectedPi);
    const wrapped = config.editMode === "line"
      ? createLineModeTools(rawTools, sandbox.cwd, sandbox.fs, { tolerance: config.toolTolerance, tracker: lineTracker })
      : config.toolTolerance
        ? rawTools.map((def) => withTolerance(def, sandbox.cwd, sandbox.fs))
        : rawTools;
    const mapHostPath = (p: unknown) => {
      if (typeof p !== "string") return p;
      if (p === hostCwd) return sandbox.cwd;
      if (p.startsWith(hostCwd + "/")) return sandbox.cwd + p.slice(hostCwd.length);
      return p;
    };
    for (const definition of wrapped) {
      const origExec = definition.execute.bind(definition) as (...a: any[]) => Promise<any>;
      pi.registerTool({
        ...definition,
        async execute(id: string, args: any, signal?: AbortSignal, onUpdate?: unknown, ctx?: any) {
          const mappedArgs = args && typeof args === "object" && !Array.isArray(args)
            ? {
                ...args,
                ...("path" in args ? { path: mapHostPath(args.path) } : {}),
                ...("file_path" in args ? { file_path: mapHostPath(args.file_path) } : {}),
              }
            : args;
          const sandboxCtx = ctx && typeof ctx === "object" ? { ...ctx, cwd: sandbox.cwd } : ctx;
          return origExec(id, mappedArgs, signal, onUpdate, sandboxCtx);
        },
      } as never);
    }
  } else {
    // The harness bash tool does not expose PI_* session variables, so its system prompt has no
    // guideline about them. Interactive pi must present the same tool.
    if (process.env.RWKV_PI_PARITY?.trim() !== "0") {
      const cwd = hostCwd;
      const bashDef = createBashToolDefinition(cwd, { exposeSessionEnvironment: false });
      const wrappedBash = cwd !== "/workspace" ? {
        ...bashDef,
        async execute(callId: string, args: any, signal?: AbortSignal, onUpdate?: any, ctx?: any) {
          const cmd = typeof args?.command === "string"
            ? args.command.replace(/(?<![\w.-])\/workspace(?=\/|$|[\s"'`),;:])/g, cwd)
            : args?.command;
          return (bashDef as any).execute(callId, { ...args, command: cmd }, signal, onUpdate, ctx);
        },
      } : bashDef;
      pi.registerTool(wrappedBash as never);
    }
    // Same recall-tolerant or anchored line-coordinate execution as the harness. ls is registered only when
    // selected so 4-tool sessions never gain an extra tool in the prompt.
    const definitions = [
      createReadToolDefinition(hostCwd),
      createEditToolDefinition(hostCwd),
      createWriteToolDefinition(hostCwd),
      ...(selectsTool("ls") ? [createLsToolDefinition(hostCwd)] : []),
    ];
    if (config.editMode === "line") {
      for (const definition of createLineModeTools(definitions, hostCwd, hostFs, { tolerance: config.toolTolerance, tracker: lineTracker })) {
        pi.registerTool(definition as never);
      }
    } else if (config.toolTolerance) {
      for (const definition of definitions) {
        pi.registerTool(withTolerance(definition, hostCwd, hostFs) as never);
      }
    }
  }
  const wantSearch = selectsTool("web_search");
  const wantFetch = selectsTool("web_fetch");
  if (wantSearch || wantFetch) {
    const onSpill = spillHostDir
      ? async (_url: string, content: string) => {
          const name = `page-${++spillSeq}.txt`;
          writeFileSync(join(spillHostDir, name), content, "utf8");
          return `/tmp/dsh-spill/${name}`;
        }
      : undefined;
    const webFormat = process.env.RWKV_WEB_FORMAT?.trim().toLowerCase() === "json" ? "json" : "dsh";
    const webCtrl = createWebToolsController(process.env, {
      search: wantSearch,
      fetch: wantFetch,
      format: webFormat,
      getUserQuestion: () => latestUserQuestion,
      onSpill,
    });
    pi.on("input", async (ev) => {
      if (ev.source !== "extension") {
        latestUserQuestion = ev.text;
        webCtrl.resetTurn(ev.text);
      }
      return { action: "continue" };
    });
    for (const definition of webCtrl.tools) {
      pi.registerTool(definition as never);
    }
  }
  if (worklistState) {
    for (const definition of createWorklistTools(worklistState)) {
      pi.registerTool(definition as never);
    }
    pi.on("tool_execution_end", (event) => {
      if (!isWorklistTool(event.toolName)) worklistState.recordWorkspaceTool();
    });
  }
  // Same guards as the harness (src/harness.ts beforeToolCall): pi records a block as the call's
  // tool error result, so the model sees it and the session file keeps it.
  if (config.identicalCallLimit > 0 || config.targetFailureLimit > 0 || worklistMode) {
    pi.on("tool_call", async (event, ctx) => {
      const messages = ctx.sessionManager.getBranch().flatMap((entry: any) => entry.type === "message" ? [entry.message as Message] : []);
      const own = messages.flatMap(m => m.role === "assistant" ? m.content : []).find((b: any) => b.type === "toolCall" && b.id === event.toolCallId) as any;
      const name = own?.name ?? event.toolName, args = own?.arguments ?? (event as any).input;
      const reason = (worklistMode ? repeatedWorklistSpin(messages, name) : undefined)
        ?? (config.identicalCallLimit > 0 ? repeatedIdenticalCall(messages, name, args, config.identicalCallLimit) : undefined)
        ?? (config.targetFailureLimit > 0 ? repeatedTargetFailure(messages, name, args, config.targetFailureLimit, config.editMode) : undefined);
      return reason ? { block: true, reason } : undefined;
    });
  }
  installRwkvSubagentCommand(pi, hostCwd);

  const tuiStateFile = process.env.RWKV_TUI_STATE_FILE?.trim();
  if (tuiStateFile) {
    let completedTurns = 0;
    const writeState = (state: string) => {
      try { writeFileSync(tuiStateFile, JSON.stringify({ state, completedTurns, ts: Date.now() }), "utf8"); } catch {}
    };
    pi.on("session_start", () => writeState("ready"));
    pi.on("agent_start", () => writeState("running"));
    pi.on("agent_end", () => { completedTurns++; writeState("idle"); });
  }
}
