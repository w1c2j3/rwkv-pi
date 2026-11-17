import {
  createBashToolDefinition, createEditToolDefinition, createLsToolDefinition, createReadToolDefinition,
  createWriteToolDefinition, type ExtensionAPI,
} from "@earendil-works/pi-coding-agent";
import type { Message } from "@earendil-works/pi-ai";
import { readConfig } from "./config.js";
import { repeatedIdenticalCall, repeatedTargetFailure, repeatedWorklistSpin } from "./loop-guard.js";
import { createRwkvProvider } from "./provider.js";
import { hostFs, withTolerance } from "./tolerant-tools.js";
import { createWorklistTools, isWorklistTool, PiWorklistState } from "./worklist.js";

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
  pi.registerProvider(createRwkvProvider(config));
  // The harness bash tool does not expose PI_* session variables, so its system prompt has no
  // guideline about them. Interactive pi must present the same tool.
  if (process.env.RWKV_PI_PARITY?.trim() !== "0") pi.registerTool(createBashToolDefinition(process.cwd(), { exposeSessionEnvironment: false }) as never);
  // Same recall-tolerant execution as the harness. ls is registered only when selected so 4-tool
  // sessions never gain an extra tool in the prompt.
  if (config.toolTolerance) {
    const definitions = [
      createReadToolDefinition(process.cwd()),
      createEditToolDefinition(process.cwd()),
      createWriteToolDefinition(process.cwd()),
      ...(selectsTool("ls") ? [createLsToolDefinition(process.cwd())] : []),
    ];
    for (const definition of definitions) {
      pi.registerTool(withTolerance(definition, process.cwd(), hostFs) as never);
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
        ?? (config.targetFailureLimit > 0 ? repeatedTargetFailure(messages, name, args, config.targetFailureLimit) : undefined);
      return reason ? { block: true, reason } : undefined;
    });
  }
}
