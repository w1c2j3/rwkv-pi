import {
  createBashToolDefinition, createEditToolDefinition, createReadToolDefinition, createWriteToolDefinition, type ExtensionAPI,
} from "@earendil-works/pi-coding-agent";
import type { Message } from "@earendil-works/pi-ai";
import { readConfig } from "./config.js";
import { repeatedIdenticalCall, repeatedTargetFailure } from "./loop-guard.js";
import { createRwkvProvider } from "./provider.js";
import { hostFs, withTolerance } from "./tolerant-tools.js";

export default function rwkvExtension(pi: ExtensionAPI): void {
  const config = readConfig();
  pi.registerProvider(createRwkvProvider(config));
  // The harness bash tool does not expose PI_* session variables, so its system prompt has no
  // guideline about them. Interactive pi must present the same tool.
  if (process.env.RWKV_PI_PARITY?.trim() !== "0") pi.registerTool(createBashToolDefinition(process.cwd(), { exposeSessionEnvironment: false }) as never);
  // Same recall-tolerant execution as the harness. Only tools present in every tool set are
  // replaced: registering a tool the session did not select could add it to the prompt.
  if (config.toolTolerance) {
    for (const definition of [createReadToolDefinition(process.cwd()), createEditToolDefinition(process.cwd()), createWriteToolDefinition(process.cwd())]) {
      pi.registerTool(withTolerance(definition, process.cwd(), hostFs) as never);
    }
  }
  // Same guards as the harness (src/harness.ts beforeToolCall): pi records a block as the call's
  // tool error result, so the model sees it and the session file keeps it.
  if (config.identicalCallLimit > 0 || config.targetFailureLimit > 0) {
    pi.on("tool_call", async (event, ctx) => {
      const messages = ctx.sessionManager.getBranch().flatMap((entry: any) => entry.type === "message" ? [entry.message as Message] : []);
      const own = messages.flatMap(m => m.role === "assistant" ? m.content : []).find((b: any) => b.type === "toolCall" && b.id === event.toolCallId) as any;
      const name = own?.name ?? event.toolName, args = own?.arguments ?? (event as any).input;
      const reason = (config.identicalCallLimit > 0 ? repeatedIdenticalCall(messages, name, args, config.identicalCallLimit) : undefined)
        ?? (config.targetFailureLimit > 0 ? repeatedTargetFailure(messages, name, args, config.targetFailureLimit) : undefined);
      return reason ? { block: true, reason } : undefined;
    });
  }
}
