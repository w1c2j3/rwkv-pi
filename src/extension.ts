import { createBashToolDefinition, type ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { createRwkvProvider } from "./provider.js";

export default function rwkvExtension(pi: ExtensionAPI): void {
  pi.registerProvider(createRwkvProvider());
  // The harness bash tool does not expose PI_* session variables, so its system prompt has no
  // guideline about them. Interactive pi must present the same tool.
  if (process.env.RWKV_PI_PARITY?.trim() !== "0") pi.registerTool(createBashToolDefinition(process.cwd(), { exposeSessionEnvironment: false }) as never);
}
