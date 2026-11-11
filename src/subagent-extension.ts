import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { registerSubagentTools } from "./orchestrator-extension.js";

/** Opt-in RWKV delegation without enabling a strong-model provider or automatic takeover. */
export default function subagentExtension(pi: ExtensionAPI): Promise<void> {
  return registerSubagentTools(pi, "rwkv");
}
