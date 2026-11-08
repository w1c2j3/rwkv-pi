/**
 * pi built-in tools rwkv-pi runs identically in the CLI and the isolated harness, in the one
 * order both use. pi lists tools in the system prompt and request in allowlist order, and all
 * existing training data uses read, write, edit, bash; the search tools follow.
 */
export const PI_TOOL_NAMES = ["read", "write", "edit", "bash", "grep", "find", "ls"] as const;
export type PiToolName = (typeof PI_TOOL_NAMES)[number];

/** The selection of the frozen baseline and all existing data. */
export const DEFAULT_PI_TOOLS: readonly PiToolName[] = ["read", "write", "edit", "bash"];

/**
 * Parse a comma-separated tool list into pi's canonical order. The system prompt and tool
 * schemas depend on the selection and its order, so every path (CLI, harness, converters)
 * must normalise through this function.
 */
export function parseToolList(value: string | readonly string[]): PiToolName[] {
  const names = (typeof value === "string" ? value.split(",") : [...value]).map((name) => name.trim()).filter(Boolean);
  if (!names.length) throw new Error("Tool list is empty.");
  const unknown = names.filter((name) => !(PI_TOOL_NAMES as readonly string[]).includes(name));
  if (unknown.length) throw new Error(`Unsupported tools: ${unknown.join(", ")}. Supported: ${PI_TOOL_NAMES.join(", ")}.`);
  if (new Set(names).size !== names.length) throw new Error("Tool list repeats a tool.");
  return PI_TOOL_NAMES.filter((name) => names.includes(name));
}
