/** Optional capabilities of the RWKV CLI. Credentials alone never enable a plugin. */
export const PLUGIN_NAMES = ["strong-model", "subagents"] as const;
export type PluginName = typeof PLUGIN_NAMES[number];

function pluginList(value: string): Set<PluginName> {
  if (!value.trim() || value.trim() === "none") return new Set();
  const names = value.split(",").map(name => name.trim());
  if (names.some(name => !PLUGIN_NAMES.includes(name as PluginName))) {
    throw new Error("RWKV plugins must be none, strong-model, subagents, or strong-model,subagents.");
  }
  return new Set(names as PluginName[]);
}

/** Environment defaults, then explicit CLI switches in order; preserve pi's other arguments. */
export function resolvePlugins(args: string[], env: NodeJS.ProcessEnv = process.env): { enabled: PluginName[]; args: string[] } {
  let enabled = pluginList(env.RWKV_PLUGINS ?? "none");
  const rest: string[] = [];
  for (let i = 0; i < args.length; i++) {
    const arg = args[i]!;
    if (arg === "--") { rest.push(...args.slice(i)); break; }
    if (arg === "--plugins" || arg.startsWith("--plugins=")) {
      const value = arg === "--plugins" ? args[++i] : arg.slice("--plugins=".length);
      if (value === undefined || !value.trim() || value.startsWith("--")) throw new Error("--plugins requires none, strong-model, subagents, or strong-model,subagents.");
      enabled = pluginList(value);
    } else if (arg === "--collab") enabled.add("strong-model");
    else if (arg === "--no-collab") enabled.delete("strong-model");
    else if (arg === "--subagents") enabled.add("subagents");
    else if (arg === "--no-subagents") enabled.delete("subagents");
    else rest.push(arg);
  }
  return { enabled: PLUGIN_NAMES.filter(name => enabled.has(name)), args: rest };
}
