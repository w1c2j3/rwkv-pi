#!/usr/bin/env node
/**
 * One-screen readiness check for a turnkey install: toolchain, pinned Pi source and patches, the Lightning-native
 * tokenizer, configuration, data root, sandbox, and (unless --offline) the live engine.
 *
 *   npm run doctor [-- --offline] [-- --json]
 *
 * RWKV token counts come only from the Lightning-native tokenizer. When the offline build is present, the doctor
 * also checks that it and the engine's /v1/tokens/count agree on the same probe text.
 */
import { execFileSync, spawnSync } from "node:child_process";
import { accessSync, constants, existsSync, readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { loadEnvFile } from "node:process";
import { fileURLToPath, pathToFileURL } from "node:url";
import { parseArgs } from "node:util";
import { readConfig, requireDataRoot, type RwkvConfig } from "./config.js";
import { countPrompt } from "./lightning.js";
import { sha256 } from "./provenance.js";
import { applyProfile, configuredByEnv, readProfile } from "./profile.js";

export type Status = "ok" | "warn" | "fail";
export interface Check { name: string; status: Status; detail: string }

export const PACKAGE_ROOT = fileURLToPath(new URL("../", import.meta.url));
const MIN_NODE = [22, 19] as const;
const PROBE = "System: doctor probe\n\nUser: 你好，RWKV 😀 <tool_call>{\"name\":\"read\"}</tool_call>\n\nAssistant: <think";

export function nativeTokenizerPaths(root = PACKAGE_ROOT) {
  return {
    binary: join(root, ".upstream/lightning-tokenizer/tokenize-native"),
    vocab: join(root, ".upstream/rwkv-lightning/assets/rwkv_vocab_v20230424.txt"),
  };
}

/** Token ids from the pinned Lightning tokenizer source, built locally by `npm run lightning:tokenizer`. */
export function nativeTokenize(text: string, root = PACKAGE_ROOT): number[] {
  const { binary, vocab } = nativeTokenizerPaths(root);
  const out = execFileSync(binary, [vocab], { input: Buffer.from(text).toString("hex") + "\n", encoding: "utf8" });
  return JSON.parse(out.trim());
}

const json = (path: string) => JSON.parse(readFileSync(path, "utf8"));

function nodeCheck(): Check {
  const [major = 0, minor = 0] = process.versions.node.split(".").map(Number);
  const ok = major > MIN_NODE[0] || (major === MIN_NODE[0] && minor >= MIN_NODE[1]);
  return { name: "node", status: ok ? "ok" : "fail", detail: `${process.versions.node} (need >= ${MIN_NODE.join(".")})` };
}

function piSourceCheck(root: string): Check {
  const lock = json(join(root, "pi-source.lock.json"));
  const checkout = join(root, ".upstream/pi");
  if (!existsSync(checkout)) return { name: "pi source", status: "fail", detail: "missing; run npm run pi:setup" };
  const head = spawnSync("git", ["-C", checkout, "rev-parse", "HEAD"], { encoding: "utf8" }).stdout.trim();
  if (head !== lock.commit) return { name: "pi source", status: "fail", detail: `HEAD ${head.slice(0, 12)} differs from lock ${lock.commit.slice(0, 12)}` };
  const dir = join(root, "patches/pi");
  const patches = existsSync(dir) ? readdirSync(dir).filter(name => name.endsWith(".patch")).sort() : [];
  const missing = patches.filter(name => spawnSync("git", ["-C", checkout, "apply", "--reverse", "--check", join(dir, name)]).status !== 0);
  if (missing.length) return { name: "pi source", status: "fail", detail: `patches not applied: ${missing.join(", ")}; run npm run pi:setup` };
  return { name: "pi source", status: "ok", detail: `v${lock.version} ${lock.commit.slice(0, 12)}, ${patches.length} patch(es) applied` };
}

function tokenizerCheck(root: string): Check {
  const lock = json(join(root, "lightning-source.lock.json"));
  const { binary, vocab } = nativeTokenizerPaths(root);
  if (!existsSync(vocab)) return { name: "native tokenizer", status: "fail", detail: "pinned Lightning source missing; run npm run lightning:setup" };
  if (sha256(readFileSync(vocab)) !== lock.vocabSha256) return { name: "native tokenizer", status: "fail", detail: "vocabulary differs from lightning-source.lock.json" };
  if (!existsSync(binary)) return { name: "native tokenizer", status: "fail", detail: "not built; run npm run lightning:tokenizer" };
  const proof = existsSync(binary + ".json") ? json(binary + ".json") : {};
  if (proof.binarySha256 !== sha256(readFileSync(binary)) || proof.engineRevision !== lock.revision || proof.sourceSha256 !== lock.tokenizerSourceSha256) {
    return { name: "native tokenizer", status: "fail", detail: "build proof does not match the lock; rebuild with npm run lightning:tokenizer" };
  }
  try { nativeTokenize(PROBE, root); } catch (error) { return { name: "native tokenizer", status: "fail", detail: `probe failed: ${(error as Error).message}` }; }
  return { name: "native tokenizer", status: "ok", detail: `Lightning ${lock.revision.slice(0, 12)}, vocab ${lock.vocabSha256.slice(0, 12)}` };
}

function sandboxCheck(): Check {
  const result = spawnSync("bwrap", ["--version"], { encoding: "utf8" });
  return result.status === 0
    ? { name: "sandbox", status: "ok", detail: result.stdout.trim() }
    : { name: "sandbox", status: "warn", detail: "bubblewrap not found; harness runs and sandboxed subagents need it (interactive Pi does not)" };
}

function dataRootCheck(env: NodeJS.ProcessEnv): Check {
  try {
    const root = requireDataRoot(env);
    accessSync(root, constants.W_OK);
    return { name: "data root", status: "ok", detail: root };
  } catch (error) { return { name: "data root", status: "fail", detail: (error as Error).message }; }
}

async function engineChecks(config: RwkvConfig, root: string, fetcher: typeof fetch): Promise<Check[]> {
  const checks: Check[] = [];
  const headers = { Authorization: `Bearer ${config.apiKey}` };
  try {
    const response = await fetcher(config.baseUrl + "/models", { headers, signal: AbortSignal.timeout(15000) });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const ids = ((await response.json() as { data?: { id?: string }[] }).data ?? []).map(model => model.id);
    checks.push(ids.length === 0 || ids.includes(config.modelId)
      ? { name: "engine", status: "ok", detail: `${config.baseUrl} serves ${ids.join(", ") || "(unnamed)"}` }
      : { name: "engine", status: "warn", detail: `${config.baseUrl} serves ${ids.join(", ")}; RWKV_MODEL is ${config.modelId}` });
  } catch (error) {
    checks.push({ name: "engine", status: "fail", detail: `${config.baseUrl}/models: ${(error as Error).message}` });
    return checks;
  }
  try {
    const served = await countPrompt(PROBE, config, fetcher);
    let detail = `/v1/tokens/count = ${served}`, status: Status = "ok";
    if (existsSync(nativeTokenizerPaths(root).binary)) {
      const local = nativeTokenize(PROBE, root).length;
      if (local !== served) { status = "fail"; detail += `, offline native build = ${local}: packing and serving disagree`; }
      else detail += ", matches the offline native build";
    }
    checks.push({ name: "tokenizer parity", status, detail });
  } catch (error) { checks.push({ name: "tokenizer parity", status: "fail", detail: (error as Error).message }); }
  return checks;
}

export async function runDoctor(options: { env?: NodeJS.ProcessEnv; offline?: boolean; root?: string; fetch?: typeof fetch } = {}): Promise<Check[]> {
  const env = options.env ?? process.env, root = options.root ?? PACKAGE_ROOT;
  const checks: Check[] = [nodeCheck(), piSourceCheck(root), tokenizerCheck(root)];
  let config: RwkvConfig | undefined;
  try {
    config = readConfig(env);
    checks.push(config.backend === "lightning"
      ? { name: "config", status: "ok", detail: `${config.modelId} @ ${config.baseUrl}, ${config.toolFormat}, ctx ${config.contextWindow}, out ${config.maxTokens}` }
      : { name: "config", status: "fail", detail: "RWKV_BACKEND must be lightning for RWKV (Lightning-native tokenizer only)" });
  } catch (error) { checks.push({ name: "config", status: "fail", detail: (error as Error).message }); }
  checks.push(dataRootCheck(env), sandboxCheck());
  if (!options.offline && config?.backend === "lightning") checks.push(...await engineChecks(config, root, options.fetch ?? fetch));
  return checks;
}

async function main() {
  const { values } = parseArgs({ options: { offline: { type: "boolean", default: false }, json: { type: "boolean", default: false } } });
  if (existsSync(".env")) loadEnvFile(".env");
  const profile = configuredByEnv() ? undefined : readProfile();
  if (profile) applyProfile(profile);
  const checks = await runDoctor({ offline: values.offline });
  if (values.json) console.log(JSON.stringify(checks, null, 2));
  else for (const check of checks) console.log(`${{ ok: "  ok ", warn: " warn", fail: " FAIL" }[check.status]}  ${check.name.padEnd(17)} ${check.detail}`);
  process.exitCode = checks.some(check => check.status === "fail") ? 1 : 0;
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) await main();
