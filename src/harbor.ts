import { spawn } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdir, mkdtemp, readdir, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, dirname, join, posix, resolve } from "node:path";
import { InMemoryCredentialStore, InMemoryModelsStore, type AssistantMessage, type Message } from "@earendil-works/pi-ai";
import {
  createAgentSession, createBashToolDefinition, createEditToolDefinition, createExtensionRuntime, createLsToolDefinition,
  createReadToolDefinition, createWriteToolDefinition, ModelRuntime, SessionManager, SettingsManager,
  type ResourceLoader, type ToolDefinition,
} from "@earendil-works/pi-coding-agent";
import { readConfig, type RwkvConfig } from "./config.js";
import { foldEpochRequest } from "./epoch.js";
import { repeatedIdenticalCall, repeatedTargetFailure, repeatedWorklistSpin } from "./loop-guard.js";
import { promptVersion } from "./prompt.js";
import { runtimeProvenance } from "./provenance.js";
import { createRwkvProvider } from "./provider.js";
import { object } from "./tasks.js";
import { budgetRwkvRequest, type TokenBudget } from "./token-budget.js";
import { MAX_LISTED_FILES, SKIPPED_DIRECTORIES, TOLERANT_TOOL_NAMES, withTolerance, type WorkspaceFs } from "./tolerant-tools.js";
import { DEFAULT_PI_TOOLS, parseToolList, PI_TOOL_NAMES, type PiToolName } from "./tools.js";
import { createWebTools } from "./web-tools.js";
import { createWorklistTools, foldWorklistRequest, isWorklistTool, PiWorklistState, WORKLIST_TOOL_NAMES } from "./worklist.js";

const byCodePoint = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0);

// ---------------------------------------------------------------------------------------------
// Minimal zero-dependency TOML subset parser for Harbor task.toml (v1.0 - v1.4)

function stripTomlComment(line: string): string {
  let quote: '"' | "'" | undefined;
  let escaped = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i]!;
    if (escaped) { escaped = false; continue; }
    if (quote === '"' && ch === "\\") { escaped = true; continue; }
    if (!quote && (ch === '"' || ch === "'")) { quote = ch; continue; }
    if (quote && ch === quote) { quote = undefined; continue; }
    if (!quote && ch === "#") return line.slice(0, i);
  }
  return line;
}

function splitTomlArrayItems(inner: string): string[] {
  const items: string[] = [];
  let cur = "", quote: '"' | "'" | undefined, escaped = false, depth = 0;
  for (let i = 0; i < inner.length; i++) {
    const ch = inner[i]!;
    if (escaped) { cur += ch; escaped = false; continue; }
    if (quote === '"' && ch === "\\") { cur += ch; escaped = true; continue; }
    if (!quote && (ch === '"' || ch === "'")) { quote = ch; cur += ch; continue; }
    if (quote && ch === quote) { quote = undefined; cur += ch; continue; }
    if (!quote && ch === "[") { depth++; cur += ch; continue; }
    if (!quote && ch === "]") { depth--; cur += ch; continue; }
    if (!quote && depth === 0 && ch === ",") {
      if (cur.trim()) items.push(cur.trim());
      cur = "";
      continue;
    }
    cur += ch;
  }
  if (cur.trim()) items.push(cur.trim());
  return items;
}

function parseTomlValue(raw: string): unknown {
  const text = raw.trim();
  if (text === "true") return true;
  if (text === "false") return false;
  if (text.startsWith('"') && text.endsWith('"') && text.length >= 2) {
    return JSON.parse(text);
  }
  if (text.startsWith("'") && text.endsWith("'") && text.length >= 2) {
    return text.slice(1, -1);
  }
  if (text.startsWith("[") && text.endsWith("]")) {
    return splitTomlArrayItems(text.slice(1, -1)).map(parseTomlValue);
  }
  if (/^[+-]?\d[\d_]*(?:\.[\d_]+)?(?:[eE][+-]?\d+)?$/.test(text)) {
    return Number(text.replaceAll("_", ""));
  }
  return text;
}

function parseTomlKeyPath(raw: string): string[] {
  const parts: string[] = [];
  let cur = "", quote: '"' | "'" | undefined;
  for (let i = 0; i < raw.length; i++) {
    const ch = raw[i]!;
    if (!quote && (ch === '"' || ch === "'")) { quote = ch; continue; }
    if (quote && ch === quote) { quote = undefined; continue; }
    if (!quote && ch === ".") {
      if (cur.trim()) parts.push(cur.trim());
      cur = "";
      continue;
    }
    cur += ch;
  }
  if (cur.trim()) parts.push(cur.trim());
  return parts;
}

function ensureTable(root: Record<string, unknown>, path: readonly string[], isArrayHeader: boolean): Record<string, unknown> {
  let cursor: Record<string, unknown> = root;
  for (let i = 0; i < path.length; i++) {
    const key = path[i]!;
    const last = i === path.length - 1;
    if (last && isArrayHeader) {
      const existing = cursor[key];
      const list = Array.isArray(existing) ? existing : [];
      if (!Array.isArray(existing)) cursor[key] = list;
      const entry: Record<string, unknown> = {};
      list.push(entry);
      return entry;
    }
    const existing = cursor[key];
    if (Array.isArray(existing)) {
      const tail = existing.at(-1);
      if (!object(tail)) throw new Error(`Invalid TOML table array target: ${path.join(".")}`);
      cursor = tail;
    } else if (object(existing)) {
      cursor = existing;
    } else {
      const next: Record<string, unknown> = {};
      cursor[key] = next;
      cursor = next;
    }
  }
  return cursor;
}

export function parseHarborToml(raw: string): Record<string, unknown> {
  const root: Record<string, unknown> = {};
  let current: Record<string, unknown> = root;
  const lines = raw.replaceAll("\r\n", "\n").split("\n");
  for (let idx = 0; idx < lines.length; idx++) {
    const stripped = stripTomlComment(lines[idx]!).trim();
    if (!stripped) continue;
    if (stripped.startsWith("[[") && stripped.endsWith("]]")) {
      current = ensureTable(root, parseTomlKeyPath(stripped.slice(2, -2)), true);
      continue;
    }
    if (stripped.startsWith("[") && stripped.endsWith("]")) {
      current = ensureTable(root, parseTomlKeyPath(stripped.slice(1, -1)), false);
      continue;
    }
    const eq = stripped.indexOf("=");
    if (eq < 1) continue;
    const keyPath = parseTomlKeyPath(stripped.slice(0, eq));
    let rhs = stripped.slice(eq + 1).trim();
    // Multi-line triple-quoted string or multi-line array continuation.
    if (rhs.startsWith('"""') || rhs.startsWith("'''")) {
      const delim = rhs.slice(0, 3);
      let body = rhs.slice(3);
      if (body.endsWith(delim) && body.length >= 3) {
        body = body.slice(0, -3);
      } else {
        const chunks = [body];
        while (++idx < lines.length) {
          const line = lines[idx]!;
          const endAt = line.indexOf(delim);
          if (endAt >= 0) { chunks.push(line.slice(0, endAt)); break; }
          chunks.push(line);
        }
        body = chunks.join("\n").replace(/^\n/, "");
      }
      const target = keyPath.length > 1 ? ensureTable(current, keyPath.slice(0, -1), false) : current;
      target[keyPath.at(-1)!] = body;
      continue;
    }
    if (rhs.startsWith("[") && !rhs.endsWith("]")) {
      const chunks = [rhs];
      while (++idx < lines.length) {
        const next = stripTomlComment(lines[idx]!).trim();
        if (next) chunks.push(next);
        if (next.endsWith("]")) break;
      }
      rhs = chunks.join(" ");
    }
    const target = keyPath.length > 1 ? ensureTable(current, keyPath.slice(0, -1), false) : current;
    target[keyPath.at(-1)!] = parseTomlValue(rhs);
  }
  return root;
}

// ---------------------------------------------------------------------------------------------
// Harbor Task Specification & Discovery

export interface HarborStepSpec {
  name: string;
  instruction: string;
  setupScriptPath?: string;
  testsDir?: string;
  agentTimeoutSec?: number;
  verifierTimeoutSec?: number;
}

export interface HarborTaskSpec {
  id: string;
  taskDir: string;
  schemaVersion: string;
  metadata: Record<string, unknown>;
  steps: HarborStepSpec[];
  rewardStrategy: "final" | "mean";
  agentTimeoutSec: number;
  verifierTimeoutSec: number;
  environment: {
    dockerImage?: string;
    dockerfilePath?: string;
    contextDir: string;
    cpus: number;
    memoryMb: number;
    allowInternet: boolean;
    env: Record<string, string>;
    buildTimeoutSec: number;
  };
}

function parseMemoryMb(value: unknown, fallback = 4096): number {
  if (typeof value === "number" && Number.isFinite(value) && value > 0) return Math.round(value);
  if (typeof value === "string") {
    const match = /^\s*(\d+(?:\.\d+)?)\s*([kmgKMG]?)[bB]?\s*$/.exec(value);
    if (match) {
      const num = Number(match[1]);
      const unit = (match[2] ?? "m").toLowerCase();
      const mult = unit === "g" ? 1024 : unit === "k" ? 1 / 1024 : 1;
      return Math.max(128, Math.round(num * mult));
    }
  }
  return fallback;
}

function positiveNum(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : fallback;
}

function parseTurnsJson(raw: string): string[] {
  const parsed = JSON.parse(raw) as unknown;
  const list = Array.isArray(parsed) ? parsed : object(parsed) && Array.isArray(parsed.turns) ? parsed.turns : undefined;
  if (!list || !list.length) throw new Error("turns.json must contain a non-empty array of turns.");
  return list.map((item, idx) => {
    if (typeof item === "string" && item.trim()) return item.trim();
    if (object(item)) {
      const text = item.prompt ?? item.instruction ?? item.content ?? item.user;
      if (typeof text === "string" && text.trim()) return text.trim();
    }
    throw new Error(`Invalid turn at index ${idx} in turns.json.`);
  });
}

export async function loadHarborTask(taskDir: string, idOverride?: string): Promise<HarborTaskSpec> {
  const dir = resolve(taskDir);
  const tomlPath = join(dir, "task.toml");
  const toml = existsSync(tomlPath) ? parseHarborToml(await readFile(tomlPath, "utf8")) : {};
  const id = idOverride ?? basename(dir);
  const schemaVersion = typeof toml.schema_version === "string" ? toml.schema_version : typeof toml.version === "string" ? toml.version : "1.0";
  const metadata = object(toml.metadata) ? toml.metadata : {};
  const agentTbl = object(toml.agent) ? toml.agent : {};
  const verifierTbl = object(toml.verifier) ? toml.verifier : {};
  const envTbl = object(toml.environment) ? toml.environment : {};
  const agentTimeoutSec = positiveNum(agentTbl.timeout_sec, 900);
  const verifierTimeoutSec = positiveNum(verifierTbl.timeout_sec, 300);
  const rewardStrategy = toml.multi_step_reward_strategy === "mean" ? "mean" : "final";

  const topTestsDir = existsSync(join(dir, "tests", "test.sh")) ? join(dir, "tests") : undefined;
  const stepsRoot = join(dir, "steps");
  const turnsJsonPath = join(dir, "environment", "turns.json");
  const steps: HarborStepSpec[] = [];

  if (existsSync(stepsRoot) && (await stat(stepsRoot)).isDirectory()) {
    const declared = Array.isArray(toml.steps) ? toml.steps.filter(object) : [];
    const stepNames = declared.length > 0
      ? declared.map((s, i) => (typeof s.name === "string" && s.name ? s.name : `step-${i + 1}`))
      : (await readdir(stepsRoot)).filter((n) => !n.startsWith(".")).sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
    for (let i = 0; i < stepNames.length; i++) {
      const name = stepNames[i]!;
      const stepDir = join(stepsRoot, name);
      const instruction = (await readFile(join(stepDir, "instruction.md"), "utf8")).trim();
      const setupCandidate = join(stepDir, "workdir", "setup.sh");
      const stepTestsCandidate = join(stepDir, "tests", "test.sh");
      const stepCfg = declared[i];
      const stepAgent = stepCfg && object(stepCfg.agent) ? stepCfg.agent : undefined;
      const stepVerifier = stepCfg && object(stepCfg.verifier) ? stepCfg.verifier : undefined;
      steps.push({
        name,
        instruction,
        ...(existsSync(setupCandidate) ? { setupScriptPath: setupCandidate } : {}),
        ...(existsSync(stepTestsCandidate) ? { testsDir: join(stepDir, "tests") } : {}),
        ...(stepAgent && typeof stepAgent.timeout_sec === "number" ? { agentTimeoutSec: stepAgent.timeout_sec } : {}),
        ...(stepVerifier && typeof stepVerifier.timeout_sec === "number" ? { verifierTimeoutSec: stepVerifier.timeout_sec } : {}),
      });
    }
  } else if (existsSync(turnsJsonPath)) {
    const turns = parseTurnsJson(await readFile(turnsJsonPath, "utf8"));
    for (let i = 0; i < turns.length; i++) {
      steps.push({
        name: `turn-${i + 1}`,
        instruction: turns[i]!,
        ...(i === turns.length - 1 && topTestsDir ? { testsDir: topTestsDir } : {}),
      });
    }
  } else {
    const instructionPath = join(dir, "instruction.md");
    if (!existsSync(instructionPath)) throw new Error(`Harbor task ${id} has neither instruction.md, steps/, nor environment/turns.json.`);
    const instruction = (await readFile(instructionPath, "utf8")).trim();
    steps.push({
      name: "main",
      instruction,
      ...(topTestsDir ? { testsDir: topTestsDir } : {}),
    });
  }

  if (!steps.length) throw new Error(`Harbor task ${id} contains zero executable steps.`);

  const contextDir = join(dir, "environment");
  const dockerfileCandidate = join(contextDir, "Dockerfile");
  const dockerfilePath = existsSync(dockerfileCandidate) ? dockerfileCandidate : undefined;
  const dockerImage = typeof envTbl.docker_image === "string" && envTbl.docker_image.trim()
    ? envTbl.docker_image.trim()
    : typeof envTbl.image === "string" && envTbl.image.trim()
      ? envTbl.image.trim()
      : undefined;
  if (!dockerfilePath && !dockerImage) {
    throw new Error(`Harbor task ${id} defines neither environment/Dockerfile nor [environment].docker_image.`);
  }

  const envVars: Record<string, string> = {};
  if (object(envTbl.env)) {
    for (const [k, v] of Object.entries(envTbl.env)) {
      if (v !== undefined && v !== null) envVars[k] = String(v);
    }
  }

  return {
    id,
    taskDir: dir,
    schemaVersion,
    metadata,
    steps,
    rewardStrategy,
    agentTimeoutSec,
    verifierTimeoutSec,
    environment: {
      ...(dockerImage ? { dockerImage } : {}),
      ...(dockerfilePath ? { dockerfilePath } : {}),
      contextDir,
      cpus: positiveNum(envTbl.cpus, 2),
      memoryMb: parseMemoryMb(envTbl.memory_mb ?? envTbl.memory, 4096),
      allowInternet: envTbl.allow_internet !== false,
      env: envVars,
      buildTimeoutSec: positiveNum(envTbl.build_timeout_sec, 900),
    },
  };
}

export async function discoverHarborTasks(tasksRoot: string): Promise<HarborTaskSpec[]> {
  const root = resolve(tasksRoot);
  if (existsSync(join(root, "task.toml")) || existsSync(join(root, "instruction.md"))) {
    return [await loadHarborTask(root)];
  }
  const found: { id: string; dir: string }[] = [];
  const walk = async (dir: string, prefix: string, depth: number): Promise<void> => {
    if (depth > 3) return;
    const entries = (await readdir(dir, { withFileTypes: true })).sort((a, b) => byCodePoint(a.name, b.name));
    for (const entry of entries) {
      if (!entry.isDirectory() || entry.name.startsWith(".")) continue;
      const full = join(dir, entry.name);
      const rel = prefix ? `${prefix}__${entry.name}` : entry.name;
      if (existsSync(join(full, "task.toml")) || existsSync(join(full, "instruction.md"))) {
        found.push({ id: rel, dir: full });
      } else {
        await walk(full, rel, depth + 1);
      }
    }
  };
  await walk(root, "", 0);
  const specs: HarborTaskSpec[] = [];
  for (const item of found) specs.push(await loadHarborTask(item.dir, item.id));
  return specs;
}

// ---------------------------------------------------------------------------------------------
// Deduplicated Image Building ("Build-Once, Run-Only")

export interface SplitDockerfileResult {
  baseDockerfile: string;
  copiesWorkspaceOnly: boolean;
}

function logicalDockerfileInstructions(normalized: string): { rawLines: string[]; joined: string }[] {
  const out: { rawLines: string[]; joined: string }[] = [];
  const lines = normalized.split("\n");
  let buf: string[] = [];
  let rawBuf: string[] = [];
  for (const line of lines) {
    const trimmed = line.trim();
    if (!buf.length && (!trimmed || trimmed.startsWith("#"))) continue;
    rawBuf.push(line);
    if (trimmed.startsWith("#")) continue;
    if (trimmed.endsWith("\\")) {
      buf.push(trimmed.slice(0, -1).trimEnd());
    } else {
      buf.push(trimmed);
      out.push({ rawLines: rawBuf, joined: buf.join(" ") });
      buf = [];
      rawBuf = [];
    }
  }
  if (buf.length) out.push({ rawLines: rawBuf, joined: buf.join(" ") });
  return out;
}

/**
 * Identifies Dockerfiles that either have zero COPY/ADD instructions or only a trailing
 * `COPY workspace/ /app/`, allowing hundreds of tasks to share a single cached base image
 * built in an empty context while seeding `/app` at container startup.
 */
export function splitReusableDockerfile(dockerfileText: string): SplitDockerfileResult | undefined {
  const normalized = dockerfileText.replaceAll("\r\n", "\n");
  const instructions = logicalDockerfileInstructions(normalized);
  const copyIndices: number[] = [];
  for (let i = 0; i < instructions.length; i++) {
    if (/^(?:COPY|ADD)\s+/i.test(instructions[i]!.joined)) copyIndices.push(i);
  }
  if (copyIndices.length === 0) {
    return { baseDockerfile: normalized.trimEnd() + "\n", copiesWorkspaceOnly: false };
  }
  if (copyIndices.length === 1 && copyIndices[0] === instructions.length - 1) {
    const last = instructions[copyIndices[0]!]!;
    if (/^COPY(?:\s+--chown=\S+)?\s+workspace\/?\s+\/app\/?\s*$/i.test(last.joined)) {
      const kept = instructions.slice(0, -1).flatMap((inst) => inst.rawLines).join("\n").trimEnd() + "\n";
      return { baseDockerfile: kept, copiesWorkspaceOnly: true };
    }
  }
  return undefined;
}

async function hashDirectoryManifest(dir: string): Promise<string> {
  const hash = createHash("sha256");
  const walk = async (current: string, relPrefix: string): Promise<void> => {
    let entries;
    try { entries = await readdir(current, { withFileTypes: true }); } catch { return; }
    entries.sort((a, b) => byCodePoint(a.name, b.name));
    for (const entry of entries) {
      if (entry.name === "turns.json" && relPrefix === "") continue;
      const full = join(current, entry.name);
      const rel = relPrefix ? `${relPrefix}/${entry.name}` : entry.name;
      if (entry.isDirectory()) {
        hash.update(`D:${rel}\n`);
        await walk(full, rel);
      } else if (entry.isFile()) {
        const content = await readFile(full);
        const fileSha = createHash("sha256").update(content).digest("hex");
        hash.update(`F:${rel}:${fileSha}\n`);
      }
    }
  };
  await walk(dir, "");
  return hash.digest("hex");
}

export interface PreparedTaskImage {
  imageTag: string;
  seedWorkspaceDir?: string;
  isContextSpecific: boolean;
}

function runProcess(
  command: string,
  args: readonly string[],
  options: { timeoutMs?: number; input?: string | Buffer; signal?: AbortSignal; env?: NodeJS.ProcessEnv; onData?: (chunk: Buffer) => void } = {},
): Promise<{ exitCode: number; stdout: Buffer; stderr: Buffer }> {
  if (options.signal?.aborted) return Promise.reject(new Error(`${command} aborted.`));
  return new Promise((accept, reject) => {
    const child = spawn(command, [...args], {
      stdio: "pipe",
      env: options.env ?? process.env,
    });
    const stdout: Buffer[] = [], stderr: Buffer[] = [];
    let total = 0, failed: Error | undefined;
    const stop = (reason: string) => {
      failed ??= new Error(reason);
      try { child.kill("SIGKILL"); } catch { /* exited */ }
    };
    const onAbort = () => stop(`${command} aborted.`);
    const timer = options.timeoutMs ? setTimeout(() => stop(`${command} timed out after ${options.timeoutMs}ms.`), options.timeoutMs) : undefined;
    options.signal?.addEventListener("abort", onAbort, { once: true });
    const collect = (bufs: Buffer[], chunk: Buffer) => {
      total += chunk.length;
      if (total > 32 * 1024 * 1024) { stop(`${command} output exceeded 32 MiB.`); return; }
      bufs.push(chunk);
      options.onData?.(chunk);
    };
    child.stdout.on("data", (c: Buffer) => collect(stdout, c));
    child.stderr.on("data", (c: Buffer) => collect(stderr, c));
    child.once("error", (err) => { failed = err; });
    child.stdin.on("error", () => {});
    child.stdin.end(options.input);
    child.once("close", (code) => {
      if (timer) clearTimeout(timer);
      options.signal?.removeEventListener("abort", onAbort);
      if (failed) reject(failed);
      else accept({ exitCode: code ?? 137, stdout: Buffer.concat(stdout), stderr: Buffer.concat(stderr) });
    });
  });
}

export class HarborImageCache {
  private readonly inFlight = new Map<string, Promise<string>>();
  private readonly remainingRefs = new Map<string, number>();

  registerPlannedTasks(tasks: readonly { task: HarborTaskSpec; count: number }[]): void {
    for (const { task, count } of tasks) {
      this.remainingRefs.set(task.id, (this.remainingRefs.get(task.id) ?? 0) + count);
    }
  }

  async prepare(task: HarborTaskSpec): Promise<PreparedTaskImage> {
    if (task.environment.dockerImage && !task.environment.dockerfilePath) {
      const imageTag = task.environment.dockerImage;
      await this.once(imageTag, async () => {
        const inspect = await runProcess("docker", ["image", "inspect", imageTag], { timeoutMs: 30_000 });
        if (inspect.exitCode !== 0) {
          const pull = await runProcess("docker", ["pull", imageTag], { timeoutMs: task.environment.buildTimeoutSec * 1000 });
          if (pull.exitCode !== 0) throw new Error(`Failed to pull ${imageTag}: ${pull.stderr.toString().slice(-800)}`);
        }
        return imageTag;
      });
      return { imageTag, isContextSpecific: false };
    }

    const dockerfileRaw = await readFile(task.environment.dockerfilePath!, "utf8");
    const split = splitReusableDockerfile(dockerfileRaw);
    if (split) {
      const digest = createHash("sha256").update(split.baseDockerfile).digest("hex").slice(0, 12);
      const imageTag = `rwkv-pi-harbor-cache:${digest}`;
      await this.once(imageTag, async () => {
        const inspect = await runProcess("docker", ["image", "inspect", imageTag], { timeoutMs: 30_000 });
        if (inspect.exitCode === 0) return imageTag;
        const emptyCtx = await mkdtemp(join(tmpdir(), "rwkv-harbor-base-"));
        try {
          const dfPath = join(emptyCtx, "Dockerfile");
          await writeFile(dfPath, split.baseDockerfile, "utf8");
          const build = await runProcess(
            "docker",
            ["build", "--network", "host", "-t", imageTag, "-f", dfPath, emptyCtx],
            { timeoutMs: task.environment.buildTimeoutSec * 1000 },
          );
          if (build.exitCode !== 0) {
            throw new Error(`Docker base build failed (${imageTag}): ${(build.stdout.toString() + build.stderr.toString()).slice(-1200)}`);
          }
          return imageTag;
        } finally {
          await rm(emptyCtx, { recursive: true, force: true });
        }
      });
      const wsCandidate = join(task.environment.contextDir, "workspace");
      return {
        imageTag,
        ...(split.copiesWorkspaceOnly && existsSync(wsCandidate) ? { seedWorkspaceDir: wsCandidate } : {}),
        isContextSpecific: false,
      };
    }

    const ctxHash = await hashDirectoryManifest(task.environment.contextDir);
    const digest = createHash("sha256").update(dockerfileRaw).update("\0").update(ctxHash).digest("hex").slice(0, 12);
    const imageTag = `rwkv-pi-harbor-ctx:${digest}`;
    await this.once(imageTag, async () => {
      const inspect = await runProcess("docker", ["image", "inspect", imageTag], { timeoutMs: 30_000 });
      if (inspect.exitCode === 0) return imageTag;
      const build = await runProcess(
        "docker",
        ["build", "--network", "host", "-t", imageTag, "-f", task.environment.dockerfilePath!, task.environment.contextDir],
        { timeoutMs: task.environment.buildTimeoutSec * 1000 },
      );
      if (build.exitCode !== 0) {
        throw new Error(`Docker context build failed (${task.id}): ${(build.stdout.toString() + build.stderr.toString()).slice(-1200)}`);
      }
      return imageTag;
    });
    return { imageTag, isContextSpecific: true };
  }

  async release(task: HarborTaskSpec, prepared: PreparedTaskImage, cleanupContextImages = false): Promise<void> {
    const left = (this.remainingRefs.get(task.id) ?? 1) - 1;
    this.remainingRefs.set(task.id, Math.max(0, left));
    if (cleanupContextImages && prepared.isContextSpecific && left <= 0) {
      await runProcess("docker", ["image", "rm", "-f", prepared.imageTag], { timeoutMs: 30_000 }).catch(() => {});
      this.inFlight.delete(prepared.imageTag);
    }
  }

  private once(key: string, fn: () => Promise<string>): Promise<string> {
    const existing = this.inFlight.get(key);
    if (existing) return existing;
    const promise = fn().catch((err) => {
      this.inFlight.delete(key);
      throw err;
    });
    this.inFlight.set(key, promise);
    return promise;
  }
}

// ---------------------------------------------------------------------------------------------
// Endpoint Pool with Least-In-Flight Load Balancing & Sticky Session Affinity

export interface EndpointLease {
  endpoint: string;
  sessionId: string;
  release: () => void;
}

export class EndpointPool {
  private readonly endpoints: string[];
  private readonly active: number[];
  private rrCursor = 0;

  constructor(rawEndpoints: readonly string[], env: NodeJS.ProcessEnv = process.env) {
    const cleaned = rawEndpoints.map((e) => e.trim()).filter(Boolean);
    if (!cleaned.length) throw new Error("At least one RWKV endpoint URL is required.");
    this.endpoints = cleaned.map((url) => readConfig({ ...env, RWKV_BASE_URL: url }).baseUrl);
    this.active = this.endpoints.map(() => 0);
  }

  get size(): number {
    return this.endpoints.length;
  }

  acquire(trialId: string): EndpointLease {
    let bestIdx = this.rrCursor % this.endpoints.length;
    let bestLoad = this.active[bestIdx]!;
    for (let offset = 1; offset < this.endpoints.length; offset++) {
      const idx = (this.rrCursor + offset) % this.endpoints.length;
      const load = this.active[idx]!;
      if (load < bestLoad) {
        bestIdx = idx;
        bestLoad = load;
      }
    }
    this.rrCursor = (bestIdx + 1) % this.endpoints.length;
    this.active[bestIdx] = bestLoad + 1;
    let released = false;
    const safeId = trialId.replace(/[^A-Za-z0-9_.-]/g, "_").slice(0, 64);
    const sessionId = `harbor-${safeId}-${randomUUID().slice(0, 8)}`;
    return {
      endpoint: this.endpoints[bestIdx]!,
      sessionId,
      release: () => {
        if (released) return;
        released = true;
        this.active[bestIdx] = Math.max(0, (this.active[bestIdx] ?? 1) - 1);
      },
    };
  }
}

// ---------------------------------------------------------------------------------------------
// Persistent Docker Sandbox (1 Container per Trial, Pure POSIX Tools, Zero Host Node Mounts)

export interface HarborSandboxExecOptions {
  cwd?: string;
  env?: Record<string, string>;
  signal?: AbortSignal;
  timeoutMs?: number;
  onData?: (data: Buffer) => void;
  input?: string | Buffer;
}

export interface HarborSandbox {
  readonly cwd: string;
  readonly fs: WorkspaceFs;
  tools(names: readonly PiToolName[]): ToolDefinition[];
  exec(argv: readonly string[], options?: HarborSandboxExecOptions): Promise<{ exitCode: number; stdout: Buffer; stderr: Buffer }>;
  writeFile(path: string, content: string | Buffer): Promise<void>;
  readFile(path: string): Promise<Buffer>;
  uploadDir(hostSrcDir: string, containerDestDir: string): Promise<void>;
  downloadFile(containerPath: string): Promise<Buffer | undefined>;
  close(): Promise<void>;
}

const POSIX_READDIR_SH = `cd -- "$1" || exit 1
for f in * .[!.]* ..?*; do
  [ -e "$f" ] || [ -L "$f" ] || continue
  if [ -d "$f" ]; then printf '1%s\\0' "$f"; else printf '0%s\\0' "$f"; fi
done`;

const POSIX_LIST_FILES_SH = `cd -- "$1" 2>/dev/null || exit 0
find . \\( -name .git -o -name node_modules -o -name .venv -o -name venv -o -name __pycache__ -o -name .pytest_cache -o -name .mypy_cache \\) -prune -o -type f -print 2>/dev/null`;

const POSIX_KILL_EXEC_SH = `for p in /proc/[0-9]*; do
  [ -r "$p/environ" ] || continue
  if tr '\\0' '\\n' < "$p/environ" 2>/dev/null | grep -qxF "RWKV_EXEC_ID=$1"; then
    pid="\${p#/proc/}"
    [ "$pid" != "$$" ] && kill -9 "$pid" 2>/dev/null || true
  fi
done`;

export interface DockerSandboxStartOptions {
  imageTag: string;
  seedWorkspaceDir?: string;
  cpus: number;
  memoryMb: number;
  networkMode: "bridge" | "none" | "host";
  env: Record<string, string>;
  lifetimeSec: number;
}

export class PersistentDockerSandbox implements HarborSandbox {
  readonly cwd: string;
  private readonly statCache = new Map<string, boolean>();
  private closed = false;

  private constructor(
    private readonly containerName: string,
    cwd: string,
    private readonly hasBash: boolean,
  ) {
    this.cwd = cwd;
  }

  static async start(options: DockerSandboxStartOptions): Promise<PersistentDockerSandbox> {
    const inspect = await runProcess("docker", ["image", "inspect", "--format", "{{.Config.WorkingDir}}", options.imageTag], { timeoutMs: 15_000 });
    const configuredWorkdir = inspect.exitCode === 0 ? inspect.stdout.toString().trim() : "";
    const cwd = configuredWorkdir && configuredWorkdir !== "/" ? configuredWorkdir : "/app";
    const containerName = `rwkv-pi-harbor-${randomUUID()}`;
    const args = [
      "run", "-d", "--rm", "--init",
      "--name", containerName,
      "--network", options.networkMode,
      "--cpus", String(options.cpus),
      "--memory", `${options.memoryMb}m`,
      "--pids-limit", "1024",
    ];
    for (const [k, v] of Object.entries(options.env)) {
      args.push("-e", `${k}=${v}`);
    }
    args.push("--entrypoint", "/bin/sh", options.imageTag, "-c", `mkdir -p "${cwd}" /logs/agent /logs/verifier && exec sleep ${Math.max(60, Math.ceil(options.lifetimeSec))}`);
    const started = await runProcess("docker", args, { timeoutMs: 60_000 });
    if (started.exitCode !== 0) {
      await runProcess("docker", ["rm", "-f", containerName], { timeoutMs: 15_000 }).catch(() => {});
      throw new Error(`Failed to start Docker container (${options.imageTag}): ${started.stderr.toString().trim()}`);
    }
    try {
      const probe = await runProcess("docker", ["exec", containerName, "/bin/sh", "-c", "mkdir -p -- \"$1\" /logs/agent /logs/verifier && rm -rf /rae /tests && if [ -x /bin/bash ]; then printf bash; else printf sh; fi", "sh", cwd], { timeoutMs: 20_000 });
      if (probe.exitCode !== 0) {
        throw new Error(`Container probe failed: ${probe.stderr.toString().trim()}`);
      }
      const sandbox = new PersistentDockerSandbox(containerName, cwd, probe.stdout.toString().trim() === "bash");
      if (options.seedWorkspaceDir) {
        await sandbox.uploadDir(options.seedWorkspaceDir, cwd);
      }
      await sandbox.exec(["/bin/sh", "-c", "rm -rf /rae /tests \"$1/turns.json\"", "sh", cwd], { timeoutMs: 10_000 });
      return sandbox;
    } catch (error) {
      await runProcess("docker", ["rm", "-f", containerName], { timeoutMs: 15_000 }).catch(() => {});
      throw error;
    }
  }

  async exec(argv: readonly string[], options: HarborSandboxExecOptions = {}): Promise<{ exitCode: number; stdout: Buffer; stderr: Buffer }> {
    if (this.closed) throw new Error("Sandbox container is already closed.");
    const execId = randomUUID();
    const args = ["exec", "-i", "-w", options.cwd ?? this.cwd, "-e", `RWKV_EXEC_ID=${execId}`];
    if (options.env) {
      for (const [k, v] of Object.entries(options.env)) args.push("-e", `${k}=${v}`);
    }
    args.push(this.containerName, ...argv);
    try {
      return await runProcess("docker", args, {
        timeoutMs: options.timeoutMs ?? 60_000,
        signal: options.signal,
        input: options.input,
        onData: options.onData,
      });
    } catch (error) {
      if (!this.closed) {
        await runProcess("docker", ["exec", this.containerName, "/bin/sh", "-c", POSIX_KILL_EXEC_SH, "sh", execId], { timeoutMs: 5_000 }).catch(() => {});
      }
      throw error;
    }
  }

  private async checked(argv: readonly string[], input?: string | Buffer): Promise<Buffer> {
    const res = await this.exec(argv, { input, timeoutMs: 30_000 });
    if (res.exitCode !== 0) throw new Error(res.stderr.toString() || `Container command exited ${res.exitCode}.`);
    return res.stdout;
  }

  readFile = (path: string): Promise<Buffer> =>
    this.checked(["/bin/sh", "-c", 'cat -- "$1"', "sh", path]);

  writeFile = async (path: string, content: string | Buffer): Promise<void> => {
    await this.checked(["/bin/sh", "-c", 'mkdir -p -- "$(dirname -- "$1")" && cat > "$1"', "sh", path], content);
  };

  private access = async (path: string): Promise<void> => {
    await this.checked(["/bin/sh", "-c", 'test -e "$1"', "sh", path]);
  };

  private mkdir = async (path: string): Promise<void> => {
    await this.checked(["/bin/sh", "-c", 'mkdir -p -- "$1"', "sh", path]);
  };

  private exists = async (path: string): Promise<boolean> => {
    const res = await this.exec(["/bin/sh", "-c", 'test -e "$1"', "sh", path], { timeoutMs: 15_000 });
    return res.exitCode === 0;
  };

  private isDirectory = async (path: string): Promise<boolean> => {
    const res = await this.exec(["/bin/sh", "-c", 'test -d "$1"', "sh", path], { timeoutMs: 15_000 });
    return res.exitCode === 0;
  };

  private readdir = async (path: string): Promise<string[]> => {
    const out = await this.checked(["/bin/sh", "-c", POSIX_READDIR_SH, "sh", path]);
    const parts = out.toString("utf8").split("\0").filter(Boolean);
    this.statCache.clear();
    const names: string[] = [];
    for (const entry of parts) {
      const isDir = entry[0] === "1";
      const name = entry.slice(1);
      this.statCache.set(posix.join(path, name), isDir);
      names.push(name);
    }
    return names;
  };

  private listFiles = async (root: string): Promise<string[]> => {
    const out = await this.checked(["/bin/sh", "-c", POSIX_LIST_FILES_SH, "sh", root]);
    return out
      .toString("utf8")
      .split("\n")
      .map((line) => line.replace(/^\.\//, "").trim())
      .filter(Boolean)
      .sort(byCodePoint)
      .slice(0, MAX_LISTED_FILES);
  };

  readonly fs: WorkspaceFs = {
    readFile: (path) => this.readFile(path),
    exists: (path) => this.exists(path),
    listFiles: (root) => this.listFiles(root),
  };

  async uploadDir(hostSrcDir: string, containerDestDir: string): Promise<void> {
    await this.mkdir(containerDestDir);
    const cp = await runProcess("docker", ["cp", `${resolve(hostSrcDir)}/.`, `${this.containerName}:${containerDestDir}`], { timeoutMs: 60_000 });
    if (cp.exitCode !== 0) throw new Error(`docker cp upload failed: ${cp.stderr.toString().trim()}`);
  }

  async downloadFile(containerPath: string): Promise<Buffer | undefined> {
    if (!(await this.exists(containerPath))) return undefined;
    return this.readFile(containerPath);
  }

  tools(names: readonly PiToolName[] = DEFAULT_PI_TOOLS): ToolDefinition[] {
    const shellArgv = (command: string) =>
      this.hasBash ? ["/bin/bash", "--noprofile", "--norc", "-c", command] : ["/bin/sh", "-c", command];
    const make: Partial<Record<PiToolName, () => unknown>> = {
      read: () => createReadToolDefinition(this.cwd, { operations: { readFile: this.readFile, access: this.access, detectImageMimeType: async () => null } }),
      write: () => createWriteToolDefinition(this.cwd, { operations: { writeFile: this.writeFile, mkdir: this.mkdir } }),
      edit: () => createEditToolDefinition(this.cwd, { operations: { readFile: this.readFile, writeFile: this.writeFile, access: this.access } }),
      bash: () => createBashToolDefinition(this.cwd, {
        exposeSessionEnvironment: false,
        operations: {
          exec: async (command, _cwd, opts) => this.exec(shellArgv(command), {
            signal: opts.signal,
            timeoutMs: (opts.timeout ?? 60) * 1000,
            onData: opts.onData,
          }),
        },
      }),
      ls: () => createLsToolDefinition(this.cwd, {
        operations: {
          exists: this.exists,
          readdir: this.readdir,
          stat: async (path: string) => {
            const cached = this.statCache.get(path);
            this.statCache.delete(path);
            const dir = cached ?? await this.isDirectory(path);
            return { isDirectory: () => dir };
          },
        },
      }),
    };
    const unsupported = names.filter((n) => !make[n]);
    if (unsupported.length) {
      throw new Error(`Harbor container tools do not support ${unsupported.join(", ")} (use bash/grep/find via bash tool).`);
    }
    return PI_TOOL_NAMES.filter((n) => names.includes(n)).map((n) => make[n]!()) as unknown as ToolDefinition[];
  }

  async close(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    await runProcess("docker", ["rm", "-f", this.containerName], { timeoutMs: 20_000 }).catch(() => {});
  }
}

// ---------------------------------------------------------------------------------------------
// Harbor Benchmark Runner, Verifier Contract, Resume & Live Scoreboard

export type HarborTrialStatus =
  | "completed"
  | "agent_timeout"
  | "budget_exhausted"
  | "agent_error"
  | "infra_error";

export interface HarborTrialRecord {
  schemaVersion: 1;
  trialId: string;
  taskId: string;
  repetition: number;
  endpoint: string;
  sessionId: string;
  model: string;
  promptVersion?: string;
  historyMode: "full" | "epoch" | "worklist";
  thinkingLevel: "high" | "off";
  status: HarborTrialStatus;
  reward: number;
  rewards?: Record<string, number>;
  requests: number;
  stepsCompleted: number;
  stepsTotal: number;
  verifierExitCode?: number;
  verifierOutputTail?: string;
  error?: string;
  startedAt: string;
  finishedAt: string;
  durationSec: number;
}

export interface HarborSummary {
  schemaVersion: 1;
  updatedAt: string;
  provenance: ReturnType<typeof runtimeProvenance>;
  model: string;
  concurrency: number;
  endpoints: string[];
  repetitions: number;
  historyMode: "full" | "epoch" | "worklist";
  thinkingLevel: "high" | "off";
  totals: {
    planned: number;
    finished: number;
    valid: number;
    infraErrors: number;
    passed: number;
    meanReward: number;
    passAtK: number;
    avgRequests: number;
  };
  byStatus: Record<string, number>;
  tasks: {
    taskId: string;
    trials: number;
    meanReward: number;
    maxReward: number;
    passedAny: boolean;
    statuses: string[];
  }[];
}

export interface HarborBenchmarkOptions {
  tasksPath: string;
  outputDir: string;
  baseConfig?: RwkvConfig;
  endpoints?: readonly string[];
  concurrency?: number;
  repetitions?: number;
  filter?: string;
  limit?: number;
  tools?: readonly PiToolName[];
  enableWebTools?: boolean;
  webTools?: ToolDefinition[];
  historyMode?: "full" | "epoch" | "worklist";
  thinkingLevel?: "high" | "off";
  maxRequests?: number;
  maxAgentTimeoutSec?: number;
  maxVerifierTimeoutSec?: number;
  containerNetwork?: "auto" | "bridge" | "none" | "host";
  verifierEnv?: Record<string, string>;
  appendSystemPrompt?: string;
  resume?: boolean;
  cleanupContextImages?: boolean;
  sandboxFactory?: (task: HarborTaskSpec) => Promise<{ sandbox: HarborSandbox; release?: () => Promise<void> }>;
  onTrialComplete?: (record: HarborTrialRecord, summary: HarborSummary) => void;
}

function isInfraTransportError(message: string): boolean {
  return /ECONNREFUSED|ECONNRESET|EHOSTUNREACH|ENETUNREACH|UND_ERR_CONNECT|fetch failed|Lightning tokenizer returned HTTP 5\d\d/i.test(message);
}

function assistantText(msg: AssistantMessage | undefined): string {
  if (!msg || !Array.isArray(msg.content)) return "";
  return msg.content
    .filter((b): b is { type: "text"; text: string } => b.type === "text" && typeof b.text === "string")
    .map((b) => b.text)
    .join("");
}

function serializeHarborEventLines(
  trialId: string,
  modelId: string,
  startedAt: string,
  events: readonly Record<string, unknown>[],
  status: string,
  requests: number,
  finalText: string,
): string {
  const all = [
    { type: "session", trialId, model: modelId, startedAt },
    ...events,
    { type: "status", status, requests },
    { type: "final", text: finalText },
  ];
  return all.map((e) => JSON.stringify(e)).join("\n") + "\n";
}

function parseVerifierReward(rewardJsonBuf?: Buffer, rewardTxtBuf?: Buffer): { reward: number; rewards?: Record<string, number> } {
  if (rewardJsonBuf) {
    try {
      const parsed = JSON.parse(rewardJsonBuf.toString("utf8")) as unknown;
      if (typeof parsed === "number" && Number.isFinite(parsed)) return { reward: parsed };
      if (object(parsed)) {
        const numeric: Record<string, number> = {};
        for (const [k, v] of Object.entries(parsed)) {
          if (typeof v === "number" && Number.isFinite(v)) numeric[k] = v;
        }
        const primary = numeric.reward ?? Object.values(numeric)[0] ?? 0;
        return { reward: primary, ...(Object.keys(numeric).length ? { rewards: numeric } : {}) };
      }
    } catch { /* fall through */ }
  }
  if (rewardTxtBuf) {
    const firstToken = rewardTxtBuf.toString("utf8").trim().split(/\s+/)[0] ?? "";
    const val = Number(firstToken);
    if (firstToken && Number.isFinite(val)) return { reward: val };
  }
  return { reward: 0 };
}

async function atomicWriteJson(filePath: string, value: unknown): Promise<void> {
  await mkdir(dirname(filePath), { recursive: true });
  const tmp = `${filePath}.${randomUUID().slice(0, 8)}.tmp`;
  await writeFile(tmp, JSON.stringify(value, null, 2) + "\n", "utf8");
  await rename(tmp, filePath);
}

async function atomicWriteText(filePath: string, content: string): Promise<void> {
  await mkdir(dirname(filePath), { recursive: true });
  const tmp = `${filePath}.${randomUUID().slice(0, 8)}.tmp`;
  await writeFile(tmp, content, "utf8");
  await rename(tmp, filePath);
}

function buildSummary(
  records: readonly HarborTrialRecord[],
  plannedTotal: number,
  meta: {
    model: string;
    concurrency: number;
    endpoints: string[];
    repetitions: number;
    historyMode: "full" | "epoch" | "worklist";
    thinkingLevel: "high" | "off";
  },
): HarborSummary {
  const byStatus: Record<string, number> = {};
  const byTask = new Map<string, HarborTrialRecord[]>();
  let infraErrors = 0;
  const valid: HarborTrialRecord[] = [];

  for (const rec of records) {
    byStatus[rec.status] = (byStatus[rec.status] ?? 0) + 1;
    if (rec.status === "infra_error") infraErrors++;
    else valid.push(rec);
    const list = byTask.get(rec.taskId) ?? [];
    list.push(rec);
    byTask.set(rec.taskId, list);
  }

  const tasks = [...byTask.entries()]
    .sort((a, b) => byCodePoint(a[0], b[0]))
    .map(([taskId, list]) => {
      const nonInfra = list.filter((r) => r.status !== "infra_error");
      const denom = nonInfra.length || 1;
      const meanReward = nonInfra.reduce((s, r) => s + r.reward, 0) / denom;
      const maxReward = nonInfra.reduce((m, r) => Math.max(m, r.reward), 0);
      return {
        taskId,
        trials: list.length,
        meanReward: Number(meanReward.toFixed(6)),
        maxReward: Number(maxReward.toFixed(6)),
        passedAny: maxReward >= 1,
        statuses: list.map((r) => r.status),
      };
    });

  const validTasks = tasks.filter((t) => t.statuses.some((s) => s !== "infra_error"));
  const meanReward = valid.length ? valid.reduce((s, r) => s + r.reward, 0) / valid.length : 0;
  const passAtK = validTasks.length ? validTasks.filter((t) => t.passedAny).length / validTasks.length : 0;
  const avgRequests = valid.length ? valid.reduce((s, r) => s + r.requests, 0) / valid.length : 0;

  return {
    schemaVersion: 1,
    updatedAt: new Date().toISOString(),
    provenance: runtimeProvenance(),
    model: meta.model,
    concurrency: meta.concurrency,
    endpoints: meta.endpoints,
    repetitions: meta.repetitions,
    historyMode: meta.historyMode,
    thinkingLevel: meta.thinkingLevel,
    totals: {
      planned: plannedTotal,
      finished: records.length,
      valid: valid.length,
      infraErrors,
      passed: valid.filter((r) => r.reward >= 1).length,
      meanReward: Number(meanReward.toFixed(6)),
      passAtK: Number(passAtK.toFixed(6)),
      avgRequests: Number(avgRequests.toFixed(2)),
    },
    byStatus,
    tasks,
  };
}

function renderScoreboardMarkdown(summary: HarborSummary): string {
  const pct = (v: number) => `${(v * 100).toFixed(2)}%`;
  const lines = [
    `# RWKV-Pi Native Harbor Scoreboard`,
    ``,
    `- **Model**: \`${summary.model}\``,
    `- **Progress**: \`${summary.totals.finished} / ${summary.totals.planned}\` (Valid: \`${summary.totals.valid}\`, Infra Errors: \`${summary.totals.infraErrors}\`)`,
    `- **Mean Reward (avg@${summary.repetitions})**: **${pct(summary.totals.meanReward)}**`,
    `- **Pass@${summary.repetitions} (reward = 1.0)**: **${pct(summary.totals.passAtK)}** (\`${summary.totals.passed}/${summary.totals.valid}\` trials)`,
    `- **Avg Requests / Trial**: \`${summary.totals.avgRequests}\``,
    `- **Config**: concurrency=\`${summary.concurrency}\`, endpoints=\`${summary.endpoints.length}\`, history=\`${summary.historyMode}\`, thinking=\`${summary.thinkingLevel}\``,
    `- **Updated**: \`${summary.updatedAt}\``,
    ``,
    `| Task | Trials | Mean Reward | Max Reward | Statuses |`,
    `| :--- | ---: | ---: | ---: | :--- |`,
    ...summary.tasks.map((t) => `| \`${t.taskId}\` | ${t.trials} | ${t.meanReward.toFixed(3)} | ${t.maxReward.toFixed(3)} | ${t.statuses.join(", ")} |`),
    ``,
  ];
  return lines.join("\n");
}

export async function runHarborBenchmark(options: HarborBenchmarkOptions): Promise<HarborSummary> {
  const baseConfig = options.baseConfig ?? readConfig();
  const concurrency = Math.max(1, Math.trunc(options.concurrency ?? 4));
  const repetitions = Math.max(1, Math.trunc(options.repetitions ?? 1));
  const maxRequests = Math.max(1, Math.trunc(options.maxRequests ?? 48));
  const thinkingLevel = options.thinkingLevel ?? "high";
  const effectiveHistoryMode = options.historyMode ?? baseConfig.historyMode;
  const piTools = parseToolList(options.tools ?? DEFAULT_PI_TOOLS);
  const endpointsList = options.endpoints?.length ? [...options.endpoints] : [baseConfig.baseUrl];
  const pool = new EndpointPool(endpointsList);
  const outputDir = resolve(options.outputDir);
  await mkdir(join(outputDir, "trials"), { recursive: true });

  let tasks = await discoverHarborTasks(options.tasksPath);
  if (options.filter) {
    const needle = options.filter.toLowerCase();
    tasks = tasks.filter((t) => t.id.toLowerCase().includes(needle));
  }
  if (options.limit !== undefined && options.limit > 0) {
    tasks = tasks.slice(0, options.limit);
  }
  if (!tasks.length) throw new Error(`No Harbor tasks matched in ${options.tasksPath}.`);

  const imageCache = new HarborImageCache();
  const jobs: { trialId: string; task: HarborTaskSpec; repetition: number }[] = [];
  const completedRecords = new Map<string, HarborTrialRecord>();

  for (const task of tasks) {
    let pendingForTask = 0;
    for (let rep = 0; rep < repetitions; rep++) {
      const safeTaskId = task.id.replaceAll("/", "__");
      const trialId = repetitions === 1 ? safeTaskId : `${safeTaskId}__r${rep}`;
      const existingPath = join(outputDir, "trials", trialId, "trial.json");
      if (options.resume !== false && existsSync(existingPath)) {
        try {
          const prev = JSON.parse(await readFile(existingPath, "utf8")) as HarborTrialRecord;
          if (prev && prev.schemaVersion === 1 && prev.status !== "infra_error") {
            completedRecords.set(trialId, prev);
            continue;
          }
        } catch { /* corrupt trial.json -> rerun */ }
      }
      jobs.push({ trialId, task, repetition: rep });
      pendingForTask++;
    }
    if (pendingForTask > 0) imageCache.registerPlannedTasks([{ task, count: pendingForTask }]);
  }

  const totalPlanned = tasks.length * repetitions;
  let flushChain = Promise.resolve();
  const persistSummary = async (): Promise<HarborSummary> => {
    const summary = buildSummary([...completedRecords.values()], totalPlanned, {
      model: baseConfig.modelId,
      concurrency,
      endpoints: endpointsList,
      repetitions,
      historyMode: effectiveHistoryMode,
      thinkingLevel,
    });
    await atomicWriteJson(join(outputDir, "summary.json"), summary);
    await atomicWriteText(join(outputDir, "SCOREBOARD.md"), renderScoreboardMarkdown(summary));
    return summary;
  };

  if (jobs.length === 0) return persistSummary();

  let jobIndex = 0;
  const runOneTrial = async (job: { trialId: string; task: HarborTaskSpec; repetition: number }): Promise<void> => {
    const startedMs = Date.now();
    const startedAt = new Date(startedMs).toISOString();
    const trialDir = join(outputDir, "trials", job.trialId);
    await mkdir(trialDir, { recursive: true });
    const lease = pool.acquire(job.trialId);
    const trialConfig: RwkvConfig = {
      ...baseConfig,
      baseUrl: lease.endpoint,
      sessionId: lease.sessionId,
      historyMode: effectiveHistoryMode,
    };
    const provider = createRwkvProvider(trialConfig);
    const model = provider.getModels()[0]!;

    let sandboxHandle: { sandbox: HarborSandbox; release?: () => Promise<void> } | undefined;
    let status: HarborTrialStatus = "completed";
    let errorMsg: string | undefined;
    let requests = 0;
    let stepsCompleted = 0;
    let reward = 0;
    let rewards: Record<string, number> | undefined;
    let verifierExitCode: number | undefined;
    let verifierOutputTail: string | undefined;
    const harborEvents: Record<string, unknown>[] = [];
    let finalAssistant: AssistantMessage | undefined;

    try {
      // 1. Container & image preparation (errors here are strictly infra_error).
      try {
        if (options.sandboxFactory) {
          sandboxHandle = await options.sandboxFactory(job.task);
        } else {
          const prepared = await imageCache.prepare(job.task);
          const totalAgentSec = job.task.steps.reduce(
            (sum, s) => sum + Math.min(s.agentTimeoutSec ?? job.task.agentTimeoutSec, options.maxAgentTimeoutSec ?? Infinity),
            0,
          );
          const totalVerifierSec = Math.min(job.task.verifierTimeoutSec, options.maxVerifierTimeoutSec ?? Infinity);
          const netMode = options.containerNetwork === "none" || !job.task.environment.allowInternet
            ? "none"
            : options.containerNetwork === "host"
              ? "host"
              : "bridge";
          const sandbox = await PersistentDockerSandbox.start({
            imageTag: prepared.imageTag,
            seedWorkspaceDir: prepared.seedWorkspaceDir,
            cpus: job.task.environment.cpus,
            memoryMb: job.task.environment.memoryMb,
            networkMode: netMode,
            env: job.task.environment.env,
            lifetimeSec: totalAgentSec + totalVerifierSec + 300,
          });
          sandboxHandle = {
            sandbox,
            release: () => imageCache.release(job.task, prepared, options.cleanupContextImages),
          };
        }
      } catch (err) {
        status = "infra_error";
        errorMsg = err instanceof Error ? err.message : String(err);
        return;
      }

      const { sandbox } = sandboxHandle;
      const settingsManager = SettingsManager.inMemory({
        compaction: { enabled: false },
        retry: { enabled: false, provider: { maxRetries: 0 } },
        cacheWarming: "off",
      });
      const resourceLoader: ResourceLoader = {
        getExtensions: () => ({ extensions: [], errors: [], runtime: createExtensionRuntime() }),
        getSkills: () => ({ skills: [], diagnostics: [] }),
        getPrompts: () => ({ prompts: [], diagnostics: [] }),
        getThemes: () => ({ themes: [], diagnostics: [] }),
        getAgentsFiles: () => ({ agentsFiles: [] }),
        getSystemPrompt: () => undefined,
        getSystemPromptSource: () => undefined,
        getAppendSystemPrompt: () => (options.appendSystemPrompt ? [options.appendSystemPrompt] : []),
        getAppendSystemPromptSources: () => [],
        extendResources: () => {},
        reload: async () => {},
      };
      const modelRuntime = await ModelRuntime.create({
        credentials: new InMemoryCredentialStore(),
        modelsStore: new InMemoryModelsStore(),
        modelsPath: null,
        refreshOnCreate: false,
        allowModelNetwork: false,
      });
      modelRuntime.registerNativeProvider(provider);

      const rawTools = sandbox.tools(piTools);
      const baseTools = trialConfig.toolTolerance
        ? rawTools.map((t) => (TOLERANT_TOOL_NAMES.includes(t.name) ? withTolerance(t, sandbox.cwd, sandbox.fs) : t))
        : rawTools;
      const extraWebTools = options.webTools ?? (options.enableWebTools ? createWebTools() : []);
      const worklistMode = effectiveHistoryMode === "worklist";
      const worklistState = worklistMode ? new PiWorklistState() : undefined;
      const customTools = [...baseTools, ...extraWebTools, ...(worklistState ? createWorklistTools(worklistState) : [])];
      const sessionToolNames = [...piTools, ...extraWebTools.map((t) => t.name), ...(worklistMode ? WORKLIST_TOOL_NAMES : [])];

      const { session } = await createAgentSession({
        cwd: sandbox.cwd,
        agentDir: join(trialDir, "pi-agent"),
        modelRuntime,
        model,
        thinkingLevel,
        settingsManager,
        resourceLoader,
        sessionManager: SessionManager.inMemory(sandbox.cwd),
        tools: sessionToolNames,
        customTools,
      });

      const { identicalCallLimit, targetFailureLimit } = trialConfig;
      if (identicalCallLimit || targetFailureLimit || worklistMode) {
        const prevBefore = session.agent.beforeToolCall;
        session.agent.beforeToolCall = async (ctx, sig) => {
          const msgs = ctx.context.messages as readonly Message[];
          const { name, arguments: args } = ctx.toolCall;
          const reason = (worklistMode ? repeatedWorklistSpin(msgs, name) : undefined)
            ?? (identicalCallLimit ? repeatedIdenticalCall(msgs, name, args, identicalCallLimit) : undefined)
            ?? (targetFailureLimit ? repeatedTargetFailure(msgs, name, args, targetFailureLimit) : undefined);
          if (reason) return { block: true, reason };
          if (worklistState && !isWorklistTool(name)) worklistState.recordWorkspaceTool();
          return prevBefore?.(ctx, sig);
        };
      }

      let interrupted: "budget_exhausted" | "agent_timeout" | undefined;
      const budgetController = new AbortController();
      const prevPayload = session.agent.onPayload;
      session.agent.onPayload = async (payload, reqModel) => {
        if (requests >= maxRequests) {
          interrupted = "budget_exhausted";
          throw new Error("Harbor trial request budget exhausted.");
        }
        const updated = await prevPayload?.(payload, reqModel);
        let request: Record<string, unknown> = (updated ?? payload) as Record<string, unknown>;
        if (!provider.rwkvPiTransportFolded) {
          if (effectiveHistoryMode === "epoch") request = foldEpochRequest(request).request;
          else if (effectiveHistoryMode === "worklist") request = foldWorklistRequest(request).request;
        }
        const prepared: { request: Record<string, unknown>; budget: TokenBudget } = await budgetRwkvRequest(request, trialConfig, budgetController.signal);
        requests++;
        return prepared.request;
      };

      const unsubscribe = session.subscribe((event) => {
        if (event.type !== "message_end") return;
        const msg = event.message;
        if (msg.role === "assistant") {
          finalAssistant = msg;
          for (const block of msg.content) {
            if (block.type === "thinking" && block.thinking) harborEvents.push({ type: "reasoning", text: block.thinking });
            else if (block.type === "text" && block.text) harborEvents.push({ type: "text", text: block.text });
            else if (block.type === "toolCall") harborEvents.push({ type: "tool_call", id: block.id, name: block.name, arguments: block.arguments });
          }
        } else if (msg.role === "toolResult") {
          const text = Array.isArray(msg.content)
            ? msg.content.filter((b): b is { type: "text"; text: string } => b.type === "text" && typeof b.text === "string").map((b) => b.text).join("\n")
            : "";
          harborEvents.push({
            type: "tool_result",
            toolCallId: msg.toolCallId,
            toolName: msg.toolName,
            isError: Boolean(msg.isError),
            text: text.slice(0, 16_384),
          });
        }
      });

      try {
        for (let i = 0; i < job.task.steps.length; i++) {
          const step = job.task.steps[i]!;
          if (step.setupScriptPath) {
            const script = await readFile(step.setupScriptPath);
            await sandbox.writeFile("/tmp/.rwkv-step-setup.sh", script);
            const setupRes = await sandbox.exec(["/bin/sh", "-c", "chmod +x /tmp/.rwkv-step-setup.sh && /tmp/.rwkv-step-setup.sh && rm -f /tmp/.rwkv-step-setup.sh"], {
              cwd: sandbox.cwd,
              timeoutMs: 300_000,
            });
            if (setupRes.exitCode !== 0) {
              throw new Error(`Step ${step.name} setup.sh failed: ${(setupRes.stdout.toString() + setupRes.stderr.toString()).slice(-800)}`);
            }
          }
          const stepTimeoutMs = Math.min(step.agentTimeoutSec ?? job.task.agentTimeoutSec, options.maxAgentTimeoutSec ?? Infinity) * 1000;
          const timer = setTimeout(() => {
            interrupted = "agent_timeout";
            budgetController.abort();
            void session.abort();
          }, stepTimeoutMs);
          try {
            harborEvents.push({ type: "user", turn: i + 1, step: step.name, text: step.instruction });
            await session.prompt(step.instruction);
          } finally {
            clearTimeout(timer);
          }
          const lastStop = finalAssistant?.stopReason;
          if (interrupted) {
            status = interrupted;
            break;
          }
          if (lastStop !== "stop") {
            const errDetail = finalAssistant?.errorMessage ?? `Turn ended with ${lastStop ?? "no response"}.`;
            if (requests <= 1 && stepsCompleted === 0 && isInfraTransportError(errDetail)) {
              status = "infra_error";
              errorMsg = errDetail;
              return;
            }
            status = "agent_error";
            errorMsg = errDetail;
            break;
          }
          stepsCompleted++;
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        if (interrupted) {
          status = interrupted;
          errorMsg = msg;
        } else if (requests <= 1 && stepsCompleted === 0 && isInfraTransportError(msg)) {
          status = "infra_error";
          errorMsg = msg;
          return;
        } else {
          status = "agent_error";
          errorMsg = msg;
        }
      } finally {
        budgetController.abort();
        unsubscribe();
        session.dispose();
      }

      // 2. Write Harbor agent logs (/logs/agent/final.txt & /logs/agent/events.jsonl) and run anti-leak verifier.
      const finalText = assistantText(finalAssistant);
      const eventsJsonl = serializeHarborEventLines(job.trialId, trialConfig.modelId, startedAt, harborEvents, status, requests, finalText);
      await writeFile(join(trialDir, "final.txt"), finalText, "utf8");
      await writeFile(join(trialDir, "events.jsonl"), eventsJsonl, "utf8");
      await sandbox.writeFile("/logs/agent/final.txt", finalText);
      await sandbox.writeFile("/logs/agent/events.jsonl", eventsJsonl);

      const stepTests = job.task.rewardStrategy === "mean"
        ? job.task.steps.filter((s) => s.testsDir)
        : [job.task.steps.at(-1)].filter((s): s is HarborStepSpec => Boolean(s?.testsDir));

      if (stepTests.length > 0) {
        const stepScores: number[] = [];
        for (const stepSpec of stepTests) {
          await sandbox.exec(["/bin/sh", "-c", "rm -rf /tests /logs/verifier && mkdir -p /logs/verifier"], { timeoutMs: 15_000 });
          await sandbox.uploadDir(stepSpec.testsDir!, "/tests");
          const vTimeoutMs = Math.min(stepSpec.verifierTimeoutSec ?? job.task.verifierTimeoutSec, options.maxVerifierTimeoutSec ?? Infinity) * 1000;
          const vRes = await sandbox.exec(
            ["/bin/sh", "-c", "chmod +x /tests/test.sh 2>/dev/null || true; if [ -x /bin/bash ]; then exec /bin/bash /tests/test.sh; else exec /bin/sh /tests/test.sh; fi"],
            { cwd: sandbox.cwd, env: options.verifierEnv, timeoutMs: vTimeoutMs },
          );
          verifierExitCode = vRes.exitCode;
          verifierOutputTail = (vRes.stdout.toString() + vRes.stderr.toString()).slice(-4000);
          const rewardJsonBuf = await sandbox.downloadFile("/logs/verifier/reward.json");
          const rewardTxtBuf = await sandbox.downloadFile("/logs/verifier/reward.txt");
          if (rewardJsonBuf) await writeFile(join(trialDir, "reward.json"), rewardJsonBuf);
          if (rewardTxtBuf) await writeFile(join(trialDir, "reward.txt"), rewardTxtBuf);
          const parsed = parseVerifierReward(rewardJsonBuf, rewardTxtBuf);
          stepScores.push(parsed.reward);
          if (parsed.rewards) rewards = parsed.rewards;
        }
        reward = stepScores.reduce((a, b) => a + b, 0) / stepScores.length;
      }
    } finally {
      if (sandboxHandle) {
        await sandboxHandle.sandbox.close().catch(() => {});
        await sandboxHandle.release?.().catch(() => {});
      }
      lease.release();
      const finishedMs = Date.now();
      const record: HarborTrialRecord = {
        schemaVersion: 1,
        trialId: job.trialId,
        taskId: job.task.id,
        repetition: job.repetition,
        endpoint: lease.endpoint,
        sessionId: lease.sessionId,
        model: trialConfig.modelId,
        ...(trialConfig.toolFormat === "xml" || trialConfig.toolFormat === "json" ? { promptVersion: promptVersion(trialConfig.toolFormat) } : {}),
        historyMode: effectiveHistoryMode,
        thinkingLevel,
        status,
        reward: Number(reward.toFixed(6)),
        ...(rewards ? { rewards } : {}),
        requests,
        stepsCompleted,
        stepsTotal: job.task.steps.length,
        ...(verifierExitCode !== undefined ? { verifierExitCode } : {}),
        ...(verifierOutputTail ? { verifierOutputTail } : {}),
        ...(errorMsg ? { error: errorMsg } : {}),
        startedAt,
        finishedAt: new Date(finishedMs).toISOString(),
        durationSec: Number(((finishedMs - startedMs) / 1000).toFixed(2)),
      };
      await atomicWriteJson(join(trialDir, "trial.json"), record);
      completedRecords.set(job.trialId, record);
      flushChain = flushChain.then(async () => {
        const summary = await persistSummary();
        options.onTrialComplete?.(record, summary);
      });
      await flushChain;
    }
  };

  const workers = Array.from({ length: Math.min(concurrency, jobs.length) }, async () => {
    while (true) {
      const idx = jobIndex++;
      const job = jobs[idx];
      if (!job) break;
      await runOneTrial(job);
    }
  });
  await Promise.all(workers);
  return persistSummary();
}
