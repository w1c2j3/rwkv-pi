// RWKV subagents for a Qwen main agent (see docs/SUBAGENTS.md).
//
// Each subagent is a separate non-interactive pi process (`--mode json -p --no-session`) driven by the
// RWKV provider with pi's unchanged system prompt; the delegated instruction is its user message, as in
// training. It runs under bubblewrap: the host is read-only, except a private pi agent dir and, for
// write subagents, a private copy of the workspace mounted at the workspace path. Changes from a copy
// reach the real workspace only through apply(), which refuses files the main workspace changed since
// the copy was made. Traces are kept under <RWKV_PI_DATA_ROOT>/runs/subagents/.
import { spawn, spawnSync, type ChildProcess } from "node:child_process";
import { createHash } from "node:crypto";
import { copyFileSync, cpSync, existsSync, lstatSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, unlinkSync, writeFileSync, appendFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dataRoot, envInteger, readConfig } from "./config.js";
import { countMessage, emptyCounters, type WorkCounters } from "./project-participation.js";

export const SUBAGENT_TOOLS = ["read", "write", "edit", "bash", "grep", "find", "ls"] as const;

export interface SubagentLimits { concurrency: number; maxRequests: number; timeoutSeconds: number; maxTasks: number; sandbox: "bwrap" | "none"; mount: string }
export function subagentLimits(env: NodeJS.ProcessEnv = process.env): SubagentLimits {
  const sandbox = env.RWKV_SUBAGENT_SANDBOX?.trim() || "bwrap";
  if (sandbox !== "bwrap" && sandbox !== "none") throw new Error("RWKV_SUBAGENT_SANDBOX must be bwrap or none.");
  return {
    concurrency: envInteger(env, "RWKV_SUBAGENT_CONCURRENCY", 16, 1),
    maxRequests: envInteger(env, "RWKV_SUBAGENT_MAX_REQUESTS", 32, 1),
    timeoutSeconds: envInteger(env, "RWKV_SUBAGENT_TIMEOUT_SECONDS", 1200, 1),
    maxTasks: envInteger(env, "RWKV_SUBAGENT_MAX_TASKS", 64, 1),
    sandbox,
    // Inside the sandbox the workspace appears at this short path (RWKV miscopies long absolute paths).
    // An empty value keeps the real path.
    mount: env.RWKV_SUBAGENT_MOUNT === undefined ? "/srv/workspace" : env.RWKV_SUBAGENT_MOUNT.trim(),
  };
}

export function subagentRoot(env: NodeJS.ProcessEnv = process.env): string {
  const data = dataRoot(env);
  // Never fall back to other storage when the data root is unavailable.
  if (!existsSync(data)) throw new Error(`Subagent data root unavailable: ${data}`);
  return join(data, "runs", "subagents");
}

export type SubagentState = "queued" | "running" | "completed" | "failed" | "budget" | "timeout" | "cancelled";

export interface SubagentSpec { task: string; name?: string; write?: boolean; maxRequests?: number; timeoutSeconds?: number }
export interface FileChange { path: string; kind: "added" | "modified" | "deleted" }

export interface Subagent {
  id: string; name: string; task: string; write: boolean; maxRequests: number; timeoutSeconds: number;
  state: SubagentState; dir: string; createdAt: number; startedAt?: number; finishedAt?: number;
  requests: number; toolCalls: number; toolErrors: number; lastTool?: string; answer?: string;
  stopReason?: string; error?: string; exitCode?: number | null; changes?: FileChange[]; applied?: boolean;
  work: WorkCounters; appliedFiles?: string[];
}

// Version-control internals and interpreter caches are never reported or applied as changes.
const IGNORED_DIRS = new Set([".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"]);
const sha = (path: string) => createHash("sha256").update(readFileSync(path)).digest("hex");

/** Regular files and symlinks under root, excluding .git internals, with content hashes. */
function manifest(root: string): Map<string, string> {
  const out = new Map<string, string>();
  const walk = (dir: string) => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const path = join(dir, entry.name);
      if (entry.isDirectory()) { if (!IGNORED_DIRS.has(entry.name)) walk(path); continue; }
      if (entry.name.endsWith(".pyc")) continue;
      const rel = relative(root, path);
      if (entry.isSymbolicLink()) out.set(rel, "link:" + lstatSync(path).size);
      else if (entry.isFile()) out.set(rel, sha(path));
    }
  };
  walk(root);
  return out;
}

function textOf(message: any): string {
  return (message?.content ?? []).filter((p: any) => p?.type === "text").map((p: any) => p.text).join("").trim();
}

/** The pi command the rwkv-pi CLI uses, in non-interactive JSON mode. */
function piCommand(task: string): { command: string; args: string[] } {
  const config = readConfig();
  const source = process.env.RWKV_PI_SOURCE ? resolve(process.env.RWKV_PI_SOURCE) : undefined;
  const piEntry = fileURLToPath(import.meta.resolve("@earendil-works/pi-coding-agent"));
  const piCli = source ? join(source, "packages/coding-agent/src/cli.ts") : join(dirname(piEntry), "bundle", "cli.js");
  const preload = source ? ["--import", pathToFileURL(join(source, "packages/coding-agent/src/experimental/source-resolver.ts")).href] : [];
  const extension = fileURLToPath(new URL("./extension.js", import.meta.url));
  return {
    command: process.execPath,
    args: [...preload, piCli, "--no-extensions", "--extension", extension, "--provider", "rwkv", "--model", config.modelId, "--thinking", "high",
      "--tools", SUBAGENT_TOOLS.join(","), "--no-skills", "--no-context-files", "--no-prompt-templates",
      "--mode", "json", "-p", "--no-session", task],
  };
}

/** pi's fd binary, so a fresh agent dir does not download it at startup. */
function seedFd(agentDir: string, cache: string) {
  const target = join(agentDir, "bin", "fd");
  const cached = join(cache, "fd");
  if (!existsSync(cached)) {
    const found = [join(homedir(), ".pi", "agent", "bin", "fd"), process.env.PI_CODING_AGENT_DIR && join(process.env.PI_CODING_AGENT_DIR, "bin", "fd")]
      .find((p): p is string => !!p && existsSync(p));
    if (!found) return;
    mkdirSync(cache, { recursive: true }); copyFileSync(found, cached);
  }
  mkdirSync(dirname(target), { recursive: true }); copyFileSync(cached, target);
}

export class SubagentManager {
  readonly session: string;
  /** Workspace path as the subagents see it. */
  readonly inner: string;
  readonly root: string;
  private readonly agents = new Map<string, Subagent>();
  private readonly children = new Map<string, ChildProcess>();
  private readonly bases = new Map<string, Map<string, string>>();
  private readonly waiters = new Set<() => void>();
  private counter = 0;

  constructor(readonly cwd: string, readonly limits: SubagentLimits = subagentLimits(), root = subagentRoot(), private readonly onChange?: (agent: Subagent) => void) {
    this.inner = limits.sandbox === "bwrap" && limits.mount ? limits.mount : cwd;
    if (this.inner !== cwd && !(this.inner.startsWith("/srv/") || this.inner.startsWith("/tmp/"))) throw new Error("RWKV_SUBAGENT_MOUNT must be under /srv or /tmp (the sandbox mounts a tmpfs there).");
    this.session = new Date().toISOString().replace(/[:.]/g, "-") + "-" + process.pid;
    this.root = join(root, this.session);
    mkdirSync(this.root, { recursive: true });
    if (limits.sandbox === "bwrap" && spawnSync("bwrap", ["--version"]).status !== 0) throw new Error("bubblewrap (bwrap) is required for RWKV subagents; set RWKV_SUBAGENT_SANDBOX=none only for local tests.");
  }

  list(ids?: readonly string[]): Subagent[] {
    const all = [...this.agents.values()];
    if (!ids?.length) return all;
    const missing = ids.filter((id) => !this.agents.has(id));
    if (missing.length) throw new Error(`Unknown subagent id: ${missing.join(", ")}`);
    return ids.map((id) => this.agents.get(id)!);
  }

  start(specs: readonly SubagentSpec[]): Subagent[] {
    if (!specs.length) throw new Error("No tasks given.");
    if (this.agents.size + specs.length > this.limits.maxTasks) throw new Error(`At most ${this.limits.maxTasks} subagents per session.`);
    const created = specs.map((spec) => {
      if (!spec.task?.trim()) throw new Error("Every subagent needs a non-empty task.");
      const id = `rwkv-${++this.counter}`;
      const agent: Subagent = {
        id, name: spec.name?.trim() || id, task: spec.task, write: spec.write === true,
        maxRequests: Math.min(spec.maxRequests ?? this.limits.maxRequests, this.limits.maxRequests),
        timeoutSeconds: Math.min(spec.timeoutSeconds ?? this.limits.timeoutSeconds, this.limits.timeoutSeconds),
        state: "queued", dir: join(this.root, id), createdAt: Date.now(), requests: 0, toolCalls: 0, toolErrors: 0, work: emptyCounters(),
      };
      mkdirSync(agent.dir, { recursive: true });
      writeFileSync(join(agent.dir, "task.json"), JSON.stringify({ ...agent, cwd: this.cwd }, null, 2) + "\n");
      this.agents.set(id, agent);
      return agent;
    });
    this.pump();
    return created;
  }

  private running(): number { return [...this.agents.values()].filter((a) => a.state === "running").length; }

  private pump() {
    for (const agent of this.agents.values()) {
      if (this.running() >= this.limits.concurrency) break;
      if (agent.state === "queued") {
        try { this.launch(agent); }
        catch (error) { this.finish(agent, "failed", String(error instanceof Error ? error.message : error)); }
      }
    }
  }

  private launch(agent: Subagent) {
    const agentDir = join(agent.dir, "pi-agent");
    mkdirSync(agentDir, { recursive: true });
    seedFd(agentDir, join(dirname(this.root), ".cache"));
    let mount: string[] = ["--ro-bind", this.cwd, this.inner];
    if (agent.write) {
      const fork = join(agent.dir, "workspace");
      cpSync(this.cwd, fork, { recursive: true, preserveTimestamps: true, verbatimSymlinks: true });
      this.bases.set(agent.id, manifest(fork));
      mount = ["--bind", fork, this.inner];
    }
    // Paths the main agent wrote in terms of the real workspace refer to the mount inside the sandbox.
    const task = this.inner === this.cwd ? agent.task : agent.task.split(this.cwd).join(this.inner);
    const { command, args } = piCommand(task);
    const env: NodeJS.ProcessEnv = { ...process.env, PI_CODING_AGENT_DIR: agentDir };
    for (const key of Object.keys(env)) if (key.startsWith("TEACHER_") || key.startsWith("DEEPSEEK_") || key.startsWith("STEP_")) delete env[key];
    // A read-only host mount still permits reading files: mask the paid main-agent key as well.
    const mainKey = process.env.TEACHER_API_KEY_FILE;
    const hiddenKeys = mainKey && existsSync(mainKey) ? ["--ro-bind", "/dev/null", resolve(mainKey)] : [];
    const sandboxed = this.limits.sandbox === "bwrap";
    const argv = sandboxed
      ? ["--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp",
        ...(this.inner.startsWith("/srv/") ? ["--tmpfs", "/srv"] : []), ...mount, ...hiddenKeys, "--bind", agentDir, agentDir,
        "--unshare-pid", "--die-with-parent", "--chdir", this.inner, "--", command, ...args]
      : args;
    const child = spawn(sandboxed ? "bwrap" : command, argv, {
      cwd: agent.write && !sandboxed ? join(agent.dir, "workspace") : this.cwd, env, stdio: ["ignore", "pipe", "pipe"], detached: true,
    });
    this.children.set(agent.id, child);
    agent.state = "running"; agent.startedAt = Date.now();
    this.changed(agent);
    const events = join(agent.dir, "events.jsonl"), stderr = join(agent.dir, "stderr.log");
    let buffer = "";
    let lastAssistant: any;
    let generationPending = false;
    child.stdout!.on("data", (chunk: Buffer) => {
      appendFileSync(events, chunk);
      buffer += chunk.toString("utf8");
      let newline: number;
      while ((newline = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, newline); buffer = buffer.slice(newline + 1);
        let event: any; try { event = JSON.parse(line); } catch { continue; }
        if (event.type === "message_start" && event.message?.role === "assistant") generationPending = true;
        if (event.type === "message_end" && event.message?.role === "assistant") {
          generationPending = false; countMessage(agent.work, event.message);
          agent.requests++; lastAssistant = event.message;
          if (agent.requests >= agent.maxRequests && event.message.stopReason === "toolUse") this.kill(agent, "budget", `Reached ${agent.maxRequests} model requests.`);
        } else if (event.type === "tool_execution_start") { agent.toolCalls++; agent.work.toolCalls++; agent.lastTool = `${event.toolName} ${JSON.stringify(event.args ?? {}).slice(0, 160)}`; }
        else if (event.type === "tool_execution_end" && event.isError) { agent.toolErrors++; agent.work.toolErrors++; }
        this.changed(agent);
      }
    });
    child.stderr!.on("data", (chunk: Buffer) => appendFileSync(stderr, chunk));
    const timer = setTimeout(() => this.kill(agent, "timeout", `Exceeded ${agent.timeoutSeconds} s.`), agent.timeoutSeconds * 1000);
    timer.unref();
    child.once("error", (error) => { clearTimeout(timer); this.finish(agent, "failed", error.message); });
    child.once("close", (code) => {
      clearTimeout(timer);
      if (generationPending) { agent.work.responses++; agent.work.unknownUsage++; }
      this.children.delete(agent.id);
      agent.exitCode = code;
      agent.answer = textOf(lastAssistant) || agent.answer;
      agent.stopReason = lastAssistant?.stopReason;
      const stop = this.stopping.get(agent.id);
      if (stop) { this.stopping.delete(agent.id); this.finish(agent, stop.state, stop.reason); }
      else {
        const error = lastAssistant?.errorMessage as string | undefined;
        if (code === 0 && lastAssistant?.stopReason === "stop") this.finish(agent, "completed");
        else this.finish(agent, "failed", error || (lastAssistant?.stopReason === "length" ? "Output limit reached." : `pi exited with code ${code}.`));
      }
    });
  }

  private readonly stopping = new Map<string, { state: SubagentState; reason: string }>();

  private kill(agent: Subagent, state: SubagentState, reason: string) {
    if (agent.finishedAt || this.stopping.has(agent.id)) return;
    const child = this.children.get(agent.id);
    if (!child?.pid) return this.finish(agent, state, reason);
    this.stopping.set(agent.id, { state, reason });
    try { process.kill(-child.pid, "SIGTERM"); } catch { /* already gone */ }
    setTimeout(() => { try { process.kill(-child.pid!, "SIGKILL"); } catch { /* gone */ } }, 5000).unref();
  }

  private finish(agent: Subagent, state: SubagentState, error?: string) {
    if (agent.finishedAt) return;
    agent.state = state; agent.finishedAt = Date.now();
    if (error) agent.error = error;
    if (agent.write) {
      try { agent.changes = this.diffList(agent); } catch (e) { agent.error = (agent.error ? agent.error + "; " : "") + `change scan failed: ${e}`; }
    }
    writeFileSync(join(agent.dir, "result.json"), JSON.stringify(agent, null, 2) + "\n");
    this.changed(agent);
    this.pump();
  }

  private changed(agent: Subagent) {
    this.onChange?.(agent);
    for (const wake of [...this.waiters]) wake();
  }

  cancel(ids?: readonly string[]): Subagent[] {
    const targets = this.list(ids).filter((a) => !a.finishedAt);
    for (const agent of targets) {
      if (agent.state === "queued") this.finish(agent, "cancelled", "Cancelled before start.");
      else this.kill(agent, "cancelled", "Cancelled.");
    }
    return targets;
  }

  /** Resolves when every listed subagent is final, or after timeoutMs (pending ones are returned as they are). */
  async wait(ids: readonly string[] | undefined, timeoutMs: number, signal?: AbortSignal): Promise<Subagent[]> {
    const targets = this.list(ids);
    const done = () => targets.every((a) => a.finishedAt !== undefined);
    if (done()) return targets;
    await new Promise<void>((resolveWait) => {
      const finish = () => { clearTimeout(timer); this.waiters.delete(check); signal?.removeEventListener("abort", finish); resolveWait(); };
      const check = () => { if (done()) finish(); };
      const timer = setTimeout(finish, timeoutMs);
      this.waiters.add(check);
      signal?.addEventListener("abort", finish, { once: true });
    });
    return targets;
  }

  private diffList(agent: Subagent): FileChange[] {
    const base = this.bases.get(agent.id);
    if (!base) return [];
    const now = manifest(join(agent.dir, "workspace"));
    const changes: FileChange[] = [];
    for (const [path, hash] of now) {
      if (!base.has(path)) changes.push({ path, kind: "added" });
      else if (base.get(path) !== hash) changes.push({ path, kind: "modified" });
    }
    for (const path of base.keys()) if (!now.has(path)) changes.push({ path, kind: "deleted" });
    return changes.sort((a, b) => a.path.localeCompare(b.path));
  }

  /** Unified diff of the subagent's copy against the current workspace (text files). */
  diff(id: string, maxChars = 12000): string {
    const agent = this.list([id])[0]!;
    if (!agent.write) return "(read-only subagent: no file changes)";
    const fork = join(agent.dir, "workspace");
    const parts: string[] = [];
    for (const change of agent.changes ?? this.diffList(agent)) {
      const a = change.kind === "added" ? "/dev/null" : join(this.cwd, change.path);
      const b = change.kind === "deleted" ? "/dev/null" : join(fork, change.path);
      const out = spawnSync("diff", ["-u", "--label", `a/${change.path}`, "--label", `b/${change.path}`, existsSync(a) ? a : "/dev/null", b], { encoding: "utf8", maxBuffer: 16 << 20 });
      parts.push(out.stdout || `${change.kind}: ${change.path}${out.stderr ? ` (${out.stderr.trim()})` : ""}`);
    }
    const text = parts.join("\n") || "(no changes)";
    return text.length > maxChars ? text.slice(0, maxChars) + `\n… diff truncated (${text.length} chars); full copy at ${fork}` : text;
  }

  /** Copies a finished write subagent's changes into the workspace; files changed there since the copy are conflicts. */
  apply(id: string, files?: readonly string[]): { applied: FileChange[]; conflicts: string[] } {
    const agent = this.list([id])[0]!;
    if (!agent.write) throw new Error(`${id} is read-only; nothing to apply.`);
    if (!agent.finishedAt) throw new Error(`${id} is still ${agent.state}; wait for it first.`);
    const base = this.bases.get(id)!;
    const fork = join(agent.dir, "workspace");
    const wanted = files?.length ? new Set(files) : undefined;
    const applied: FileChange[] = [], conflicts: string[] = [];
    for (const change of agent.changes ?? []) {
      if (wanted && !wanted.has(change.path)) continue;
      const target = join(this.cwd, change.path);
      const current = existsSync(target) && statSync(target).isFile() ? sha(target) : existsSync(target) ? "other" : undefined;
      if (current !== base.get(change.path)) { conflicts.push(change.path); continue; }
      if (change.kind === "deleted") unlinkSync(target);
      else { mkdirSync(dirname(target), { recursive: true }); rmSync(target, { force: true }); cpSync(join(fork, change.path), target, { preserveTimestamps: true, verbatimSymlinks: true }); }
      applied.push(change);
    }
    agent.appliedFiles = [...new Set([...(agent.appliedFiles ?? []), ...applied.map(c => c.path)])];
    agent.applied = (agent.changes?.length ?? 0) > 0 && agent.changes!.every(c => agent.appliedFiles!.includes(c.path));
    writeFileSync(join(agent.dir, "result.json"), JSON.stringify(agent, null, 2) + "\n");
    this.changed(agent);
    return { applied, conflicts };
  }

  shutdown() { this.cancel(); }
}
