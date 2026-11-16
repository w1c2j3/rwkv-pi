import { spawn, type ChildProcessByStdio } from "node:child_process";
import { randomUUID } from "node:crypto";
import { existsSync, mkdtempSync, readdirSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { basename, delimiter, join, posix, resolve } from "node:path";
import type { Readable } from "node:stream";
import {
  createBashToolDefinition, createEditToolDefinition, createFindToolDefinition, createGrepToolDefinition,
  createLsToolDefinition, createReadToolDefinition, createWriteToolDefinition, type ToolDefinition,
} from "@earendil-works/pi-coding-agent";
import { type PiToolName, PI_TOOL_NAMES } from "./tools.js";
import { dockerHostPath, type TaskEnvironment } from "./environment.js";
import { LIST_FILES_SCRIPT, MAX_LISTED_FILES, SKIPPED_DIRECTORIES, type WorkspaceFs } from "./tolerant-tools.js";

/**
 * Where bubblewrap shows the workspace to the agent. A short fixed path keeps the long host run
 * directory (with its run UUID) out of the prompt and out of every path the model has to repeat.
 */
export const WORKSPACE_MOUNT = "/workspace";

type Spawned = ChildProcessByStdio<null, Readable, Readable>;
type SpawnHook = { spawn: (command: string, args: string[]) => Spawned };

/** Host binary pi itself would run (PATH, then pi's managed bin directory). */
function hostBinary(name: "rg" | "fd"): string | undefined {
  const names = name === "fd" ? ["fd", "fdfind"] : ["rg"];
  const dirs = [...(process.env.PATH ?? "").split(delimiter), join(process.env.PI_CODING_AGENT_DIR ?? join(homedir(), ".pi", "agent"), "bin")];
  for (const dir of dirs) for (const candidate of names) {
    const path = join(dir, candidate);
    if (dir && existsSync(path)) return realpathSync(path);
  }
  return undefined;
}

/**
 * Native pi schemas/rendering, with file and process operations isolated from the host.
 *
 * Without an environment every call runs in a fresh bubblewrap namespace over the host's
 * read-only system directories. With a task environment every call runs in a fresh
 * `docker run --rm` container of the pinned image instead (no network, no capabilities,
 * read-only root, the caller's uid), with the workspace mounted at the image's workdir;
 * `cwd` is then that in-container path. Without one the workspace is mounted at /workspace.
 * Both give the same semantics: only the workspace persists between calls, and a call's
 * processes end with it.
 */
export class WorkspaceSandbox {
  readonly kind: "bubblewrap" | "docker";
  readonly cwd: string;
  private readonly prefix: string[];
  private readonly searchBinaries = new Set<string>();
  private readonly statCache = new Map<string, boolean>();
  constructor(hostCwd: string, private readonly environment?: TaskEnvironment, workspaceMount = WORKSPACE_MOUNT) {
    // A caller can test an unseen path without aliases for paths remembered from training.
    // Keep custom mounts away from the system directories needed by native tools.
    if (workspaceMount !== WORKSPACE_MOUNT && (!workspaceMount.startsWith("/srv/") ||
        posix.normalize(workspaceMount) !== workspaceMount || workspaceMount.includes("\0"))) {
      throw new Error("Custom workspace mount must be a normalized absolute path under /srv/.");
    }
    if (environment && workspaceMount !== WORKSPACE_MOUNT) throw new Error("Container tasks use their image's workdir.");
    this.kind = environment ? "docker" : "bubblewrap";
    this.cwd = environment ? environment.workdir : workspaceMount;
    if (environment) { this.prefix = this.dockerPrefix(hostCwd, environment); return; }
    this.prefix = ["--unshare-all", "--die-with-parent", "--new-session", "--cap-drop", "ALL"];
    // /etc/alternatives holds the targets of Debian alternatives links such as awk, cc and java;
    // the JDK reads its security/config files through /etc/java-*-openjdk. Both are read-only system config.
    const jdkConfig = readdirSync("/etc").filter((name) => /^java-\d+-openjdk$/.test(name)).map((name) => `/etc/${name}`);
    for (const path of ["/usr", "/bin", "/lib", "/lib64", "/etc/alternatives", ...jdkConfig]) {
      if (existsSync(path)) this.prefix.push("--ro-bind", path, path);
    }
    this.prefix.push(
      "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--dir", "/home/sandbox",
      "--dir", "/opt/rwkv-pi/bin", "--ro-bind", process.execPath, "/opt/rwkv-pi/bin/node",
    );
    // Search tools run the same rg/fd binaries pi would use, but inside the namespace.
    for (const name of ["rg", "fd"] as const) {
      const binary = hostBinary(name);
      if (binary) { this.prefix.push("--ro-bind", binary, `/opt/rwkv-pi/bin/${name}`); this.searchBinaries.add(name); }
    }
    this.prefix.push(
      "--bind", resolve(hostCwd), this.cwd, "--chdir", this.cwd, "--clearenv",
      "--setenv", "HOME", "/home/sandbox", "--setenv", "PATH", "/opt/rwkv-pi/bin:/usr/local/bin:/usr/bin:/bin",
      "--setenv", "LANG", "C.UTF-8", "--",
    );
  }

  private dockerPrefix(hostCwd: string, env: TaskEnvironment): string[] {
    const prefix = ["run", "--rm", "-i", "--network", "none", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
      "--read-only", "--tmpfs", "/tmp:rw,exec,size=2g", "--tmpfs", "/home/sandbox:rw,size=256m", "--pids-limit", "1024",
      "--memory", "8g", "--user", `${process.getuid?.() ?? 1000}:${process.getgid?.() ?? 1000}`, "--entrypoint", "",
      "-v", `${dockerHostPath(resolve(hostCwd))}:${env.workdir}`, "-w", env.workdir,
      "-v", `${dockerHostPath(process.execPath)}:/opt/rwkv-pi/bin/node:ro`];
    for (const name of ["rg", "fd"] as const) {
      const binary = hostBinary(name);
      if (binary) { prefix.push("-v", `${dockerHostPath(binary)}:/opt/rwkv-pi/bin/${name}:ro`); this.searchBinaries.add(name); }
    }
    prefix.push("-e", "HOME=/home/sandbox", "-e", `PATH=/opt/rwkv-pi/bin:${env.path}`, "-e", "LANG=C.UTF-8");
    return prefix;
  }

  /** The launcher argv for one isolated call, and how to end it early. */
  private launch(argv: string[]): { command: string; args: string[]; env: NodeJS.ProcessEnv; kill: () => void } {
    if (!this.environment) return { command: "bwrap", args: [...this.prefix, ...argv], env: { PATH: "/usr/bin:/bin" }, kill: () => {} };
    // Killing the docker client does not stop its container; stop it by name.
    const name = `rwkv-pi-${randomUUID()}`;
    const kill = () => { spawn("docker", ["kill", name], { stdio: "ignore", detached: true }).on("error", () => {}).unref(); };
    const env = { PATH: "/usr/bin:/bin", ...(process.env.HOME ? { HOME: process.env.HOME } : {}), ...(process.env.DOCKER_HOST ? { DOCKER_HOST: process.env.DOCKER_HOST } : {}) };
    return { command: "docker", args: [...this.prefix, "--name", name, this.environment.image, ...argv], env, kill };
  }

  async exec(argv: string[], options: {
    signal?: AbortSignal; timeoutMs?: number; onData?: (data: Buffer) => void; input?: string;
  } = {}): Promise<{ exitCode: number; stdout: Buffer; stderr: Buffer }> {
    if (options.signal?.aborted) throw new Error("Sandbox execution aborted.");
    return new Promise((accept, reject) => {
      const launched = this.launch(argv);
      const child = spawn(launched.command, launched.args, { stdio: "pipe", detached: true, env: launched.env });
      const stdout: Buffer[] = [], stderr: Buffer[] = [];
      let size = 0, failure: Error | undefined;
      const stop = (reason: string) => {
        failure ??= new Error(reason);
        launched.kill();
        if (child.pid) { try { process.kill(-child.pid, "SIGKILL"); } catch { /* Already exited. */ } }
      };
      const abort = () => stop("Sandbox execution aborted.");
      const timer = setTimeout(() => stop("Sandbox command timed out."), options.timeoutMs ?? 30_000);
      options.signal?.addEventListener("abort", abort, { once: true });
      const collect = (target: Buffer[], chunk: Buffer) => {
        size += chunk.length;
        if (size > 16 * 1024 * 1024) { stop("Sandbox output exceeded 16 MiB."); return; }
        target.push(chunk);
        options.onData?.(chunk);
      };
      child.stdout.on("data", (chunk: Buffer) => collect(stdout, chunk));
      child.stderr.on("data", (chunk: Buffer) => collect(stderr, chunk));
      child.once("error", (error) => { failure = error; });
      child.stdin.on("error", () => {});
      child.stdin.end(options.input);
      child.once("close", (code) => {
        clearTimeout(timer);
        options.signal?.removeEventListener("abort", abort);
        if (failure) reject(failure);
        else accept({ exitCode: code ?? 137, stdout: Buffer.concat(stdout), stderr: Buffer.concat(stderr) });
      });
    });
  }

  async checked(argv: string[], input?: string): Promise<Buffer> {
    const result = await this.exec(argv, { input });
    if (result.exitCode !== 0) throw new Error(result.stderr.toString() || `Sandbox command exited ${result.exitCode}.`);
    return result.stdout;
  }

  async probe(): Promise<void> {
    try { await this.checked(["/opt/rwkv-pi/bin/node", "-e", "process.stdout.write('ready')"]); }
    catch (error) { throw new Error(`${this.kind} isolation unavailable; no fallback to host tools. ${String(error)}`); }
  }

  readFile = (path: string) => this.checked(["/opt/rwkv-pi/bin/node", "-e",
    "process.stdout.write(require('node:fs').readFileSync(process.argv[1]))", path]);
  private access = async (path: string) => { await this.checked(["/opt/rwkv-pi/bin/node", "-e",
    "require('node:fs').accessSync(process.argv[1])", path]); };
  private writeFile = async (path: string, content: string) => { await this.checked(["/opt/rwkv-pi/bin/node", "-e",
    "const fs=require('node:fs'); fs.writeFileSync(process.argv[1],fs.readFileSync(0))", path], content); };
  private mkdir = async (path: string) => { await this.checked(["/opt/rwkv-pi/bin/node", "-e",
    "require('node:fs').mkdirSync(process.argv[1],{recursive:true})", path]); };

  /** Start rg/fd inside the namespace; pi passes the command name it resolved on the host. */
  private spawnSearch = (command: string, args: string[]): Spawned => {
    const name = basename(command) === "fdfind" ? "fd" : basename(command);
    if (!this.searchBinaries.has(name)) throw new Error(`${name} is not available inside the sandbox.`);
    const launched = this.launch([`/opt/rwkv-pi/bin/${name}`, ...args]);
    const child = spawn(launched.command, launched.args, { stdio: ["ignore", "pipe", "pipe"], env: launched.env }) as Spawned;
    child.once("exit", (_code, signal) => { if (signal) launched.kill(); });
    return child;
  };

  private isDirectory = async (path: string): Promise<boolean> => {
    const out = await this.checked(["/opt/rwkv-pi/bin/node", "-e",
      "process.stdout.write(String(require('node:fs').statSync(process.argv[1]).isDirectory()))", path]);
    return out.toString() === "true";
  };

  /** One namespace call per listing; entry types are cached for ls's per-entry stat calls. */
  private readdir = async (path: string): Promise<string[]> => {
    const out = await this.checked(["/opt/rwkv-pi/bin/node", "-e", `const fs=require('node:fs'),p=require('node:path');
      const d=process.argv[1];process.stdout.write(JSON.stringify(fs.readdirSync(d).map(n=>{let dir=false;
      try{dir=fs.statSync(p.join(d,n)).isDirectory()}catch{}return [n,dir]})))`, path]);
    const entries = JSON.parse(out.toString()) as [string, boolean][];
    this.statCache.clear();
    for (const [name, dir] of entries) this.statCache.set(join(path, name), dir);
    return entries.map(([name]) => name);
  };

  private exists = async (path: string): Promise<boolean> => {
    try { await this.access(path); return true; } catch { return false; }
  };

  private listFiles = async (root: string): Promise<string[]> =>
    JSON.parse((await this.checked(["/opt/rwkv-pi/bin/node", "-e", LIST_FILES_SCRIPT, root,
      JSON.stringify(SKIPPED_DIRECTORIES), String(MAX_LISTED_FILES)])).toString()) as string[];

  /** The workspace as the agent sees it, for the recall-tolerant tool wrappers. */
  readonly fs: WorkspaceFs = { readFile: (path) => this.readFile(path), exists: (path) => this.exists(path), listFiles: (root) => this.listFiles(root) };

  /** Fail closed when the loaded pi ignores the spawn hook (it would search the host instead). */
  async probeSearch(names: readonly PiToolName[]): Promise<void> {
    const needed = names.filter((name): name is "grep" | "find" => name === "grep" || name === "find");
    if (!needed.length) return;
    const dir = mkdtempSync(join(tmpdir(), "rwkv-pi-search-probe-"));
    try {
      writeFileSync(join(dir, "probe.txt"), "needle\n");
      for (const name of needed) {
        if (!this.searchBinaries.has(name === "grep" ? "rg" : "fd")) throw new Error(`${name} requires ${name === "grep" ? "rg" : "fd"} on the host.`);
        let called = false;
        const hook: SpawnHook = { spawn: (command, args) => { called = true; return spawn(command, args, { stdio: ["ignore", "pipe", "pipe"] }) as Spawned; } };
        const definition = name === "grep"
          ? createGrepToolDefinition(dir, { operations: { isDirectory: async () => true, readFile: async () => "", ...hook } as never })
          : createFindToolDefinition(dir, { operations: { exists: async () => true, glob: async () => [], ...hook } as never });
        const params = name === "grep" ? { pattern: "needle" } : { pattern: "*.txt" };
        await (definition.execute as (...a: unknown[]) => Promise<unknown>)("probe", params, undefined, undefined, undefined);
        if (!called) throw new Error(`The loaded pi ${name} tool ignores operations.spawn; run with the patched pi source (npm run pi:setup, *:source scripts).`);
      }
    } finally { rmSync(dir, { recursive: true, force: true }); }
  }

  tools(names: readonly PiToolName[] = ["read", "write", "edit", "bash"]): ToolDefinition[] {
    const search: SpawnHook = { spawn: this.spawnSearch };
    const make: Record<PiToolName, () => unknown> = {
      read: () => createReadToolDefinition(this.cwd, { operations: { readFile: this.readFile, access: this.access, detectImageMimeType: async () => null } }),
      write: () => createWriteToolDefinition(this.cwd, { operations: { writeFile: this.writeFile, mkdir: this.mkdir } }),
      edit: () => createEditToolDefinition(this.cwd, { operations: { readFile: this.readFile, writeFile: this.writeFile, access: this.access } }),
      bash: () => createBashToolDefinition(this.cwd, { exposeSessionEnvironment: false, operations: {
        exec: async (command, _cwd, options) => this.exec(["/bin/bash", "--noprofile", "--norc", "-c", command], {
          signal: options.signal, timeoutMs: (options.timeout ?? 60) * 1000, onData: options.onData,
        }),
      } }),
      // spawn is the patched-pi hook (patches/pi); the published 1.5.0 types do not declare it.
      grep: () => createGrepToolDefinition(this.cwd, { operations: {
        isDirectory: this.isDirectory, readFile: async (path: string) => (await this.readFile(path)).toString("utf8"), ...search,
      } as never }),
      find: () => createFindToolDefinition(this.cwd, { operations: { exists: this.exists, glob: async () => [], ...search } as never }),
      ls: () => createLsToolDefinition(this.cwd, { operations: {
        exists: this.exists, readdir: this.readdir,
        stat: async (path: string) => {
          // Entries from the listing just read are used once; anything else is a fresh namespace stat.
          const cached = this.statCache.get(path);
          this.statCache.delete(path);
          const dir = cached ?? await this.isDirectory(path);
          return { isDirectory: () => dir };
        },
      } }),
    };
    for (const name of names) if (!PI_TOOL_NAMES.includes(name)) throw new Error(`Unsupported sandbox tool: ${name}`);
    return PI_TOOL_NAMES.filter((name) => names.includes(name)).map((name) => make[name]()) as unknown as ToolDefinition[];
  }
}
