import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { promisify } from "node:util";

const run = promisify(execFile);

/**
 * A task that runs inside a pinned container image (e.g. a SWE-smith repository environment).
 * The image supplies the system and toolchain; the agent sees the workspace at `workdir`.
 * `setupPatch` (applied once, then hidden behind a fresh single-commit history) turns the
 * image's clean checkout into the task's starting state.
 */
export interface TaskEnvironment {
  kind: "docker";
  image: string;
  /** sha256 of the image's layer diff_id list (canonical JSON): the same on every daemon. */
  rootfsDigest: string;
  workdir: string;
  path: string;
  setupPatch?: string;
}

export function parseEnvironment(value: unknown): TaskEnvironment {
  const env = value as Record<string, unknown>;
  if (typeof env !== "object" || env === null || env.kind !== "docker") throw new Error("Task environment must be {kind: 'docker', ...}.");
  for (const field of ["image", "rootfsDigest", "workdir", "path"] as const) {
    if (typeof env[field] !== "string" || !env[field]) throw new Error(`Task environment requires ${field}.`);
  }
  if (!/^sha256:[0-9a-f]{64}$/.test(String(env.rootfsDigest))) throw new Error("Task environment rootfsDigest must be sha256:<hex>.");
  if (!String(env.workdir).startsWith("/") || String(env.workdir) === "/") throw new Error("Task environment workdir must be an absolute directory.");
  if (env.setupPatch !== undefined && typeof env.setupPatch !== "string") throw new Error("Task environment setupPatch must be text.");
  return env as unknown as TaskEnvironment;
}

/**
 * Host path of a path this process sees. When the collector itself runs inside a container that
 * talks to the host Docker daemon, bind-mount sources must be host paths:
 * RWKV_DOCKER_PATH_MAP="/mnt/f=/srv/rwkv-pi-data;/opt/rwkv-pi=/srv/runtime" (longest prefix wins).
 */
export function dockerHostPath(path: string, map = process.env.RWKV_DOCKER_PATH_MAP ?? ""): string {
  const pairs = map.split(";").filter(Boolean).map((pair) => pair.split("=") as [string, string])
    .sort((a, b) => b[0].length - a[0].length);
  for (const [inner, host] of pairs) {
    if (path === inner || path.startsWith(inner.endsWith("/") ? inner : inner + "/")) return host + path.slice(inner.length);
  }
  return path;
}

/** Digest of an image's layer diff_ids as this daemon reports them. */
export async function rootfsDigest(image: string): Promise<string> {
  const { stdout } = await run("docker", ["image", "inspect", "--format", "{{json .RootFS.Layers}}", image]);
  return "sha256:" + createHash("sha256").update(JSON.stringify(JSON.parse(stdout))).digest("hex");
}

const GIT = ["-c", "user.name=workspace", "-c", "user.email=workspace@localhost", "-c", "commit.gpgsign=false"];

/** Copy the image's workdir into the empty host workspace and apply the task's setup patch. */
export async function materializeEnvironment(env: TaskEnvironment, cwd: string): Promise<void> {
  const actual = await rootfsDigest(env.image);
  if (actual !== env.rootfsDigest) throw new Error(`Image ${env.image} has layers ${actual}, task requires ${env.rootfsDigest}.`);
  const user = `${process.getuid?.() ?? 1000}:${process.getgid?.() ?? 1000}`;
  await run("docker", ["run", "--rm", "--network", "none", "--cap-drop", "ALL", "--user", user, "-v", `${dockerHostPath(cwd)}:/out`,
    "--entrypoint", "/bin/sh", env.image, "-c", `tar -C '${env.workdir}' -cf - . | tar -C /out -xf - --no-same-owner`],
  { maxBuffer: 16 * 1024 * 1024 });
  // The image's history contains the clean revision; the task starts from one opaque commit.
  await rm(join(cwd, ".git"), { recursive: true, force: true });
  if (env.setupPatch) {
    const patch = join(cwd, ".rwkv-setup.patch");
    await writeFile(patch, env.setupPatch);
    try { await run("git", ["apply", "--whitespace=nowarn", patch], { cwd }); }
    finally { await rm(patch, { force: true }); }
  }
  const date = { ...process.env, GIT_AUTHOR_DATE: "2025-01-01T00:00:00Z", GIT_COMMITTER_DATE: "2025-01-01T00:00:00Z" };
  await run("git", ["init", "-q", "-b", "main"], { cwd });
  await run("git", [...GIT, "add", "-A"], { cwd, maxBuffer: 64 * 1024 * 1024 });
  await run("git", [...GIT, "commit", "-q", "--no-verify", "-m", "Initial state"], { cwd, env: date });
}
