import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { readdirSync, readFileSync } from "node:fs";
import { basename, dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export function digest(value: unknown): string {
  return sha256(JSON.stringify(value));
}

export function sha256(value: string | Buffer): string {
  return createHash("sha256").update(value).digest("hex");
}

function hashTree(root: string): string {
  const hash = createHash("sha256");
  function walk(directory: string): void {
    for (const entry of readdirSync(directory, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
      const path = join(directory, entry.name);
      if (entry.isDirectory()) walk(path);
      else if (entry.isFile()) hash.update(relative(root, path)).update("\0").update(readFileSync(path)).update("\0");
    }
  }
  walk(root);
  return hash.digest("hex");
}

let cachedProvenance: { key: string; value: ReturnType<typeof computeRuntimeProvenance> } | undefined;

function computeRuntimeProvenance() {
  const root = fileURLToPath(new URL("../", import.meta.url));
  const piEntry = fileURLToPath(import.meta.resolve("@earendil-works/pi-coding-agent"));
  const piRoot = dirname(dirname(piEntry));
  const pkg = JSON.parse(readFileSync(join(piRoot, "package.json"), "utf8")) as { version: string };
  const source = basename(dirname(piEntry)) === "src" ? resolve(piRoot, "../..") : undefined;
  if (process.env.RWKV_PI_SOURCE && (!source || source !== resolve(process.env.RWKV_PI_SOURCE))) {
    throw new Error("RWKV_PI_SOURCE does not match the loaded pi runtime. Use npm run harness:source (or smoke:source).");
  }
  return {
    node: process.version,
    piVersion: pkg.version,
    piMode: source ? "source" : "npm",
    piEntry,
    piRevision: source ? execFileSync("git", ["-C", source, "rev-parse", "HEAD"], { encoding: "utf8" }).trim() : null,
    // Include actual source bytes, including local edits and generated model catalogs.
    piRuntimeSha256: source
      ? digest(["agent", "ai", "coding-agent", "schema", "telemetry", "tui", "codemode", "mcp"].map((name) => [name, hashTree(join(source, "packages", name, "src"))]))
      : sha256(readFileSync(join(root, "package-lock.json"))),
    harnessSha256: hashTree(dirname(fileURLToPath(import.meta.url))),
    dependencyLockSha256: sha256(readFileSync(join(root, "package-lock.json"))),
  };
}

export function runtimeProvenance() {
  const key = process.env.RWKV_PI_SOURCE ?? "";
  if (!cachedProvenance || cachedProvenance.key !== key) {
    cachedProvenance = { key, value: computeRuntimeProvenance() };
  }
  return { ...cachedProvenance.value };
}
