import { execFileSync } from "node:child_process";
import { cpSync, existsSync, mkdirSync, readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
const lock = JSON.parse(readFileSync(join(root, "pi-source.lock.json"), "utf8"));
const checkout = join(root, ".upstream/pi");
const git = (args) => execFileSync("git", args, { cwd: root, encoding: "utf8" }).trim();
if (!existsSync(checkout)) {
  mkdirSync(dirname(checkout), { recursive: true });
  git(["clone", "--depth", "1", "--branch", lock.tag, lock.repository, checkout]);
  git(["-C", checkout, "switch", "-c", "rwkv-pi"]);
}
const revision = git(["-C", checkout, "rev-parse", "HEAD"]);
if (revision !== lock.commit) throw new Error("pi checkout differs from pi-source.lock.json; refusing to change an existing checkout.");
const patches = join(root, "patches/pi");
if (existsSync(patches)) {
  for (const name of readdirSync(patches).filter((name) => name.endsWith(".patch")).sort()) {
    const path = join(patches, name);
    try { git(["-C", checkout, "apply", "--reverse", "--check", path]); }
    catch {
      git(["-C", checkout, "apply", "--check", path]);
      git(["-C", checkout, "apply", path]);
    }
  }
}

// Upstream excludes generated catalog JSON from Git. Use the same pinned npm release.
const aiRoot = dirname(dirname(fileURLToPath(import.meta.resolve("@earendil-works/pi-ai"))));
const aiPackage = JSON.parse(readFileSync(join(aiRoot, "package.json"), "utf8"));
if (aiPackage.version !== lock.version) throw new Error("Installed pi-ai version differs from the source lock.");
cpSync(join(aiRoot, "dist/providers/data"), join(checkout, "packages/ai/src/providers/data"), {
  recursive: true, force: false,
});
console.log(`pi source ready: ${checkout}\nrevision: ${revision}\nLocal source changes are preserved.`);
