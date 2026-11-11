import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { execFileSync } from "node:child_process";
const lock = JSON.parse(await readFile("lightning-source.lock.json", "utf8"));
const root = ".upstream/rwkv-lightning";
if (!existsSync(root)) {
  execFileSync("git", ["clone", "--no-checkout", "--filter=blob:none", lock.repository, root], { stdio: "inherit" });
  execFileSync("git", ["-C", root, "checkout", "--detach", lock.revision], { stdio: "inherit" });
}
const actual = execFileSync("git", ["-C", root, "rev-parse", "HEAD"], { encoding: "utf8" }).trim();
if (actual !== lock.revision) throw Error("Existing Lightning checkout differs; preserve local work and select the pinned revision explicitly.");
execFileSync(process.execPath, ["scripts/build-lightning-tokenizer.mjs"], { stdio: "inherit" });
console.log("Pinned source and native CPU tokenizer ready. The CUDA package is installed separately; see README.md (Engine installation).");
