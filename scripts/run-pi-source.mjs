import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const source = resolve(process.env.RWKV_PI_SOURCE || fileURLToPath(new URL("../.upstream/pi", import.meta.url)));
const resolver = resolve(source, "packages/coding-agent/src/experimental/source-resolver.ts");
if (!existsSync(resolver)) throw new Error("Run npm run pi:setup before using pi source.");
const child = spawn(process.execPath, ["--import", pathToFileURL(resolver).href, ...process.argv.slice(2)], {
  stdio: "inherit", env: { ...process.env, RWKV_PI_SOURCE: source },
});
const forward = (signal) => child.kill(signal);
process.on("SIGINT", forward);
process.on("SIGTERM", forward);
child.once("error", (error) => { console.error(error.message); process.exitCode = 1; });
child.once("exit", (code, signal) => {
  process.removeListener("SIGINT", forward);
  process.removeListener("SIGTERM", forward);
  if (signal) process.kill(process.pid, signal);
  else process.exitCode = code ?? 1;
});
