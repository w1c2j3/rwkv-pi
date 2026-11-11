import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import test from "node:test";
import { WorkspaceSandbox } from "../dist/sandbox.js";
import { runTask } from "../dist/harness.js";
import { createRwkvProvider, readConfig } from "../dist/provider.js";
import { copyFileResponses, mockServer } from "./helpers/mock-server.mjs";

test("native pi tools and verification run with host files, credentials, processes, and network isolated", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "rwkv-pi-isolation-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const cwd = join(root, "workspace");
  await mkdir(cwd);
  const sandbox = new WorkspaceSandbox(cwd);
  try { await sandbox.probe(); }
  catch (error) {
    if (process.env.REQUIRE_SANDBOX_TESTS === "1" || !/spawn bwrap ENOENT|Operation not permitted|No permissions to create/.test(String(error))) throw error;
    t.skip(String(error)); return;
  }
  await writeFile(join(root, "secret"), "must stay outside");
  await symlink(join(root, "secret"), join(cwd, "escape"));
  await assert.rejects(sandbox.readFile(join(root, "secret")), /ENOENT/);
  assert.equal(sandbox.cwd, "/workspace");
  await assert.rejects(sandbox.readFile(join(sandbox.cwd, "escape")), /ENOENT/);
  const isolated = await sandbox.exec(["node", "-e", `
    const fs=require('node:fs');
    if (Object.keys(process.env).some(k => /KEY|TOKEN|SSH|CODEX/.test(k))) process.exit(1);
    if (fs.existsSync('/proc/${process.pid}')) process.exit(2);
    require('node:net').connect(9,'127.0.0.1').on('connect',()=>process.exit(3)).on('error',()=>console.log('isolated'));
  `]);
  assert.equal(isolated.exitCode, 0, isolated.stderr.toString());
  assert.equal(isolated.stdout.toString(), "isolated\n");
  const server = await mockServer(copyFileResponses);
  t.after(server.close);
  const { record, path } = await runTask({
    schemaVersion: 1, id: "sandbox-copy", family: "test-fixture", split: "smoke", source: "mock test only",
    prompt: "Read input.txt and write its uppercase contents to output.txt.", files: { "input.txt": "hello pi\n" },
    tools: ["read", "write", "edit", "bash"],
    checks: [{ type: "file_equals", path: "output.txt", expected: "HELLO PI\n" },
      { type: "command", argv: ["node", "-e", "process.stdout.write(require('node:fs').readFileSync('output.txt'))"], timeoutMs: 2000, expectedStdout: "HELLO PI\n" }],
  }, { provider: createRwkvProvider(readConfig({ RWKV_BACKEND: "openai", RWKV_BASE_URL: server.url })), role: "smoke", outputDir: join(root, "runs"), maxRequests: 4, timeoutMs: 10_000 });
  assert.equal(record.status, "passed", JSON.stringify(record.checks));
  assert.equal(JSON.parse(await readFile(path, "utf8")).isolation, "bubblewrap");
  await assert.rejects(sandbox.exec(["bash", "-c", "sleep 30"], { timeoutMs: 50 }), /timed out/);
});

test("Debian alternatives (awk, cc, javac) resolve inside the sandbox", async (t) => {
  const cwd = await mkdtemp(join(tmpdir(), "rwkv-pi-alternatives-"));
  t.after(() => rm(cwd, { recursive: true, force: true }));
  const sandbox = new WorkspaceSandbox(cwd);
  try { await sandbox.probe(); }
  catch (error) { t.skip(String(error)); return; }
  const awk = await sandbox.exec(["sh", "-c", "echo 2 3 | awk '{print $1 * $2}'"]);
  assert.equal(awk.stdout.toString().trim(), "6", awk.stderr.toString());
  const { existsSync } = await import("node:fs");
  if (existsSync("/etc/alternatives/cc")) assert.equal((await sandbox.exec(["sh", "-c", "command -v cc"])).exitCode, 0);
  if (existsSync("/etc/alternatives/javac")) {
    await writeFile(join(cwd, "A.java"), 'public class A { public static void main(String[] a) { System.out.println("java ok"); } }');
    const java = await sandbox.exec(["sh", "-c", "javac A.java && java A"], { timeoutMs: 60_000 });
    assert.equal(java.stdout.toString().trim(), "java ok", java.stderr.toString());
  }
});
