import assert from "node:assert/strict";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import test from "node:test";
import { createFindToolDefinition, createGrepToolDefinition, createLsToolDefinition } from "@earendil-works/pi-coding-agent";
import { WorkspaceSandbox } from "../dist/sandbox.js";
import { parseToolList } from "../dist/tools.js";

const run = async (definition, params) => {
  try {
    const result = await definition.execute("t", params, undefined, undefined, undefined);
    return result.content.map((part) => part.text).join("");
  } catch (error) { return `ERROR ${error.message}`; }
};

test("tool lists normalise to the order shared by the CLI and harness", () => {
  assert.deepEqual(parseToolList("ls,bash,read,grep"), ["read", "bash", "grep", "ls"]);
  assert.deepEqual(parseToolList(["read", "write", "edit", "bash"]), ["read", "write", "edit", "bash"]);
  assert.throws(() => parseToolList("read,powershell"), /Unsupported tools: powershell/);
  assert.throws(() => parseToolList("read,read"), /repeats/);
});

test("sandboxed grep/find/ls match pi's host output and cannot read outside the workspace", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "rwkv-pi-search-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const cwd = join(root, "workspace");
  await mkdir(join(cwd, "src", "pkg"), { recursive: true });
  await writeFile(join(cwd, "README.md"), "Project notes\nneedle in readme\n");
  await writeFile(join(cwd, "src", "pkg", "core.py"), "def needle():\n    return 1\n");
  await writeFile(join(cwd, "src", "app.py"), "from pkg.core import needle\n");
  await writeFile(join(root, "secret.txt"), "needle outside the workspace\n");
  const sandbox = new WorkspaceSandbox(cwd);
  try { await sandbox.probe(); await sandbox.probeSearch(["grep", "find"]); }
  catch (error) {
    if (process.env.REQUIRE_SANDBOX_TESTS === "1" || !/spawn bwrap ENOENT|Operation not permitted|No permissions to create|requires (rg|fd)/.test(String(error))) throw error;
    t.skip(String(error)); return;
  }
  const [grep, find, ls] = sandbox.tools(["grep", "find", "ls"]);
  assert.deepEqual([grep.name, find.name, ls.name], ["grep", "find", "ls"]);
  const host = { grep: createGrepToolDefinition(cwd), find: createFindToolDefinition(cwd), ls: createLsToolDefinition(cwd) };
  for (const [name, params] of [
    ["grep", { pattern: "needle" }], ["grep", { pattern: "NEEDLE", ignoreCase: true, context: 1 }], ["grep", { pattern: "needle", path: "src", glob: "*.py" }],
    ["find", { pattern: "*.py" }], ["find", { pattern: "src/**/*.py" }], ["ls", {}], ["ls", { path: "src" }],
  ]) {
    const sandboxed = await run({ grep, find, ls }[name], params);
    const expected = await run(host[name], params);
    // rg may visit files in a different order on each parallel traversal. Every line here carries
    // its path and line number, so compare those exact observations without requiring file order.
    if (name === "grep") assert.deepEqual(sandboxed.split("\n").sort(), expected.split("\n").sort(), `${name} ${JSON.stringify(params)}`);
    else assert.equal(sandboxed, expected, `${name} ${JSON.stringify(params)}`);
    assert.doesNotMatch(sandboxed, /^ERROR/, sandboxed);
  }
  // The host tools can see the secret; the sandboxed ones cannot.
  assert.match(await run(host.grep, { pattern: "needle", path: root }), /secret\.txt/);
  const leaked = await run(grep, { pattern: "needle", path: root });
  assert.doesNotMatch(leaked, /outside the workspace/);
  assert.doesNotMatch(await run(grep, { pattern: "root", path: "/etc" }), /passwd/);
  assert.doesNotMatch(await run(find, { pattern: "secret.txt", path: root }), /secret\.txt/);
  // bwrap creates the workspace's parent directories empty: only the workspace itself is visible.
  assert.equal(await run(ls, { path: root }), "workspace/");
  assert.match(await run(host.ls, { path: root }), /secret\.txt/);
});
