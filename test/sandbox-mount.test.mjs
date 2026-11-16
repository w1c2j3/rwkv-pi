import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { mkdtemp, mkdir, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { WorkspaceSandbox } from "../dist/sandbox.js";

test("an unseen workspace path exposes one project, without familiar-path aliases", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "rwkv-unseen-mount-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const project = join(root, "project");
  await mkdir(project);
  await writeFile(join(project, "input.txt"), "current project\n");
  await writeFile(join(root, "unrelated.txt"), "other project\n");
  await symlink(join(root, "unrelated.txt"), join(project, "escape"));
  const mount = `/srv/run-${randomUUID()}/project with spaces`;
  const sandbox = new WorkspaceSandbox(project, undefined, mount);
  try { await sandbox.probe(); }
  catch (error) {
    if (process.env.REQUIRE_SANDBOX_TESTS === "1" || !/spawn bwrap ENOENT|Operation not permitted|No permissions to create/.test(String(error))) throw error;
    t.skip(String(error)); return;
  }
  assert.equal((await sandbox.readFile(`${mount}/input.txt`)).toString(), "current project\n");
  assert.equal((await sandbox.readFile("input.txt")).toString(), "current project\n");
  for (const path of ["/app/input.txt", "/workspace/input.txt", "/testbed/input.txt", "/srv/workspace/input.txt", join(root, "unrelated.txt"), `${mount}/escape`]) {
    await assert.rejects(sandbox.readFile(path), /ENOENT/);
  }
  const result = await sandbox.exec(["/bin/bash", "--noprofile", "--norc", "-c", "pwd; printf 'saved\\n' > output.txt"]);
  assert.equal(result.exitCode, 0, result.stderr.toString());
  assert.equal(result.stdout.toString(), `${mount}\n`);
  assert.equal(await readFile(join(project, "output.txt"), "utf8"), "saved\n");
});

test("custom mounts cannot cover system directories or conceal path traversal", () => {
  for (const path of ["/", "/usr", "/srv/../usr", "relative", "/srv/a/../../usr", "/srv/a\0b"]) {
    assert.throws(() => new WorkspaceSandbox("/tmp", undefined, path), /Custom workspace mount/);
  }
});
