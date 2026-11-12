import assert from "node:assert/strict";
import { copyFile, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { PACKAGE_ROOT, runDoctor } from "../dist/doctor.js";
import { dataRoot, requireDataRoot } from "../dist/config.js";

async function bareRoot(t) {
  const root = await mkdtemp(join(tmpdir(), "rwkv-pi-doctor-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  for (const lock of ["pi-source.lock.json", "lightning-source.lock.json"]) await copyFile(join(PACKAGE_ROOT, lock), join(root, lock));
  return root;
}

const byName = checks => Object.fromEntries(checks.map(check => [check.name, check]));

test("a bare checkout reports each missing setup step with the command that fixes it", async t => {
  const root = await bareRoot(t);
  const checks = byName(await runDoctor({ root, offline: true, env: { RWKV_PI_DATA_ROOT: root } }));
  assert.equal(checks.node.status, "ok");
  assert.match(checks["pi source"].detail, /npm run pi:setup/);
  assert.match(checks["native tokenizer"].detail, /npm run lightning:setup/);
  assert.equal(checks.config.status, "ok");
  assert.equal(checks["data root"].status, "ok");
  assert.equal(checks.engine, undefined);
});

test("RWKV must use the Lightning backend and therefore its native tokenizer", async t => {
  const root = await bareRoot(t);
  const checks = byName(await runDoctor({ root, offline: true, env: { RWKV_BACKEND: "openai", RWKV_PI_DATA_ROOT: root } }));
  assert.equal(checks.config.status, "fail");
  assert.match(checks.config.detail, /Lightning-native tokenizer/);
});

test("engine checks read /v1/models and the served token count", async t => {
  const root = await bareRoot(t);
  const seen = [];
  const fetch = async (url, init) => {
    seen.push(String(url));
    if (String(url).endsWith("/models")) return Response.json({ data: [{ id: "rwkv7" }] });
    assert.equal(typeof JSON.parse(init.body).text, "string");
    return Response.json({ count: 42 });
  };
  const checks = byName(await runDoctor({ root, fetch, env: { RWKV_BASE_URL: "http://engine.test/v1", RWKV_PI_DATA_ROOT: root } }));
  assert.deepEqual(seen, ["http://engine.test/v1/models", "http://engine.test/v1/tokens/count"]);
  assert.equal(checks.engine.status, "ok");
  assert.equal(checks["tokenizer parity"].status, "ok");
  assert.match(checks["tokenizer parity"].detail, /= 42/);
});

test("data root: portable default, and an explicit root that is missing never falls back", async t => {
  const root = await bareRoot(t);
  assert.equal(dataRoot({ XDG_DATA_HOME: root }), join(root, "rwkv-pi"));
  assert.equal(requireDataRoot({ XDG_DATA_HOME: root }), join(root, "rwkv-pi"));
  assert.throws(() => requireDataRoot({ RWKV_PI_DATA_ROOT: join(root, "unmounted") }), /not available/);
});
