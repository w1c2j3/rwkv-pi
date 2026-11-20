import assert from "node:assert/strict";
import test from "node:test";
import { execFile } from "node:child_process";
import { once } from "node:events";
import { createServer } from "node:http";
import { mkdtemp, rm, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { applyProfile, decodeToken, encodeToken, readProfile, verifyProfile, writeProfile } from "../dist/profile.js";

const profile = { url: "http://10.0.0.2:8000/v1/", key: "secret", model: "rwkv7-g1l", adapter: { id: "nora-cot", scale: 1 }, maxTokens: 8192, name: "157" };
const cli = new URL("../dist/cli.js", import.meta.url).pathname;

test("connection tokens round-trip and reject damage", () => {
  const token = encodeToken(profile);
  assert.match(token, /^rwkvpi1\.[A-Za-z0-9_-]+$/);
  assert.deepEqual(decodeToken(` ${token}\n`), { ...profile, url: "http://10.0.0.2:8000/v1" });
  assert.throws(() => decodeToken("sk-123"), /starts with rwkvpi1\./);
  assert.throws(() => decodeToken(token.slice(0, -6)), /damaged|Invalid/);
  assert.throws(() => encodeToken({ ...profile, url: "ftp://x/v1" }), /http/);
  assert.throws(() => encodeToken({ ...profile, key: "" }), /key/);
  assert.throws(() => encodeToken({ ...profile, maxTokens: 0 }), /maxTokens/);
});

test("profile fills only unset variables", () => {
  const env = { RWKV_MODEL: "mine", RWKV_API_KEY_FILE: "/k" };
  applyProfile(decodeToken(encodeToken(profile)), env);
  assert.equal(env.RWKV_BASE_URL, "http://10.0.0.2:8000/v1");
  assert.equal(env.RWKV_MODEL, "mine");
  assert.equal(env.RWKV_API_KEY, undefined, "a key file wins over the profile key");
  assert.equal(env.RWKV_ADAPTER_ID, "nora-cot");
  assert.equal(env.RWKV_ADAPTER_SCALE, "1");
  assert.equal(env.RWKV_MAX_TOKENS, "8192");
});

test("saved profile is private and readable", async () => {
  const dir = await mkdtemp(join(tmpdir(), "rwkv-profile-"));
  try {
    const env = { RWKV_PI_CONFIG_DIR: join(dir, "cfg") };
    const path = writeProfile(profile, env);
    assert.equal((await stat(path)).mode & 0o777, 0o600);
    assert.equal(readProfile(env).key, "secret");
  } finally { await rm(dir, { recursive: true, force: true }); }
});

test("verification distinguishes rejected tokens from unreachable servers", async () => {
  const server = createServer((req, res) => {
    const ok = req.headers.authorization === "Bearer secret";
    res.writeHead(ok ? 200 : 401, { "content-type": "application/json" });
    res.end(JSON.stringify(ok ? { data: [{ id: "rwkv7-g1l" }] } : { error: "no" }));
  }).listen(0, "127.0.0.1");
  await once(server, "listening");
  const url = `http://127.0.0.1:${server.address().port}/v1`;
  try {
    assert.deepEqual(await verifyProfile({ url, key: "secret" }), ["rwkv7-g1l"]);
    await assert.rejects(verifyProfile({ url, key: "wrong" }), /rejected this token/);
    await assert.rejects(verifyProfile({ url: "http://127.0.0.1:9/v1", key: "secret" }), /Cannot reach/);
  } finally { server.close(); }
});

test("login, status and logout through the launcher", async () => {
  const server = createServer((req, res) => {
    res.writeHead(req.headers.authorization === "Bearer secret" ? 200 : 401, { "content-type": "application/json" });
    res.end(JSON.stringify({ data: [{ id: "rwkv7-g1l" }] }));
  }).listen(0, "127.0.0.1");
  await once(server, "listening");
  const dir = await mkdtemp(join(tmpdir(), "rwkv-login-"));
  const env = { ...process.env, RWKV_PI_CONFIG_DIR: dir, RWKV_BASE_URL: "" };
  const run = (...args) => new Promise(resolve => {
    const child = execFile(process.execPath, [cli, ...args], { cwd: dir, env, encoding: "utf8" },
      (error, stdout, stderr) => resolve({ status: error ? error.code : 0, stdout, stderr }));
    child.stdin.end();
  });
  try {
    const token = encodeToken({ ...profile, url: `http://127.0.0.1:${server.address().port}/v1` });
    const bad = await run("login", encodeToken({ ...profile, url: `http://127.0.0.1:${server.address().port}/v1`, key: "x" }));
    assert.equal(bad.status, 1);
    assert.match(bad.stderr, /rejected this token/);
    const ok = await run("login", token);
    assert.equal(ok.status, 0, ok.stderr);
    assert.match(ok.stdout, /Connected: 157/);
    assert.match((await run("status")).stdout, /serves rwkv7-g1l/);
    assert.match((await run("logout")).stdout, /Removed/);
    const first = await run("hello");
    assert.equal(first.status, 1);
    assert.match(first.stderr, /No connection configured\. Run: rwkv-pi login <token>/);
  } finally {
    server.close();
    await rm(dir, { recursive: true, force: true });
  }
});
