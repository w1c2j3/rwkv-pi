import assert from "node:assert/strict";
import test from "node:test";
import { readConfig } from "../dist/config.js";

test("rejects invalid token budgets before starting a model request", () => {
  for (const value of ["-1", "0", "NaN", "1.5", "9007199254740993"]) {
    assert.throws(() => readConfig({ RWKV_MAX_TOKENS: value }), /positive integer/);
  }
  assert.throws(() => readConfig({ RWKV_CONTEXT_WINDOW: "4096" }), /leave room for the prompt/);
});

test("rejects incorrect API URLs without echoing embedded credentials", () => {
  for (const url of ["invalid", "file:///tmp/model", "https://user:secret@example.com/v1", "https://example.com/v1?key=secret", "http://localhost:8000/v1/chat/completions"]) {
    assert.throws(() => readConfig({ RWKV_BASE_URL: url }), (error) => {
      assert.match(error.message, /RWKV_BASE_URL/);
      assert.doesNotMatch(error.message, /secret/);
      return true;
    });
  }
  assert.equal(readConfig({ RWKV_BASE_URL: "http://localhost:8000/v1/" }).baseUrl, "http://localhost:8000/v1");
});

test("parses and validates RWKV sampling overrides and sticky session ID", () => {
  const base = readConfig({});
  assert.equal(base.historyMode, "epoch");
  assert.equal(base.epochBudgetTokens, 9600);
  assert.equal(base.sampling, undefined);
  assert.equal(base.sessionId, undefined);

  const custom = readConfig({
    RWKV_TEMPERATURE: "0.25",
    RWKV_TOP_P: "0.4",
    RWKV_TOP_K: "4096",
    RWKV_ALPHA_PRESENCE: "0.6",
    RWKV_ALPHA_FREQUENCY: "0.15",
    RWKV_ALPHA_DECAY: "0.995",
    RWKV_SESSION_ID: "trial-abc-123",
  });
  assert.deepEqual(custom.sampling, {
    temperature: 0.25,
    topP: 0.4,
    topK: 4096,
    alphaPresence: 0.6,
    alphaFrequency: 0.15,
    alphaDecay: 0.995,
  });
  assert.equal(custom.sessionId, "trial-abc-123");
  assert.equal(readConfig({ RWKV_TEMPERATURE: "0" }).sampling?.temperature, 0);

  assert.throws(() => readConfig({ RWKV_TEMPERATURE: "-0.1" }), /RWKV_TEMPERATURE/);
  assert.throws(() => readConfig({ RWKV_TOP_P: "0" }), /RWKV_TOP_P/);
  assert.throws(() => readConfig({ RWKV_TOP_P: "1.1" }), /RWKV_TOP_P/);
  assert.throws(() => readConfig({ RWKV_TOP_K: "0" }), /RWKV_TOP_K/);
  assert.throws(() => readConfig({ RWKV_ALPHA_PRESENCE: "-1" }), /RWKV_ALPHA_PRESENCE/);
  assert.throws(() => readConfig({ RWKV_ALPHA_FREQUENCY: "NaN" }), /RWKV_ALPHA_FREQUENCY/);
  assert.throws(() => readConfig({ RWKV_ALPHA_DECAY: "0" }), /RWKV_ALPHA_DECAY/);
  assert.throws(() => readConfig({ RWKV_SESSION_ID: "bad\nid" }), /RWKV_SESSION_ID/);
});
