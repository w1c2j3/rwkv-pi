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
