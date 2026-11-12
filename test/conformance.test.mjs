import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import test from "node:test";
import { check } from "../scripts/conformance.mjs";
import { nativeTokenizerPaths } from "../dist/doctor.js";

test("renderer and parser reproduce the committed wire conformance fixtures", () => {
  assert.deepEqual(check({ tokens: false }), []);
});

test("the Lightning-native tokenizer reproduces the committed token ids", t => {
  if (!existsSync(nativeTokenizerPaths().binary)) { t.skip("native tokenizer not built; run npm run lightning:setup && npm run lightning:tokenizer"); return; }
  assert.deepEqual(check({ tokens: true }), []);
});
