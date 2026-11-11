import assert from "node:assert/strict";
import test from "node:test";
import { createModels } from "@earendil-works/pi-ai";
import { createRwkvProvider, readConfig } from "../dist/provider.js";
import { mockServer } from "./helpers/mock-server.mjs";

// Explicit legacy native-OpenAI compatibility; Lightning uses the separate continuation adapter.
test("native client preserves XML and JSON text instead of inventing tool calls", async t => {
  const texts = [
    '<tool_call>{"name":"read","arguments":{"path":"input.txt"}}</tool_call>',
    '{"name":"read","arguments":{"path":"input.txt"}}',
  ];
  const server = await mockServer((_body, index) => ({ deltas: [{ content: texts[index - 1] }] }));
  t.after(server.close);
  const provider = createRwkvProvider(readConfig({ RWKV_BACKEND: "openai", RWKV_BASE_URL: server.url }));
  const models = createModels(); models.setProvider(provider);
  for (const text of texts) {
    const result = await models.completeSimple(provider.getModels()[0], { messages: [{ role: "user", content: "Read.", timestamp: 1 }] });
    assert.equal(result.stopReason, "stop");
    assert.deepEqual(result.content, [{ type: "text", text }]);
    assert.equal(result.diagnostics?.some(d => d.type === "rwkv_pi_tool_adapter"), undefined);
  }
});

test("legacy native-OpenAI path rejects incompatible client conversion modes", () => {
  const config = readConfig({ RWKV_BACKEND: "openai",});
  assert.equal(config.toolFormat, "native");
  assert.equal(config.promptTemplate, "xml");
  for (const value of ["xml", "json"]) assert.throws(() => readConfig({ RWKV_BACKEND: "openai", RWKV_TOOL_FORMAT: value }), /legacy OpenAI/);
});
