import test from "node:test";
import assert from "node:assert/strict";
import { assertPublicHttpUrl, createWebTools } from "../dist/web-tools.js";

test("web_search and web_fetch format DeepSeek tool responses and block private fetch URLs", async () => {
  const calls = [];
  const fakeFetch = async (url, init) => {
    calls.push({ url: String(url), method: init.method, headers: init.headers, body: JSON.parse(init.body) });
    if (String(url).endsWith("/search")) {
      return new Response(JSON.stringify({
        results: [{ title: "RWKV-7", url: "https://rwkv.com/g1", date: "2026-10-31", snippet: "s", summary: "x".repeat(2500) }],
      }), { status: 200 });
    }
    return new Response(JSON.stringify({ title: "Page", content: "a".repeat(5000) + "b".repeat(5000) + "c".repeat(1234) }), { status: 200 });
  };
  const tools = createWebTools(
    { RWKV_WEB_API_URL: "http://172.17.0.1:18481/v1/tools", RWKV_WEB_MAX_CALLS: "5" },
    { fetchImpl: fakeFetch },
  );
  assert.deepEqual(tools.map((t) => t.name), ["web_search", "web_fetch"]);

  const searchRes = await tools[0].execute("c1", { query: "  RWKV-7 G1L  ", count: 5 });
  const searchPayload = JSON.parse(searchRes.content[0].text);
  assert.equal(calls[0].url, "http://172.17.0.1:18481/v1/tools/search");
  assert.deepEqual(calls[0].body, { query: "RWKV-7 G1L", count: 5, need_content: false });
  assert.equal(searchPayload.externalUntrustedData, true);
  assert.equal(searchPayload.results.length, 1);
  assert.ok(searchPayload.results[0].summary.endsWith("[工具长度限制：后文已截断]"));

  await assert.rejects(() => Promise.resolve().then(() => assertPublicHttpUrl("http://127.0.0.1:8000/v1/models")), /Public HTTP/);
  await assert.rejects(() => Promise.resolve().then(() => assertPublicHttpUrl("http://172.17.0.1:18480/")), /Public HTTP/);

  const fetchRes = await tools[1].execute("c2", { url: "https://rwkv.com/g1" });
  const first = JSON.parse(fetchRes.content[0].text);
  assert.equal(first.truncated, true);
  assert.equal(first.nextOffset, 5000);
  assert.equal(first.totalChars, 11234);
  assert.ok(first.content.startsWith("a".repeat(5000)));
  assert.match(first.content, /offset=5000/);

  // Following pages come from the page already fetched: one billed request, contiguous text, a clean last page.
  const second = JSON.parse((await tools[1].execute("c3", { url: "https://rwkv.com/g1", offset: first.nextOffset })).content[0].text);
  assert.ok(second.content.startsWith("b".repeat(5000)));
  assert.equal(second.nextOffset, 10000);
  const last = JSON.parse((await tools[1].execute("c4", { url: "https://rwkv.com/g1", offset: second.nextOffset })).content[0].text);
  assert.equal(last.content, "c".repeat(1234));
  assert.equal(last.truncated, false);
  assert.equal(last.nextOffset, undefined);
  assert.equal(calls.filter((c) => c.url.endsWith("/fetch")).length, 1);
  await assert.rejects(() => tools[1].execute("c5", { url: "https://rwkv.com/g1", offset: 99999 }), /beyond the end/);

  await assert.rejects(() => tools[0].execute("c6", { query: "over budget" }), /budget reached/);
});
