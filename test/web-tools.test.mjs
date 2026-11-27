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

  const spilledFiles = new Map();
  const spillTools = createWebTools(
    { RWKV_WEB_API_URL: "http://172.17.0.1:18481/v1/tools", RWKV_WEB_MAX_CALLS: "5" },
    {
      fetchImpl: fakeFetch,
      onSpill: async (_url, content) => {
        const path = "/tmp/dsh-spill-test123.txt";
        spilledFiles.set(path, content);
        return path;
      },
    },
  );
  const spillFirst = JSON.parse((await spillTools[1].execute("s1", { url: "https://rwkv.com/rfc" })).content[0].text);
  assert.equal(spillFirst.spilledTo, "/tmp/dsh-spill-test123.txt");
  assert.match(spillFirst.content, /\/tmp\/dsh-spill-test123\.txt/);
  assert.equal(spilledFiles.get("/tmp/dsh-spill-test123.txt")?.length, 11234);
});

test("DSH retrieval mode merges multi-query searches, focuses long pages without offset traps, and blocks duplicate calls", async () => {
  const calls = [];
  const fakeFetch = async (url, init) => {
    const body = JSON.parse(init.body);
    calls.push({ url: String(url), body });
    if (String(url).endsWith("/search")) {
      return new Response(JSON.stringify({
        results: [{
          title: `Result for ${body.query}`,
          url: `https://docs.python.org/${encodeURIComponent(body.query)}`,
          date: "2024-10-07",
          summary: "Python 3.13.0 final was released on October 7, 2024 with experimental free-threaded PEP 703 support.",
        }],
      }), { status: 200 });
    }
    const longDoc = "Overview header\n" + "filler paragraph\n".repeat(800) + "PEP 703 free-threaded mode arrived in Python 3.13.0 on 2024-10-07.\n" + "tail section\n".repeat(200);
    return new Response(JSON.stringify({ title: "What's New In Python 3.13", content: longDoc }), { status: 200 });
  };

  const dshTools = createWebTools(
    { RWKV_WEB_API_URL: "http://172.17.0.1:18481/v1/tools", RWKV_WEB_FORMAT: "dsh" },
    {
      fetchImpl: fakeFetch,
      getUserQuestion: () => "对比 Python 3.12 与 3.13 发布日期及 PEP 703",
      onSpill: async () => "/tmp/dsh-spill/page-1.txt",
    },
  );

  const s1 = await dshTools[0].execute("d1", { queries: ["Python 3.12.0 release date", "Python 3.13.0 PEP 703"] });
  assert.match(s1.content[0].text, /\[S1\]/);
  assert.match(s1.content[0].text, /\[S2\]/);
  assert.equal(calls.filter((c) => c.url.endsWith("/search")).length, 2);
  assert.equal(calls[0].body.need_content, true);
  assert.ok(calls[0].body.ori_question.includes("PEP 703"));

  // Duplicate search in same turn returns DSH duplicate reminder instead of re-billing
  const dupSearch = await dshTools[0].execute("d2", { query: "Python 3.12.0 release date" });
  assert.match(dupSearch.content[0].text, /repeats a successful call/);

  // Long page fetch keeps head + PEP 703 focus section + tail + spill file notice (no nextOffset loop)
  const f1 = await dshTools[1].execute("d3", { url: "https://docs.python.org/3.13/whatsnew/3.13.html", focus: "PEP 703" });
  assert.match(f1.content[0].text, /PEP 703 free-threaded mode arrived in Python 3.13\.0/);
  assert.match(f1.content[0].text, /\/tmp\/dsh-spill\/page-1\.txt/);
  assert.doesNotMatch(f1.content[0].text, /nextOffset/);
});
