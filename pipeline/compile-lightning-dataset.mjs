#!/usr/bin/env node
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { execFileSync } from "node:child_process";
import { resolve, join } from "node:path";
import { parseArgs } from "node:util";
import { renderPrompt, PROMPT_VERSION } from "../dist/prompt.js";
import { sha256 } from "../dist/provenance.js";

const { values: a } = parseArgs({ options: { dataset: { type: "string" }, out: { type: "string" }, tokenizer: { type: "string", default: ".upstream/lightning-tokenizer/tokenize-native" }, vocab: { type: "string", default: ".upstream/rwkv-lightning/assets/rwkv_vocab_v20230424.txt" }, "max-tokens": { type: "string", default: "0" } } });
if (!a.dataset || !a.out) throw Error("Usage: npm run tokenize -- --dataset datasets/verified --out datasets/lightning-v2");
const lock = JSON.parse(await readFile("lightning-source.lock.json", "utf8"));
const vocabHash = sha256(await readFile(a.vocab));
if (vocabHash !== lock.vocabSha256) throw Error("Vocabulary differs from the pinned backend.");
const tokenizerHash = sha256(await readFile(a.tokenizer));
const tokenizerProof = JSON.parse(await readFile(a.tokenizer + ".json", "utf8"));
if (tokenizerProof.binarySha256 !== tokenizerHash || tokenizerProof.engineRevision !== lock.revision || tokenizerProof.sourceSha256 !== lock.tokenizerSourceSha256) throw Error("Native tokenizer proof mismatch; run npm run lightning:tokenizer.");
const maxTokens = Number(a["max-tokens"]);
if (!Number.isSafeInteger(maxTokens) || maxTokens < 0 || maxTokens === 1) throw Error("Invalid context budget.");
const manifest = JSON.parse(await readFile(join(a.dataset, "manifest.json"), "utf8"));
if (manifest.format !== "openai-messages-with-last-assistant-loss") throw Error("Expected verified intermediate dataset.");
const tokenize = texts => execFileSync(resolve(a.tokenizer), [resolve(a.vocab)], { input: texts.map(t => Buffer.from(t).toString("hex")).join("\n") + "\n", encoding: "utf8", maxBuffer: 64 * 1024 * 1024 }).trim().split("\n").map(JSON.parse);
if (JSON.stringify(tokenize(["<EOD>"])[0]) !== "[0]") throw Error("Unexpected EOS encoding.");
const outputs = {}, counts = {}, rows = [], inputs = [], families = new Map();
for (const split of ["train", "validation", "test"]) {
  const raw = await readFile(join(a.dataset, split + ".jsonl"), "utf8");
  inputs.push({ split, sha256: sha256(raw) });
  const samples = raw.trim() ? raw.trim().split("\n").map(JSON.parse) : [];
  if (samples.length !== manifest.counts[split]) throw Error("Input count mismatch.");
  outputs[split] = [];
  for (const sample of samples) {
    if (sample.split !== split || sample.role === "smoke" || sample.loss?.scope !== "last_assistant" || sample.loss.messageIndex !== sample.messages.length - 1) throw Error("Invalid supervision contract.");
    if (families.has(sample.family) && families.get(sample.family) !== split) throw Error("Task family crosses splits.");
    families.set(sample.family, split);
    const prefix = renderPrompt(sample.messages.slice(0, -1), sample.tools, true);
    const full = renderPrompt(sample.messages, sample.tools, true, false);
    if (!full.startsWith(prefix)) throw Error("Target is not a strict prompt continuation.");
    const continuation = full.slice(prefix.length);
    const [p, f, c, joined] = tokenize([prefix, full, continuation + "<EOD>", full + "<EOD>"]);
    if (f.includes(0) || !p.length || f.length <= p.length || !p.every((id, i) => f[i] === id)) throw Error("Token prefix mismatch or embedded EOS.");
    if (JSON.stringify([...p, ...c]) !== JSON.stringify(joined) || joined.at(-1) !== 0 || joined.length !== f.length + 1) throw Error("Segment tokenization changes boundaries.");
    if (maxTokens && joined.length > maxTokens) throw Error(`Sample ${sample.id} exceeds context; truncation forbidden.`);
    outputs[split].push({ segments: [{ text: prefix, train: false }, { text: continuation + "<EOD>", train: true }] });
    rows.push({ id: sample.id, family: sample.family, split, sourceSampleSha256: sha256(JSON.stringify(sample)), tokens: joined.length, promptTokens: p.length, supervisedTokens: c.length, tokenIdsSha256: sha256(JSON.stringify(joined)) });
  }
  counts[split] = outputs[split].length;
}
if (!counts.train) throw Error("No verified training samples.");
await mkdir(a.out); // Never overwrite an existing compiled dataset.
const files = {};
for (const [split, samples] of Object.entries(outputs)) {
  const raw = samples.map(s => JSON.stringify(s) + "\n").join("");
  await writeFile(join(a.out, split + ".jsonl"), raw, { mode: 0o600 });
  files[split + ".jsonl"] = sha256(raw);
}
const record = { schemaVersion: 1, format: "rwkv-lightning-segments-v1", engineRevision: lock.revision, createdAt: new Date().toISOString(), promptVersion: PROMPT_VERSION, promptSha256: sha256(await readFile("dist/prompt.js")), compilerSha256: sha256(await readFile(new URL(import.meta.url))), tokenizerProof, vocabSha256: vocabHash, eosTokenId: 0, eosText: "<EOD>", maxTokens, inputs, files, counts, rows, verification: "Native tokenizer strict prefix and independent-segment concatenation equal full tokenization for every row. Only final assistant continuation and EOS have loss." };
await writeFile(join(a.out, "manifest.json"), JSON.stringify(record, null, 2) + "\n");
console.log(JSON.stringify({ out: a.out, counts, maxSequenceTokens: Math.max(...rows.map(r => r.tokens)) }));
