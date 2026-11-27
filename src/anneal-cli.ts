#!/usr/bin/env node
import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { createReadStream, createWriteStream, existsSync, readdirSync, readFileSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import {
  computeTokenLayout,
  convertPackedSpanRow,
  type LightningSegment,
  type PackedSpanRow,
  type TrajectoryHistory,
} from "./anneal-data.js";
import { PROMPT_VERSION } from "./prompt.js";
import { digest } from "./provenance.js";

const REPO_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");

function sha256Hex(input: string | Buffer): string {
  return createHash("sha256").update(input).digest("hex");
}

async function streamFileSha256(filePath: string): Promise<string> {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(filePath)) hash.update(chunk);
  return hash.digest("hex");
}

function parseSpanGroup(row: PackedSpanRow, lineIdx: number): { baseId: string; orderKey: [number, number, number] } {
  const id = typeof row.id === "string" ? row.id : "";
  const m = /^(.*)__span(\d+)$/.exec(id);
  const baseId = m ? m[1]! : (id || `row_${lineIdx}`);
  const firstText = Array.isArray(row.segments) && typeof row.segments[0]?.text === "string"
    ? row.segments[0].text
    : "";
  const ledgerStart = firstText.indexOf("[State Ledger (");
  const hasLedger = ledgerStart >= 0 ? 1 : 0;
  const ledgerEnd = ledgerStart >= 0 ? firstText.indexOf("]\n", ledgerStart) : -1;
  const ledgerLen = ledgerStart >= 0 && ledgerEnd > ledgerStart ? (ledgerEnd - ledgerStart) : 0;
  return { baseId, orderKey: [hasLedger, ledgerLen, firstText.length] };
}

function bucketReason(reason: string): string {
  if (reason.startsWith("line_edit_apply_rejected:No edit needed")) return "line_edit_apply_rejected:noop_identical";
  if (reason.startsWith("line_edit_apply_rejected:") && reason.includes("duplicates line")) return "line_edit_apply_rejected:boundary_duplication";
  if (reason.startsWith("line_edit_apply_rejected:") && reason.includes("overlap")) return "line_edit_apply_rejected:overlap";
  if (reason.startsWith("line_edit_apply_rejected:") && reason.includes("start_text")) return "line_edit_apply_rejected:anchor_mismatch";
  if (reason.startsWith("line_edit_apply_rejected:")) return "line_edit_apply_rejected:other";
  return reason;
}

const RECOVERY_MODES = ["unread_first", "anchor_mismatch", "boundary_dup", "noop_recovery"] as const;

function tokenizeBatchNative(tokenizerBin: string, vocabPath: string, texts: readonly string[]): Promise<number[][]> {
  if (texts.length === 0) return Promise.resolve([]);
  return new Promise((accept, reject) => {
    const child = spawn(tokenizerBin, [vocabPath], { stdio: ["pipe", "pipe", "pipe"] });
    const chunks: Buffer[] = [];
    const errChunks: Buffer[] = [];
    child.stdout.on("data", (c: Buffer) => chunks.push(c));
    child.stderr.on("data", (c: Buffer) => errChunks.push(c));
    child.once("error", reject);
    child.once("close", (code) => {
      if (code !== 0) {
        reject(new Error(`tokenize-native exited ${code}: ${Buffer.concat(errChunks).toString("utf8")}`));
        return;
      }
      const outStr = Buffer.concat(chunks).toString("utf8").trim();
      const lines = outStr ? outStr.split("\n").map((l) => JSON.parse(l) as number[]) : [];
      if (lines.length !== texts.length) {
        reject(new Error(`tokenize-native count mismatch (${lines.length} !== ${texts.length})`));
        return;
      }
      accept(lines);
    });
    const payload = texts.map((t) => Buffer.from(t, "utf8").toString("hex")).join("\n") + "\n";
    child.stdin.end(payload);
  });
}

interface MeasuredSpanMeta {
  tokens: number;
  supervisedTokens: number;
  trainedTurns: number;
  maxPromptTokens: number;
  maxGeneratedTokens: number;
  legacyPositions: number;
  lengthBucket: string;
  segmentsSha256: string;
  tokenIdsSha256: string;
  labelsSha256: string;
  maskSha256: string;
}

async function measureBatchWithTokenizer(
  tokenizerBin: string,
  vocabPath: string,
  batch: readonly { segments: LightningSegment[] }[],
  maxContextTokens: number,
  maxTurnTokens: number,
): Promise<(MeasuredSpanMeta | { error: string })[]> {
  const flatTexts: string[] = [];
  const slices: { fullIdx: number; segStart: number; segCount: number }[] = [];
  for (const item of batch) {
    const joined = item.segments.slice(0, -1).map((s) => s.text).join("") + "<EOD>";
    const fullIdx = flatTexts.length;
    flatTexts.push(joined);
    const segStart = flatTexts.length;
    for (const s of item.segments) flatTexts.push(s.text);
    slices.push({ fullIdx, segStart, segCount: item.segments.length });
  }
  const encodedAll = await tokenizeBatchNative(tokenizerBin, vocabPath, flatTexts);
  return batch.map((item, idx) => {
    const sl = slices[idx]!;
    const full = encodedAll[sl.fullIdx]!;
    const segEncoded = encodedAll.slice(sl.segStart, sl.segStart + sl.segCount);
    try {
      const layout = computeTokenLayout(item.segments, segEncoded);
      if (
        layout.tokens.length !== full.length
        || layout.tokens.some((id, k) => id !== full[k])
        || full[full.length - 1] !== 0
        || full.slice(0, -1).includes(0)
      ) {
        return { error: "segment_tokenization_boundary_mismatch" };
      }
      if (full.length > maxContextTokens) {
        return { error: `exceeds_ctx_${maxContextTokens}` };
      }
      let offset = 0;
      let maxPrompt = 0;
      let maxGen = 0;
      let legacyPos = 0;
      let trainedTurns = 0;
      for (let i = 0; i < item.segments.length; i++) {
        const s = item.segments[i]!;
        const len = segEncoded[i]!.length;
        if (s.train) {
          trainedTurns++;
          const genLen = len + 1;
          if (genLen > maxTurnTokens) return { error: `turn_exceeds_max_${maxTurnTokens}` };
          maxGen = Math.max(maxGen, genLen);
          maxPrompt = Math.max(maxPrompt, offset);
          legacyPos += offset + genLen - 1;
        }
        offset += len;
      }
      let supToks = 0;
      for (let k = 1; k < layout.train.length; k++) supToks += layout.train[k]!;
      const n = full.length;
      return {
        tokens: n,
        supervisedTokens: supToks,
        trainedTurns,
        maxPromptTokens: maxPrompt,
        maxGeneratedTokens: maxGen,
        legacyPositions: legacyPos,
        lengthBucket: n <= 20480 ? "20k" : n <= 32768 ? "32k" : n <= 65536 ? "64k" : "over64k",
        segmentsSha256: digest(item.segments),
        tokenIdsSha256: sha256Hex(JSON.stringify(full)),
        labelsSha256: sha256Hex(JSON.stringify(layout.labels)),
        maskSha256: sha256Hex(JSON.stringify(layout.train)),
      };
    } catch (err) {
      return { error: `token_layout_error:${err instanceof Error ? err.message : String(err)}` };
    }
  });
}

function makeStreamWriter(outStream: ReturnType<typeof createWriteStream>) {
  return (row: unknown): Promise<void> =>
    new Promise((res, rej) => {
      const line = JSON.stringify(row) + "\n";
      if (outStream.write(line)) {
        res();
        return;
      }
      const onDrain = () => {
        outStream.off("error", onErr);
        res();
      };
      const onErr = (err: Error) => {
        outStream.off("drain", onDrain);
        rej(err);
      };
      outStream.once("drain", onDrain).once("error", onErr);
    });
}

async function runCompiledDatasetMode(opts: {
  datasetIn: string;
  outDir: string;
  recoveryRatio: number;
  allowBlindEdits: boolean;
  limit: number;
  tokenizerBin: string;
  vocabPath: string;
}): Promise<void> {
  const { datasetIn, outDir, recoveryRatio, allowBlindEdits, limit, tokenizerBin, vocabPath } = opts;
  const srcManifest = JSON.parse(readFileSync(join(datasetIn, "manifest.json"), "utf8"));
  const srcRows: Record<string, any>[] = srcManifest.rows ?? [];
  const inJsonl = join(datasetIn, "train.jsonl");
  const maxContextTokens = Number(srcManifest.generation?.contextWindow ?? 20480);
  const maxTurnTokens = Number(srcManifest.generation?.maxTokens ?? 8192);

  const lock = JSON.parse(readFileSync(join(REPO_ROOT, "lightning-source.lock.json"), "utf8"));
  const proof = JSON.parse(readFileSync(`${tokenizerBin}.json`, "utf8"));
  const vocabSha = sha256Hex(readFileSync(vocabPath));
  if (vocabSha !== lock.vocabSha256 || sha256Hex(readFileSync(tokenizerBin)) !== proof.binarySha256) {
    throw new Error("Native tokenizer or vocab SHA256 mismatch against lightning-source.lock.json.");
  }

  // Pass 1: count spans per trajectory baseId from `srcRows`
  const expectedSpans = new Map<string, number>();
  const maxScan = Math.min(srcRows.length, limit);
  for (let i = 0; i < maxScan; i++) {
    const rawId = String(srcRows[i]?.id ?? `row_${i + 1}`);
    const m = /^(.*)__span(\d+)$/.exec(rawId);
    const baseId = m ? m[1]! : rawId;
    expectedSpans.set(baseId, (expectedSpans.get(baseId) ?? 0) + 1);
  }

  await mkdir(outDir, { recursive: true });
  const outTrainPath = join(outDir, "train.jsonl");
  const outValPath = join(outDir, "validation.jsonl");
  const outTestPath = join(outDir, "test.jsonl");
  const outStream = createWriteStream(outTrainPath, { encoding: "utf8", mode: 0o600 });
  const writeRow = makeStreamWriter(outStream);

  let totalIn = 0;
  let keptBase = 0;
  let augmentedCount = 0;
  let convertedEdits = 0;
  let convertedReads = 0;
  let synthesizedPostFoldReads = 0;
  let editRowIdx = 0;
  const droppedReasons: Record<string, number> = {};
  const seenTokenHashes = new Set<string>();
  const outManifestRows: Record<string, any>[] = [];

  interface CandidateItem {
    meta: Record<string, any>;
    segments: LightningSegment[];
    isAugmented: boolean;
  }
  const pendingTokenBatch: CandidateItem[] = [];

  const flushTokenBatch = async () => {
    if (pendingTokenBatch.length === 0) return;
    const chunkSize = 64;
    const subBatches: CandidateItem[][] = [];
    for (let k = 0; k < pendingTokenBatch.length; k += chunkSize) {
      subBatches.push(pendingTokenBatch.slice(k, k + chunkSize));
    }
    pendingTokenBatch.length = 0;
    const measuredGroups = await Promise.all(
      subBatches.map((sub) => measureBatchWithTokenizer(tokenizerBin, vocabPath, sub, maxContextTokens, maxTurnTokens)),
    );
    for (let g = 0; g < subBatches.length; g++) {
      const sub = subBatches[g]!;
      const measured = measuredGroups[g]!;
      for (let j = 0; j < sub.length; j++) {
        const cand = sub[j]!;
        const m = measured[j]!;
        if ("error" in m) {
          if (!cand.isAugmented) {
            droppedReasons[m.error] = (droppedReasons[m.error] ?? 0) + 1;
          }
          continue;
        }
        if (seenTokenHashes.has(m.tokenIdsSha256)) {
          if (!cand.isAugmented) {
            droppedReasons["duplicate_token_ids"] = (droppedReasons["duplicate_token_ids"] ?? 0) + 1;
          }
          continue;
        }
        seenTokenHashes.add(m.tokenIdsSha256);
        await writeRow({ segments: cand.segments });
        if (cand.isAugmented) augmentedCount++;
        else keptBase++;
        outManifestRows.push({
          ...cand.meta,
          split: "train",
          editMode: "line",
          ...m,
        });
      }
    }
  };

  const pendingGroups = new Map<string, { orderKey: [number, number, number]; rawLine: string; row: PackedSpanRow }[]>();

  const processGroup = async (
    items: { orderKey: [number, number, number]; rawLine: string; row: PackedSpanRow }[],
  ): Promise<void> => {
    items.sort((a, b) =>
      (a.orderKey[0] - b.orderKey[0])
      || (a.orderKey[1] - b.orderKey[1])
      || (a.orderKey[2] - b.orderKey[2]));
    const trajectoryHistory: TrajectoryHistory = new Map();
    if (items.length > 1) {
      for (const item of items) {
        convertPackedSpanRow(item.row, {
          rejectBlindEdits: false,
          augmentRecovery: "none",
          trajectoryHistory,
        });
      }
    }
    for (const item of items) {
      const preCheckHasEdit = item.rawLine.includes('\\"name\\":\\"edit\\"') || item.rawLine.includes('"name":"edit"');
      let augmentMode: "none" | "unread_first" | "anchor_mismatch" | "boundary_dup" | "noop_recovery" = "none";
      if (preCheckHasEdit && recoveryRatio > 0) {
        editRowIdx++;
        const stride = Math.max(1, Math.round(1 / recoveryRatio));
        if (editRowIdx % stride === 0) {
          augmentMode = RECOVERY_MODES[Math.floor(editRowIdx / stride) % RECOVERY_MODES.length]!;
        }
      }

      const res = convertPackedSpanRow(item.row, {
        rejectBlindEdits: !allowBlindEdits,
        augmentRecovery: augmentMode,
        trajectoryHistory,
      });
      if (!res.ok) {
        const bucket = bucketReason(res.reason);
        droppedReasons[bucket] = (droppedReasons[bucket] ?? 0) + 1;
        continue;
      }
      convertedEdits += res.convertedEdits;
      convertedReads += res.convertedReads;
      synthesizedPostFoldReads += res.synthesizedPostFoldReads;

      const { segments: _dropSegs, ...cleanMeta } = res.row;
      pendingTokenBatch.push({ meta: cleanMeta, segments: res.row.segments, isAugmented: false });
      if (res.augmented) {
        const { segments: _dropAugSegs, ...augMeta } = res.augmented;
        pendingTokenBatch.push({ meta: augMeta, segments: res.augmented.segments, isAugmented: true });
      }
      if (pendingTokenBatch.length >= 256) {
        await flushTokenBatch();
      }
    }
  };

  const rl = createInterface({ input: createReadStream(inJsonl, { encoding: "utf8" }), crlfDelay: Infinity });
  for await (const rawLine of rl) {
    const trimmed = rawLine.trim();
    if (!trimmed) continue;
    if (totalIn >= maxScan) break;
    const rowMeta = srcRows[totalIn] ?? {};
    totalIn++;
    const data = JSON.parse(trimmed) as { segments: LightningSegment[] };
    const mergedRow: PackedSpanRow = { ...rowMeta, segments: data.segments };
    const { baseId, orderKey } = parseSpanGroup(mergedRow, totalIn);
    const expected = expectedSpans.get(baseId) ?? 1;
    if (expected <= 1) {
      await processGroup([{ orderKey, rawLine: trimmed, row: mergedRow }]);
    } else {
      const list = pendingGroups.get(baseId) ?? [];
      list.push({ orderKey, rawLine: trimmed, row: mergedRow });
      if (list.length >= expected) {
        pendingGroups.delete(baseId);
        await processGroup(list);
      } else {
        pendingGroups.set(baseId, list);
      }
    }
    if (totalIn % 5000 === 0) {
      console.error(JSON.stringify({ progress: totalIn, keptBase, augmentedCount, synthesizedPostFoldReads }));
    }
  }
  for (const [, leftover] of pendingGroups) {
    await processGroup(leftover);
  }
  pendingGroups.clear();
  await flushTokenBatch();

  await new Promise<void>((res, rej) => outStream.end((err?: Error | null) => (err ? rej(err) : res())));
  await writeFile(outValPath, "", { mode: 0o600 });
  await writeFile(outTestPath, "", { mode: 0o600 });

  const files = {
    "train.jsonl": await streamFileSha256(outTrainPath),
    "validation.jsonl": await streamFileSha256(outValPath),
    "test.jsonl": await streamFileSha256(outTestPath),
  };

  const totalTokens = outManifestRows.reduce((s, r) => s + Number(r.tokens ?? 0), 0);
  const totalSupTokens = outManifestRows.reduce((s, r) => s + Number(r.supervisedTokens ?? 0), 0);
  const totalTrainedTurns = outManifestRows.reduce((s, r) => s + Number(r.trainedTurns ?? 0), 0);
  const totalLegacyPos = outManifestRows.reduce((s, r) => s + Number(r.legacyPositions ?? 0), 0);
  const maxRowToks = outManifestRows.reduce((m, r) => Math.max(m, Number(r.tokens ?? 0)), 0);
  const uniqueFamilies = new Set(outManifestRows.map((r) => r.family).filter(Boolean)).size;
  const uniqueTasks = new Set(outManifestRows.map((r) => r.taskId).filter(Boolean)).size;

  const tierTokensRaw: Record<string, number> = {};
  const cohortTokensRaw: Record<string, number> = {};
  for (const r of outManifestRows) {
    const toks = Number(r.tokens ?? 0);
    const tier = String(r.tier ?? "unspecified");
    const cohort = String(r.cohort ?? "unspecified");
    tierTokensRaw[tier] = (tierTokensRaw[tier] ?? 0) + toks;
    cohortTokensRaw[cohort] = (cohortTokensRaw[cohort] ?? 0) + toks;
  }
  const tierTokensM = Object.fromEntries(
    Object.entries(tierTokensRaw)
      .sort((a, b) => b[1] - a[1])
      .map(([k, v]) => [k, Math.round((v / 1e6) * 100) / 100]),
  );
  const topBucketsTokensM = Object.fromEntries(
    Object.entries(cohortTokensRaw)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 25)
      .map(([k, v]) => [k, Math.round((v / 1e6) * 100) / 100]),
  );

  const annealingSummary = {
    physicalRows: outManifestRows.length,
    uniqueSelectedRows: outManifestRows.length,
    keptBaseSpans: keptBase,
    recoveryAugmentedSpans: augmentedCount,
    totalPackedTokens: totalTokens,
    totalSupervisedTokens: totalSupTokens,
    maxRowTokens: maxRowToks,
    requiredCtx: maxContextTokens,
    trainedTurns: totalTrainedTurns,
    convertedReads,
    convertedEdits,
    synthesizedPostFoldReads,
    tierTokensM,
    topBucketsTokensM,
    droppedByReason: droppedReasons,
  };

  const nativeAudit = {
    at: new Date().toISOString(),
    rows: outManifestRows.length,
    passed: true,
    failed: [],
    productionLimits: { contextWindow: maxContextTokens, maxTokens: maxTurnTokens },
    nativeTokenizer: tokenizerBin,
    vocabSha256: vocabSha,
  };

  const manifest = {
    schemaVersion: 2,
    format: "rwkv-lightning-segments-eos-target-v2",
    requiredTrainingCapability: "segment-end-target-v1",
    engineRevision: lock.revision,
    vocabSha256: vocabSha,
    promptVersion: PROMPT_VERSION,
    editMode: "line",
    tokenizerProof: proof,
    createdAt: new Date().toISOString(),
    name: basename(outDir),
    sourceDataset: datasetIn,
    trainingReady: true,
    generation: { contextWindow: maxContextTokens, maxTokens: maxTurnTokens },
    maxTokens: maxRowToks,
    requiredCtx: maxContextTokens,
    counts: {
      rows: outManifestRows.length,
      train: outManifestRows.length,
      validation: 0,
      test: 0,
      tokens: totalTokens,
      supervisedTokens: totalSupTokens,
      uniqueRows: outManifestRows.length,
      uniqueQuestionFamilies: uniqueFamilies,
      uniqueTrajectories: uniqueTasks,
      trainedTurns: totalTrainedTurns,
      legacyPositions: totalLegacyPos,
    },
    files,
    nativeAudit,
    annealingSummary,
    rows: outManifestRows,
  };

  await writeFile(join(outDir, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n", "utf8");
  await writeFile(join(outDir, "native-audit.json"), JSON.stringify(nativeAudit, null, 2) + "\n", "utf8");
  await writeFile(join(outDir, "summary-report.json"), JSON.stringify(annealingSummary, null, 2) + "\n", "utf8");
  console.log(JSON.stringify(annealingSummary, null, 2));
}

async function runShardPoolOrJsonlMode(opts: {
  inPaths: string[];
  outDir: string;
  preserveShards: boolean;
  recoveryRatio: number;
  allowBlindEdits: boolean;
  limit: number;
}): Promise<void> {
  const { inPaths, outDir, preserveShards, recoveryRatio, allowBlindEdits, limit } = opts;

  const expectedSpans = new Map<string, number>();
  let scanned = 0;
  for (const inPath of inPaths) {
    const scanRl = createInterface({ input: createReadStream(inPath, { encoding: "utf8" }), crlfDelay: Infinity });
    for await (const rawLine of scanRl) {
      const trimmed = rawLine.trim();
      if (!trimmed) continue;
      if (scanned >= limit) break;
      scanned++;
      const idMatch = /"id"\s*:\s*"([^"]+)"/.exec(trimmed);
      const rawId = idMatch ? idMatch[1]! : `row_${scanned}`;
      const spanMatch = /^(.*)__span(\d+)$/.exec(rawId);
      const baseId = spanMatch ? spanMatch[1]! : rawId;
      expectedSpans.set(baseId, (expectedSpans.get(baseId) ?? 0) + 1);
    }
    if (scanned >= limit) break;
  }

  await mkdir(outDir, { recursive: true });
  if (preserveShards) await mkdir(join(outDir, "shards"), { recursive: true });

  let singleStream: ReturnType<typeof createWriteStream> | undefined;
  let singleWrite: ((row: unknown) => Promise<void>) | undefined;
  if (!preserveShards) {
    const outJsonl = join(outDir, "train.jsonl");
    singleStream = createWriteStream(outJsonl, { encoding: "utf8", mode: 0o600 });
    singleWrite = makeStreamWriter(singleStream);
  }

  let totalIn = 0;
  let keptBase = 0;
  let augmentedCount = 0;
  let convertedEdits = 0;
  let convertedReads = 0;
  let synthesizedPostFoldReads = 0;
  let editRowIdx = 0;
  const droppedReasons: Record<string, number> = {};
  const pendingGroups = new Map<string, { orderKey: [number, number, number]; rawLine: string; row: PackedSpanRow; shardName: string }[]>();
  const shardStreams = new Map<string, { stream: ReturnType<typeof createWriteStream>; write: (r: unknown) => Promise<void> }>();

  const getWriter = (shardName: string) => {
    if (!preserveShards) return singleWrite!;
    let entry = shardStreams.get(shardName);
    if (!entry) {
      const s = createWriteStream(join(outDir, "shards", shardName), { encoding: "utf8", mode: 0o600 });
      entry = { stream: s, write: makeStreamWriter(s) };
      shardStreams.set(shardName, entry);
    }
    return entry.write;
  };

  const processGroup = async (
    items: { orderKey: [number, number, number]; rawLine: string; row: PackedSpanRow; shardName: string }[],
  ): Promise<void> => {
    items.sort((a, b) =>
      (a.orderKey[0] - b.orderKey[0])
      || (a.orderKey[1] - b.orderKey[1])
      || (a.orderKey[2] - b.orderKey[2]));
    const trajectoryHistory: TrajectoryHistory = new Map();
    if (items.length > 1) {
      for (const item of items) {
        convertPackedSpanRow(item.row, {
          rejectBlindEdits: false,
          augmentRecovery: "none",
          trajectoryHistory,
        });
      }
    }
    for (const item of items) {
      const preCheckHasEdit = item.rawLine.includes('\\"name\\":\\"edit\\"') || item.rawLine.includes('"name":"edit"');
      let augmentMode: "none" | "unread_first" | "anchor_mismatch" | "boundary_dup" | "noop_recovery" = "none";
      if (preCheckHasEdit && recoveryRatio > 0) {
        editRowIdx++;
        const stride = Math.max(1, Math.round(1 / recoveryRatio));
        if (editRowIdx % stride === 0) {
          augmentMode = RECOVERY_MODES[Math.floor(editRowIdx / stride) % RECOVERY_MODES.length]!;
        }
      }

      const res = convertPackedSpanRow(item.row, {
        rejectBlindEdits: !allowBlindEdits,
        augmentRecovery: augmentMode,
        trajectoryHistory,
      });
      if (!res.ok) {
        const bucket = bucketReason(res.reason);
        droppedReasons[bucket] = (droppedReasons[bucket] ?? 0) + 1;
        continue;
      }
      const writer = getWriter(item.shardName);
      await writer(res.row);
      keptBase++;
      convertedEdits += res.convertedEdits;
      convertedReads += res.convertedReads;
      synthesizedPostFoldReads += res.synthesizedPostFoldReads;
      if (res.augmented) {
        await writer(res.augmented);
        augmentedCount++;
      }
    }
  };

  for (const inPath of inPaths) {
    const shardName = basename(inPath);
    const rl = createInterface({ input: createReadStream(inPath, { encoding: "utf8" }), crlfDelay: Infinity });
    for await (const rawLine of rl) {
      const trimmed = rawLine.trim();
      if (!trimmed) continue;
      if (totalIn >= limit) break;
      totalIn++;
      const row = JSON.parse(trimmed) as PackedSpanRow;
      const { baseId, orderKey } = parseSpanGroup(row, totalIn);
      const expected = expectedSpans.get(baseId) ?? 1;
      if (expected <= 1) {
        await processGroup([{ orderKey, rawLine: trimmed, row, shardName }]);
      } else {
        const list = pendingGroups.get(baseId) ?? [];
        list.push({ orderKey, rawLine: trimmed, row, shardName });
        if (list.length >= expected) {
          pendingGroups.delete(baseId);
          await processGroup(list);
        } else {
          pendingGroups.set(baseId, list);
        }
      }
    }
    if (totalIn >= limit) break;
  }
  for (const [, leftover] of pendingGroups) {
    await processGroup(leftover);
  }
  pendingGroups.clear();

  if (singleStream) {
    await new Promise<void>((res, rej) => singleStream!.end((err?: Error | null) => (err ? rej(err) : res())));
  }
  for (const [, entry] of shardStreams) {
    await new Promise<void>((res, rej) => entry.stream.end((err?: Error | null) => (err ? rej(err) : res())));
  }

  const manifest = {
    schemaVersion: 2,
    format: "rwkv-lightning-segments-eos-target-v2",
    editMode: "line",
    createdAt: new Date().toISOString(),
    sourcePaths: inPaths,
    recoveryRatio,
    stats: {
      totalInputSpans: totalIn,
      keptBaseSpans: keptBase,
      recoveryAugmentedSpans: augmentedCount,
      totalOutputSpans: keptBase + augmentedCount,
      convertedReads,
      convertedEdits,
      synthesizedPostFoldReads,
      droppedByReason: droppedReasons,
    },
  };
  await writeFile(join(outDir, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n", "utf8");
  console.log(JSON.stringify(manifest.stats, null, 2));
}

try {
  const { values } = parseArgs({
    options: {
      in: { type: "string" },
      "dataset-in": { type: "string" },
      "shards-in": { type: "string" },
      out: { type: "string" },
      "recovery-ratio": { type: "string", default: "0.15" },
      "allow-blind-edits": { type: "boolean", default: false },
      tokenizer: { type: "string", default: join(REPO_ROOT, "artifacts/lightning-migration/tokenize-native") },
      vocab: { type: "string", default: join(REPO_ROOT, ".upstream/rwkv-lightning/assets/rwkv_vocab_v20230424.txt") },
      limit: { type: "string" },
      help: { type: "boolean", default: false },
    },
  });

  if (values.help || !values.out || (!values.in && !values["dataset-in"] && !values["shards-in"])) {
    console.log(
      "Usage:\n" +
      "  1. Compiled dataset: npm run anneal-convert -- --dataset-in <datasets/stage2-anneal-...> --out <datasets/out>\n" +
      "  2. Shard pool dir:   npm run anneal-convert -- --shards-in <exports/live-dagger-...> --out <exports/out>\n" +
      "  3. Single JSONL:     npm run anneal-convert -- --in <train.jsonl> --out <out-dir>\n" +
      "Options: [--recovery-ratio 0.15] [--allow-blind-edits] [--limit N]",
    );
    if (!values.help) process.exitCode = 1;
  } else {
    const outDir = resolve(values.out);
    const recoveryRatio = Number(values["recovery-ratio"]);
    if (!Number.isFinite(recoveryRatio) || recoveryRatio < 0 || recoveryRatio > 1) {
      throw new Error("--recovery-ratio must be in [0, 1].");
    }
    const limit = values.limit !== undefined ? Number(values.limit) : Infinity;

    if (values["dataset-in"]) {
      const defaultTok = existsSync(resolve(values.tokenizer))
        ? resolve(values.tokenizer)
        : resolve(REPO_ROOT, ".upstream/lightning-tokenizer/tokenize-native");
      await runCompiledDatasetMode({
        datasetIn: resolve(values["dataset-in"]),
        outDir,
        recoveryRatio,
        allowBlindEdits: Boolean(values["allow-blind-edits"]),
        limit,
        tokenizerBin: defaultTok,
        vocabPath: resolve(values.vocab),
      });
    } else if (values["shards-in"]) {
      const shardsRoot = resolve(values["shards-in"]);
      const shardsDir = existsSync(join(shardsRoot, "shards")) ? join(shardsRoot, "shards") : shardsRoot;
      const files = readdirSync(shardsDir)
        .filter((f) => f.endsWith(".jsonl"))
        .sort()
        .map((f) => join(shardsDir, f));
      if (files.length === 0) throw new Error(`No .jsonl shard files found in ${shardsDir}`);
      await runShardPoolOrJsonlMode({
        inPaths: files,
        outDir,
        preserveShards: true,
        recoveryRatio,
        allowBlindEdits: Boolean(values["allow-blind-edits"]),
        limit,
      });
    } else {
      await runShardPoolOrJsonlMode({
        inPaths: [resolve(values.in!)],
        outDir,
        preserveShards: false,
        recoveryRatio,
        allowBlindEdits: Boolean(values["allow-blind-edits"]),
        limit,
      });
    }
  }
} catch (error) {
  console.error(`rwkv-pi anneal-convert: ${error instanceof Error ? error.message : String(error)}`);
  process.exitCode = 1;
}
