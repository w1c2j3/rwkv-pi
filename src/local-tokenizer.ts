import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { sha256 } from "./provenance.js";

const PACKAGE_ROOT = fileURLToPath(new URL("../", import.meta.url));
const MAX_VOCAB_LEN = 256;

interface CompiledTrie {
  readonly next: Int32Array;
  readonly term: Uint8Array;
}

let cachedTrie: CompiledTrie | null | undefined;

function processVocabFormat(input: string): string {
  if (input.length > 0 && (input[0] === "'" || input[0] === '"')) {
    return input.slice(1, input.length - 2);
  }
  if (input.length > 1 && input[0] === "b" && (input[1] === "'" || input[1] === '"')) {
    return input.slice(2, input.length - 2);
  }
  return input;
}

function processEscapes(input: string, utf8String: boolean, utf8ByteLength: number): Uint8Array {
  const rawBuf = Buffer.from(input, "utf8");
  if (utf8String && utf8ByteLength > 0 && rawBuf.length > 1 && rawBuf[0] === 0x5c && (rawBuf[1] === 0x75 || rawBuf[1] === 0x78)) {
    const out: number[] = [];
    let i = 0;
    while (i < rawBuf.length) {
      const b = rawBuf[i]!;
      if (b === 0x5c && i + 1 < rawBuf.length && (rawBuf[i + 1] === 0x75 || rawBuf[i + 1] === 0x78)) {
        i += 2;
        const end = Math.min(rawBuf.length, i + 4);
        const hexCode = rawBuf.subarray(i, end).toString("ascii");
        i = end;
        const cp = Number.parseInt(hexCode, 16) || 0;
        if (cp <= 0x7f) out.push(cp);
        else if (cp <= 0x7ff) {
          out.push(192 + (cp >> 6), 128 + (cp & 0x3f));
        } else if (cp <= 0xffff) {
          out.push(224 + (cp >> 12), 128 + ((cp >> 6) & 0x3f), 128 + (cp & 0x3f));
        } else if (cp <= 0x10ffff) {
          out.push(240 + (cp >> 18), 128 + ((cp >> 12) & 0x3f), 128 + ((cp >> 6) & 0x3f), 128 + (cp & 0x3f));
        }
      } else {
        out.push(b);
        i++;
      }
    }
    while (out.length < utf8ByteLength) out.unshift(0);
    return Uint8Array.from(out);
  }

  const out: number[] = [];
  let escape = false;
  for (let i = 0; i < rawBuf.length; i++) {
    const b = rawBuf[i]!;
    if (escape) {
      if (b === 0x78 /* 'x' */) {
        if (i + 2 < rawBuf.length) {
          const hexDigits = String.fromCharCode(rawBuf[i + 1]!, rawBuf[i + 2]!);
          i += 2;
          out.push(Number.parseInt(hexDigits, 16) & 0xff);
        } else {
          i = rawBuf.length;
        }
      } else if (b === 0x6e /* 'n' */) out.push(0x0a);
      else if (b === 0x74 /* 't' */) out.push(0x09);
      else if (b === 0x72 /* 'r' */) out.push(0x0d);
      else if (b === 0x5c /* '\\' */) out.push(0x5c);
      else if (b === 0x27 /* '\'' */) out.push(0x27);
      else if (b === 0x22 /* '"' */) out.push(0x22);
      else {
        out.push(0x5c, b);
      }
      escape = false;
    } else if (b === 0x5c /* '\\' */) {
      escape = true;
    } else {
      out.push(b);
    }
  }
  return Uint8Array.from(out);
}

function loadPinnedTrie(root = PACKAGE_ROOT): CompiledTrie | null {
  const vocabPath = join(root, ".upstream/rwkv-lightning/assets/rwkv_vocab_v20230424.txt");
  const lockPath = join(root, "lightning-source.lock.json");
  if (!existsSync(vocabPath) || !existsSync(lockPath)) return null;
  const vocabRaw = readFileSync(vocabPath);
  const lock = JSON.parse(readFileSync(lockPath, "utf8")) as { vocabSha256?: string };
  if (lock.vocabSha256 && sha256(vocabRaw) !== lock.vocabSha256) return null;

  let capNodes = 131072;
  let next = new Int32Array(capNodes * 256).fill(-1);
  let term = new Uint8Array(capNodes);
  let nodeCount = 1;

  const ensureNode = (): number => {
    if (nodeCount >= capNodes) {
      const nextCap = capNodes * 2;
      const grownNext = new Int32Array(nextCap * 256).fill(-1);
      grownNext.set(next);
      next = grownNext;
      const grownTerm = new Uint8Array(nextCap);
      grownTerm.set(term);
      term = grownTerm;
      capNodes = nextCap;
    }
    return nodeCount++;
  };

  const addBytes = (bytes: Uint8Array) => {
    if (bytes.length === 0 || bytes.length > MAX_VOCAB_LEN) return;
    let cur = 0;
    for (let i = 0; i < bytes.length; i++) {
      const slot = cur * 256 + bytes[i]!;
      let child = next[slot]!;
      if (child < 0) {
        child = ensureNode();
        next[slot] = child;
      }
      cur = child;
    }
    term[cur] = 1;
  };

  const lines = vocabRaw.toString("utf8").split("\n");
  for (const line of lines) {
    if (!line) continue;
    const firstSpace = line.indexOf(" ");
    const lastSpace = line.lastIndexOf(" ");
    if (firstSpace < 0 || lastSpace <= firstSpace) continue;
    const utf8ByteLength = Number.parseInt(line.slice(lastSpace + 1), 10);
    const utf8String = line[firstSpace + 1] !== "b";
    const formatted = processVocabFormat(line.slice(firstSpace + 1, lastSpace + 1));
    const bytes = processEscapes(formatted, utf8String, utf8ByteLength);
    addBytes(bytes);
  }
  // Exact parity with OptimizedTrieTokenizer in upstream src/utils/trie.h (`root->add("<EOD>", 0, 0)`).
  addBytes(Buffer.from("<EOD>", "utf8"));

  return {
    next: next.subarray(0, nodeCount * 256),
    term: term.subarray(0, nodeCount),
  };
}

/**
 * Counts tokens in-memory using the exact pinned Lightning `OptimizedTrieTokenizer` byte-trie
 * without any network RTT or subprocess spawn. Returns `undefined` only if the pinned local
 * vocabulary file is absent or unverified.
 */
export function countTokensLocal(text: string): number | undefined {
  if (cachedTrie === undefined) {
    try {
      cachedTrie = loadPinnedTrie();
    } catch {
      cachedTrie = null;
    }
  }
  if (!cachedTrie) return undefined;
  const src = Buffer.from(text, "utf8");
  if (src.length === 0) return 0;
  const { next, term } = cachedTrie;
  let idx = 0;
  let count = 0;
  while (idx < src.length) {
    let cur = 0;
    let bestNext = -1;
    let scan = idx;
    while (scan < src.length) {
      const child = next[cur * 256 + src[scan]!]!;
      if (child < 0) break;
      cur = child;
      scan++;
      if (term[cur]) bestNext = scan;
    }
    if (bestNext <= idx) break;
    idx = bestNext;
    count++;
  }
  return count;
}
