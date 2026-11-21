#!/usr/bin/env python3
"""
Low-RAM Disk-Spooling Builder for the 3-Round 6.50B (65亿 exact RWKV tokens) Open-Source Expansion Datasets:
  - Round 1: stage1c-2p2b-9k6-20261121 (2.20B tokens)
  - Round 2: stage1d-2p2b-9k6-20261121 (2.20B tokens)
  - Round 3: stage1e-2p1b-9k6-20261121 (2.10B tokens)

Memory-Safe Design (< 2.0 GiB total RSS across main + 4 workers):
- Verified compact JSONL rows are appended immediately to a binary spool file on /mnt/f/rwkv-pi-data/datasets/.
- Python keeps ONLY compact (tok, sup, src_id, byte_offset, byte_len) integer tuples in memory.
- Post-cutoff SWE-rebench parquet batches stream lazily via generator (4 nice workers, maxtasksperchild=12).
"""
from __future__ import annotations
import ast
import ctypes
import gzip
import hashlib
import json
import lzma
import multiprocessing as mp
import os
import pathlib
import random
import re
import shutil
import subprocess
import time
import pyarrow.parquet as pq

DATA_ROOT = pathlib.Path(os.environ.get("RWKV_PI_DATA_ROOT", "/mnt/f/rwkv-pi-data"))
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
VOCAB_PATH = pathlib.Path(os.environ.get("RWKV_VOCAB_PATH", str(REPO_ROOT / ".upstream/rwkv-lightning/assets/rwkv_vocab_v20230424.txt")))
STAGE1A_XZ = pathlib.Path(os.environ.get("RWKV_STAGE1A_XZ", str(DATA_ROOT / "stage1-1p2b-9k6-20261119.jsonl.xz")))
STAGE1B_GZ = pathlib.Path(os.environ.get("RWKV_STAGE1B_GZ", str(DATA_ROOT / "stage1b-1p4b-9k6-20261120/train.jsonl.gz")))
MEGA_DIR = pathlib.Path(os.environ.get("RWKV_MEGA_DIR", str(DATA_ROOT / "mega-corpus-9k6-20261118")))
POST_DIR = pathlib.Path(os.environ.get("RWKV_POST_DIR", str(DATA_ROOT / "raw-post-cutoff-20261121")))
DATASETS_DIR = pathlib.Path(os.environ.get("RWKV_DATASETS_DIR", str(DATA_ROOT / "datasets")))
SPOOL_PATH = DATASETS_DIR / ".s1cde-spool-20261121.bin"

NUM_WORKERS = 16

C_SRC = r"""
#include <stdint.h>
#include <string.h>

#define MAX_NODES 131072
static int32_t trie[MAX_NODES][256];
static uint8_t term[MAX_NODES];
static int32_t n_nodes = 1;

void init_trie(void) {
    memset(trie[0], -1, sizeof(trie[0]));
    term[0] = 0;
    n_nodes = 1;
}

int32_t add_token(const uint8_t* s, int len) {
    int32_t cur = 0;
    for (int i = 0; i < len; i++) {
        uint8_t c = s[i];
        int32_t nxt = trie[cur][c];
        if (nxt < 0) {
            if (n_nodes >= MAX_NODES) return -1;
            nxt = n_nodes++;
            memset(trie[nxt], -1, sizeof(trie[nxt]));
            term[nxt] = 0;
            trie[cur][c] = nxt;
        }
        cur = nxt;
    }
    term[cur] = 1;
    return n_nodes;
}

int64_t count_tokens(const uint8_t* s, int64_t n) {
    int64_t i = 0, cnt = 0;
    while (i < n) {
        int32_t cur = 0;
        int64_t best_next = i + 1;
        for (int64_t j = i; j < n; j++) {
            cur = trie[cur][s[j]];
            if (cur < 0) break;
            if (term[cur]) best_next = j + 1;
        }
        i = best_next;
        cnt++;
    }
    return cnt;
}
"""

SO_PATH = pathlib.Path("/tmp/rwkv_trie_s1cde.so")
SECRET_RE = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{24,}|ghp_[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{28,}|AKIA[0-9A-Z]{16})\b")
RSC_RE = re.compile(r"^\$[0-9a-fA-F]{1,5}$")

_WORKER_LIB = None


def compile_trie_so() -> None:
    c_path = pathlib.Path("/tmp/rwkv_trie_s1cde.c")
    c_path.write_text(C_SRC)
    subprocess.run(["gcc", "-O3", "-shared", "-fPIC", str(c_path), "-o", str(SO_PATH)], check=True)


def init_worker() -> int:
    global _WORKER_LIB
    lib = ctypes.CDLL(str(SO_PATH))
    lib.init_trie.restype = None
    lib.add_token.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.add_token.restype = ctypes.c_int32
    lib.count_tokens.argtypes = [ctypes.c_char_p, ctypes.c_int64]
    lib.count_tokens.restype = ctypes.c_int64
    lib.init_trie()
    used_nodes = 1
    with VOCAB_PATH.open("r", encoding="utf-8") as vf:
        for line in vf:
            line = line.strip()
            if not line:
                continue
            idx_end = line.find(" ")
            last_sp = line.rfind(" ")
            val = ast.literal_eval(line[idx_end + 1 : last_sp])
            raw = val.encode("utf-8") if isinstance(val, str) else bytes(val)
            used_nodes = int(lib.add_token(raw, len(raw)))
            if used_nodes < 0:
                raise RuntimeError("Trie overflow: increase MAX_NODES")
    _WORKER_LIB = lib
    return used_nodes


def tok_len(text: str) -> int:
    if not text:
        return 0
    raw = text.encode("utf-8", errors="replace")
    return int(_WORKER_LIB.count_tokens(raw, len(raw)))


def seg_fingerprint(segs: list[dict]) -> bytes:
    h = hashlib.blake2b(digest_size=12)
    h.update(str(len(segs)).encode())
    for s in segs[:3]:
        h.update(s.get("text", "")[:512].encode("utf-8", errors="ignore"))
    if len(segs) > 3:
        h.update(segs[-2].get("text", "")[:512].encode("utf-8", errors="ignore"))
    return h.digest()


def verify_and_compact_segs(segs: list[dict], max_ctx: int = 20470) -> tuple[int, int, bytes] | None:
    if not isinstance(segs, list) or len(segs) < 3:
        return None
    first_txt = segs[0].get("text", "")
    if not (first_txt.startswith("System:") or first_txt.startswith("User:")):
        return None
    last = segs[-1]
    if last.get("train", True) or last.get("text") != "<EOD>":
        return None
    total_toks = 0
    sup_toks = 0
    eos_pending = False
    clean_segs = []
    for seg in segs:
        txt = seg.get("text", "")
        if "\x00" in txt or SECRET_RE.search(txt):
            return None
        try:
            raw_txt = txt.encode("utf-8")
        except UnicodeEncodeError:
            return None
        tr = bool(seg.get("train", False))
        n_tok = int(_WORKER_LIB.count_tokens(raw_txt, len(raw_txt))) if raw_txt else 0
        if eos_pending:
            if n_tok == 0 or tr:
                return None
            sup_toks += 1
            eos_pending = False
        total_toks += n_tok
        if tr:
            sup_toks += n_tok
        item = {"text": txt, "train": tr}
        if seg.get("end_target") == 0:
            if not tr or n_tok == 0:
                return None
            item["end_target"] = 0
            eos_pending = True
        clean_segs.append(item)
        if total_toks > max_ctx:
            return None
    if eos_pending or total_toks < 128 or total_toks > max_ctx or sup_toks < 32:
        return None
    compact_bytes = (json.dumps({"segments": clean_segs}, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    return (total_toks, sup_toks, compact_bytes)


def process_mega_shard(shard_rel: str) -> tuple[str, list[tuple[str, bytes, int, int, bytes]]]:
    sh_path = MEGA_DIR / shard_rel
    out = []
    with sh_path.open("r", encoding="utf-8", errors="replace") as sf:
        for line in sf:
            if "\x00" in line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            rid = obj.get("id") or ""
            segs = obj.get("segments")
            if not segs:
                continue
            fp = seg_fingerprint(segs)
            ver = verify_and_compact_segs(segs)
            if ver is None:
                continue
            out.append((rid, fp, ver[0], ver[1], ver[2]))
    return (shard_rel, out)


def truncate_obs(text: str, max_chars: int = 7200) -> str:
    if len(text) <= max_chars:
        return text
    return text[:4400] + "\n...[observation truncated]...\n" + text[-2400:]


def clean_think(text: str) -> str:
    return text.replace("<think>", "").replace("</think>", "").strip()


def pack_turns_9k6(preamble: str, pairs: list[tuple[str, str]], id_prefix: str) -> list[tuple[str, bytes, int, int, bytes]]:
    if not pairs:
        return []
    pre_tok = tok_len(preamble)
    if pre_tok > 12000 or pre_tok == 0:
        return []
    pair_toks = [(tok_len(a), tok_len(u) if u else 1) for a, u in pairs]
    results = []
    n = len(pairs)
    idx = 0
    span_idx = 0
    assistant_tail = "\n\nAssistant: <think"
    while idx < n:
        segs: list[dict] = []
        ctx_start = max(0, idx - 2)
        cur_tok = pre_tok
        if ctx_start < idx:
            note = ""
            if ctx_start > 0:
                note = f"\n\n[Prior {ctx_start} tool steps completed; showing recent context.]"
            base_pre = preamble
            if base_pre.endswith(assistant_tail):
                base_pre = base_pre[: -len(assistant_tail)] + note + assistant_tail
            cur_text = base_pre
            for k in range(ctx_start, idx):
                a_txt, u_txt = pairs[k]
                cur_text += a_txt + (u_txt or "")
            c_tok = tok_len(cur_text)
            if c_tok <= 12500:
                segs.append({"text": cur_text, "train": False})
                cur_tok = c_tok
            else:
                segs.append({"text": preamble, "train": False})
                cur_tok = pre_tok
        else:
            segs.append({"text": preamble, "train": False})

        added = 0
        while idx < n:
            a_txt, u_txt = pairs[idx]
            a_tok, u_tok = pair_toks[idx]
            need = a_tok + (u_tok if (u_txt and idx + 1 < n) else 2)
            if cur_tok + need > 19200:
                if added == 0:
                    idx += 1
                break
            segs.append({"text": a_txt, "train": True, "end_target": 0})
            cur_tok += a_tok
            added += 1
            idx += 1
            if u_txt and idx < n and cur_tok + u_tok <= 18500 and cur_tok <= 11500:
                segs.append({"text": u_txt, "train": False})
                cur_tok += u_tok
            else:
                break
        if added > 0:
            segs.append({"text": "<EOD>", "train": False})
            ver = verify_and_compact_segs(segs)
            if ver is not None:
                rid = f"{id_prefix}__span{span_idx}"
                results.append((rid, seg_fingerprint(segs), ver[0], ver[1], ver[2]))
                span_idx += 1
    return results


def convert_swe_rebench_batch(rows_and_tools: tuple[list[dict], str]) -> list[tuple[str, bytes, int, int, bytes]]:
    rows, default_tools_json = rows_and_tools
    out = []
    tool_footer = (
        '\nWhen using tools, output one XML envelope per call: '
        '<tool_call>{"name":"read","arguments":{"path":"input.txt"}}</tool_call>. '
        'Use only the declared tool names and arguments. Tool results arrive inside <tool_response>. '
        'Otherwise answer normally.'
    )
    nl2 = "\n\n"
    nl1 = "\n"
    for row in rows:
        msgs = row.get("trajectory") or []
        if len(msgs) < 4 or msgs[0].get("role") != "system" or msgs[1].get("role") != "user":
            continue
        sys_txt = (msgs[0].get("content") or "").strip()
        usr_txt = (msgs[1].get("content") or "").strip()
        if not sys_txt or not usr_txt:
            continue
        preamble = f"System: {sys_txt}\nTools:\n{default_tools_json}{tool_footer}\n\nUser: {usr_txt}\n\nAssistant: <think"
        pairs: list[tuple[str, str]] = []
        i = 2
        bad = False
        while i < len(msgs):
            m = msgs[i]
            role = m.get("role")
            if role != "assistant":
                bad = True
                break
            i += 1
            tcalls = m.get("tool_calls") or []
            content = (m.get("content") or "").strip()
            call_lines = []
            thought_parts = []
            if tcalls:
                if content:
                    thought_parts.append(clean_think(content))
                for tc in tcalls:
                    fn_obj = tc.get("function") or {}
                    fn_name = fn_obj.get("name") or ""
                    raw_args = fn_obj.get("arguments")
                    if not fn_name:
                        bad = True
                        break
                    try:
                        args_obj = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                    except Exception:
                        bad = True
                        break
                    if fn_name == "think" and isinstance(args_obj, dict) and "thought" in args_obj and len(tcalls) > 1:
                        thought_parts.append(clean_think(str(args_obj["thought"])))
                        continue
                    call_lines.append(
                        "<tool_call>"
                        + json.dumps({"name": fn_name, "arguments": args_obj}, ensure_ascii=False, separators=(",", ":"))
                        + "</tool_call>"
                    )
                if bad:
                    break
                if not call_lines and thought_parts:
                    call_lines.append(
                        "<tool_call>"
                        + json.dumps({"name": "think", "arguments": {"thought": nl1.join(thought_parts)}}, ensure_ascii=False, separators=(",", ":"))
                        + "</tool_call>"
                    )
                    thought_parts = []
                think_str = nl2.join(p for p in thought_parts if p)
                a_body = f">{think_str}</think>\n" + nl1.join(call_lines) + nl2
            else:
                if not content:
                    bad = True
                    break
                cleaned_c = clean_think(content)
                a_body = f"></think>\n{cleaned_c}\n\n"

            obs_chunks = []
            while i < len(msgs) and msgs[i].get("role") in ("tool", "user"):
                tm = msgs[i]
                i += 1
                tc_out = truncate_obs((tm.get("content") or "").strip())
                if tm.get("role") == "tool":
                    obs_chunks.append(f"<tool_response>\n{tc_out}\n</tool_response>")
                else:
                    obs_chunks.append(tc_out)
            u_body = ("User: " + nl1.join(obs_chunks) + "\n\nAssistant: <think") if obs_chunks else ""
            pairs.append((a_body, u_body))
        if bad or len(pairs) < 2:
            continue
        tid_raw = str(row.get("trajectory_id") or "")[:24]
        tid = f"swe_rebench_{tid_raw}"
        out.extend(pack_turns_9k6(preamble, pairs, tid))
    return out


def iter_rebench_batches(pf_reb: pq.ParquetFile, tools_compact: str, batch_rows: int = 128):
    cur_batch = []
    for b in pf_reb.iter_batches(batch_size=256, columns=["trajectory_id", "resolved", "trajectory"]):
        for r in b.to_pylist():
            if r.get("resolved") == 1:
                cur_batch.append(r)
                if len(cur_batch) >= batch_rows:
                    yield (cur_batch, tools_compact)
                    cur_batch = []
    if cur_batch:
        yield (cur_batch, tools_compact)


def convert_terminalbench_clean() -> list[tuple[str, bytes, int, int, bytes]]:
    bash_tool_schema = json.dumps(
        [{"type": "function", "function": {"name": "bash", "description": "Execute a shell command in the terminal.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}],
        separators=(",", ":"),
    )
    footer = (
        '\nWhen using tools, output one XML envelope per call: '
        '<tool_call>{"name":"bash","arguments":{"command":"ls -la"}}</tool_call>. '
        'Use only the declared tool names and arguments. Tool results arrive inside <tool_response>. '
        'Otherwise answer normally.'
    )
    nl2 = "\n\n"
    nl1 = "\n"
    out = []
    for idx in (0, 1):
        pf = pq.ParquetFile(POST_DIR / f"yoonholee--terminalbench-trajectories/data/train-0000{idx}-of-00002.parquet")
        for b in pf.iter_batches(batch_size=1024):
            for r in b.to_pylist():
                if r.get("reward") != 1 or not r.get("steps") or r["steps"] == "null":
                    continue
                try:
                    steps = json.loads(r["steps"])
                except Exception:
                    continue
                if not isinstance(steps, list) or len(steps) < 3:
                    continue
                corrupted = False
                for s in steps:
                    for v in (s.get("msg"), s.get("obs")):
                        if isinstance(v, str) and (RSC_RE.match(v.strip()) or "[agent error]" in v):
                            corrupted = True
                            break
                    for tc in (s.get("tools") or []):
                        cmd = tc.get("cmd")
                        if isinstance(cmd, str) and RSC_RE.match(cmd.strip()):
                            corrupted = True
                        elif isinstance(cmd, list) and any(isinstance(x, str) and RSC_RE.match(x.strip()) for x in cmd):
                            corrupted = True
                    if corrupted:
                        break
                if corrupted:
                    continue

                sys_parts = []
                usr_parts = []
                pos = 0
                while pos < len(steps) and steps[pos].get("src") in ("system", "user"):
                    mtxt = (steps[pos].get("msg") or "").strip()
                    if mtxt and mtxt != "Warmup":
                        if steps[pos].get("src") == "system":
                            sys_parts.append(mtxt)
                        else:
                            usr_parts.append(mtxt)
                    pos += 1
                if not usr_parts:
                    continue
                sys_str = nl2.join(sys_parts) if sys_parts else "You are a helpful AI coding agent solving tasks in a Linux terminal environment."
                usr_str = nl2.join(usr_parts)
                preamble = f"System: {sys_str}\nTools:\n{bash_tool_schema}{footer}\n\nUser: {usr_str}\n\nAssistant: <think"

                pairs = []
                while pos < len(steps):
                    s = steps[pos]
                    pos += 1
                    if s.get("src") != "agent":
                        continue
                    msg = clean_think(s.get("msg") or "")
                    if msg.startswith("THOUGHT:"):
                        msg = msg[len("THOUGHT:") :].strip()
                    tcs = s.get("tools") or []
                    if tcs:
                        calls = []
                        for tc in tcs:
                            cmd = tc.get("cmd")
                            if isinstance(cmd, list):
                                cmd_str = cmd[-1] if len(cmd) >= 3 and cmd[0] in ("bash", "sh") else " ".join(str(x) for x in cmd)
                            else:
                                cmd_str = str(cmd or "")
                            if not cmd_str.strip():
                                continue
                            calls.append(
                                "<tool_call>"
                                + json.dumps({"name": "bash", "arguments": {"command": cmd_str}}, ensure_ascii=False, separators=(",", ":"))
                                + "</tool_call>"
                            )
                        if not calls:
                            continue
                        a_body = f">{msg}</think>\n" + nl1.join(calls) + nl2
                    else:
                        if not msg:
                            continue
                        a_body = f"></think>\n{msg}\n\n"
                    obs = truncate_obs((s.get("obs") or "").strip())
                    u_body = f"User: <tool_response>\n{obs}\n</tool_response>\n\nAssistant: <think" if obs else ""
                    pairs.append((a_body, u_body))
                if len(pairs) >= 2:
                    trial_raw = str(r.get("trial_id") or "")[:20]
                    tid = f"tbench_{trial_raw}"
                    out.extend(pack_turns_9k6(preamble, pairs, tid))
    return out


# Entry tuple stored in memory: (tok, sup, src_id, offset, byte_len)
SpoolEntry = tuple[int, int, int, int, int]


def write_stage_dataset(
    name: str,
    rows: list[SpoolEntry],
    src_names: list[str],
    spool_rf,
    seed: int,
    vocab_sha: str,
    dp_group: int = 144,
) -> dict:
    rng = random.Random(seed)
    rng.shuffle(rows)
    usable_n = (len(rows) // dp_group) * dp_group
    rows = rows[:usable_n]
    rows.sort(key=lambda x: x[0])
    batches = [rows[i : i + dp_group] for i in range(0, usable_n, dp_group)]
    rng.shuffle(batches)
    ordered = [item for batch in batches for item in batch]

    out_dir = DATASETS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    h = hashlib.sha256()
    total_toks = 0
    total_sup = 0
    max_tok = 0
    by_source: dict[str, dict[str, int]] = {}
    with train_path.open("wb", buffering=4 * 1024 * 1024) as tf:
        for tok, sup, src_id, offset, byte_len in ordered:
            spool_rf.seek(offset)
            bline = spool_rf.read(byte_len)
            tf.write(bline)
            h.update(bline)
            total_toks += tok
            total_sup += sup
            if tok > max_tok:
                max_tok = tok
            src = src_names[src_id]
            st = by_source.setdefault(src, {"rows": 0, "tokens": 0, "supervisedTokens": 0})
            st["rows"] += 1
            st["tokens"] += tok
            st["supervisedTokens"] += sup

    train_sha = h.hexdigest()
    empty_sha = hashlib.sha256(b"").hexdigest()
    (out_dir / "validation.jsonl").write_bytes(b"")
    (out_dir / "test.jsonl").write_bytes(b"")

    manifest = {
        "schemaVersion": 2,
        "format": "rwkv-lightning-segments-eos-target-v2",
        "requiredTrainingCapability": "segment-end-target-v1",
        "engineRevision": "d54c53696ed57001ee80209847d6cef1cf54abb6",
        "vocabSha256": vocab_sha,
        "name": name,
        "trainingReady": True,
        "dpGroupSize": dp_group,
        "createdAt": time.time(),
        "maxTokens": max_tok,
        "requiredCtx": 20480,
        "counts": {
            "rows": usable_n,
            "train": usable_n,
            "stepsAtBatch144": usable_n // dp_group,
            "validation": 0,
            "test": 0,
            "tokens": total_toks,
            "supervisedTokens": total_sup,
        },
        "bySource": by_source,
        "files": {
            "train.jsonl": train_sha,
            "validation.jsonl": empty_sha,
            "test.jsonl": empty_sha,
        },
        "rows": [{"split": "train", "tokens": max_tok}],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    sample_rep = dict(manifest["counts"])
    sample_rep["bySource"] = by_source
    sample_rep["trainSha256"] = train_sha
    (out_dir / "sample-report.json").write_text(json.dumps(sample_rep, indent=2) + "\n")
    gz_cmd = ["pigz", "-p", "16", "-1", "-c", str(train_path)] if shutil.which("pigz") else ["gzip", "-1", "-c", str(train_path)]
    with (out_dir / "train.jsonl.gz").open("wb") as gf:
        subprocess.run(gz_cmd, stdout=gf, check=True)
    return manifest


def rss_mib() -> float:
    try:
        for ln in pathlib.Path("/proc/self/status").read_text().splitlines():
            if ln.startswith("VmRSS:"):
                return int(ln.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def main() -> None:
    t0 = time.time()
    print(f"[1/6] Compiling C-Trie RWKV tokenizer (workers={NUM_WORKERS})...", flush=True)
    compile_trie_so()
    used_nodes = init_worker()
    print(f"      Trie initialized: used_nodes={used_nodes:,} / 131,072, main_rss={rss_mib():.0f} MiB", flush=True)
    vocab_sha = hashlib.sha256(VOCAB_PATH.read_bytes()).hexdigest()

    print("[2/6] Loading Stage-1a + Stage-1b IDs & segment fingerprints for 100% zero overlap...", flush=True)
    seen_ids: set[str] = set()
    seen_fps: set[bytes] = set()
    with lzma.open(STAGE1A_XZ, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("id"):
                seen_ids.add(obj["id"])
            if obj.get("segments"):
                seen_fps.add(seg_fingerprint(obj["segments"]))
    with gzip.open(STAGE1B_GZ, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("id"):
                seen_ids.add(obj["id"])
            if obj.get("segments"):
                seen_fps.add(seg_fingerprint(obj["segments"]))
    print(f"      Loaded seen_ids={len(seen_ids):,}, seen_fps={len(seen_fps):,} ({time.time()-t0:.1f}s, main_rss={rss_mib():.0f} MiB)", flush=True)

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    src_names: list[str] = []
    src_to_id: dict[str, int] = {}

    def get_src_id(s: str) -> int:
        sid = src_to_id.get(s)
        if sid is None:
            sid = len(src_names)
            src_names.append(s)
            src_to_id[s] = sid
        return sid

    spool_wf = SPOOL_PATH.open("wb", buffering=4 * 1024 * 1024)
    spool_pos = 0

    def append_spool(tok: int, sup: int, src: str, bline: bytes) -> SpoolEntry:
        nonlocal spool_pos
        off = spool_pos
        blen = len(bline)
        spool_wf.write(bline)
        spool_pos += blen
        return (tok, sup, get_src_id(src), off, blen)

    print("[3/6] Streaming post-cutoff SWE-rebench (resolved==1) + clean Terminal-Bench (reward==1) to disk spool...", flush=True)
    tools_raw = json.loads((POST_DIR / "nebius--SWE-rebench-openhands-trajectories/tools.json").read_text())
    tools_compact = json.dumps(tools_raw, ensure_ascii=False, separators=(",", ":"))
    pf_reb = pq.ParquetFile(POST_DIR / "nebius--SWE-rebench-openhands-trajectories/trajectories.parquet")

    post_rebench_rows: list[SpoolEntry] = []
    reb_src = "nebius/SWE-rebench-openhands-trajectories"
    done_batches = 0
    with mp.Pool(processes=NUM_WORKERS, initializer=init_worker, maxtasksperchild=12) as pool:
        for batch_out in pool.imap_unordered(convert_swe_rebench_batch, iter_rebench_batches(pf_reb, tools_compact, 128), chunksize=1):
            for rid, fp, tok, sup, bline in batch_out:
                if rid not in seen_ids and fp not in seen_fps:
                    seen_ids.add(rid)
                    seen_fps.add(fp)
                    post_rebench_rows.append(append_spool(tok, sup, reb_src, bline))
            done_batches += 1
            if done_batches % 50 == 0:
                print(f"      [rebench] batches={done_batches}, spans={len(post_rebench_rows):,}, spool_GB={spool_pos/1e9:.2f}, rss={rss_mib():.0f} MiB", flush=True)

    tb_raw = convert_terminalbench_clean()
    post_tbench_rows: list[SpoolEntry] = []
    tb_src = "yoonholee/terminalbench-trajectories"
    for rid, fp, tok, sup, bline in tb_raw:
        if rid not in seen_ids and fp not in seen_fps:
            seen_ids.add(rid)
            seen_fps.add(fp)
            post_tbench_rows.append(append_spool(tok, sup, tb_src, bline))
    del tb_raw

    reb_toks = sum(x[0] for x in post_rebench_rows)
    tb_toks = sum(x[0] for x in post_tbench_rows)
    print(
        f"      Post-cutoff packed: SWE-rebench rows={len(post_rebench_rows):,} ({reb_toks/1e6:.1f}M toks), "
        f"Terminal-Bench clean rows={len(post_tbench_rows):,} ({tb_toks/1e6:.1f}M toks) ({time.time()-t0:.1f}s, rss={rss_mib():.0f} MiB)",
        flush=True,
    )

    print("[4/6] Streaming & C-Trie verifying remaining mega-corpus-9k6 shards to disk spool...", flush=True)
    mega_manifest = json.loads((MEGA_DIR / "manifest.json").read_text())
    wanted_sources = {
        "nvidia/Open-SWE-Traces-claudecode",
        "SWE-bench/SWE-smith-trajectories",
        "open-thoughts/OpenThoughts-Agent-v1-SFT",
        "SI2CA-Training-Trajectories",
        "R2E-Gym/R2EGym-SFT-Trajectories",
        "RWKV-Agent-Lab/rwkv-agent-data:20261114-mix-relay",
        "RWKV-Agent-Lab/rwkv-agent-data:edit-train500-final",
        "SWE-Gym/OpenHands-SFT-Trajectories",
        "nvidia/Open-SWE-Traces-sweagent",
    }
    shard_to_src = {sh["shard"]: sh["source"] for sh in mega_manifest["shards"] if sh["source"] in wanted_sources}
    pools: dict[str, list[SpoolEntry]] = {
        "claudecode": [],
        "swesmith": [],
        "openthoughts": [],
        "diverse": [],
        "sweagent": [],
    }
    done_shards = 0
    total_shards = len(shard_to_src)
    with mp.Pool(processes=NUM_WORKERS, initializer=init_worker, maxtasksperchild=16) as pool:
        for shard_rel, items in pool.imap_unordered(process_mega_shard, list(shard_to_src.keys()), chunksize=1):
            src = shard_to_src[shard_rel]
            bkey = (
                "claudecode" if src == "nvidia/Open-SWE-Traces-claudecode"
                else "swesmith" if src == "SWE-bench/SWE-smith-trajectories"
                else "openthoughts" if src == "open-thoughts/OpenThoughts-Agent-v1-SFT"
                else "sweagent" if src == "nvidia/Open-SWE-Traces-sweagent"
                else "diverse"
            )
            for rid, fp, tok, sup, bline in items:
                if (rid and rid in seen_ids) or fp in seen_fps:
                    continue
                if rid:
                    seen_ids.add(rid)
                seen_fps.add(fp)
                pools[bkey].append(append_spool(tok, sup, src, bline))
            done_shards += 1
            if done_shards % 40 == 0 or done_shards == total_shards:
                print(f"      [mega] shards={done_shards}/{total_shards}, spool_GB={spool_pos/1e9:.2f}, rss={rss_mib():.0f} MiB", flush=True)

    spool_wf.flush()
    spool_wf.close()
    seen_ids.clear()
    seen_fps.clear()

    rng = random.Random(20261121)
    for k in ("claudecode", "swesmith", "openthoughts", "diverse"):
        rng.shuffle(pools[k])
    pools["sweagent"].sort(key=lambda x: (x[1] / max(1, x[0])) + rng.uniform(-0.03, 0.03), reverse=True)

    for k, lst in pools.items():
        print(f"      Untrained pool {k:13s}: rows={len(lst):8,d}, tokens={sum(x[0] for x in lst)/1e9:.3f}B", flush=True)

    def take_tokens(lst: list[SpoolEntry], cursor: int, budget: int) -> tuple[list[SpoolEntry], int, int]:
        picked = []
        acc = 0
        while cursor < len(lst) and acc < budget:
            picked.append(lst[cursor])
            acc += lst[cursor][0]
            cursor += 1
        return picked, cursor, acc

    print("[5/6] Partitioning exact 6.50B across Stage-1c (2.20B), Stage-1d (2.20B), Stage-1e (2.10B)...", flush=True)
    cc_cur = sm_cur = ot_cur = dv_cur = sw_cur = 0

    s1c_rows: list[SpoolEntry] = []
    part, cc_cur, _ = take_tokens(pools["claudecode"], cc_cur, 1_400_000_000)
    s1c_rows.extend(part)
    part, ot_cur, _ = take_tokens(pools["openthoughts"], ot_cur, 200_000_000)
    s1c_rows.extend(part)
    rem_1c = 2_205_000_000 - sum(x[0] for x in s1c_rows)
    part, sm_cur, _ = take_tokens(pools["swesmith"], sm_cur, rem_1c)
    s1c_rows.extend(part)

    s1d_rows: list[SpoolEntry] = []
    part, cc_cur, _ = take_tokens(pools["claudecode"], cc_cur, 10_000_000_000)
    s1d_rows.extend(part)
    part, sm_cur, _ = take_tokens(pools["swesmith"], sm_cur, 10_000_000_000)
    s1d_rows.extend(part)
    part, dv_cur, _ = take_tokens(pools["diverse"], dv_cur, 10_000_000_000)
    s1d_rows.extend(part)
    rem_1d = max(0, 2_205_000_000 - sum(x[0] for x in s1d_rows))
    part, sw_cur, _ = take_tokens(pools["sweagent"], sw_cur, rem_1d)
    s1d_rows.extend(part)

    rng.shuffle(post_rebench_rows)
    rng.shuffle(post_tbench_rows)
    s1e_rows: list[SpoolEntry] = list(post_tbench_rows)
    part_reb, _, _ = take_tokens(post_rebench_rows, 0, max(0, 1_650_000_000 - sum(x[0] for x in s1e_rows)))
    s1e_rows.extend(part_reb)
    rem_1e = max(0, 2_105_000_000 - sum(x[0] for x in s1e_rows))
    part, sw_cur, _ = take_tokens(pools["sweagent"], sw_cur, rem_1e)
    s1e_rows.extend(part)

    print("[6/6] Writing DP_GROUP=144 length-grouped train.jsonl + manifest.json + .gz archives...", flush=True)
    with SPOOL_PATH.open("rb") as spool_rf:
        m_1c = write_stage_dataset("stage1c-2p2b-9k6-20261121", s1c_rows, src_names, spool_rf, 202611211, vocab_sha)
        print(f"      DONE Stage-1c: rows={m_1c['counts']['rows']:,}, tokens={m_1c['counts']['tokens']:,}", flush=True)
        m_1d = write_stage_dataset("stage1d-2p2b-9k6-20261121", s1d_rows, src_names, spool_rf, 202611212, vocab_sha)
        print(f"      DONE Stage-1d: rows={m_1d['counts']['rows']:,}, tokens={m_1d['counts']['tokens']:,}", flush=True)
        m_1e = write_stage_dataset("stage1e-2p1b-9k6-20261121", s1e_rows, src_names, spool_rf, 202611213, vocab_sha)
        print(f"      DONE Stage-1e: rows={m_1e['counts']['rows']:,}, tokens={m_1e['counts']['tokens']:,}", flush=True)

    SPOOL_PATH.unlink(missing_ok=True)

    summary = {
        "totalExactTokens": m_1c["counts"]["tokens"] + m_1d["counts"]["tokens"] + m_1e["counts"]["tokens"],
        "totalSupervisedTokens": m_1c["counts"]["supervisedTokens"] + m_1d["counts"]["supervisedTokens"] + m_1e["counts"]["supervisedTokens"],
        "stages": {
            "stage1c": m_1c["counts"],
            "stage1d": m_1d["counts"],
            "stage1e": m_1e["counts"],
        },
        "elapsedSeconds": round(time.time() - t0, 1),
    }
    (DATASETS_DIR / "stage1cde-6p5b-summary-20261121.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("ALL 3 ROUNDS COMPLETE:", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
