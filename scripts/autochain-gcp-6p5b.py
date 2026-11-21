#!/usr/bin/env python3
"""
Turnkey Unattended 3-Round 6.5B Expansion + Spot/Cross-Server Auto-Resume Launcher
Supports:
  - Google Cloud a4-highgpu-8g (8x B200 180GB)
  - Google Cloud a3-ultragpu-8g (8x H200 141GB)
  - Google Cloud a3-megagpu-8g / a3-highgpu-8g (8x H100 80GB)
  - 4x / 8x / 16x RTX PRO 6000 / A100-80GB

Features:
  - Auto-detects GPU count & VRAM/SMs to select optimal --devices, --chunk, --micro-bsz
  - Reconstructs bit-exact rwkv7-g1l-7.2b-stage1ab-merged.pth from HF base + 2x144MB adapters
    if not already present (saves uploading 14.4 GB over local broadband!)
  - Automatically resumes from the latest checkpoint-* directory on Spot/DWS preemption
    or mid-run server migration!
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import pathlib
import re
import shutil
import subprocess
import sys
import time


def run_cmd(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print(f"[EXEC] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=env)


def detect_gpus() -> tuple[str, int, int]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.total", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().splitlines()
    indices = []
    min_mib = 999999
    for line in out:
        idx_s, mem_s = [x.strip() for x in line.split(",")]
        indices.append(idx_s)
        min_mib = min(min_mib, int(mem_s))
    return ",".join(indices), len(indices), min_mib


def find_latest_checkpoint(ckpt_root: pathlib.Path) -> pathlib.Path | None:
    if not ckpt_root.exists():
        return None
    best_step = -1
    best_dir = None
    for p in ckpt_root.iterdir():
        m = re.match(r"^checkpoint-(\d+)$", p.name)
        if m and (p / "training.pth").exists():
            st = int(m.group(1))
            if st > best_step:
                best_step = st
                best_dir = p
    return best_dir


def ensure_decompressed(gz_or_jsonl: pathlib.Path) -> pathlib.Path:
    if gz_or_jsonl.suffix == ".jsonl" and gz_or_jsonl.exists():
        return gz_or_jsonl
    jsonl_path = gz_or_jsonl.with_suffix("") if gz_or_jsonl.suffix == ".gz" else gz_or_jsonl
    if jsonl_path.exists() and jsonl_path.stat().st_size > 0:
        return jsonl_path
    if not gz_or_jsonl.exists():
        raise FileNotFoundError(f"Missing dataset archive: {gz_or_jsonl}")
    print(f"[DATA] Decompressing {gz_or_jsonl} -> {jsonl_path} ...", flush=True)
    tmp_path = jsonl_path.with_suffix(".jsonl.tmp")
    with gzip.open(gz_or_jsonl, "rb") as rf, tmp_path.open("wb") as wf:
        shutil.copyfileobj(rf, wf, length=16 * 1024 * 1024)
    tmp_path.replace(jsonl_path)
    return jsonl_path


def ensure_stage1ab_base(work: pathlib.Path, portable: pathlib.Path, env: dict[str, str]) -> pathlib.Path:
    merged_ab = work / "weights/rwkv7-g1l-7.2b-stage1ab-merged.pth"
    if merged_ab.exists() and merged_ab.stat().st_size > 14_000_000_000:
        return merged_ab
    merged_ab.parent.mkdir(parents=True, exist_ok=True)
    raw_base = work / "weights/rwkv7-g1l-7.2b-20261031-ctx34304.pth"
    if not raw_base.exists():
        print("[WEIGHTS] Downloading BlinkDL/rwkv7-g1 rwkv7-g1l-7.2b-20261031-ctx34304.pth from HuggingFace...", flush=True)
        run_cmd(
            [
                sys.executable,
                "-m",
                "huggingface_hub.commands.huggingface_cli",
                "download",
                "BlinkDL/rwkv7-g1",
                "rwkv7-g1l-7.2b-20261031-ctx34304.pth",
                "--local-dir",
                str(work / "weights"),
            ]
        )
    merge_bin = portable / "rwkv_merge_nora"
    ad_1a = work / "checkpoints-pro8-20261121/stage1a-final/adapter-final.pth"
    ad_1b = work / "checkpoints-pro8-20261121/stage1b-final/adapter-final.pth"
    merged_1a = work / "weights/rwkv7-g1l-7.2b-stage1a-merged.pth"
    if not merged_1a.exists():
        run_cmd([str(merge_bin), "--base", str(raw_base), "--adapter", str(ad_1a), "--output", str(merged_1a)], env=env)
    run_cmd([str(merge_bin), "--base", str(merged_1a), "--adapter", str(ad_1b), "--output", str(merged_ab)], env=env)
    return merged_ab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=pathlib.Path, default=pathlib.Path("/workspace/rwkv-6p5b"))
    ap.add_argument("--portable", type=pathlib.Path, default=pathlib.Path("/workspace/rwkv-6p5b/portable-v6"))
    ap.add_argument("--devices", default="")
    ap.add_argument("--batch-size", type=int, default=144)
    ap.add_argument("--save-every", type=int, default=150)
    args = ap.parse_args()

    import os
    env = dict(os.environ)
    ld_parts = [
        str(args.portable / "usr/lib/x86_64-linux-gnu"),
        str(args.portable / "deps/jsoncpp/usr/lib/x86_64-linux-gnu"),
        "/usr/local/cuda/lib64",
    ]
    if env.get("LD_LIBRARY_PATH"):
        ld_parts.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(ld_parts)

    devs, n_gpus, min_mib = detect_gpus()
    if args.devices:
        devs = args.devices
        n_gpus = len(devs.split(","))

    per_gpu = (args.batch_size + n_gpus - 1) // n_gpus
    if min_mib >= 135000:
        # B200 (180G) or H200 (141G): full stride=1 across 300k+ rows
        chunk, mbsz = 448, max(per_gpu + 4, 22)
    elif min_mib >= 92000:
        # RTX PRO 6000 (96G): chunk=384 maximizes L2 residency & keeps 225k+ rows in stride=1
        chunk, mbsz = 384, max(per_gpu + 2, 19)
    else:
        # H100 / A100 (80G): chunk=384 with adaptive stride=1/2 single-window pool
        chunk, mbsz = 384, max(per_gpu + 2, 19)

    print(f"[HW] Detected {n_gpus} GPUs ({devs}), min_VRAM={min_mib} MiB -> chunk={chunk}, micro-bsz={mbsz}", flush=True)

    tune_bin = args.portable / "rwkv_nora_tune_dp"
    merge_bin = args.portable / "rwkv_merge_nora"
    vocab = args.portable / "assets/rwkv_vocab_v20230424.txt"

    cur_base = ensure_stage1ab_base(args.work, args.portable, env)

    rounds = [
        ("stage1c", "stage1c-2p2b-9k6-20261121", "4e-5", "1e-5", "rwkv7-g1l-7.2b-stage1abc-merged.pth"),
        ("stage1d", "stage1d-2p2b-9k6-20261121", "3.5e-5", "8e-6", "rwkv7-g1l-7.2b-stage1abcd-merged.pth"),
        ("stage1e", "stage1e-2p1b-9k6-20261121", "3e-5", "5e-6", "rwkv7-g1l-7.2b-stage1abcde-6p5b-merged.pth"),
    ]

    for tag, ds_name, lr_init, lr_fin, next_merged_name in rounds:
        merged_out = args.work / "weights" / next_merged_name
        ckpt_dir = args.work / "runs" / f"nora-{tag}-7p2b-r32" / "checkpoint"
        final_adapter = ckpt_dir / "adapter-final.pth"

        if merged_out.exists() and merged_out.stat().st_size > 14_000_000_000:
            print(f"[SKIP] {tag} already merged at {merged_out}", flush=True)
            cur_base = merged_out
            continue

        if not final_adapter.exists():
            ds_dir = args.work / "datasets" / ds_name
            train_jsonl = ensure_decompressed(
                ds_dir / ("train.jsonl" if (ds_dir / "train.jsonl").exists() else "train.jsonl.gz")
            )
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                str(tune_bin),
                "--model", str(cur_base),
                "--data", str(train_jsonl),
                "--output", str(ckpt_dir),
                "--vocab", str(vocab),
                "--ctx", "20480",
                "--chunk", str(chunk),
                "--micro-bsz", str(mbsz),
                "--batch-size", str(args.batch_size),
                "--epochs", "1",
                "--save-every", str(args.save_every),
                "--warmup-steps", "40",
                "--seed", "1234",
                "--gemm-mode", "auto",
                "--devices", devs,
                "--dp-profile",
                "--rank", "32",
                "--alpha", "32",
                "--targets", "all",
                "--lr", lr_init,
                "--lr-final", lr_fin,
            ]
            resume_dir = find_latest_checkpoint(ckpt_dir)
            if resume_dir is not None:
                print(f"[RESUME] Found {resume_dir}, auto-resuming {tag}!", flush=True)
                cmd.extend(["--resume", str(resume_dir)])
            t0 = time.time()
            run_cmd(cmd, env=env)
            print(f"[DONE] {tag} training finished in {(time.time()-t0)/3600:.2f}h", flush=True)

        print(f"[MERGE] Merging {final_adapter} into {cur_base} -> {merged_out}", flush=True)
        run_cmd([str(merge_bin), "--base", str(cur_base), "--adapter", str(final_adapter), "--output", str(merged_out)], env=env)
        cur_base = merged_out

    print(f"[ALL COMPLETE] Final 6.5B expanded base ready at: {cur_base}", flush=True)


if __name__ == "__main__":
    main()
