#!/usr/bin/env python3
"""Install the exact GitHub Actions artifact, verifying outer and inner digests."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import zipfile

def sha(path):
    with path.open("rb") as f: return hashlib.file_digest(f, "sha256").hexdigest()

p = argparse.ArgumentParser(); p.add_argument("--archive", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
lock = json.loads((Path(__file__).resolve().parents[1] / "lightning-source.lock.json").read_text())
if sha(a.archive) != lock["artifactSha256"]: raise ValueError("Actions artifact SHA-256 mismatch")
if a.out.exists(): raise ValueError("Refusing to overwrite installed backend")
with tempfile.TemporaryDirectory(dir=a.archive.resolve().parent) as directory:
    temporary = Path(directory)
    with zipfile.ZipFile(a.archive) as z:
        archives = [n for n in z.namelist() if n.endswith(".tar.gz")]
        if len(archives) != 1: raise ValueError("Expected exactly one Linux bundle")
        name = archives[0]
        expected = z.read(name + ".sha256").decode().split()[0]
        bundle = temporary / "bundle.tar.gz"
        with z.open(name) as src, bundle.open("wb") as dst: shutil.copyfileobj(src, dst)
    if sha(bundle) != expected: raise ValueError("Inner bundle SHA-256 mismatch")
    unpack = temporary / "unpacked"; unpack.mkdir()
    with tarfile.open(bundle) as tar: tar.extractall(unpack, filter="data")
    roots = list(unpack.iterdir())
    if len(roots) != 1 or not roots[0].is_dir(): raise ValueError("Invalid bundle layout")
    root = roots[0]
    vocab = root / "rwkv_vocab_v20230424.txt"
    if sha(vocab) != lock["vocabSha256"]: raise ValueError("Bundled tokenizer vocabulary differs")
    names = ["rwkv_lighting_cuda", "rwkv_miss_tune", "rwkv_state_tune", "rwkv_nora_tune"]
    record = {"engineRevision": lock["revision"], "artifactSha256": lock["artifactSha256"], "bundleSha256": expected, "binaries": {name: sha(root / name) for name in names}}
    (root / "installation.json").write_text(json.dumps(record, indent=2) + "\n")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(root), a.out)
print(json.dumps({"installed": str(a.out.resolve()), **record}))
