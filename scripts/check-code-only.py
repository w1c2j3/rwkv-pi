#!/usr/bin/env python3
"""Check complete Git trees/history against an explicitly reviewed code-only manifest.

Print paths and violation kinds only. Never print matched private content.
The allowlist is a review boundary, not proof that arbitrary future code is harmless.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path


def git(*args):
    return subprocess.check_output(["git", *args])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", action="store_true", help="Check the complete staged tree")
    parser.add_argument("--refs", nargs="+", default=["HEAD"])
    args = parser.parse_args()
    root = Path(git("rev-parse", "--show-toplevel").decode().strip())
    allowed = {s.strip() for s in (root / ".github/code-only-files.txt").read_text().splitlines()
               if s.strip() and not s.startswith("#")}
    refs = [git("write-tree").decode().strip()] if args.index else git("rev-list", *args.refs).decode().splitlines()
    patterns = {
        "private-question-id": re.compile(rb"\b(?:dsa5k-[0-9]{6}|ds5k-(?:x[0-9]{3}|cxg-|[a-z]+-[a-z]+-))"),
        "api-credential": re.compile(rb"\b(?:sk-[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{24,})"),
        "private-key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    }
    violations, checked = set(), set()
    for revision in refs:
        for entry in git("ls-tree", "-rz", revision).split(b"\0"):
            if not entry:
                continue
            info, raw_path = entry.split(b"\t", 1)
            mode, kind, oid = info.decode().split()
            path = raw_path.decode()
            if path not in allowed or kind != "blob" or mode == "120000":
                violations.add((path, "not-an-approved-code-file"))
                continue
            if oid in checked:
                continue
            checked.add(oid)
            raw = git("cat-file", "blob", oid)
            for name, pattern in patterns.items():
                if pattern.search(raw):
                    violations.add((path, name))
    if violations:
        for path, kind in sorted(violations):
            print(f"BLOCKED {kind}: {path}", file=sys.stderr)
        print("Keep questions, generators, references, checks, trajectories and reports on private storage.", file=sys.stderr)
        return 1
    print(f"Code-only check passed: {len(refs)} trees, {len(checked)} unique approved blobs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
