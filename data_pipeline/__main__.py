"""Run with python3 -B -m data_pipeline. No third-party dependencies."""
import argparse
import json
from pathlib import Path

from pipeline.datakit import read_json, require_data_root, write_json
from .core import inventory, prepare_release, review_template


def main():
    parser = argparse.ArgumentParser(description="Question inventory and reviewed release; no paid calls")
    commands = parser.add_subparsers(dest="command", required=True)
    inv = commands.add_parser("inventory", help="Snapshot explicit final-bundle directories without executing them")
    inv.add_argument("--source", action="append", required=True, metavar="NAME=DIRECTORY")
    inv.add_argument("--out", type=Path, required=True)
    inv.add_argument("--heldout-manifest", type=Path)
    template = commands.add_parser("review-template", help="Create pending, hash-bound semantic review records")
    template.add_argument("--catalog", type=Path, required=True)
    template.add_argument("--id", action="append", required=True)
    template.add_argument("--out", type=Path, required=True)
    release = commands.add_parser("prepare-release", help="Freeze reviewed questions; never enqueue model work")
    release.add_argument("--catalog", type=Path, required=True)
    release.add_argument("--reviews", type=Path, required=True)
    release.add_argument("--out", type=Path, required=True)
    status = commands.add_parser("status", help="Summarize a question inventory")
    status.add_argument("catalog", type=Path)
    args = parser.parse_args()
    if args.command == "inventory":
        sources = []
        for value in args.source:
            name, sep, directory = value.partition("=")
            if not sep or not name or not directory:
                parser.error("--source requires NAME=DIRECTORY")
            sources.append((name, Path(directory)))
        result = inventory(sources, args.out, args.heldout_manifest)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return int(bool(result["errors"]))
    if args.command == "review-template":
        path = require_data_root(args.out)
        if path.exists():
            parser.error("Review file exists; refusing to replace decisions")
        result = review_template(args.catalog, args.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, result)
        print(path)
    elif args.command == "prepare-release":
        result = prepare_release(args.catalog, args.reviews, args.out)
        print(json.dumps({"questions": len(result["newBundles"]), "authorizedToStart": False,
                          "newTrainingTrajectories": 0, "path": str(args.out)}, ensure_ascii=False, indent=2))
    else:
        result = read_json(args.catalog)
        print(json.dumps({**result["summary"], "exactCollisionGroups": {
            k: len(v) for k, v in result["exactCollisions"].items()},
            "state": result["state"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
