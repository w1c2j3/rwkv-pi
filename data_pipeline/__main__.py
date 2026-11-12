"""Run with python3 -B -m data_pipeline. No third-party dependencies."""
import argparse
import json
from pathlib import Path

from pipeline.datakit import read_json, require_data_root, write_json
from .core import inventory, prepare_release, review_template
from .products import export_questions, export_training, plan_trajectories, compare_attempts


def main():
    parser = argparse.ArgumentParser(description="Questions or per-teacher trajectory datasets; collection requires --execute")
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
    product = commands.add_parser("produce", help="Export questions or reviewed per-teacher training data")
    product.add_argument("--mode", choices=["questions", "trajectories"], required=True)
    product.add_argument("--source", action="append", type=Path)
    product.add_argument("--questions", type=Path)
    product.add_argument("--profile", action="append", type=Path)
    product.add_argument("--entries", type=Path, help="Reviewed runs and existing native packed datasets")
    product.add_argument("--out", type=Path, required=True)
    product.add_argument("--execute", action="store_true", help="Run configured paid collection adapters after planning")
    comparison = commands.add_parser("compare", help="Group model attempts by exact question revision")
    comparison.add_argument("--attempts", type=Path, required=True)
    comparison.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "produce":
        if args.mode == "questions":
            if not args.source or args.questions or args.profile or args.entries or args.execute:
                parser.error("Question mode requires --source and cannot invoke teachers")
            result = export_questions(args.source, args.out)
        elif args.source:
            parser.error("Trajectory mode reads --questions, not mutable --source bundles")
        elif not args.questions:
            parser.error("Trajectory mode requires a frozen --questions directory")
        elif args.entries:
            if args.execute or args.profile:
                parser.error("Export already-reviewed entries separately from new model collection")
            result = export_training(args.questions, read_json(args.entries), args.out)
        else:
            if not args.profile:
                parser.error("Trajectory collection requires at least one --profile")
            result = plan_trajectories(args.questions, [read_json(p) for p in args.profile], args.out)
            if args.execute:
                from .execution import collect_plan
                result = collect_plan(args.out, args.profile)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "compare":
        target = require_data_root(args.out)
        if target.exists():parser.error("Comparison output already exists")
        target.parent.mkdir(parents=True, exist_ok=True)
        result = compare_attempts(read_json(args.attempts));write_json(target, result)
        print(json.dumps({"questionVersions": result["questionVersions"], "path": str(target)}, ensure_ascii=False))
        return 0
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
