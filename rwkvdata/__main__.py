"""python -m rwkvdata {select,faults,export,qc,finalize,dsh-import} ..."""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

from . import atif, bundle as bundle_mod, collect, dsh_import, faults, harbor_export, qc, test_turn


def _bundles(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in map(Path, paths):
        out += sorted(p.rglob("*.json")) if p.is_dir() else [p]
    return [p for p in out if p.name != "manifest.json"]


def cmd_select(args) -> None:
    """Pick bundles: >= N follow-ups, detectable tests, not already used, at most
    --per-environment per environment, round-robin over domains."""
    by_domain: dict[str, list[str]] = collections.defaultdict(list)
    per_env: collections.Counter = collections.Counter()
    seen: set[str] = set()
    for listing in args.exclude_list:  # bundle paths or ids already used by earlier waves
        seen |= {Path(line.strip()).stem for line in Path(listing).read_text().splitlines() if line.strip()}
    for path in _bundles(args.inputs):
        b = bundle_mod.load(path)
        if (b.id not in seen and len(b.followups) >= args.min_followups and b.domain not in args.exclude_domain
                and (args.per_environment == 0 or per_env[b.environment] < args.per_environment)
                and test_turn.detect_commands(b.files)):
            per_env[b.environment] += 1
            seen.add(b.id)
            by_domain[b.domain].append(str(path))
    count = args.count or sum(map(len, by_domain.values()))
    picked: list[str] = []
    while len(picked) < count and any(by_domain.values()):
        for domain in sorted(by_domain):
            if by_domain[domain] and len(picked) < count:
                picked.append(by_domain[domain].pop(0))
    print("\n".join(picked))


def cmd_faults(args) -> None:
    """Validate regression candidates for exported task dirs (reads environment/workspace and
    metadata.test_commands); writes <out>/<task>.json. Runs tests in Docker, no API spend."""
    import tomllib
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runner = faults.docker_runner(args.image)

    def one(task: Path) -> None:
        target = out / f"{task.name}.json"
        if target.exists():
            return
        root = task / "environment" / "workspace"
        files = {p.relative_to(root).as_posix(): p.read_text(encoding="utf-8", errors="replace")
                 for p in root.rglob("*") if p.is_file()}
        commands = tomllib.loads((task / "task.toml").read_text())["metadata"]["test_commands"]
        try:
            report = faults.validate(files, commands, faults.candidates(files, task.name), runner)
        except Exception as exc:  # one broken workspace must not stop the batch
            print(json.dumps({"task": task.name, "error": str(exc)[:200]}), flush=True)
            return
        target.write_text(json.dumps(report, ensure_ascii=False, indent=1))
        print(json.dumps({"task": task.name, "baseline_pass": report["baseline_pass"],
                          "killed": len(report["killed"]), "tried": report["tried"]}), flush=True)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(one, map(Path, args.tasks)))


def cmd_export(args) -> None:
    if args.print_base_dockerfile:
        print(harbor_export.DOCKERFILE.replace("COPY workspace/ /app/\n", ""), end="")
        return
    if not args.out or not args.inputs:
        raise SystemExit("export requires inputs and --out (unless --print-base-dockerfile is set)")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reviewed = bundle_mod.load_turns(args.turns) if args.turns else {}
    for path in _bundles(args.inputs):
        b = bundle_mod.load(path)
        if args.turns:
            if b.id not in reviewed:
                raise SystemExit(f"{b.id}: no reviewed turns in {args.turns}")
            b = bundle_mod.with_turns(b, reviewed[b.id])
        report = None
        if args.faults_dir and (Path(args.faults_dir) / f"{b.id}.json").is_file():
            report = json.loads((Path(args.faults_dir) / f"{b.id}.json").read_text())
        print(harbor_export.export(b, out, faults=report, base_image=args.base_image))


def cmd_qc(args) -> None:
    reports = [qc.evaluate(t) for t in args.trials]
    for r in reports:
        print(qc.to_json(r))
    accepted = sum(r.verdict == "accept" for r in reports)
    print(json.dumps({"trials": len(reports), "accepted": accepted}), file=sys.stderr)


def cmd_finalize(args) -> None:
    """QC every complete trial, write <trial>/qc.json and the merged, validated <trial>/trajectory.json
    (shared `extra` included), and print one summary line per trial plus totals."""
    rows = []
    tasks_dir = Path(args.tasks) if getattr(args, "tasks", None) else None
    for trial in map(Path, args.trials):
        task_id = collect.trial_task(trial)
        if not collect._trial_complete(trial, tasks_dir):
            rows.append({"trial": trial.name, "task_id": task_id, "status": "incomplete"})
            continue
        report = qc.evaluate(trial)
        (trial / "qc.json").write_text(qc.to_json(report))
        row = {"trial": trial.name, "task": task_id, "task_id": task_id, "status": "ok", "verdict": report.verdict, "reasons": report.reasons,
               "effective_calls": report.effective_calls, "repairs": report.repair_iterations,
               "final_turn_repairs": report.final_turn_repairs, "tests_pass": report.final_tests_pass,
               "fault_injected": bool(report.fault and report.fault["injected"]),
               "cost_cny": round(collect.trial_cost_cny(trial), 4)}
        try:
            out = atif.finalize(trial, task_id=task_id, qc_verdict=report.verdict,
                                pi_version=args.harness_version, teacher_model=args.model,
                                reasoning_effort=args.reasoning_effort)
            merged = json.loads(out.read_text())
            agent_steps = [s for s in merged["steps"] if s["source"] == "agent"]
            row.update(atif_valid=True, user_turns=sum(s["source"] == "user" for s in merged["steps"]),
                       agent_steps=len(agent_steps),
                       with_reasoning=sum(bool(s.get("reasoning_content")) for s in agent_steps))
        except (ValueError, FileNotFoundError) as exc:
            row.update(atif_valid=False, atif_error=str(exc)[:300])
        rows.append(row)
    if not getattr(args, "all_trials", False):
        best_by_task: dict[str, int] = {}
        for idx, row in enumerate(rows):
            if row["status"] != "ok":
                continue
            tid = row["task_id"]
            prev = best_by_task.get(tid)
            if prev is None or (row["verdict"] == "accept" or rows[prev]["verdict"] != "accept"):
                best_by_task[tid] = idx
        for idx, row in enumerate(rows):
            if row["status"] == "ok" and best_by_task.get(row["task_id"]) != idx:
                row["status"] = "superseded"
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
    ok = [r for r in rows if r["status"] == "ok"]
    all_done = [r for r in rows if r["status"] in ("ok", "superseded")]
    print(json.dumps({"trials": len(rows), "complete": len(ok),
                      "accepted": sum(r["verdict"] == "accept" for r in ok),
                      "atif_valid": sum(bool(r.get("atif_valid")) for r in ok),
                      "cost_cny": round(sum(r["cost_cny"] for r in all_done), 4)}), file=sys.stderr)


def cmd_dsh_import(args) -> None:
    print(json.dumps(dsh_import.run(Path(args.root), Path(args.out),
                                    Path(args.translations) if args.translations else None,
                                    base_image=args.base_image)))


def main() -> None:
    parser = argparse.ArgumentParser(prog="rwkvdata")
    sub = parser.add_subparsers(required=True)
    s = sub.add_parser("select")
    s.add_argument("inputs", nargs="+")
    s.add_argument("--count", type=int, default=20)
    s.add_argument("--min-followups", type=int, default=2)
    s.add_argument("--exclude-domain", action="append", default=["git"])
    s.add_argument("--per-environment", type=int, default=1, help="max bundles per environment (0 = no limit)")
    s.add_argument("--exclude-list", action="append", default=[], help="file of bundle paths/ids already used")
    s.add_argument("--all", dest="count", action="store_const", const=0, help="no total limit")
    s.set_defaults(func=cmd_select)
    e = sub.add_parser("export")
    e.add_argument("inputs", nargs="*")
    e.add_argument("--out")
    e.add_argument("--faults-dir", help="validated fault reports from `faults`; enables the debugging turn")
    e.add_argument("--turns", help="JSON-lines {id, turns} of reviewed user turns to use instead of the bundle's")
    e.add_argument("--base-image", help="prebuilt Harbor base image (skips per-task apt/Node/Pi network install)")
    e.add_argument("--print-base-dockerfile", action="store_true", help="print the prebuilt base Dockerfile and exit")
    e.set_defaults(func=cmd_export)
    f = sub.add_parser("faults")
    f.add_argument("tasks", nargs="+")
    f.add_argument("--out", required=True)
    f.add_argument("--workers", type=int, default=1, help="parallel Docker validations")
    f.add_argument("--image", required=True, help="image with the task toolchain (workspace is mounted)")
    f.set_defaults(func=cmd_faults)
    q = sub.add_parser("qc")
    q.add_argument("trials", nargs="+")
    q.set_defaults(func=cmd_qc)
    z = sub.add_parser("finalize")
    z.add_argument("trials", nargs="+")
    z.add_argument("--tasks", help="optional tasks root to verify all expected step-N trajectories exist")
    z.add_argument("--all-trials", action="store_true", help="keep superseded retry trials as status=ok")
    z.add_argument("--harness-version", default=collect.PI_VERSION)
    z.add_argument("--model", default=collect.MODEL)
    z.add_argument("--reasoning-effort", default=None)
    z.set_defaults(func=cmd_finalize)
    d = sub.add_parser("dsh-import")
    d.add_argument("root", help="rwkv-dsh-training-questions/harbor")
    d.add_argument("out")
    d.add_argument("--translations", help="JSON {task name: [Chinese turns]} for English tasks")
    d.add_argument("--base-image", help="prebuilt Harbor base image (skips per-task apt/Node/Pi network install)")
    d.set_defaults(func=cmd_dsh_import)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
