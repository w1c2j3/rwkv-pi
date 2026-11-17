"""Budget-guarded DeepSeek trajectory collection through Harbor's stock Pi agent.

Pi's native system prompt plus pi_agent.ZH_SYSTEM_PROMPT (Chinese reasoning), no --thinking
flag, Pi's default tools. The Pi package version is pinned.

Spend is computed from ATIF usage at the ledger's full (non-discounted) tariff.
Stock Pi calls DeepSeek from inside the container, so requests cannot be reserved
one by one; instead every batch reserves a worst-case amount up front and the run
stops before any batch whose reservation would cross the cap.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from .harbor_export import PI_VERSION

MODEL = "deepseek/deepseek-flash"
SELF_HOSTED_MODEL = "deepseek/rwkv-ds41f-1017"
TARIFF_CNY_PER_M = {"input": 2.0, "cached": 0.04, "output": 8.0}  # deepseek-flash peak price
OFFPEAK_FACTOR = 0.5  # DeepSeek bills every token type at half price outside weekday peaks (pricing page, 2026-11-11)
MAX_TURNS_PER_STEP = 40
INITIAL_RESERVE_CNY = 3.0     # per task until real usage is observed (33 pilot trials: max 1.02)
RESERVE_FACTOR = 2.0          # reserve = factor x most expensive task seen so far
SHANGHAI = ZoneInfo("Asia/Shanghai")
PEAKS = ((9, 12), (14, 18))   # weekday peak hours, Asia/Shanghai; weekends are off-peak
MIN_LEFT = timedelta(minutes=20)
INITIAL_BATCH_SECONDS = 3600
# Harbor's Pi installer fetches nvm from raw.githubusercontent.com in every trial, which is
# intermittently unreachable from 157. A trial that never reached step 1 made no model call,
# so it is re-queued (bounded) instead of being counted as collected.
MAX_ATTEMPTS = 4
SETUP_RETRY_TYPES = ("NetworkConnectionError", "AgentSetupTimeoutError")
RETRY_PAUSE_SECONDS = 120


def next_peak(now: datetime) -> datetime | None:
    """Start of the peak period that contains or follows `now` (None = more than a day away)."""
    for day in range(3):
        d = (now + timedelta(days=day)).date()
        if d.weekday() >= 5:
            continue
        for start, end in PEAKS:
            s = datetime(d.year, d.month, d.day, start, tzinfo=SHANGHAI)
            e = datetime(d.year, d.month, d.day, end, tzinfo=SHANGHAI)
            if now < e:
                return s if now < s else now
    return None


def admissible(now: datetime, needed: timedelta) -> bool:
    peak = next_peak(now)
    return peak is None or peak - now >= max(MIN_LEFT, needed)


def in_peak(at: datetime) -> bool:
    at = at.astimezone(SHANGHAI)
    return at.weekday() < 5 and any(start <= at.hour < end for start, end in PEAKS)


def _tokens_cny(prompt: int, cached: int, output: int) -> float:
    return ((prompt - cached) * TARIFF_CNY_PER_M["input"] + cached * TARIFF_CNY_PER_M["cached"]
            + output * TARIFF_CNY_PER_M["output"]) / 1e6


def step_cost_cny(trajectory: dict) -> float:
    """Price each model call at the tariff of the window it ran in. Copied context from earlier steps is
    skipped; if the per-call metrics do not add up to `final_metrics`, the whole step is priced at peak."""
    m = trajectory.get("final_metrics") or {}
    totals = (m.get("total_prompt_tokens") or 0, m.get("total_cached_tokens") or 0,
              m.get("total_completion_tokens") or 0)
    calls = [s for s in trajectory.get("steps", [])
             if s.get("metrics") and not (s.get("is_copied_context") or (s.get("extra") or {}).get("is_copied_context"))]
    sums = tuple(sum(c["metrics"].get(k) or 0 for c in calls)
                 for k in ("prompt_tokens", "cached_tokens", "completion_tokens"))
    if sums != totals:
        return _tokens_cny(*totals)
    total = 0.0
    for c in calls:
        cm = c["metrics"]
        try:
            factor = 1.0 if in_peak(datetime.fromisoformat(c["timestamp"].replace("Z", "+00:00"))) else OFFPEAK_FACTOR
        except (KeyError, TypeError, ValueError):
            factor = 1.0
        total += factor * _tokens_cny(cm.get("prompt_tokens") or 0, cm.get("cached_tokens") or 0,
                                      cm.get("completion_tokens") or 0)
    return total


def trial_cost_cny(trial_dir: Path) -> float:
    return sum(step_cost_cny(json.loads(p.read_text())) for p in trial_dir.glob("steps/*/agent/trajectory.json"))


def job_costs(jobs_dir: Path) -> dict[str, float]:
    return {t.name: trial_cost_cny(t) for t in jobs_dir.glob("*/*") if (t / "steps").is_dir()}


def _trial_reached_step1(trial: Path) -> bool:
    return (trial / "steps" / "step-1" / "agent" / "trajectory.json").is_file()


def _trial_complete(trial: Path, tasks_dir: Path | None = None) -> bool:
    steps = sorted((trial / "steps").glob("step-*")) if (trial / "steps").is_dir() else []
    if not steps or any(not (s / "agent" / "trajectory.json").is_file() for s in steps):
        return False
    if tasks_dir is not None:
        task_steps = tasks_dir / trial.name.split("__")[0] / "steps"
        if task_steps.is_dir():
            expected = len(list(task_steps.glob("step-*")))
            if expected and len(steps) < expected:
                return False
    return True


def collected(jobs_dir: Path, *, require_accept: bool = False, tasks_dir: Path | None = None) -> set[str]:
    """Task names with at least one complete trial (or QC-accepted trial when require_accept=True)."""
    if not require_accept:
        return {t.name.split("__")[0] for t in jobs_dir.glob("*/*__*") if _trial_complete(t, tasks_dir)}
    from . import qc
    accepted: set[str] = set()
    for t in jobs_dir.glob("*/*__*"):
        if not _trial_complete(t, tasks_dir):
            continue
        if qc.evaluate(t).verdict == "accept":
            accepted.add(t.name.split("__")[0])
    return accepted


def _harbor_env(base_url: str | None = None) -> dict[str, str]:
    """Harbor imports rwkvdata.pi_agent by path, so the package root must be importable."""
    root = str(Path(__file__).resolve().parent.parent)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [root, os.environ.get("PYTHONPATH")]))}
    if base_url:
        env["DEEPSEEK_BASE_URL"] = base_url
    return env


def harbor_command(harbor: str, tasks_dir: Path, names: list[str], jobs_dir: Path, job: str,
                   concurrency: int, *, model: str = MODEL, base_url: str | None = None) -> list[str]:
    cmd = [harbor, "run", "-p", str(tasks_dir), "-o", str(jobs_dir), "--job-name", job,
           "-a", "rwkvdata.pi_agent:PreinstalledPi", "-m", model, "--ak", f"version={PI_VERSION}",
           "--ak", f"max_turns={MAX_TURNS_PER_STEP}"]
    if base_url:
        cmd += ["--ak", f"base_url={base_url}"]
    cmd += ["--resume-trajectory",
            "-n", str(concurrency), "--max-retries", "2", "-y", "-q",
            "--agent-setup-timeout-multiplier", "2"]  # slow GitHub fetches during agent install
    for exc in SETUP_RETRY_TYPES:  # in-job retries only for failures that precede any model call
        cmd += ["--retry-include", exc]
    for name in names:
        cmd += ["-i", name]  # local datasets match on directory name
    return cmd


def run(args) -> None:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        sys.exit("DEEPSEEK_API_KEY must be set in the orchestrator environment (never in job files).")
    base_url = getattr(args, "base_url", None) or os.environ.get("DEEPSEEK_BASE_URL")
    model = getattr(args, "model", None) or (SELF_HOSTED_MODEL if base_url else MODEL)
    require_accept = bool(getattr(args, "require_accept", False))
    max_attempts = getattr(args, "max_attempts", MAX_ATTEMPTS)
    ignore_peak = bool(args.ignore_peak or base_url)

    def task_names() -> list[str]:  # re-listed every round: a wave may still be adding tasks
        return sorted(p.name for p in Path(args.tasks).iterdir() if (p / "task.toml").is_file())

    jobs_dir = Path(args.jobs)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    lock = (jobs_dir / "collector.lock").open("w")  # one collector per jobs dir, or spend is double-counted
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(f"another collector already owns {jobs_dir}")
    ledger = jobs_dir / "spend.jsonl"
    tasks_root = Path(args.tasks)
    attempts: dict[str, int] = {}
    for existing in jobs_dir.glob("*/*__*"):
        if _trial_reached_step1(existing):
            name = existing.name.split("__")[0]
            attempts[name] = attempts.get(name, 0) + 1

    def remaining() -> list[str]:
        done = collected(jobs_dir, require_accept=require_accept, tasks_dir=tasks_root)
        return [t for t in task_names()[: args.limit] if t not in done and attempts.get(t, 0) < max_attempts]

    pending = remaining()

    batch_no = len(list(jobs_dir.glob("batch-*")))
    durations = [r["seconds"] for r in map(json.loads, ledger.read_text().splitlines())
                 if r.get("event") == "end"] if ledger.exists() else []
    while pending:
        needed = timedelta(seconds=1.25 * max(durations, default=INITIAL_BATCH_SECONDS))
        now = datetime.now(SHANGHAI)
        if not ignore_peak and not admissible(now, needed):
            if not args.wait:
                print(json.dumps({"stop": "outside-offpeak-window", "at": now.isoformat()}))
                break
            time.sleep(300)
            continue
        costs = {} if base_url else job_costs(jobs_dir)
        spent = sum(costs.values())
        # Conservative until real usage exists, then twice the most expensive task seen (scaled to peak if ignore_peak).
        peak_scale = (1.0 / OFFPEAK_FACTOR) if (args.ignore_peak and not base_url) else 1.0
        per_task = 0.0 if base_url else (RESERVE_FACTOR * peak_scale * max(costs.values()) if any(costs.values()) else INITIAL_RESERVE_CNY)
        batch_cap = args.batch if per_task == 0.0 else max(0, min(args.batch, int((args.cap - spent) // per_task)))
        batch = pending[: batch_cap]  # shrink to fit the cap
        reserve = per_task * len(batch)
        if not batch:
            print(json.dumps({"stop": "cap", "spentCny": round(spent, 4), "nextReserveCny": reserve}))
            break
        batch_no += 1
        job = f"batch-{batch_no:03d}"
        cmd = harbor_command(args.harbor, Path(args.tasks), batch, jobs_dir, job, args.concurrency,
                             model=model, base_url=base_url)
        record = {"job": job, "tasks": batch, "spentBeforeCny": round(spent, 4), "reserveCny": reserve,
                  **({"baseUrl": base_url, "model": model, "requireAccept": require_accept} if base_url or require_accept else {})}
        with ledger.open("a") as f:
            f.write(json.dumps({**record, "event": "start"}) + "\n")
        started = time.time()
        code = subprocess.run(cmd, env=_harbor_env(base_url)).returncode
        durations.append(time.time() - started)
        after = 0.0 if base_url else sum(job_costs(jobs_dir).values())
        with ledger.open("a") as f:
            f.write(json.dumps({**record, "event": "end", "exit": code, "seconds": round(durations[-1]),
                                "spentAfterCny": round(after, 4)}) + "\n")
        if code != 0:
            print(json.dumps({"stop": "harbor-exit", "exit": code}))
            break
        reached = {t.name.split("__")[0] for t in (jobs_dir / job).glob("*__*") if _trial_reached_step1(t)}
        for name in batch:
            attempts[name] = attempts.get(name, 0) + 1
        failed_setup = [t for t in batch if t not in reached]
        pending = remaining()
        if failed_setup:
            print(json.dumps({"setupFailed": failed_setup, "requeued": [t for t in failed_setup if t in pending]}),
                  flush=True)
            time.sleep(RETRY_PAUSE_SECONDS)
    gave_up = [t for t in task_names()[: args.limit] if t not in collected(jobs_dir, require_accept=require_accept, tasks_dir=tasks_root)]
    if gave_up:
        print(json.dumps({"notCollected": gave_up}))
    print(json.dumps({"spentCny": 0.0 if base_url else round(sum(job_costs(jobs_dir).values()), 4), "cap": args.cap}))


def main() -> None:
    p = argparse.ArgumentParser(prog="rwkvdata.collect")
    p.add_argument("--tasks", required=True)
    p.add_argument("--jobs", required=True)
    p.add_argument("--harbor", default="harbor")
    p.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL"),
                   help="custom OpenAI-compatible DeepSeek endpoint (e.g. http://<host>:58866/v1)")
    p.add_argument("--model", default=None, help="override model (defaults to deepseek/rwkv-ds41f-1017 when --base-url is set)")
    p.add_argument("--require-accept", action="store_true",
                   help="rejection sampling: retry tasks up to --max-attempts until qc.evaluate() returns accept")
    p.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    p.add_argument("--cap", type=float, default=100.0, help="hard pilot cap in CNY")
    p.add_argument("--batch", type=int, default=5)
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--wait", action="store_true", help="wait for the next off-peak window instead of exiting")
    p.add_argument("--ignore-peak", action="store_true",
                   help="run during weekday peak hours too (needs explicit owner approval; spend is always "
                        "priced at the peak tariff)")
    p.add_argument("--dry-run", action="store_true", help="print the first Harbor command only")
    args = p.parse_args()
    if args.dry_run:
        names = sorted(p.name for p in Path(args.tasks).iterdir())[: args.batch]
        model = args.model or (SELF_HOSTED_MODEL if args.base_url else MODEL)
        print(" ".join(harbor_command(args.harbor, Path(args.tasks), names, Path(args.jobs), "batch-001",
                                      args.concurrency, model=model, base_url=args.base_url)))
        return
    run(args)


if __name__ == "__main__":
    main()
