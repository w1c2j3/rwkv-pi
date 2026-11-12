"""Merge a multi-step Harbor trial into one ATIF-v1.8 trajectory and attach shared `extra`.

Harbor writes one trajectory.json per step. With --resume-trajectory step N repeats
the events of steps 1..N-1 flagged `is_copied_context=true` and appends its own new
events; `final_metrics` counts the new events only. Taking the non-copied events of
each step in order therefore reproduces the full conversation (all user turns,
reasoning, tool calls, observations, final answers) exactly once, as the model saw it.
Steps are renumbered; nothing else is rewritten.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

SOURCE_ROOT = Path(__file__).resolve().parent
TIMEZONE = "Asia/Shanghai"


def source_sha256() -> str:
    """Content hash of the collector code (used while rwkvdata is uncommitted)."""
    h = hashlib.sha256()
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        if "tests" not in path.parts:
            h.update(path.relative_to(SOURCE_ROOT).as_posix().encode() + b"\0" + path.read_bytes())
    return h.hexdigest()


def _step_dirs(trial_dir: Path) -> list[Path]:
    return sorted((trial_dir / "steps").iterdir(), key=lambda p: int(p.name.split("-")[-1]))


def merge(trial_dir: Path) -> dict:
    parts = [json.loads((d / "agent" / "trajectory.json").read_text()) for d in _step_dirs(trial_dir)
             if (d / "agent" / "trajectory.json").is_file()]
    if not parts:
        raise FileNotFoundError(f"{trial_dir}: no step trajectories")
    steps = []
    totals = {"total_prompt_tokens": 0, "total_completion_tokens": 0, "total_cached_tokens": 0,
              "total_cost_usd": 0.0}
    reasoning_tokens = 0
    for part in parts:
        for s in part.get("steps", []):
            if s.get("is_copied_context"):
                continue
            steps.append({**s, "step_id": len(steps) + 1})
        metrics = part.get("final_metrics") or {}
        for key in totals:
            totals[key] += metrics.get(key) or 0
        reasoning_tokens += (metrics.get("extra") or {}).get("reasoning_tokens") or 0
    totals["total_steps"] = len(steps)
    totals["extra"] = {"reasoning_tokens": reasoning_tokens}
    merged = {k: v for k, v in parts[0].items() if k not in ("steps", "final_metrics", "extra")}
    # Harbor 0.25.0's Pi adapter emits ATIF-v1.7. v1.8 only adds audio content parts, so a
    # text-only v1.7 trajectory is valid v1.8 unchanged; the source version is kept in `extra`.
    merged.update(schema_version="ATIF-v1.8", steps=steps, final_metrics=totals)
    merged["_source_schema_versions"] = sorted({p.get("schema_version", "") for p in parts})
    return merged


def extra_fields(*, task_id: str, agent: dict, qc_verdict: str, pi_version: str,
                 teacher_model: str, reasoning_effort: str | None, collected_at: datetime) -> dict:
    """Shared field names; must stay identical to the DSH side's names once agreed."""
    return {
        "task_id": task_id,
        "harness_name": "pi",
        "harness_version": pi_version,
        "harness_adapter": f"harbor/{agent.get('name', 'pi')}@harbor-0.25.0",
        "collector_name": "rwkv-pi-data/rwkvdata",
        "collector_source_sha256": source_sha256(),
        "teacher_model": teacher_model,
        "reasoning_effort": reasoning_effort or "default",
        "collected_at": collected_at.astimezone(ZoneInfo(TIMEZONE)).isoformat(),
        "timezone": TIMEZONE,
        "reasoning_format": "verbatim",
        "qc_verdict": qc_verdict,
    }


def finalize(trial_dir: Path, *, task_id: str, qc_verdict: str, pi_version: str, teacher_model: str,
             reasoning_effort: str | None = None) -> Path:
    trajectory = merge(trial_dir)
    source_versions = trajectory.pop("_source_schema_versions")
    trajectory["extra"] = extra_fields(
        task_id=task_id, agent=trajectory.get("agent") or {}, qc_verdict=qc_verdict, pi_version=pi_version,
        teacher_model=teacher_model, reasoning_effort=reasoning_effort,
        collected_at=datetime.fromtimestamp((trial_dir / "result.json").stat().st_mtime, timezone.utc)
        if (trial_dir / "result.json").exists() else datetime.now(timezone.utc))
    trajectory["extra"]["source_schema_versions"] = source_versions
    trajectory["extra"]["merged_step_count"] = len(list((trial_dir / "steps").glob("*/agent/trajectory.json")))
    out = trial_dir / "trajectory.json"
    out.write_text(json.dumps(trajectory, ensure_ascii=False, indent=1))
    try:  # Harbor's validator is the acceptance gate for the shared format.
        from harbor.utils.trajectory_validator import TrajectoryValidator
    except ImportError:
        return out
    validator = TrajectoryValidator()
    if not validator.validate(trajectory):
        raise ValueError(f"{out}: ATIF validation failed: {validator.get_errors()[:5]}")
    return out
