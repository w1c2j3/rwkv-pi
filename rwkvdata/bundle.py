"""Read-only view of a frozen question bundle (schemaVersion 1).

Bundles are the pipeline's source of truth. This module never rewrites them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class Check:
    argv: tuple[str, ...]
    timeout_sec: float


@dataclass(frozen=True)
class Bundle:
    id: str
    family: str
    domain: str
    difficulty: str
    prompt: str
    followups: tuple[str, ...]
    files: dict[str, str]
    checks: tuple[Check, ...]
    environment: str
    source_path: Path

    @property
    def turns(self) -> tuple[str, ...]:
        """User requests in order: the initial prompt, then each follow-up."""
        return (self.prompt, *self.followups)


def with_turns(b: Bundle, turns: list[str]) -> Bundle:
    """The bundle with reviewed user turns (e.g. labels stripped); the frozen file is not touched.

    TASK.md embeds the first request, so it is updated when it contains the original text."""
    if not turns or len(turns) != len(b.turns):
        raise ValueError(f"{b.id}: {len(turns)} reviewed turns for {len(b.turns)} bundle turns")
    files = dict(b.files)
    if "TASK.md" in files and b.prompt in files["TASK.md"]:
        files["TASK.md"] = files["TASK.md"].replace(b.prompt, turns[0], 1)
    return replace(b, prompt=turns[0], followups=tuple(turns[1:]), files=files)


def load_turns(path: str | Path) -> dict[str, list[str]]:
    """Reviewed turns by bundle id from a JSON-lines file of {"id", "turns"} records."""
    return {r["id"]: r["turns"] for r in map(json.loads, Path(path).read_text(encoding="utf-8").splitlines()) if r}


def load(path: str | Path) -> Bundle:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schemaVersion") != 1:
        raise ValueError(f"{path}: unsupported bundle schemaVersion {raw.get('schemaVersion')!r}")
    task = raw["task"]
    for name in task["files"]:
        if name.startswith(("/", "\\")) or ".." in Path(name).parts or ":" in name:
            raise ValueError(f"{path}: unsafe workspace path {name!r}")
    checks = tuple(
        Check(tuple(c["argv"]), c.get("timeoutMs", 60000) / 1000)
        for c in task["checks"] if c.get("type") == "command"
    )
    return Bundle(
        id=task["id"],
        family=task["family"],
        domain=task["domain"],
        difficulty=task.get("difficulty", ""),
        prompt=task["prompt"],
        followups=tuple(f["prompt"] for f in raw.get("followups", [])),
        files=dict(task["files"]),
        checks=checks,
        environment=raw.get("review", {}).get("baseEnvironmentId") or task["family"],
        source_path=path,
    )
