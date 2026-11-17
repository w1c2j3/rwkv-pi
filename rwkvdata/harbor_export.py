"""Export a frozen bundle as a multi-step Harbor task (task.toml schema 1.4).

Layout:
    task.toml
    environment/Dockerfile, environment/workspace/<bundle files>
    steps/step-N/instruction.md        one per user turn, plus a final test turn
    steps/step-N/tests/{test.sh,run_checks.py,checks.json}
    steps/step-N/workdir/setup.sh       final step only, when a validated fault exists

The final turn is a debugging turn when `faults` (from faults.validate) holds killed
mutants: its setup plants one regression and the turn asks the agent to run the tests,
find the cause, fix and rerun. Without a validated fault it is a plain test turn.

Verifiers record environment signals only (bundle baseline checks, the project's own
tests after the final turn, whether a fault was planted and whether test files stayed
untouched during the debugging turn). They never gate later steps and never read agent
or harness logs; trajectory QC is the quality gate.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from . import faults as faults_mod
from . import test_turn
from .bundle import Bundle

PI_VERSION = "1.5.0"          # latest stable at least a week old (1.5.1 published 2026-11-09)
NODE_VERSION = "22.22.1"  # matches the runtime used for environment checks on 82
DOCKERFILE = f"""\
FROM ubuntu:24.04
# Hosts may inject a client-side proxy into builds; build and run without it.
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 \\
    http_proxy= https_proxy= HTTP_PROXY= HTTPS_PROXY= no_proxy= NO_PROXY=
ARG APT_MIRROR=http://cn.archive.ubuntu.com
RUN unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY \\
    && sed -i "s#http://archive.ubuntu.com#$APT_MIRROR#g" /etc/apt/sources.list.d/ubuntu.sources \\
    && apt-get -o Acquire::http::Proxy=false update \\
    && apt-get -o Acquire::http::Proxy=false install -y --no-install-recommends \\
      python3 python3-pytest build-essential make sqlite3 git curl ca-certificates xz-utils jq \\
    && rm -rf /var/lib/apt/lists/*
RUN unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY && curl -fsSL https://nodejs.org/dist/v{NODE_VERSION}/node-v{NODE_VERSION}-linux-x64.tar.xz \\
      | tar -xJ -C /usr/local --strip-components=1
# Bake the pinned Pi agent in so trials skip the stock nvm/Node/npm install (see pi_agent.PreinstalledPi).
ARG NPM_REGISTRY=https://registry.npmmirror.com
RUN unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY && npm install -g --ignore-scripts --registry "$NPM_REGISTRY" @earendil-works/pi-coding-agent@{PI_VERSION} \\
    && mkdir -p /root/.nvm && touch /root/.nvm/nvm.sh && pi --version
# Fallback installs fetch from GitHub inside the container on every trial and that route is
# intermittently unreachable from our hosts: make curl retry stalled or failed transfers, and
# have the nvm installer use plain downloads (METHOD=script) instead of a git clone.
RUN printf 'retry = 6\\nretry-all-errors\\nretry-delay = 2\\nconnect-timeout = 10\\nspeed-limit = 1000\\nspeed-time = 15\\n' > /root/.curlrc
ENV METHOD=script
WORKDIR /app
COPY workspace/ /app/
"""

# Docker on the collection host has ~15 usable address pools; a per-trial compose network caps concurrency there.
# The shared default bridge gives the same public egress without creating a network per trial.
COMPOSE_OVERRIDE = """\
services:
  main:
    network_mode: bridge
"""

TEST_SH = """\
#!/bin/bash
# Environment signals only; see run_checks.py.
python3 /tests/run_checks.py /tests/checks.json
"""

RUN_CHECKS = '''\
"""Run recorded checks from /app and write Harbor rewards. Never gates steps."""
import hashlib, json, shutil, subprocess, sys
from pathlib import Path

LOGS = Path("/logs/verifier")
LOGS.mkdir(parents=True, exist_ok=True)
spec = json.loads(Path(sys.argv[1]).read_text())
results = []
for check in spec["checks"]:
    try:
        r = subprocess.run(check["argv"], cwd="/app", capture_output=True, text=True,
                           timeout=check["timeoutSec"])
        ok, tail = r.returncode == 0, (r.stdout + r.stderr)[-4000:]
    except subprocess.TimeoutExpired:
        ok, tail = False, "timeout"
    results.append({"group": check["group"], "name": check["name"], "passed": ok, "output": tail})
rewards = {}
for group in sorted({c["group"] for c in results}):
    members = [c for c in results if c["group"] == group]
    rewards[group] = sum(c["passed"] for c in members) / len(members)
rewards["reward"] = rewards.get("tests_pass", rewards.get("bundle_checks", 0.0))
fault = Path(spec.get("faultRecord") or "/nonexistent")
if fault.is_file():  # debugging turn: planted-fault outcome and test-file integrity
    record = json.loads(fault.read_text())
    rewards["fault_injected"] = float(bool(record.get("injected")))
    changed = [rel for rel, digest in record.get("test_files_sha256", {}).items()
               if not (Path("/app") / rel).is_file()
               or hashlib.sha256((Path("/app") / rel).read_bytes()).hexdigest() != digest]
    rewards["tests_untouched"] = float(not changed)
    record["test_files_changed"] = changed
    (LOGS / "fault.json").write_text(json.dumps(record, ensure_ascii=False, indent=1))
(LOGS / "reward.json").write_text(json.dumps(rewards))
(LOGS / "checks.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
'''


def _toml_str(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)  # JSON strings are valid TOML basic strings


def _task_toml(bundle: Bundle, step_names: list[str], commands: list[str], source_sha: str,
               agent_timeout: int, verifier_timeout: int, fault_meta: dict) -> str:
    lines = [
        'schema_version = "1.4"',
        'multi_step_reward_strategy = "final"',
        "",
        "[task]",
        f"name = {_toml_str('rwkv-pi/' + bundle.id)}",
        f"description = {_toml_str(bundle.domain + ' ' + bundle.difficulty + ' multi-turn coding task')}",
        "",
    ]
    for name in step_names:
        lines += ["[[steps]]", f"name = {_toml_str(name)}", "",
                  "[steps.agent]", f"timeout_sec = {agent_timeout}.0", "",
                  "[steps.verifier]", f"timeout_sec = {verifier_timeout}.0", ""]
    lines += [
        "[metadata]",
        f"task_id = {_toml_str(bundle.id)}",
        f"family = {_toml_str(bundle.family)}",
        f"domain = {_toml_str(bundle.domain)}",
        f"difficulty_target = {_toml_str(bundle.difficulty)}",
        f"source_bundle_sha256 = {_toml_str(source_sha)}",
        f"test_commands = {json.dumps(commands, ensure_ascii=False)}",
        *(f"{k} = {json.dumps(v, ensure_ascii=False)}" for k, v in fault_meta.items()),
        "",
        "[verifier]",
        f"timeout_sec = {verifier_timeout}.0",
        'environment_mode = "shared"',
        "",
        "[agent]",
        f"timeout_sec = {agent_timeout}.0",
        "",
        "[environment]",
        "build_timeout_sec = 900.0",
        "cpus = 2",
        "memory_mb = 4096",
        "storage_mb = 8192",
        "",
        "[environment.env]",  # runtime override for host-injected proxies
        *(f'{k} = ""' for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")),
        "",
    ]
    return "\n".join(lines)


def _checks(bundle: Bundle, commands: list[str] | None, fault_record: str | None = None) -> dict:
    checks = [{"group": "bundle_checks", "name": f"bundle-{i}", "argv": list(c.argv),
               "timeoutSec": c.timeout_sec} for i, c in enumerate(bundle.checks)]
    for cmd in commands or []:
        checks.append({"group": "tests_pass", "name": cmd, "argv": ["bash", "-lc", cmd],
                       "timeoutSec": 300})
    spec = {"schemaVersion": 1, "checks": checks}
    if fault_record:
        spec["faultRecord"] = fault_record
    return spec


def task_dockerfile(base_image: str | None = None) -> str:
    """Task Dockerfile: either the full standalone recipe or a fast layer on top of a prebuilt base image."""
    if not base_image:
        return DOCKERFILE
    return (
        f"FROM {base_image}\n"
        "ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 \\\n"
        "    http_proxy= https_proxy= HTTP_PROXY= HTTPS_PROXY= no_proxy= NO_PROXY=\n"
        "WORKDIR /app\n"
        "COPY workspace/ /app/\n"
    )


def export(bundle: Bundle, out_root: Path, *, faults: dict | None = None, base_image: str | None = None,
           agent_timeout: int = 1800, verifier_timeout: int = 600) -> Path:
    """Write one Harbor task directory and return its path. Overwrites nothing.

    `faults` is the report from faults.validate(); with killed mutants the final turn
    becomes a debugging turn.
    """
    commands = test_turn.detect_commands(bundle.files)
    killed = [k for k in (faults or {}).get("killed") or [] if "__main__" not in k["old"]]
    final_turn = (test_turn.debug_instruction(bundle.id, commands) if killed
                  else test_turn.instruction(bundle.id, commands))
    turns = [*bundle.turns, final_turn]
    fault_meta = {
        "final_turn": "debug" if killed else "test",
        "fault_candidates": len(killed),
        "fault_baseline_pass": bool((faults or {}).get("baseline_pass")),
        "fault_tried": int((faults or {}).get("tried") or 0),
        "fault_survived": int((faults or {}).get("survived") or 0),
    }
    names = [f"step-{i + 1}" for i in range(len(turns))]

    task_dir = out_root / bundle.id
    if task_dir.exists():
        raise FileExistsError(task_dir)
    tmp = out_root / f".{bundle.id}.partial"
    shutil.rmtree(tmp, ignore_errors=True)
    workspace = tmp / "environment" / "workspace"
    for rel, text in bundle.files.items():
        target = workspace / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    (tmp / "environment" / "Dockerfile").write_text(task_dockerfile(base_image))
    (tmp / "environment" / "docker-compose.yaml").write_text(COMPOSE_OVERRIDE)

    source_sha = hashlib.sha256(bundle.source_path.read_bytes()).hexdigest()
    (tmp / "task.toml").write_text(
        _task_toml(bundle, names, commands, source_sha, agent_timeout, verifier_timeout, fault_meta))
    for index, (name, text) in enumerate(zip(names, turns)):
        step = tmp / "steps" / name
        tests = step / "tests"
        tests.mkdir(parents=True)
        (step / "instruction.md").write_text(text.rstrip() + "\n", encoding="utf-8")
        final = index == len(turns) - 1
        record = faults_mod.FAULT_RECORD if final and killed else None
        (tests / "checks.json").write_text(
            json.dumps(_checks(bundle, commands if final else None, record), ensure_ascii=False, indent=1))
        if record:
            (step / "workdir").mkdir()
            (step / "workdir" / "setup.sh").write_text(faults_mod.setup_script(
                killed, (faults or {}).get("passing_commands") or commands))
        (tests / "run_checks.py").write_text(RUN_CHECKS)
        (tests / "test.sh").write_text(TEST_SH)
        (tests / "test.sh").chmod(0o755)
    tmp.rename(task_dir)
    return task_dir
