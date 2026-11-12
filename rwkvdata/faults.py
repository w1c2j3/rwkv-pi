"""Deterministic regression injection for the debugging turn.

Why: frozen environments mostly pass their own tests on the first run, so a plain
"run the tests" turn rarely produces a feedback loop. Instead, before the final turn a
small, realistic regression is planted in non-test source ("a teammate's change") and
the agent is asked to run the tests, diagnose the failure, fix it and rerun.

Pipeline:
  1. `candidates()` proposes single-line mutants of source files (never test files)
     with language-agnostic token operators (comparison boundaries, boolean logic,
     arithmetic, literals).
  2. `validate()` runs the project's test commands in the task image: the baseline must
     pass and a mutant is kept only if the tests then fail ("killed"). No LLM, no spend.
  3. `setup_script()` renders the step's `workdir/setup.sh`. At step start it applies the
     first validated mutant whose exact line still exists once (earlier turns may have
     rewritten code), re-runs the tests to confirm they fail, otherwise reverts and tries
     the next. It records the outcome outside /app, hashes the test
     files (to detect tampering during the debug turn) and deletes itself.

The record lives at FAULT_RECORD and is collected as a step artifact; the step verifier
reads it to report `fault_injected` and `tests_untouched`. It is environment state, not
a harness log, and QC treats any agent access to it as harness probing.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

FAULT_DIR = "/var/lib/rwkvdata"
FAULT_RECORD = f"{FAULT_DIR}/fault.json"
MAX_TRIED = 24      # mutants validated per task
KEEP = 4            # validated mutants shipped per task (fallbacks for edited code)

TEST_FILE = re.compile(
    r"(^|/)(tests?|__tests__|spec|fixtures?)/|(^|/)test_[^/]*$|_test\.[a-z]+$|\.(test|spec)\.[a-z]+$")
SOURCE_SUFFIXES = {".py", ".js", ".mjs", ".cjs", ".ts", ".c", ".h", ".go", ".sh", ".rs"}
_COMMENT = re.compile(r"^\s*(#|//|/\*|\*|--|\"\"\"|''')")

# (name, pattern, replacement). Patterns avoid matching inside longer operators.
OPERATORS: tuple[tuple[str, str, str], ...] = (
    ("lt->le", r"(?<![<>=!-])<(?![<=>-])", "<="),
    ("le->lt", r"(?<![<>=!])<=(?!=)", "<"),
    ("gt->ge", r"(?<![<>=!-])>(?![>=])", ">="),
    ("ge->gt", r"(?<![<>=!])>=(?!=)", ">"),
    ("eq->ne", r"(?<![<>=!])==(?!=)", "!="),
    ("ne->eq", r"!=(?!=)", "=="),
    ("and->or", r"\band\b", "or"),
    ("or->and", r"\bor\b", "and"),
    ("&&->||", r"&&", "||"),
    ("||->&&", r"\|\|", "&&"),
    ("plus1->minus1", r"\+ 1\b", "- 1"),
    ("minus1->plus1", r"- 1\b", "+ 1"),
    ("true->false", r"\bTrue\b", "False"),
    ("false->true", r"\bFalse\b", "True"),
    ("js-true->false", r"\btrue\b", "false"),
    ("js-false->true", r"\bfalse\b", "true"),
    ("sh-gt->ge", r"-gt\b", "-ge"),
    ("sh-lt->le", r"-lt\b", "-le"),
    ("sh-eq->ne", r"-eq\b", "-ne"),
    ("not-drop", r"\bnot\s+(?=[A-Za-z_(])", ""),
)


@dataclass(frozen=True)
class Mutant:
    path: str
    line_no: int
    old: str          # full original line, without newline
    new: str
    operator: str


def _outside_strings(line: str, start: int) -> bool:
    """Crude but conservative: even number of quotes before the match position."""
    prefix = line[:start]
    return all(prefix.count(q) % 2 == 0 for q in ('"', "'", "`"))


def is_test_path(path: str) -> bool:
    return bool(TEST_FILE.search(path))


def candidates(files: dict[str, str], seed: str) -> list[Mutant]:
    """All single-site mutants of non-test source lines, in a seeded order."""
    out: list[Mutant] = []
    for path, text in sorted(files.items()):
        if is_test_path(path) or Path(path).suffix not in SOURCE_SUFFIXES:
            continue
        lines = text.split("\n")
        counts: dict[str, int] = {}
        for line in lines:
            counts[line] = counts.get(line, 0) + 1
        for no, line in enumerate(lines, 1):
            stripped = line.strip()
            if (not stripped or _COMMENT.match(line) or counts[line] != 1 or len(stripped) > 160
                    or stripped.startswith(("import ", "from ", "#include", "assert ", "print(", "console."))
                    or "__main__" in line):
                continue
            for name, pattern, repl in OPERATORS:
                for m in re.finditer(pattern, line):
                    if not _outside_strings(line, m.start()):
                        continue
                    new = line[: m.start()] + repl + line[m.end():]
                    if new != line:
                        out.append(Mutant(path, no, line, new, name))
                    break  # one site per operator per line
    rng = random.Random(hashlib.sha256(seed.encode()).hexdigest())
    rng.shuffle(out)
    # Spread across files: round-robin by path after the seeded shuffle.
    by_path: dict[str, list[Mutant]] = {}
    for m in out:
        by_path.setdefault(m.path, []).append(m)
    ordered: list[Mutant] = []
    while any(by_path.values()):
        for path in list(by_path):
            if by_path[path]:
                ordered.append(by_path[path].pop(0))
    return ordered


Runner = Callable[[Path, list[str]], list[tuple[int, str]]]


def docker_runner(image: str, timeout: int = 300) -> Runner:
    """Run test commands in `image` on a private copy of the workspace (read-only mount copied
    to /app inside the container), no network; nothing is written back to the host."""
    def run(workdir: Path, commands: list[str]) -> list[tuple[int, str]]:
        results = []
        for cmd in commands:
            # A planted fault can loop forever. Killing the docker client leaves the container running,
            # so the command is bounded inside the container and the named container removed on expiry.
            name = f"rwkvdata-fault-{uuid.uuid4().hex[:12]}"
            try:
                r = subprocess.run(
                    ["docker", "run", "--rm", "--name", name, "--network", "none", "-v", f"{workdir}:/src:ro", image,
                     "timeout", "-k", "10", str(timeout), "bash", "-lc",
                     f"mkdir -p /app && cp -a /src/. /app/ && cd /app && {cmd}"],
                    capture_output=True, text=True, timeout=timeout + 30)
                results.append((r.returncode, (r.stdout + r.stderr)[-2000:]))
            except subprocess.TimeoutExpired:
                subprocess.run(["docker", "rm", "-f", name], capture_output=True)
                results.append((124, "timeout"))
        return results
    return run


def _write_tree(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")


def validate(files: dict[str, str], commands: list[str], mutants: list[Mutant], runner: Runner,
             *, keep: int = KEEP, max_tried: int = MAX_TRIED) -> dict:
    """Return {"baseline_pass", "passing_commands", "killed": [...], "tried", "survived"};
    never raises on test failure."""
    report = {"baseline_pass": False, "passing_commands": [], "killed": [], "tried": 0, "survived": 0}
    with tempfile.TemporaryDirectory(prefix="rwkvdata-fault-") as tmp:
        base = Path(tmp) / "base"
        _write_tree(base, files)
        # Only commands that pass on the untouched workspace can detect a regression; a command
        # that already fails (missing build step, genuinely failing test) is left out.
        commands = [c for c, (rc, _) in zip(commands, runner(base, commands)) if rc == 0]
        report["passing_commands"] = commands
        if not commands:
            return report
        report["baseline_pass"] = True
        used_paths: set[str] = set()
        for m in mutants[:max_tried]:
            if len(report["killed"]) >= keep:
                break
            report["tried"] += 1
            work = Path(tmp) / f"m{report['tried']}"
            shutil.copytree(base, work)
            target = work / m.path
            text = target.read_text(encoding="utf-8")
            target.write_text(text.replace(m.old + "\n", m.new + "\n", 1)
                              if (m.old + "\n") in text else text.replace(m.old, m.new, 1), encoding="utf-8")
            results = runner(work, commands)
            shutil.rmtree(work, ignore_errors=True)
            failing = [(cmd, out) for cmd, (rc, out) in zip(commands, results) if rc not in (0, 124)]
            if failing and m.path not in used_paths:
                used_paths.add(m.path)
                report["killed"].append({**asdict(m), "failing_command": failing[0][0],
                                         "failure_tail": failing[0][1][-600:]})
            elif not failing:
                report["survived"] += 1
    return report


SETUP_TEMPLATE = r'''#!/bin/bash
# Plants one validated regression before this turn; see rwkvdata/faults.py.
set -u
mkdir -p __FAULT_DIR__ && chmod 700 __FAULT_DIR__
python3 - <<'PY'
import hashlib, json, os, re, subprocess
MUTANTS = json.loads(__MUTANTS__)
COMMANDS = json.loads(__COMMANDS__)
TEST_FILE = re.compile(__TEST_FILE__)
COMPILED = (".c", ".h", ".cc", ".cpp", ".go", ".rs")
record = {"injected": False, "attempts": []}

def tests_fail():
    for cmd in COMMANDS:
        try:
            r = subprocess.run(["bash", "-lc", cmd], cwd=__APP__, capture_output=True, timeout=300)
        except subprocess.TimeoutExpired:
            continue
        if r.returncode != 0:
            return cmd
    return None

for m in MUTANTS:
    path = os.path.join(__APP__, m["path"])
    try:
        text = open(path, encoding="utf-8").read()
    except (FileNotFoundError, UnicodeDecodeError):
        record["attempts"].append({"path": m["path"], "result": "missing"}); continue
    lines = text.split("\n")
    if lines.count(m["old"]) != 1:
        record["attempts"].append({"path": m["path"], "result": "line-changed"}); continue
    st = os.stat(path)
    lines[lines.index(m["old"])] = m["new"]
    open(path, "w", encoding="utf-8").write("\n".join(lines))
    # Keep the old mtime so "newest file" does not give the fault away; compiled sources
    # need a fresh one or make would not rebuild them.
    if not path.endswith(COMPILED):
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
    failing = tests_fail()
    if failing:
        record.update(injected=True, mutant=m, failing_command=failing)
        break
    open(path, "w", encoding="utf-8").write(text)
    if not path.endswith(COMPILED):
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
    record["attempts"].append({"path": m["path"], "result": "survived"})

digests = {}
for root, dirs, names in os.walk(__APP__):
    dirs[:] = [d for d in dirs if d not in (".git", "node_modules", "__pycache__", ".pytest_cache")]
    for name in names:
        rel = os.path.relpath(os.path.join(root, name), __APP__)
        if TEST_FILE.search(rel):
            digests[rel] = hashlib.sha256(open(os.path.join(root, name), "rb").read()).hexdigest()
record["test_files_sha256"] = digests
json.dump(record, open("__FAULT_RECORD__", "w"), ensure_ascii=False, indent=1)
PY
# Remove byproducts of the confirmation run so the workspace looks as the agent left it.
find __APPSH__ -name __pycache__ -type d -newer "$0" -prune -exec rm -rf {} + 2>/dev/null
rm -rf __APPSH__/.pytest_cache
rm -f -- "$0"
exit 0
'''


def setup_script(killed: list[dict], commands: list[str], *, app: str = "/app",
                 fault_dir: str = FAULT_DIR) -> str:
    mutants = [{k: m[k] for k in ("path", "line_no", "old", "new", "operator")} for m in killed]
    return (SETUP_TEMPLATE
            .replace("__FAULT_DIR__", fault_dir)
            .replace("__FAULT_RECORD__", f"{fault_dir}/fault.json")
            .replace("__APPSH__", app)
            .replace("__APP__", repr(app))
            .replace("__MUTANTS__", repr(json.dumps(mutants, ensure_ascii=False)))
            .replace("__COMMANDS__", repr(json.dumps(commands, ensure_ascii=False)))
            .replace("__TEST_FILE__", repr(TEST_FILE.pattern)))
