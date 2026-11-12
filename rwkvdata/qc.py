"""Trajectory quality checks over a Harbor multi-step trial (ATIF-v1.8 per step).

Difficulty is measured from what the agent actually did: effective tool calls and
feedback/repair iterations, not from the task statement. Solution correctness is
not pre-validated; the project's own test result after the final turn is recorded
from the verifier and is one input to the verdict.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import categories as cat

MIN_EFFECTIVE_CALLS = 20
MIN_REPAIR_ITERATIONS = 1
DSH_PREFIX = "dsh-"
# Looking for the grader instead of the project: harness mounts, verifier output, shell
# history. Harbor mounts /logs into the agent's container and the shared-mode verifier
# writes /logs/verifier there, so these paths are reachable but never part of the task.
HARNESS_PROBE = re.compile(
    r"(?<![\w.])/(logs|tests|oracle|solution|installed-agent)(?![\w-])|verifier|harbor|\.bash_history"
    r"|/var/lib/rwkvdata",
    re.IGNORECASE)


# A feedback loop is: a check (test/build/lint) command fails -> the agent modifies code
# -> the check is rerun. Checks are recognised by the runner they invoke.
CHECK_COMMAND = re.compile(
    r"\bmake\s+(test|check|all)\b|\bmake\s*($|&&|;)|\bunittest\b|\bpytest\b|\bnpm\s+(run\s+)?test\b"
    r"|\bnode\s+--test\b|\bgo\s+test\b|\bcargo\s+test\b|\bctest\b|\btsc\b|\bbash\s+\S*tests?/")
FAILURE_OUTPUT = re.compile(
    r"\bFAILED\b|\bFAIL\b|Traceback \(most recent call last\)|\bfailures?=\d|\bnot ok\b|\bError:|"
    r"npm ERR!|make: \*\*\*|AssertionError|\d+ failed|# fail [1-9]")
# bash used as an editor (heredoc rewrites, sed -i, redirects into source files).
BASH_WRITE = re.compile(r"write_text\(|\.write\(|sed\s+-i|\bpatch\b|\bcat\s*>|\btee\b|>\s*[\w./-]+\.\w+")
TEST_PATH = re.compile(
    r"(^|/)(tests?|__tests__|spec)/|(^|/)test_[^/]*$|_test\.[a-z]+$|\.(test|spec)\.[a-z]+$")


@dataclass
class Call:
    step: str
    name: str
    category: str | None
    signature: str
    is_error: bool
    arguments: str = ""
    output: str = ""

    @property
    def command(self) -> str:
        try:
            return str(json.loads(self.arguments).get("command") or "")
        except (json.JSONDecodeError, AttributeError):
            return ""

    @property
    def is_check(self) -> bool:
        return self.category == cat.COMMAND_EXECUTION and bool(CHECK_COMMAND.search(self.command))

    @property
    def failed(self) -> bool:
        return self.is_error or bool(FAILURE_OUTPUT.search(self.output[-4000:]))

    @property
    def modifies(self) -> bool:
        """Edit/write tools, or bash used as an editor (possibly in the same command as a rerun)."""
        if self.category == cat.FILE_MODIFICATION:
            return True
        return self.category == cat.COMMAND_EXECUTION and bool(BASH_WRITE.search(self.command))

    @property
    def touches_tests(self) -> bool:
        if self.category == cat.FILE_MODIFICATION:
            try:
                return bool(TEST_PATH.search(str(json.loads(self.arguments).get("path") or "")))
            except (json.JSONDecodeError, AttributeError):
                return False
        return self.modifies and bool(re.search(r"(^|[\s/'\"])(tests?|__tests__)/|test_\w+\.py", self.command))


@dataclass
class Report:
    trial: str
    steps: int = 0
    tool_calls: int = 0
    effective_calls: int = 0
    repeated_calls: int = 0
    repair_iterations: int = 0
    final_turn_repairs: int = 0
    debug_resolved: bool = False
    fault: dict | None = None
    test_edits_in_debug_turn: int = 0
    harness_probes: list[str] = field(default_factory=list)
    reasoning_steps: int = 0          # agent steps carrying reasoning text
    reasoning_cjk_ratio: float | None = None  # recorded, not gating: share of CJK characters in reasoning
    by_category: dict[str, int] = field(default_factory=dict)
    unknown_tools: list[str] = field(default_factory=list)
    unanswered_turns: list[str] = field(default_factory=list)
    final_tests_pass: float | None = None
    verdict: str = "reject"
    reasons: list[str] = field(default_factory=list)


def _calls(step_name: str, trajectory: dict) -> tuple[list[Call], bool]:
    """Tool calls in order, skipping copied context; also whether the turn got a final answer."""
    calls: list[Call] = []
    answered = False
    for s in trajectory.get("steps", []):
        if s.get("is_copied_context") or s.get("llm_call_count") == 0 or s.get("source") != "agent":
            continue
        results = {r.get("source_call_id"): r for r in (s.get("observation") or {}).get("results", [])}
        for tc in s.get("tool_calls") or []:
            name = tc["function_name"]
            calls.append(Call(step_name, name, cat.category(name),
                              name + json.dumps(tc.get("arguments", {}), sort_keys=True, ensure_ascii=False),
                              bool((results.get(tc["tool_call_id"], {}).get("extra") or {}).get("is_error")),
                              json.dumps(tc.get("arguments", {}), ensure_ascii=False),
                              str(results.get(tc["tool_call_id"], {}).get("content") or "")))
        if not s.get("tool_calls") and str(s.get("message") or "").strip():
            answered = True
    return calls, answered


def _effective(calls: list[Call]) -> tuple[int, int]:
    """An identical call repeated with no file modification in between is not progress."""
    seen: set[str] = set()
    repeated = 0
    for c in calls:
        if c.category == cat.FILE_MODIFICATION:
            seen.clear()
        if c.signature in seen:
            repeated += 1
        seen.add(c.signature)
    return len(calls) - repeated, repeated


def _repairs(calls: list[Call]) -> tuple[int, bool]:
    """Count feedback cycles: failing check -> >=1 modification -> check rerun.

    Returns (cycles, resolved) where resolved means a check ran green after the last
    failing one. Reading/searching between steps is allowed and expected (diagnosis).
    """
    cycles, state, resolved, saw_failure = 0, "idle", False, False
    for c in calls:
        if c.modifies and state == "failed":  # one bash command may patch and rerun: order is edit, then check
            state = "modified"
        if c.is_check:
            if state == "modified":
                cycles += 1
            if c.failed:
                state, resolved, saw_failure = "failed", False, True
            else:
                state, resolved = "idle", saw_failure
    return cycles, resolved


def evaluate(trial_dir: str | Path) -> Report:
    trial_dir = Path(trial_dir)
    report = Report(trial=trial_dir.name)
    step_dirs = sorted((trial_dir / "steps").iterdir(), key=lambda p: int(p.name.split("-")[-1]))
    calls: list[Call] = []
    reasoning: list[str] = []
    for step_dir in step_dirs:
        path = step_dir / "agent" / "trajectory.json"
        if not path.is_file():
            report.reasons.append(f"{step_dir.name}: no trajectory")
            report.unanswered_turns.append(step_dir.name)
            continue
        trajectory = json.loads(path.read_text())
        for s in trajectory.get("steps", []):
            if s.get("source") == "agent" and not s.get("is_copied_context") and s.get("reasoning_content"):
                report.reasoning_steps += 1
                reasoning.append(str(s["reasoning_content"]))
        step_calls, answered = _calls(step_dir.name, trajectory)
        calls += step_calls
        if not answered:
            report.unanswered_turns.append(step_dir.name)
    report.steps = len(step_dirs)
    text = "".join(reasoning)
    if text:
        report.reasoning_cjk_ratio = round(sum("\u4e00" <= ch <= "\u9fff" for ch in text) / len(text), 3)

    report.tool_calls = len(calls)
    report.effective_calls, report.repeated_calls = _effective(calls)
    report.repair_iterations, report.debug_resolved = _repairs(calls)
    final_name = step_dirs[-1].name if step_dirs else ""
    final_calls = [c for c in calls if c.step == final_name]
    report.final_turn_repairs, _ = _repairs(final_calls)
    fault_path = step_dirs[-1] / "verifier" / "fault.json" if step_dirs else None
    if fault_path and fault_path.is_file():
        record = json.loads(fault_path.read_text())
        report.fault = {"injected": bool(record.get("injected")),
                        "operator": (record.get("mutant") or {}).get("operator"),
                        "path": (record.get("mutant") or {}).get("path"),
                        "test_files_changed": record.get("test_files_changed", [])}
        report.test_edits_in_debug_turn = sum(c.touches_tests for c in final_calls)
        fault_path_name = report.fault["path"] or ""
        report.fault["located"] = bool(fault_path_name) and any(
            c.modifies and (fault_path_name in c.arguments or Path(fault_path_name).name in c.arguments)
            for c in final_calls)
    report.by_category = {k: sum(c.category == k for c in calls) for k in cat.CATEGORIES}
    report.unknown_tools = sorted({c.name for c in calls if c.category is None})
    report.harness_probes = [f"{c.step}:{c.name}:{c.arguments[:160]}" for c in calls
                             if HARNESS_PROBE.search(c.arguments)]

    if step_dirs:
        reward = step_dirs[-1] / "verifier" / "reward.json"
        if reward.is_file():
            report.final_tests_pass = json.loads(reward.read_text()).get("tests_pass")

    # Imported DSH tasks (dsh_import) are short skill tasks graded by their own judge: no length/repair floor.
    short_task = trial_dir.name.startswith(DSH_PREFIX)
    if not short_task and report.effective_calls < MIN_EFFECTIVE_CALLS:
        report.reasons.append(f"effective_calls {report.effective_calls} < {MIN_EFFECTIVE_CALLS}")
    if not short_task and report.repair_iterations < MIN_REPAIR_ITERATIONS:
        report.reasons.append("no feedback/repair iteration")
    if report.unanswered_turns:
        report.reasons.append("unanswered turns: " + ",".join(report.unanswered_turns))
    if report.final_tests_pass != 1:
        report.reasons.append(f"final tests_pass={report.final_tests_pass}")
    if report.fault and report.fault["injected"]:
        if report.final_turn_repairs < 1:
            report.reasons.append("planted fault but no fail->fix->rerun cycle in the debugging turn")
        elif not report.fault["located"]:
            report.reasons.append("debugging turn never modified the file holding the planted fault")
        if report.fault["test_files_changed"]:
            report.reasons.append("test files changed during debugging turn: "
                                  + ",".join(report.fault["test_files_changed"][:5]))
    if report.harness_probes:
        report.reasons.append(f"harness probing: {len(report.harness_probes)} call(s)")
    if report.unknown_tools:
        report.reasons.append("unmapped tools: " + ",".join(report.unknown_tools))
    report.verdict = "accept" if not report.reasons else "reject"
    return report


def to_json(report: Report) -> str:
    return json.dumps(asdict(report), ensure_ascii=False, indent=1)
