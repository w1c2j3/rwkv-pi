import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from rwkvdata import bundle, faults, harbor_export, qc

SRC = "def count_over(xs, limit):\n    return sum(1 for x in xs if x > limit)\n"
TEST = ("import unittest\nfrom app.core import count_over\n\nclass T(unittest.TestCase):\n"
        "    def test_count(self):\n        self.assertEqual(count_over([1, 5, 9], 5), 1)\n")
FILES = {"app/__init__.py": "", "app/core.py": SRC, "tests/__init__.py": "", "tests/test_core.py": TEST}
COMMANDS = ["python3 -m unittest discover -s tests -t ."]


def local_runner(workdir: Path, commands):
    out = []
    for cmd in commands:
        r = subprocess.run(["bash", "-lc", cmd], cwd=workdir, capture_output=True, text=True, timeout=60)
        out.append((r.returncode, r.stdout + r.stderr))
    return out


class TestCandidates(unittest.TestCase):
    def test_never_mutates_tests_and_is_deterministic(self):
        a = faults.candidates(FILES, "seed")
        self.assertTrue(a)
        self.assertTrue(all(m.path == "app/core.py" for m in a))
        self.assertEqual(a, faults.candidates(FILES, "seed"))
        self.assertIn("gt->ge", {m.operator for m in a})

    def test_skips_strings_and_comments(self):
        files = {"m.py": "# if a < b\nmsg = 'a < b'\nok = a < b\n"}
        self.assertEqual([(m.line_no, m.operator) for m in faults.candidates(files, "s")], [(3, "lt->le")])


class TestValidateAndSetup(unittest.TestCase):
    def test_killed_mutant_planted_then_recorded(self):
        report = faults.validate(FILES, COMMANDS, faults.candidates(FILES, "seed"), local_runner)
        self.assertTrue(report["baseline_pass"])
        self.assertTrue(report["killed"])
        with tempfile.TemporaryDirectory() as tmp:
            app, state = Path(tmp) / "app", Path(tmp) / "state"
            faults._write_tree(app, FILES)
            script = app / "setup.sh"
            # First mutant's line was "rewritten by the agent": setup must fall back.
            stale = {**report["killed"][0], "old": "    no such line"}
            script.write_text(faults.setup_script([stale] + report["killed"], COMMANDS, app=str(app),
                                                  fault_dir=str(state)))
            subprocess.run(["bash", str(script)], check=True, capture_output=True)
            record = json.loads((state / "fault.json").read_text())
            self.assertTrue(record["injected"])
            self.assertEqual(record["attempts"][0]["result"], "line-changed")
            self.assertIn("tests/test_core.py", record["test_files_sha256"])
            self.assertFalse(script.exists())
            self.assertNotEqual(local_runner(app, COMMANDS)[0][0], 0)

    def test_baseline_failure_yields_nothing(self):
        broken = {**FILES, "tests/test_core.py": TEST + "        self.fail()\n"}
        report = faults.validate(broken, COMMANDS, faults.candidates(broken, "s"), local_runner)
        self.assertFalse(report["baseline_pass"])
        self.assertEqual(report["killed"], [])


class TestDebugExport(unittest.TestCase):
    def test_final_step_is_debug_turn_with_setup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = {"schemaVersion": 1,
                   "task": {"id": "unit-0002", "family": "f", "domain": "python", "difficulty": "L2",
                            "prompt": "实现功能A", "checks": [], "files": {**FILES, "Makefile": "test:\n\ttrue\n"}},
                   "followups": [{"prompt": "再加一个功能"}], "review": {"baseEnvironmentId": "e"}}
            (root / "b.json").write_text(json.dumps(raw, ensure_ascii=False))
            killed = [{"path": "app/core.py", "line_no": 2, "old": "    if x < lo:", "new": "    if x <= lo:",
                       "operator": "lt->le"}]
            task = harbor_export.export(bundle.load(root / "b.json"), root / "out",
                                        faults={"baseline_pass": True, "killed": killed, "tried": 3, "survived": 2})
            self.assertTrue((task / "steps/step-3/workdir/setup.sh").is_file())
            self.assertFalse((task / "steps/step-2/workdir").exists())
            text = (task / "steps/step-3/instruction.md").read_text()
            self.assertIn("`make test`", text)
            self.assertNotIn("app/core.py", text)  # never reveals the planted location
            spec = json.loads((task / "steps/step-3/tests/checks.json").read_text())
            self.assertEqual(spec["faultRecord"], faults.FAULT_RECORD)
            import tomllib
            meta = tomllib.loads((task / "task.toml").read_text())["metadata"]
            self.assertEqual((meta["final_turn"], meta["fault_candidates"]), ("debug", 1))


def _call(i, name, args, *, error=False, output=""):
    return {"step_id": i, "source": "agent", "message": "",
            "tool_calls": [{"tool_call_id": f"c{i}", "function_name": name, "arguments": args}],
            "observation": {"results": [{"source_call_id": f"c{i}", "content": output,
                                         "extra": {"is_error": error}}]}}


class TestFeedbackLoop(unittest.TestCase):
    def _trial(self, root: Path, step_lists, fault=None):
        for n, steps in enumerate(step_lists, 1):
            d = root / "steps" / f"step-{n}"
            (d / "agent").mkdir(parents=True)
            (d / "verifier").mkdir(parents=True)
            (d / "agent/trajectory.json").write_text(json.dumps({"steps": steps}))
            (d / "verifier/reward.json").write_text(json.dumps({"tests_pass": 1.0}))
        if fault is not None:
            (root / "steps" / f"step-{len(step_lists)}" / "verifier/fault.json").write_text(json.dumps(fault))
        return root

    def test_full_cycle_in_debug_turn(self):
        reads = [_call(k, "read", {"path": f"app/f{k}.py"}) for k in range(1, 21)]
        debug = [
            _call(30, "bash", {"command": "make test"}, output="FAILED (failures=1)"),  # exit 0 but failing output
            _call(31, "read", {"path": "app/core.py"}),
            _call(32, "bash", {"command": "grep -n clamp app/core.py"}),
            _call(33, "bash", {"command": "python3 - <<'PY'\np = Path('app/core.py')\np.write_text(s)\nPY"}),  # bash as editor
            _call(34, "bash", {"command": "make test"}, output="OK"),
            {"step_id": 35, "source": "agent", "message": "根因是边界条件，已修复"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            r = qc.evaluate(self._trial(Path(tmp), [reads + [{"step_id": 99, "source": "agent", "message": "好"}],
                                                    debug],
                                        fault={"injected": True, "mutant": {"operator": "lt->le", "path": "app/core.py"},
                                               "test_files_changed": []}))
            self.assertEqual((r.repair_iterations, r.final_turn_repairs, r.debug_resolved), (1, 1, True))
            self.assertEqual(r.verdict, "accept", r.reasons)

    def test_patch_and_rerun_in_one_command_counts(self):
        calls = [_call(1, "bash", {"command": "make test"}, error=True, output="FAILED (failures=2)"),
                 _call(2, "bash", {"command": "python3 - <<'PY'\np.write_text(s)\nPY\nmake test"}, output="OK")]
        self.assertEqual(qc._repairs(qc._calls("step-1", {"steps": calls})[0]), (1, True))

    def test_planted_fault_without_cycle_or_with_test_edit_rejected(self):
        reads = [_call(k, "read", {"path": f"app/f{k}.py"}) for k in range(1, 21)]
        debug = [
            _call(30, "bash", {"command": "make test"}, error=True, output="AssertionError"),
            _call(31, "edit", {"path": "tests/test_core.py"}),
            _call(32, "bash", {"command": "make test"}, output="OK"),
            {"step_id": 35, "source": "agent", "message": "完成"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            r = qc.evaluate(self._trial(Path(tmp), [reads + [{"step_id": 99, "source": "agent", "message": "好"}],
                                                    debug],
                                        fault={"injected": True, "test_files_changed": ["tests/test_core.py"]}))
            self.assertEqual(r.test_edits_in_debug_turn, 1)
            self.assertTrue(any("test files changed" in x for x in r.reasons), r.reasons)
            self.assertEqual(r.verdict, "reject")


if __name__ == "__main__":
    unittest.main()
