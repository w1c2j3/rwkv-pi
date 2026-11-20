import json
import tempfile
import unittest
from pathlib import Path

from rwkvdata import atif, bundle, collect, harbor_export, qc, test_turn


def _bundle_file(root: Path, followups=("加一个功能",)) -> Path:
    raw = {
        "schemaVersion": 1,
        "task": {
            "id": "unit-0001", "family": "unit-family", "domain": "python", "difficulty": "L2",
            "prompt": "实现功能A", "checks": [{"type": "command", "argv": ["true"], "timeoutMs": 1000}],
            "files": {"app/Makefile": "test:\n\tpython3 -m unittest\n", "app/main.py": "print(1)\n"},
        },
        "followups": [{"prompt": p} for p in followups],
        "review": {"baseEnvironmentId": "env-1"},
    }
    path = root / "unit-0001.json"
    path.write_text(json.dumps(raw, ensure_ascii=False))
    return path


class ReviewedTurns(unittest.TestCase):
    def test_reviewed_turns_replace_bundle_turns_and_task_md(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _bundle_file(Path(tmp))
            raw = json.loads(path.read_text()); raw["task"]["files"]["TASK.md"] = "# t\n\n实现功能A\n"
            path.write_text(json.dumps(raw, ensure_ascii=False))
            b = bundle.with_turns(bundle.load(path), ["实现功能A（清理后）", "加一个功能"])
            self.assertEqual(b.turns, ("实现功能A（清理后）", "加一个功能"))
            self.assertIn("实现功能A（清理后）", b.files["TASK.md"])
            with self.assertRaises(ValueError):
                bundle.with_turns(b, ["只有一轮"])


class TestTurn(unittest.TestCase):
    def test_detects_make_and_npm(self):
        files = {"a/Makefile": "check:\n\ttrue\n", "b/package.json": '{"scripts":{"test":"node --test"}}'}
        self.assertEqual(test_turn.detect_commands(files), ["cd a && make check", "cd b && npm test"])

    def test_falls_back_to_unittest(self):
        files = {"tests/test_x.py": "import unittest\n"}
        self.assertEqual(test_turn.detect_commands(files), ["python3 -m unittest discover -s tests -v"])

    def test_instruction_is_deterministic_and_names_commands(self):
        text = test_turn.instruction("id-1", ["make test"])
        self.assertIn("`make test`", text)
        self.assertEqual(text, test_turn.instruction("id-1", ["make test"]))
        with self.assertRaises(ValueError):
            test_turn.instruction("id-1", [])


class TestExport(unittest.TestCase):
    def test_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = harbor_export.export(bundle.load(_bundle_file(root)), root / "out")
            self.assertTrue((task / "environment/workspace/app/main.py").is_file())
            steps = sorted(p.name for p in (task / "steps").iterdir())
            self.assertEqual(steps, ["step-1", "step-2", "step-3"])
            final = json.loads((task / "steps/step-3/tests/checks.json").read_text())
            self.assertEqual({c["group"] for c in final["checks"]}, {"bundle_checks", "tests_pass"})
            first = json.loads((task / "steps/step-1/tests/checks.json").read_text())
            self.assertEqual({c["group"] for c in first["checks"]}, {"bundle_checks"})
            self.assertIn('schema_version = "1.4"', (task / "task.toml").read_text())
            with self.assertRaises(FileExistsError):
                harbor_export.export(bundle.load(_bundle_file(root)), root / "out")
            self.assertNotIn("golang", (task / "environment/Dockerfile").read_text())
            self.assertNotIn("task-setup", (task / "environment/Dockerfile").read_text())
            fast = harbor_export.export(bundle.load(_bundle_file(root)), root / "out-fast",
                                        base_image="rwkv-pi-harbor-base:1.5.0")
            self.assertIn("FROM rwkv-pi-harbor-base:1.5.0\n", (fast / "environment/Dockerfile").read_text())
            self.assertNotIn("apt-get", (fast / "environment/Dockerfile").read_text())

    def test_shebang_files_are_executable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = _bundle_file(root)
            raw = json.loads(path.read_text())
            raw["task"]["files"]["app/tests/run.sh"] = "#!/bin/sh\nexit 0\n"
            path.write_text(json.dumps(raw, ensure_ascii=False))
            ws = harbor_export.export(bundle.load(path), root / "out") / "environment/workspace/app"
            self.assertTrue((ws / "tests/run.sh").stat().st_mode & 0o111)
            self.assertFalse((ws / "main.py").stat().st_mode & 0o111)

    def test_toolchains_and_setup_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = _bundle_file(root)
            raw = json.loads(path.read_text())
            raw["task"]["toolchains"] = ["rust", "go"]
            raw["task"]["files"][".task-setup.sh"] = "git init -q\n"
            path.write_text(json.dumps(raw, ensure_ascii=False))
            text = harbor_export.dockerfile(bundle.load(path))
            self.assertLess(text.index("golang-go"), text.index("rustc cargo"))
            self.assertLess(text.index("rustc cargo"), text.index("COPY workspace/"))
            self.assertTrue(text.endswith("COPY workspace/ /app/\nRUN bash .task-setup.sh && rm -f .task-setup.sh\n"))
            raw["task"]["toolchains"] = ["web"]
            path.write_text(json.dumps(raw, ensure_ascii=False))
            text = harbor_export.dockerfile(bundle.load(path))
            self.assertIn("react@18.3.1", text)
            self.assertLess(text.index("ARG NPM_REGISTRY"), text.index("/opt/web"))
            self.assertIn("COPY workspace/ /app/\nRUN ln -s /opt/web/node_modules /app/node_modules\n", text)
            raw["task"]["toolchains"] = ["cobol"]
            path.write_text(json.dumps(raw, ensure_ascii=False))
            with self.assertRaises(ValueError):
                harbor_export.dockerfile(bundle.load(path))


def _step(i, calls=(), message="", errors=()):
    tool_calls = [{"tool_call_id": f"c{i}-{n}", "function_name": name, "arguments": args}
                  for n, (name, args) in enumerate(calls)]
    results = [{"source_call_id": f"c{i}-{n}", "content": "", "extra": {"is_error": n in errors}}
               for n in range(len(calls))]
    s = {"step_id": i, "source": "agent", "message": message}
    if tool_calls:
        s["tool_calls"], s["observation"] = tool_calls, {"results": results}
    return s


class TestQC(unittest.TestCase):
    def _trial(self, root: Path, steps: list[list[dict]], tests_pass=1.0) -> Path:
        for n, trajectory_steps in enumerate(steps, 1):
            d = root / "steps" / f"step-{n}"
            (d / "agent").mkdir(parents=True)
            (d / "verifier").mkdir(parents=True)
            (d / "agent/trajectory.json").write_text(json.dumps({"steps": trajectory_steps}))
            (d / "verifier/reward.json").write_text(json.dumps({"tests_pass": tests_pass}))
        return root

    def test_repair_iteration_and_repeats(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = [("read", {"path": f"f{k}"}) for k in range(20)]
            s1 = [_step(1, calls), _step(2, [("bash", {"command": "make test"})], errors=(0,)),
                  _step(3, [("edit", {"path": "f1"})]), _step(4, [("bash", {"command": "make test"})]),
                  _step(5, [("bash", {"command": "make test"})]), _step(6, message="完成")]
            r = qc.evaluate(self._trial(Path(tmp), [s1]))
            self.assertEqual(r.repair_iterations, 1)
            self.assertEqual(r.repeated_calls, 1)  # second identical run after the fix, no edit between
            self.assertEqual(r.effective_calls, 23)
            self.assertEqual(r.verdict, "accept", r.reasons)

    def test_web_tools_mapped_to_web_browsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = [("read", {"path": f"f{k}"}) for k in range(18)]
            calls += [("web_search", {"query": "RWKV"}), ("web_fetch", {"url": "https://example.com"})]
            s1 = [_step(1, calls), _step(2, [("bash", {"command": "make test"})], errors=(0,)),
                  _step(3, [("edit", {"path": "f1"})]), _step(4, [("bash", {"command": "make test"})]),
                  _step(5, message="完成")]
            r = qc.evaluate(self._trial(Path(tmp), [s1]))
            self.assertEqual(r.by_category["web_browsing"], 2)
            self.assertEqual(r.unknown_tools, [])
            self.assertEqual(r.verdict, "accept", r.reasons)

    def test_copied_context_excluded_and_unanswered_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            copied = _step(1, [("read", {"path": "x"})] * 30)
            copied["is_copied_context"] = True
            r = qc.evaluate(self._trial(Path(tmp), [[copied]], tests_pass=0.0))
            self.assertEqual(r.tool_calls, 0)
            self.assertEqual(r.verdict, "reject")
            self.assertIn("step-1", r.unanswered_turns)

    def test_harness_probing_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = [("read", {"path": f"src/f{k}.py"}) for k in range(20)]
            calls += [("bash", {"command": "ls -la /logs/verifier"}), ("bash", {"command": "cat ~/.bash_history"}),
                      ("bash", {"command": "cd /app && python3 -m unittest discover -s tests"}),
                      ("read", {"path": "/app/tests/test_x.py"})]
            r = qc.evaluate(self._trial(Path(tmp), [[_step(1, calls, message="完成")]]))
            self.assertEqual(len(r.harness_probes), 2, r.harness_probes)
            self.assertIn("harness probing: 2 call(s)", r.reasons)


class TestMerge(unittest.TestCase):
    def test_copied_context_kept_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            u1 = {"step_id": 1, "source": "user", "message": "第一轮"}
            a1 = _step(2, [("read", {"path": "a"})], message="好")
            u2 = {"step_id": 3, "source": "user", "message": "第二轮"}
            a2 = _step(4, [("bash", {"command": "make test"})], message="完成")
            copied = [{**u1, "is_copied_context": True}, {**a1, "is_copied_context": True}]
            for n, (steps, prompt) in enumerate([([u1, a1], 10), (copied + [u2, a2], 30)], 1):
                d = root / "steps" / f"step-{n}" / "agent"
                d.mkdir(parents=True)
                (d / "trajectory.json").write_text(json.dumps({
                    "schema_version": "ATIF-v1.7", "session_id": "s", "agent": {"name": "pi"},
                    "steps": steps, "final_metrics": {"total_prompt_tokens": prompt}}))
            merged = atif.merge(root)
            self.assertEqual([s["message"] for s in merged["steps"]], ["第一轮", "好", "第二轮", "完成"])
            self.assertEqual([s["step_id"] for s in merged["steps"]], [1, 2, 3, 4])
            self.assertEqual(merged["final_metrics"]["total_prompt_tokens"], 40)
            self.assertEqual(merged["schema_version"], "ATIF-v1.8")


class TestSpendPricing(unittest.TestCase):
    """Spend must follow DeepSeek's billing: off-peak calls at half price, copied context never re-billed."""

    @staticmethod
    def _call(ts, prompt, cached, out, copied=False):
        step = {"timestamp": ts, "metrics": {"prompt_tokens": prompt, "cached_tokens": cached,
                                             "completion_tokens": out}}
        if copied:
            step["is_copied_context"] = True
        return step

    def test_offpeak_half_and_copied_context_skipped(self):
        peak, offpeak = "2026-11-11T02:00:00Z", "2026-11-11T04:30:00Z"  # Wed 10:00 / 12:30 Shanghai
        traj = {"steps": [self._call(peak, 10**6, 0, 0, copied=True), self._call(peak, 10**6, 0, 0),
                          self._call(offpeak, 0, 0, 10**6)],
                "final_metrics": {"total_prompt_tokens": 10**6, "total_cached_tokens": 0,
                                  "total_completion_tokens": 10**6}}
        self.assertAlmostEqual(collect.step_cost_cny(traj), 2.0 + 4.0)

    def test_mismatched_metrics_priced_at_peak(self):
        traj = {"steps": [self._call("2026-11-14T04:00:00Z", 10, 0, 0)],  # Saturday
                "final_metrics": {"total_prompt_tokens": 10**6, "total_cached_tokens": 0,
                                  "total_completion_tokens": 0}}
        self.assertAlmostEqual(collect.step_cost_cny(traj), 2.0)


class TestSelfHostedDeepSeek(unittest.TestCase):
    def test_harbor_command_and_models_script_for_self_hosted_ds41f(self):
        from rwkvdata import dsh_import, pi_agent
        cmd = collect.harbor_command(
            "harbor", Path("/tmp/task"), ["q1"], Path("/tmp/jobs"), "job-1", 32,
            model="deepseek/rwkv-ds41f-1017",
            base_url="http://127.0.0.1:58866/v1",
        )
        self.assertIn("deepseek/rwkv-ds41f-1017", cmd)
        self.assertIn("base_url=http://127.0.0.1:58866/v1", cmd)
        script = pi_agent._models_json_script("http://127.0.0.1:58866/v1/")
        self.assertIn("http://127.0.0.1:58866/v1/", script)
        self.assertIn("x-litellm-session-id", script)
        self.assertIn("rwkv-ds41f-1017", script)
        self.assertIn("requiresReasoningContentOnAssistantMessages", script)
        self.assertIn("exposeSessionEnvironment: false", pi_agent.PARITY_EXTENSION)
        mapped = dsh_import._pi_expect_tools({"turns": [{"required_tools": ["read_file", ["replace_lines", "write_file"]]}]})
        self.assertEqual(mapped["turns"][0]["required_tools"], [["read_file", "read"], ["replace_lines", "write_file", "edit", "write"]])

    def test_collected_requires_all_expected_steps_with_trajectories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks_dir = root / "tasks"
            for n in (1, 2):
                (tasks_dir / "q1" / "steps" / f"step-{n}").mkdir(parents=True)
            trial = root / "jobs" / "batch-001" / "q1__abc"
            (trial / "steps" / "step-1" / "agent").mkdir(parents=True)
            # Empty step-1 without trajectory.json is neither reached nor collected.
            self.assertEqual(collect.collected(root / "jobs", tasks_dir=tasks_dir), set())
            (trial / "steps" / "step-1" / "agent" / "trajectory.json").write_text("{}")
            # Partial trial (1 of 2 steps) is not marked collected when tasks_dir has 2 steps.
            self.assertEqual(collect.collected(root / "jobs", tasks_dir=tasks_dir), set())
            (trial / "steps" / "step-2" / "agent").mkdir(parents=True)
            (trial / "steps" / "step-2" / "agent" / "trajectory.json").write_text("{}")
            self.assertEqual(collect.collected(root / "jobs", tasks_dir=tasks_dir), {"q1"})


if __name__ == "__main__":
    unittest.main()
