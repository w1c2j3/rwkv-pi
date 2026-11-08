"""Engineering fixtures only. These never become candidate or training data."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from pipeline.datakit import sha256_file, write_json
from data_pipeline.core import (
    REVIEW_CHECKS, check_calibration, inventory, metadata, prepare_release, review_template,
)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        parent = Path("/mnt/f/rwkv-pi-data/artifacts/data-pipeline-tests")
        if not Path("/mnt/f").is_mount():
            self.fail("F: must be mounted for test workspaces")
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.holdout = self.root / "heldout.json"
        write_json(self.holdout, {"families": ["evaluation-family"]})
        self.bundle = {
            "schemaVersion": 1, "author": "Engineering fixture; not a model output",
            "task": {
                "id": "test-question-one", "schemaVersion": 1, "split": "train", "family": "test-family",
                "prompt": "测试流水线存储，不执行本题，不生成训练数据。", "difficulty": "L1", "domain": "python",
                "taskCategory": "fix", "mechanism": "test:branch-coverage", "scenario": "test-scenario",
                "files": {"input.txt": "fixture only\n"}, "checks": [{"command": "exit 1"}],
                "qualityProfile": {"deliverables": []},
            },
            "review": {"difficulty": "L1", "domain": "python"},
            "controls": {"reference": {"files": {"fixed.txt": "fixture"}}, "wrong": [{"why": "test-only"}]},
        }

    def save_bundle(self, bundle=None, name="one.json"):
        path = self.source / name
        write_json(path, bundle or self.bundle)
        return path

    def catalog(self):
        self.save_bundle()
        path = self.root / "inventory"
        result = inventory([("test", self.source)], path, self.holdout)
        return path / "catalog.json", result

    def reviewed(self):
        catalog, _ = self.catalog()
        decisions = review_template(catalog, [self.bundle["task"]["id"]])
        review = decisions["reviews"][0]
        receipt = {
            "id": review["id"], "bundleSha256": review["bundleSha256"], "executionHost": "engineering-fixture",
            "passed": True, "controls": [
                {"control": "baseline", "expectedPass": False, "exit": 1},
                {"control": "reference", "expectedPass": True, "exit": 0},
                {"control": "wrong-1", "expectedPass": False, "exit": 1},
            ],
        }
        evidence = self.root / "calibration.json"
        write_json(evidence, receipt)
        review.update(verdict="pass", reviewer="unit test", notes="Synthetic gate exercise, not a semantic approval",
                      checks=dict.fromkeys(REVIEW_CHECKS, True), calibrationReceipt=str(evidence),
                      calibrationSha256=sha256_file(evidence))
        reviews = self.root / "reviews.json"
        write_json(reviews, decisions)
        return catalog, reviews, decisions, evidence, receipt

    def test_copies_and_revisions_are_one_question(self):
        self.save_bundle()
        self.save_bundle(name="copy.json")
        revision = copy.deepcopy(self.bundle)
        revision["task"]["files"]["input.txt"] = "corrected fixture"
        self.save_bundle(revision, "revision.json")
        report = inventory([("test", self.source)], self.root / "inventory", self.holdout)
        self.assertEqual(report["summary"]["filesRead"], 3)
        self.assertEqual(report["summary"]["distinctQuestions"], 1)
        self.assertEqual(report["summary"]["bundleRevisions"], 2)
        with self.assertRaisesRegex(ValueError, "explicit revision"):
            review_template(self.root / "inventory/catalog.json", [self.bundle["task"]["id"]])

    def test_inheritance_preserves_bytes_and_never_executes_check(self):
        marker = self.root / "must-not-exist"
        self.bundle["task"]["checks"] = [{"command": f"touch {marker}"}]
        path = self.save_bundle()
        original = path.read_bytes()
        catalog, result = self.catalog()
        revision = result["questions"][0]["revisions"][0]
        self.assertEqual((catalog.parent / revision["object"]).read_bytes(), original)
        self.assertEqual(path.read_bytes(), original)
        self.assertFalse(marker.exists())

    def test_duplicate_body_cannot_hide_behind_new_mechanism_label(self):
        self.save_bundle()
        other = copy.deepcopy(self.bundle)
        other["task"].update(id="test-question-two", mechanism="different:label", scenario="different-story")
        self.save_bundle(other, "two.json")
        report = inventory([("test", self.source)], self.root / "inventory", self.holdout)
        self.assertEqual(len(report["exactCollisions"]["content"]), 1)

    def test_mechanism_case_and_unicode_spacing_are_normalized(self):
        self.save_bundle()
        other = copy.deepcopy(self.bundle)
        other["task"].update(id="test-question-two", mechanism="ＴＥＳＴ:BRANCH-COVERAGE  ", prompt="不同的工程测试题面")
        self.save_bundle(other, "two.json")
        report = inventory([("test", self.source)], self.root / "inventory", self.holdout)
        self.assertEqual(len(report["exactCollisions"]["mechanism"]), 1)

    def test_holdout_family_is_excluded_without_loading_evaluation_tasks(self):
        self.bundle["task"]["family"] = "evaluation-family"
        _, result = self.catalog()
        self.assertEqual(result["summary"]["distinctQuestions"], 0)
        self.assertEqual(result["summary"]["excludedFiles"], 1)

    def test_pending_template_cannot_release(self):
        catalog, _ = self.catalog()
        reviews = self.root / "pending.json"
        write_json(reviews, review_template(catalog, [self.bundle["task"]["id"]]))
        with self.assertRaisesRegex(ValueError, "No reviewed"):
            prepare_release(catalog, reviews, self.root / "release")

    def test_legacy_implicit_path_is_preserved_for_semantic_review(self):
        self.bundle["task"]["prompt"] = "请在 repo 目录下创建 ops.sh，并运行它完成验证。"
        self.bundle["task"]["qualityProfile"]["deliverables"] = ["repo/ops.sh"]
        _, result = self.catalog()
        self.assertEqual(result["summary"]["invalidFiles"], 0)
        self.assertEqual(result["questions"][0]["revisions"][0]["deliverablesNotLiteralInPrompt"], ["repo/ops.sh"])

    def test_calibration_and_semantic_review_are_both_required(self):
        catalog, reviews, decisions, _, _ = self.reviewed()
        decisions["reviews"][0]["checks"]["acceptance_independent"] = False
        write_json(reviews, decisions)
        with self.assertRaisesRegex(ValueError, "Semantic review is incomplete"):
            prepare_release(catalog, reviews, self.root / "release")

    def test_same_id_old_calibration_is_rejected(self):
        catalog, reviews, decisions, evidence, receipt = self.reviewed()
        receipt["bundleSha256"] = "0" * 64
        write_json(evidence, receipt)
        decisions["reviews"][0]["calibrationSha256"] = sha256_file(evidence)
        write_json(reviews, decisions)
        with self.assertRaisesRegex(ValueError, "different question revision"):
            prepare_release(catalog, reviews, self.root / "release")

    def test_all_wrong_controls_required_and_timeouts_are_not_success(self):
        _, _, _, _, receipt = self.reviewed()
        row = {**metadata(self.bundle), "bundleSha256": receipt["bundleSha256"]}
        for code in (0, 124, 137, -9, None, False):
            bad = copy.deepcopy(receipt)
            bad["controls"][-1]["exit"] = code
            with self.subTest(code=code), self.assertRaises(ValueError):
                check_calibration(bad, row)
        receipt["controls"].pop()
        with self.assertRaisesRegex(ValueError, "coverage is incomplete"):
            check_calibration(receipt, row)

    def test_release_keeps_bytes_and_cannot_start_collection(self):
        catalog, reviews, _, evidence, _ = self.reviewed()
        output = self.root / "release"
        result = prepare_release(catalog, reviews, output)
        self.assertFalse(result["authorizedToStart"])
        self.assertEqual(result["newTrainingTrajectories"], 0)
        self.assertEqual(result["solver"]["model"], "deepseek-flash")
        self.assertEqual(result["solver"]["protocol"], "rwkv-pi-xml-continuation-v3")
        self.assertEqual((output / "bundles/test-question-one.json").read_bytes(), (self.source / "one.json").read_bytes())
        # Native admission groups by domain/category and reads this exact receipt path.
        row = result["newBundles"][0]
        self.assertEqual((row["domain"], row["category"]), ("python", "fix"))
        exported = output / "calibration-results/test-question-one.json"
        self.assertEqual(exported.read_bytes(), evidence.read_bytes())
        self.assertEqual(sha256_file(exported), row["calibrationSha256"])
        with self.assertRaisesRegex(ValueError, "Release exists"):
            prepare_release(catalog, reviews, output)

    def test_changed_object_evidence_or_holdout_invalidates_release(self):
        catalog, reviews, _, evidence, _ = self.reviewed()
        evidence.write_text(evidence.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            prepare_release(catalog, reviews, self.root / "release")
        write_json(self.holdout, {"families": ["test-family"]})
        with self.assertRaisesRegex(ValueError, "exclusions changed"):
            prepare_release(catalog, reviews, self.root / "release")

    def test_invalid_bundle_is_reported_and_release_is_blocked(self):
        (self.source / "bad.json").write_text("{broken")
        catalog, result = self.catalog()
        self.assertEqual(result["summary"]["invalidFiles"], 1)
        reviews = self.root / "reviews.json"
        write_json(reviews, {"reviews": []})
        with self.assertRaisesRegex(ValueError, "Incomplete inventory"):
            prepare_release(catalog, reviews, self.root / "release")


if __name__ == "__main__":
    unittest.main()
