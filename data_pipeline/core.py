"""No model calls and no execution of imported checks. Final bundles only."""
import collections
import hashlib
import json
import re
import shutil
import tempfile
import unicodedata
from pathlib import Path

from pipeline.datakit import read_json, require_data_root, sha256_file, write_json

DOMAINS = (
    "linux-shell", "git", "python", "node-typescript", "sqlite-sql",
    "data-processing", "config-deploy", "compiled-languages",
    "data-management", "agent-orchestration",
)
FOCUS = {"data-management", "agent-orchestration"}
REVIEW_CHECKS = (
    "prompt_fixture_consistent", "acceptance_independent", "correct_alternatives_accepted",
    "wrong_behaviors_rejected", "difficulty_appropriate", "mechanism_distinct",
)
SOLVER = {
    "model": "deepseek-flash", "promptProfile": "zh-reasoning-v2",
    "protocol": "rwkv-pi-xml-continuation-v3", "thinking": "high",
    "systemPrompt": "complete native Pi system prompt with existing Chinese constraints",
}


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def safe_relative(name):
    path = Path(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts and "\\" not in name


def metadata(bundle):
    """Structural screening, deliberately NOT semantic review or acceptance."""
    task, review, controls = bundle["task"], bundle["review"], bundle["controls"]
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,110}", task["id"]):
        raise ValueError("Invalid task ID")
    if task.get("split") != "train" or task.get("schemaVersion") != 1:
        raise ValueError("Only native training-question bundles can be inventoried")
    if task.get("difficulty") not in ("L1", "L2", "L3") or task.get("domain") not in DOMAINS:
        raise ValueError("Missing/unsupported difficulty or domain")
    if review.get("difficulty") != task["difficulty"] or review.get("domain") != task["domain"]:
        raise ValueError("Review metadata contradicts task metadata")
    for key in ("prompt", "family", "mechanism", "scenario", "taskCategory"):
        if not isinstance(task.get(key), str) or not task[key].strip():
            raise ValueError(f"Missing {key}")
    if task["taskCategory"] not in ("understand", "fix", "ops"):
        raise ValueError("Unsupported task category")
    if not isinstance(bundle.get("author"), str) or not bundle["author"].strip():
        raise ValueError("Author provenance is required")
    if not task.get("files") or not task.get("checks") or not controls.get("reference") or not controls.get("wrong"):
        raise ValueError("Need initial files, acceptance, reference and wrong controls")
    if not all(safe_relative(k) and isinstance(v, str) for k, v in task["files"].items()):
        raise ValueError("Initial files must be relative text files")
    for control in [controls["reference"], *controls["wrong"]]:
        names = [*control.get("files", {}), *control.get("delete", []), *control.get("modes", {})]
        if not all(safe_relative(k) for k in names):
            raise ValueError("Invalid control path")
    outputs = task.get("qualityProfile", {}).get("deliverables", [])
    if any(not safe_relative(p) for p in outputs):
        raise ValueError("Deliverables must be safe relative paths")
    body = {"prompt": normalized(task["prompt"]), "files": task["files"]}
    return {
        "id": task["id"], "author": bundle["author"],
        **{k: task[k] for k in ("family", "mechanism", "scenario", "taskCategory", "domain", "difficulty")},
        "contentFingerprint": digest(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()),
        "wrongControls": len(controls["wrong"]), "state": "inventoried_only",
        # Legacy questions may say “ops.sh in repo/” or name stdout as a deliverable.
        # Literal mismatch needs semantic inspection, not silent rewriting or rejection.
        "deliverablesNotLiteralInPrompt": [p for p in outputs if p not in task["prompt"]],
    }


def collisions(questions):
    groups = {k: collections.defaultdict(set) for k in ("mechanism", "scenario", "content")}
    for question in questions:
        for revision in question["revisions"]:
            groups["mechanism"][normalized(revision["mechanism"])].add(question["id"])
            groups["scenario"][normalized(revision["taskCategory"] + ":" + revision["scenario"])].add(question["id"])
            groups["content"][revision["contentFingerprint"]].add(question["id"])
    return {kind: [{"key": k, "ids": sorted(ids)} for k, ids in sorted(group.items()) if len(ids) > 1]
            for kind, group in groups.items()}


def inventory(sources, destination, heldout_manifest=None):
    """Snapshot final bundles by SHA; same-ID revisions never increase task count."""
    destination = require_data_root(destination)
    if destination.exists():
        raise ValueError("Inventory destination exists; use a new snapshot path")
    heldout = set(read_json(heldout_manifest)["families"]) if heldout_manifest else set()
    destination.mkdir(parents=True)
    objects = destination / "objects"
    objects.mkdir()
    questions, errors, excluded, seen = {}, [], [], 0
    for source, directory in sources:
        directory = Path(directory).resolve(strict=True)
        files = sorted(directory.glob("*.json"))
        if not files:
            raise ValueError(f"No bundles in {directory}")
        for path in files:
            seen += 1
            try:
                raw = path.read_bytes()
                bundle = json.loads(raw)
                meta = metadata(bundle)
                if meta["family"] in heldout:
                    excluded.append({"path": str(path), "id": meta["id"], "reason": "heldout_family"})
                    continue
                sha = digest(raw)
                revisions = questions.setdefault(meta["id"], {})
                if sha not in revisions:
                    target = objects / f"{sha}.json"
                    if not target.exists():
                        target.write_bytes(raw)
                    revisions[sha] = {**meta, "bundleSha256": sha, "object": f"objects/{sha}.json", "origins": []}
                origin = {"source": source, "path": str(path)}
                if origin not in revisions[sha]["origins"]:
                    revisions[sha]["origins"].append(origin)
            except (ValueError, KeyError, TypeError, AttributeError, OSError) as error:
                errors.append({"path": str(path), "source": source, "error": str(error)})
    rows = [{"id": key, "revisions": sorted(value.values(), key=lambda r: r["bundleSha256"])}
            for key, value in sorted(questions.items())]
    report = {
        "schemaVersion": 1, "kind": "question-inventory-v1", "state": "inventoried_only",
        "sources": [{"name": name, "path": str(Path(path).resolve())} for name, path in sources],
        "heldoutManifest": ({"path": str(Path(heldout_manifest).resolve()), "sha256": sha256_file(heldout_manifest)}
                            if heldout_manifest else None),
        "summary": {"filesRead": seen, "distinctQuestions": len(rows),
                    "bundleRevisions": sum(len(q["revisions"]) for q in rows),
                    "ambiguousRevisionIds": sum(len(q["revisions"]) > 1 for q in rows),
                    "invalidFiles": len(errors), "excludedFiles": len(excluded),
                    "newTrainingTrajectories": 0},
        "questions": rows, "exactCollisions": collisions(rows), "errors": errors, "excluded": excluded,
        "semanticReview": "Not inferred from syntax, source author, or prior self-reported pass labels",
    }
    write_json(destination / "catalog.json", report)
    return report


def review_template(catalog_path, ids):
    catalog = read_json(catalog_path)
    questions = {q["id"]: q for q in catalog["questions"]}
    rows = []
    for task_id in ids:
        revisions = questions[task_id]["revisions"]
        if len(revisions) != 1:
            raise ValueError(f"{task_id}: select and review an explicit revision; refusing latest-wins")
        row = revisions[0]
        rows.append({"id": task_id, "bundleSha256": row["bundleSha256"], "verdict": "pending",
                     "reviewer": "", "notes": "", "checks": dict.fromkeys(REVIEW_CHECKS, False),
                     "calibrationReceipt": "", "calibrationSha256": ""})
    return {"schemaVersion": 1, "catalogSha256": sha256_file(catalog_path), "reviews": rows}


def check_calibration(receipt, row):
    """Accept the existing 82 calibrator's real receipt, not a bare `passed` flag."""
    if receipt.get("id") != row["id"] or receipt.get("bundleSha256") != row["bundleSha256"]:
        raise ValueError("Calibration belongs to a different question revision")
    if receipt.get("passed") is not True or not receipt.get("executionHost"):
        raise ValueError("Calibration has no successful execution receipt")
    results = receipt.get("controls", [])
    by_name = {c.get("control"): c for c in results}
    expected = {"baseline": False, "reference": True,
                **{f"wrong-{i + 1}": False for i in range(row["wrongControls"])}}
    if len(by_name) != len(results) or set(by_name) != set(expected):
        raise ValueError("Baseline/reference/wrong control coverage is incomplete")
    for name, should_pass in expected.items():
        control = by_name[name]
        status = control.get("exit")
        if (type(status) is not int or control.get("expectedPass") is not should_pass
                or (status == 0) != should_pass or not 0 <= status < 124):
            raise ValueError(f"Invalid result for {name}; infrastructure timeouts are not wrong-control success")


def prepare_release(catalog_path, reviews_path, destination):
    """Freeze question-only hand-off. Does not admit, solve, spend, pack or train."""
    catalog_path, reviews_path = Path(catalog_path), Path(reviews_path)
    catalog, decisions = read_json(catalog_path), read_json(reviews_path)
    if catalog.get("kind") != "question-inventory-v1" or catalog["errors"]:
        raise ValueError("Incomplete inventory cannot produce a reviewed release")
    if not catalog.get("heldoutManifest"):
        raise ValueError("Release needs an inventory checked against the evaluation family manifest")
    manifest = catalog["heldoutManifest"]
    if sha256_file(manifest["path"]) != manifest["sha256"]:
        raise ValueError("Evaluation exclusions changed; build a new inventory")
    if decisions.get("catalogSha256") != sha256_file(catalog_path):
        raise ValueError("Review belongs to a different inventory snapshot")
    revisions = {(q["id"], r["bundleSha256"]): r for q in catalog["questions"] for r in q["revisions"]}
    blocked = {task_id for group in collisions(catalog["questions"]).values() for c in group for task_id in c["ids"]}
    selected, seen = [], set()
    for review in decisions["reviews"]:
        if review.get("verdict") != "pass":
            continue
        task_id = review["id"]
        if task_id in seen:
            raise ValueError("A revision is not a new question; duplicate ID in release")
        seen.add(task_id)
        if task_id in blocked:
            raise ValueError(f"{task_id}: unresolved exact mechanism/scenario/content collision")
        row = revisions[(task_id, review["bundleSha256"])]
        if not review.get("reviewer", "").strip() or not review.get("notes", "").strip():
            raise ValueError("Independent semantic reviewer and concrete notes are required")
        if any(review.get("checks", {}).get(k) is not True for k in REVIEW_CHECKS):
            raise ValueError("Semantic review is incomplete")
        source = (catalog_path.parent / row["object"]).resolve()
        if not source.is_relative_to(catalog_path.parent.resolve() / "objects"):
            raise ValueError("Bundle object is outside the catalog")
        if sha256_file(source) != row["bundleSha256"]:
            raise ValueError("Frozen question changed after review")
        actual = metadata(read_json(source))
        if any(actual[k] != row[k] for k in actual):
            raise ValueError("Inventory metadata differs from the bundle")
        evidence = Path(review["calibrationReceipt"]).resolve(strict=True)
        if sha256_file(evidence) != review["calibrationSha256"]:
            raise ValueError("Calibration evidence changed after review")
        raw_receipt = evidence.read_bytes()
        check_calibration(json.loads(raw_receipt), row)
        selected.append((row, source.read_bytes(), review, raw_receipt))
    if not selected:
        raise ValueError("No reviewed, calibrated questions selected")
    destination = require_data_root(destination)
    if destination.exists():
        raise ValueError("Release exists; never overwrite a frozen hand-off")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".release-", dir=destination.parent))
    try:
        (staging / "bundles").mkdir()
        (staging / "evidence").mkdir()
        rows = []
        for row, raw, review, receipt in selected:
            filename = row["id"] + ".json"
            (staging / "bundles" / filename).write_bytes(raw)
            (staging / "evidence" / filename).write_bytes(receipt)
            rows.append({"id": row["id"], "file": filename, "sha256": row["bundleSha256"],
                         "domain": row["domain"], "difficulty": row["difficulty"],
                         "questionSemanticReview": "passed", "solverTrajectoryGenerated": False,
                         "origins": row["origins"]})
        focus = sum(r["domain"] in FOCUS for r in rows)
        result = {
            "schemaVersion": 1, "purpose": "Reviewed questions for explicit native-Pi solver admission",
            "newBundles": rows, "allowedAuthors": sorted({r[0]["author"] for r in selected}),
            "authorizedToStart": False, "allowFutureInbox": False, "solver": SOLVER,
            "catalogSha256": sha256_file(catalog_path), "reviewSha256": sha256_file(reviews_path),
            "mix": {"focus": focus, "established": len(rows) - focus, "targetFocusToEstablished": "1:2",
                    "matchesTarget": len(rows) - focus == 2 * focus},
            "newTrainingTrajectories": 0,
        }
        write_json(staging / "semantic-reviews.json", decisions)
        write_json(staging / "release.json", result)
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return result
