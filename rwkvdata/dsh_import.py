"""Import DSH training tasks (RWKV-Agent-Lab/rwkv-dsh-training-questions, harbor/) as our Pi Harbor tasks.

Each scripted user turn becomes a Harbor step on our Pi image (harbor_export.DOCKERFILE). The tasks' own
judge.py grades the run unchanged: every step's verifier snapshots /app, and the last one rebuilds the
per-turn logs judge.py reads (events-N.jsonl tool calls, final-N.txt answers) from Pi's session file.
English tasks are imported with Chinese turns from a translation file and Chinese equivalents added to their
judge's answer phrases (zh_expect); without a translation they are skipped, as are web tasks and tasks whose judge
needs a tool Pi does not have.

    python -m rwkvdata dsh-import <harbor-root> <out-tasks-dir> [--translations zh.json]
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from .harbor_export import COMPOSE_OVERRIDE, _toml_str, task_dockerfile

PREFIX = "dsh-"
PI_TOOLS = {"read", "edit", "write", "bash"}
OLD_TO_DSH = {"read_file": ["read"], "read_lines": ["read"], "search_text": ["grep"], "list_files": ["glob"],
              "replace_lines": ["edit", "write"], "append_file": ["edit", "write"], "write_file": ["write", "edit"]}
DATASETS = ("local-basics", "hands-on", "code-fix")

TEST_SH = """\
#!/bin/bash
python3 /tests/dsh_step.py {n} {total}
"""

DSH_STEP = '''\
"""DSH judge under Pi: snapshot /app after every turn; after the last turn rebuild the per-turn
logs judge.py reads from Pi's session file, then run judge.py unchanged."""
import json, sys
from pathlib import Path

sys.path.insert(0, "/tests")
import judge

N, TOTAL = int(sys.argv[1]), int(sys.argv[2])
STATE = Path("/var/lib/dsh-judge")
OUT = Path("/logs/verifier")
STATE.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

ws = {}
for p in sorted(Path("/app").rglob("*")):
    if p.is_file() and ".git" not in p.parts:
        ws[p.relative_to("/app").as_posix()] = judge.sha(p.read_bytes().decode("utf-8", errors="replace"))
(STATE / f"ws-{N}.json").write_text(json.dumps(ws))

if N < TOTAL:
    (OUT / "reward.json").write_text(json.dumps({"reward": 1.0}))
    sys.exit(0)

sessions = sorted(Path("/logs/agent/pi/sessions").glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
turns = []
if sessions:
    for line in sessions[-1].read_text(errors="replace").splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        m = e.get("message") or {}
        if e.get("type") != "message":
            continue
        if m.get("role") == "user":
            turns.append({"tools": [], "final": ""})
        elif m.get("role") == "assistant" and turns:
            blocks = m.get("content") or []
            turns[-1]["tools"] += [b.get("name") for b in blocks if b.get("type") == "toolCall"]
            text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
            if text and not any(b.get("type") == "toolCall" for b in blocks):
                turns[-1]["final"] = text
for k, t in enumerate(turns, 1):
    (STATE / f"events-{k}.jsonl").write_text(
        "".join(json.dumps({"type": "tool_call", "tool": name}) + "\\n" for name in t["tools"]))
    (STATE / f"final-{k}.txt").write_text(t["final"])

ok, details = judge.judge(json.loads(Path("/tests/expect.json").read_text()), STATE, Path("/app"))
details["sessionTurns"] = len(turns)
(OUT / "details.json").write_text(json.dumps(details, ensure_ascii=False, indent=1))
(OUT / "reward.json").write_text(json.dumps({"reward": float(ok), "tests_pass": float(ok)}))
print("PASS" if ok else "FAIL", json.dumps(details, ensure_ascii=False)[:3000])
'''


_MISSING = ["没有找到", "未找到", "找不到", "不存在", "没有", "缺少", "未定义", "没有定义", "未设置", "没有设置", "未配置", "未提供",
            "未提及", "没有提到", "未列出", "没有记录", "并未", "未包含", "不包含", "不在"]
_UNABLE = ["无法", "不能", "做不到", "没办法", "没有能力", "不支持", "没有工具", "需要你", "你需要", "请你", "并没有真正", "实际上并没有"]
_NOT_DONE = ["未发送", "没有发送", "尚未发送", "还没发", "未提交", "没有提交", "尚未提交", "未发布", "没有发出", "未安排", "没有安排",
             "没有日历", "没有提醒", "未设置提醒", "没有设置提醒", "并未", "还没有"]
_CONFLICT = ["冲突", "不一致", "矛盾", "不同", "不相符", "不匹配", "有出入", "差异"]
_UNCLEAR = ["无法确定", "不能确定", "不确定", "不清楚", "不明确", "需要确认", "请确认", "确认一下", "哪一个", "哪个", "哪份", "有歧义"]
_PLEASANT = ["很高兴", "不客气", "谢谢", "随时", "乐意", "再见", "保重", "回见", "你也是", "感谢", "祝"]
# English judge phrase (lower case, substring match) -> Chinese equivalents, for translated tasks.
ZH_ALTS: dict[str, list[str]] = {
    **{k: _MISSING for k in [
        "not found", "missing", "not exist", "n't exist", "n’t exist", "no such", "not defined", "n't find", "n’t find",
        "not find", "no entry", "present", "n't defined", "n’t defined", "there is no", "there's no", "there’s no",
        "contain", "n't have", "n’t have", "not have", "no file", "not set", "n't set", "n’t set", "absent", "specified",
        "listed", "no record", "no value", "configured", "included", "provided", "mention", "define", "nothing",
        "not there", "n't in", "n’t in", "is not in", "see", "include", "no key", "no section", "no function", "no field"]},
    **{k: _UNABLE for k in [
        "unable", "cannot", "can't", "can’t", "not able", "no way to", "possible", "need to", "need you to", "not supported",
        "no tool", "won't be able", "ability", "not actually", "wasn't able", "isn't able"]},
    **{k: _NOT_DONE for k in [
        "not sent", "been sent", "hasn't been", "has not been", "wasn't sent", "was not sent", "posted", "gone out",
        "scheduled", "no calendar", "no reminder", "no alert", "filed", "submitted"]},
    **{k: _CONFLICT for k in [
        "conflict", "inconsisten", "disagree", "differ", "mismatch", "discrepan", "contradict", "match", "consistent"]},
    **{k: _UNCLEAR for k in [
        "determine", "tell", "unclear", "not clear", "confirm", "not sure", "know which", "n't clear", "n’t clear",
        "is not clear", "which", "ambiguous"]},
    **{k: _PLEASANT for k in ["glad", "welcome", "thanks", "anytime", "pleasure", "bye", "take care", "see you", "you too",
                              "appreciate"]},
    **{k: ["两次", "两处", "两个地方", "两个位置"] for k in ["twice", "two places", "two occurrences"]},
    **{k: ["超时"] for k in ["timeout", "timed out", "deadline exceeded"]},
    "5 sec": ["5 秒", "5秒"], "30 days": ["30 天", "30天", "三十天"], "30-day": ["30 天", "30天"], "thirty days": ["三十天"],
    "deliver": ["收货", "交付", "送达", "签收", "到货"], "month": ["每月", "每个月", "按月"], "1.5 percent": ["1.5%"],
    "dollars": ["美元"], "sunday": ["周日", "星期日", "星期天"], "upper": ["上限", "上界"], "above": ["超过", "高于", "以上"],
    "exceed": ["超过", "超出"], "higher": ["高于", "大于"], "bound": ["边界", "上下限"], "range": ["范围", "区间"],
    "limit": ["限制", "限定"], "facilities": ["设施", "行政"], "fork": ["复刻", "派生"], "branch": ["分支"],
    "commit": ["提交"], "pull request": ["拉取请求", "合并请求"], "issue": ["议题", "问题"], "log rotation": ["日志轮转"],
    "backup": ["备份"], "1 am": ["凌晨1点", "凌晨 1 点"], "2 am": ["凌晨2点", "凌晨 2 点"], "4 am": ["凌晨4点", "凌晨 4 点"],
    "snapshot": ["快照"], "percent": ["%", "百分之"], "three quarters": ["四分之三"],
}
NUMBER_WORDS = {"two": ["两", "二"], "three": ["三"], "five": ["五"], "six": ["六"], "ten": ["十"], "thirty": ["三十"],
                "forty-five": ["四十五"], "forty five": ["四十五"], "fifty-two": ["五十二"], "fifty two": ["五十二"],
                "seventy-five percent": ["百分之七十五"], "seventy five percent": ["百分之七十五"]}


def _alts(value: str) -> list[str]:
    low = value.lower()
    if low in NUMBER_WORDS:
        return NUMBER_WORDS[low]
    return [z for k, zs in ZH_ALTS.items() if k in low for z in zs]


def _extend(values: list) -> list:
    out = list(values)
    for v in values:
        if isinstance(v, str) and re.search(r"[A-Za-z]{2,}", v) and not re.search(r"[\u4e00-\u9fff]", v):
            out += [z for z in _alts(v) if z not in out]
    return out


def zh_expect(expect: dict) -> dict:
    """Judge expectations for a translated task: Chinese equivalents next to English answer phrases."""
    expect = json.loads(json.dumps(expect))
    for turn in expect.get("turns") or []:
        if not turn:
            continue
        for key in ("output_contains_any", "output_contains_any_token"):
            if isinstance(turn.get(key), list):
                turn[key] = _extend(turn[key])
        if isinstance(turn.get("output_contains_groups"), list):
            turn["output_contains_groups"] = [_extend(g) for g in turn["output_contains_groups"]]
        turn.pop("output_format", None)  # "en"; judge.py does not implement it
    return expect


def _zh(text: str) -> bool:
    return len(re.findall(r"[一-鿿]", text)) > len(re.findall(r"[A-Za-z]+", text)) / 2


def _pi_expect_tools(expect: dict) -> dict:
    """Expand DSH required_tools entries with their Pi tool names so judge.py matches Pi session logs."""
    expect = json.loads(json.dumps(expect))
    for turn in expect.get("turns") or []:
        if not turn or not isinstance(turn.get("required_tools"), list):
            continue
        mapped = []
        for want in turn["required_tools"]:
            alts = list(want) if isinstance(want, list) else [want]
            expanded = list(dict.fromkeys([*alts, *(d for a in alts for d in OLD_TO_DSH.get(a, [a]) if d in PI_TOOLS)]))
            mapped.append(expanded if len(expanded) > 1 or isinstance(want, list) else expanded[0])
        turn["required_tools"] = mapped
    return expect


def _mappable(expect: dict) -> bool:
    for turn in expect.get("turns") or []:
        for want in (turn or {}).get("required_tools") or []:
            alts = set(want) if isinstance(want, list) else {want}
            if not {d for a in alts for d in OLD_TO_DSH.get(a, [a])} & PI_TOOLS:
                return False
    return True


def skip_reason(row: dict, task: Path, translations: dict | None = None) -> str | None:
    if row["dataset"] not in DATASETS or row.get("web"):
        return "web_or_other_dataset"
    if not (task / "environment" / "workspace").is_dir():
        return "no_workspace"
    turns_file = task / "environment" / "turns.json"
    turns = json.loads(turns_file.read_text()) if turns_file.is_file() else [(task / "instruction.md").read_text()]
    if not all(_zh(t) for t in turns) and f"{PREFIX}{row['dataset']}-{row['task_id']}" not in (translations or {}):
        return "not_chinese"
    expect = json.loads((task / "tests" / "expect.json").read_text())
    if len(expect.get("turns") or []) != len(turns):
        return "turn_count_mismatch"
    if not _mappable(expect):
        return "needs_non_pi_tool"
    return None


def import_task(row: dict, task: Path, out_root: Path, agent_timeout: int = 900, verifier_timeout: int = 300,
                translated: list[str] | None = None, base_image: str | None = None) -> Path:
    name = f"{PREFIX}{row['dataset']}-{row['task_id']}"
    task_dir = out_root / name
    if task_dir.exists():
        raise FileExistsError(task_dir)
    turns_file = task / "environment" / "turns.json"
    turns = json.loads(turns_file.read_text()) if turns_file.is_file() else [(task / "instruction.md").read_text()]
    expect = _pi_expect_tools(json.loads((task / "tests" / "expect.json").read_text()))
    if translated:
        if len(translated) != len(turns):
            raise ValueError(f"{name}: translation turn count")
        turns, expect = translated, zh_expect(expect)
    tmp = out_root / f".{name}.partial"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(task / "environment" / "workspace", tmp / "environment" / "workspace", symlinks=True)
    (tmp / "environment" / "Dockerfile").write_text(task_dockerfile(base_image))
    (tmp / "environment" / "docker-compose.yaml").write_text(COMPOSE_OVERRIDE)
    steps = [f"step-{i + 1}" for i in range(len(turns))]
    meta = {k: row.get(k) for k in ("dataset", "task_id", "bucket", "category", "requirement", "origin_bank",
                                     "origin_case", "split", "trained_in", "pipeline_commit")}
    lines = ['schema_version = "1.4"', 'multi_step_reward_strategy = "final"', "", "[task]",
             f"name = {_toml_str('rwkv-pi/' + name)}", f"description = {_toml_str(row.get('description') or '')}", ""]
    for s in steps:
        lines += ["[[steps]]", f"name = {_toml_str(s)}", "", "[steps.agent]", f"timeout_sec = {agent_timeout}.0", "",
                  "[steps.verifier]", f"timeout_sec = {verifier_timeout}.0", ""]
    lines += ["[metadata]", 'source = "RWKV-Agent-Lab/rwkv-dsh-training-questions"',
              *(f"dsh_{k} = {json.dumps(v if v is not None else '', ensure_ascii=False)}" for k, v in meta.items()),
              f"dsh_translated_from_english = {'true' if translated else 'false'}", "",
              "[verifier]", f"timeout_sec = {verifier_timeout}.0", 'environment_mode = "shared"', "",
              "[agent]", f"timeout_sec = {agent_timeout}.0", "",
              "[environment]", "build_timeout_sec = 900.0", "cpus = 2", "memory_mb = 4096", "storage_mb = 8192", "",
              "[environment.env]", *(f'{k} = ""' for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")), ""]
    (tmp / "task.toml").write_text("\n".join(lines))
    for i, (s, text) in enumerate(zip(steps, turns), 1):
        tests = tmp / "steps" / s / "tests"
        tests.mkdir(parents=True)
        (tmp / "steps" / s / "instruction.md").write_text(text.rstrip() + "\n", encoding="utf-8")
        shutil.copy(task / "tests" / "judge.py", tests / "judge.py")
        (tests / "expect.json").write_text(json.dumps(expect, ensure_ascii=False, indent=2))
        (tests / "dsh_step.py").write_text(DSH_STEP)
        (tests / "test.sh").write_text(TEST_SH.format(n=i, total=len(turns)))
        (tests / "test.sh").chmod(0o755)
    tmp.rename(task_dir)
    return task_dir


def run(root: Path, out_root: Path, translations: Path | None = None, *, base_image: str | None = None) -> dict:
    """Import every eligible task; English tasks only when `translations` ({task name: [Chinese turns]}) has them."""
    out_root.mkdir(parents=True, exist_ok=True)
    zh = json.loads(translations.read_text()) if translations else {}
    counts: dict[str, int] = {}
    for line in (root / "index.jsonl").read_text().splitlines():
        row = json.loads(line)
        task = root / row["dataset"] / row["task_id"]
        reason = skip_reason(row, task, zh)
        if reason is None and (out_root / f"{PREFIX}{row['dataset']}-{row['task_id']}").exists():
            reason = "exists"
        if reason is None:
            name = f"{PREFIX}{row['dataset']}-{row['task_id']}"
            import_task(row, task, out_root, translated=zh.get(name), base_image=base_image)
            reason = "imported_translated" if name in zh else "imported"
        counts[reason] = counts.get(reason, 0) + 1
    return counts
