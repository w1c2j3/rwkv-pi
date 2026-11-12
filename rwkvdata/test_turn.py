"""Build the final "run these tests" user turn from the workspace itself.

The turn names concrete commands found in the project (Makefile targets,
package.json scripts, test files). No reference solution is involved; the
agent's own feedback loop on these commands is what the trajectory records.
The instruction text is model-facing and therefore Chinese.
"""
from __future__ import annotations

import hashlib
import json
import posixpath
import re

_MAKE_TARGET = re.compile(r"^(test|check)\s*:", re.M)


def _cd(directory: str, command: str) -> str:
    return command if directory in ("", ".") else f"cd {directory} && {command}"


def detect_commands(files: dict[str, str]) -> list[str]:
    """Return de-duplicated test commands, most specific first."""
    commands: list[str] = []
    for path, text in sorted(files.items()):
        directory, name = posixpath.split(path)
        if name == "Makefile":
            for target in _MAKE_TARGET.findall(text):
                commands.append(_cd(directory, f"make {target}"))
        elif name == "package.json":
            try:
                scripts = json.loads(text).get("scripts", {})
            except (json.JSONDecodeError, AttributeError):
                continue
            if "test" in scripts:
                commands.append(_cd(directory, "npm test"))
    if commands:
        return list(dict.fromkeys(commands))

    # No declared target: fall back to test files the project ships.
    py_tests = sorted(p for p in files if re.search(r"(^|/)tests?/test_[^/]+\.py$", p))
    if py_tests:
        root = py_tests[0].split("/tests/")[0] if "/tests/" in py_tests[0] else ""
        uses_pytest = any("import pytest" in files[p] for p in py_tests)
        start = "tests" if posixpath.basename(posixpath.dirname(py_tests[0])) == "tests" else "test"
        cmd = "python3 -m pytest -q" if uses_pytest else f"python3 -m unittest discover -s {start} -v"
        commands.append(_cd(root, cmd))
    for path in sorted(p for p in files if re.search(r"(^|/)tests?/[^/]+\.sh$", p)):
        commands.append(f"bash {path}")
    for path in sorted(p for p in files if re.search(r"(^|/)tests?/[^/]+\.test\.(m?js|ts)$", p)):
        commands.append(f"node --test {path}")
    return list(dict.fromkeys(commands))


_TEMPLATES = (
    "前面的改动都完成了。现在请在项目里运行下面的测试：\n\n{commands}\n\n"
    "根据输出定位失败原因并修复，修复后重新运行，直到这些测试全部通过。"
    "如果某个失败来自测试本身或环境而不是代码，请给出依据并说明，不要为了通过而删改测试。"
    "最后汇报你实际运行的命令和最终结果。",
    "请运行项目现有的测试来验证这几轮修改：\n\n{commands}\n\n"
    "如果有失败，逐个排查原因、修改代码并重新运行，直到全部通过。"
    "不要删除或放宽测试；若确认是测试或环境本身的问题，说明理由。"
    "完成后报告运行过的命令和最终通过情况。",
    "接下来跑一下测试，确认之前的功能都没问题：\n\n{commands}\n\n"
    "看到失败就根据报错修复，再重新运行，循环到测试全部通过为止。"
    "测试代码不要为了通过而改动；如果判断是测试或环境的问题，请说明依据。"
    "最后告诉我执行了哪些命令、结果如何。",
)


def instruction(task_id: str, commands: list[str]) -> str:
    if not commands:
        raise ValueError(f"{task_id}: no test command detected")
    pick = int(hashlib.sha256(task_id.encode()).hexdigest(), 16) % len(_TEMPLATES)
    listed = "\n".join(f"- `{c}`" for c in commands)
    return _TEMPLATES[pick].format(commands=listed)


# Debugging turn: a regression was planted before this turn (rwkvdata/faults.py). The text
# reports the symptom the way a teammate would; it never names the file or the change, and
# it stays truthful if injection was impossible (the agent then just confirms green tests).
_DEBUG_TEMPLATES = (
    "刚才同事合入了一个改动，之后 CI 好像开始报错了。请先运行项目的测试：\n\n{commands}\n\n"
    "如果有失败，根据失败输出定位根因（不要只看报错那一行，要确认是哪处代码行为不对），修复代码后重新运行，"
    "直到全部通过。不要修改、删除或跳过测试来让它通过。"
    "最后说明问题出在哪里、你是怎么确认的、改了什么，以及最终的测试结果。",
    "有人反馈最新代码上测试不过了，但不清楚是哪次改动引起的。请运行：\n\n{commands}\n\n"
    "看到失败就沿着报错排查到具体原因，修好以后再跑一遍确认；如果修完还有失败，继续排查，循环到全部通过为止。"
    "测试本身不要动。完成后告诉我根因、修改内容和你实际运行过的命令及结果。",
    "合并之后流水线变红了，麻烦排查一下。先在项目里跑测试：\n\n{commands}\n\n"
    "根据失败信息找到出问题的代码并修复，修复后重新运行测试验证，直到全部通过。"
    "请不要为了通过而改测试或跳过用例；如果你认为是测试或环境的问题，给出依据。"
    "最后汇报：失败现象、根因、修复方式、最终测试结果。",
)


def debug_instruction(task_id: str, commands: list[str]) -> str:
    if not commands:
        raise ValueError(f"{task_id}: no test command detected")
    pick = int(hashlib.sha256(("debug:" + task_id).encode()).hexdigest(), 16) % len(_DEBUG_TEMPLATES)
    listed = "\n".join(f"- `{c}`" for c in commands)
    return _DEBUG_TEMPLATES[pick].format(commands=listed)
