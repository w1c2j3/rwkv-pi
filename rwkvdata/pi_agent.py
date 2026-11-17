"""Harbor's stock Pi agent, minus the per-trial network install when the task image already ships Pi.

harbor_export.DOCKERFILE installs the pinned Pi into /usr/local and an empty ~/.nvm/nvm.sh (the stock run
command sources it). Stock install() downloads nvm, Node and Pi from GitHub/nodejs.org/npm in every container,
which timed out for 59 wave-1 tasks. Images without Pi, or with another version, fall back to the stock install.

ZH_SYSTEM_PROMPT is appended to Pi's system prompt so the teacher reasons in Chinese (wave 1 was 7% Chinese).
"""
import json
import os
import shlex

try:
    from harbor.agents.installed.pi import Pi
except ModuleNotFoundError:  # pragma: no cover - unit tests outside Harbor venv
    Pi = object  # type: ignore[misc,assignment]

ZH_SYSTEM_PROMPT = (
    "本次任务的工作语言是简体中文。所有分析、推理和最终答复均须用简体中文书写，包括每次工具调用前后的分析。"
    "只有代码、命令、路径、标识符以及必须引用的原文可以保留英文；不要用英文句子起草分析后只翻译最终答复。"
    "根据实际需要选择工具，不重复操作凑次数。根据真实工具观察完成任务与必要验证，然后简洁结束。"
)
PARITY_EXTENSION_PATH = "/root/.pi/agent/harbor-parity.mjs"
PARITY_EXTENSION = (
    'import { createBashToolDefinition } from "@earendil-works/pi-coding-agent";\n'
    "export default function (pi) {\n"
    "  pi.registerTool(createBashToolDefinition(process.cwd(), { exposeSessionEnvironment: false }));\n"
    "}\n"
)


def _parity_extension_script() -> str:
    """Install the bash parity extension so Harbor hides PI_* session vars and matches rwkv-pi's <rules>."""
    return f"mkdir -p ~/.pi/agent && printf %s {shlex.quote(PARITY_EXTENSION)} > {shlex.quote(PARITY_EXTENSION_PATH)}"


def _models_json_script(base_url: str) -> str:
    """Idempotent container command writing ~/.pi/agent/models.json with a sticky per-trial x-litellm-session-id."""
    model_spec = {
        "reasoning": True,
        "input": ["text"],
        "contextWindow": 131072,
        "maxTokens": 16384,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "samplingParams": {"thinking": {"type": "enabled"}, "reasoning_effort": "high", "parallel_tool_calls": False},
        "compat": {
            "supportsStore": False,
            "supportsDeveloperRole": False,
            "supportsReasoningEffort": False,
            "supportsStrictMode": False,
            "maxTokensField": "max_tokens",
            "requiresReasoningContentOnAssistantMessages": True,
        },
    }
    template = {
        "providers": {
            "deepseek": {
                "baseUrl": base_url,
                "api": "openai-completions",
                "apiKey": "${DEEPSEEK_API_KEY}",
                "authHeader": True,
                "headers": {"x-litellm-session-id": "__SESSION_ID__"},
                "models": [
                    {"id": "rwkv-ds41f-1017", "name": "rwkv-ds41f-1017", **model_spec},
                    {"id": "deepseek-flash", "name": "deepseek-flash", **model_spec},
                ],
            }
        }
    }
    raw = shlex.quote(json.dumps(template, ensure_ascii=False))
    return (
        'if [ ! -s ~/.pi/agent/models.json ]; then '
        'mkdir -p ~/.pi/agent && SID="pi-harbor-$(cat /proc/sys/kernel/random/uuid)" && '
        f'printf %s {raw} | sed "s/__SESSION_ID__/$SID/g" > ~/.pi/agent/models.json && chmod 600 ~/.pi/agent/models.json; '
        'fi'
    )


class PreinstalledPi(Pi):
    def __init__(self, *args, base_url: str | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL")

    async def install(self, environment) -> None:
        want = shlex.quote(self._version or "")
        try:
            await self.exec_as_agent(
                environment, command=f'. ~/.nvm/nvm.sh; [ -z {want} ] || pi --version | grep -qxF {want}')
        except Exception:
            await super().install(environment)
        if self._base_url:
            await self.exec_as_agent(environment, command=_models_json_script(self._base_url))
        await self.exec_as_agent(environment, command=_parity_extension_script())

    def build_cli_flags(self) -> str:
        flags = super().build_cli_flags()
        extra = (
            f"--tools read,write,edit,bash --no-extensions --extension {shlex.quote(PARITY_EXTENSION_PATH)} "
            f"--no-skills --no-context-files --no-prompt-templates --no-mcp "
            f"--append-system-prompt {shlex.quote(ZH_SYSTEM_PROMPT)}"
        )
        return f"{flags} {extra}" if flags else extra
