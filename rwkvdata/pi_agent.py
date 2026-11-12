"""Harbor's stock Pi agent, minus the per-trial network install when the task image already ships Pi.

harbor_export.DOCKERFILE installs the pinned Pi into /usr/local and an empty ~/.nvm/nvm.sh (the stock run
command sources it). Stock install() downloads nvm, Node and Pi from GitHub/nodejs.org/npm in every container,
which timed out for 59 wave-1 tasks. Images without Pi, or with another version, fall back to the stock install.

ZH_SYSTEM_PROMPT is appended to Pi's system prompt so the teacher reasons in Chinese (wave 1 was 7% Chinese).
"""
import shlex

from harbor.agents.installed.pi import Pi

ZH_SYSTEM_PROMPT = (
    "本次任务的工作语言是简体中文。所有分析、推理和最终答复均须用简体中文书写，包括每次工具调用前后的分析。"
    "只有代码、命令、路径、标识符以及必须引用的原文可以保留英文；不要用英文句子起草分析后只翻译最终答复。"
    "根据实际需要选择工具，不重复操作凑次数。根据真实工具观察完成任务与必要验证，然后简洁结束。"
)


class PreinstalledPi(Pi):
    async def install(self, environment) -> None:
        want = shlex.quote(self._version or "")
        try:
            await self.exec_as_agent(
                environment, command=f'. ~/.nvm/nvm.sh; [ -z {want} ] || pi --version | grep -qxF {want}')
        except Exception:
            await super().install(environment)

    def build_cli_flags(self) -> str:
        flags = super().build_cli_flags()
        extra = f"--append-system-prompt {shlex.quote(ZH_SYSTEM_PROMPT)}"
        return f"{flags} {extra}" if flags else extra
