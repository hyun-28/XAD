"""Claude Code CLI 어댑터 — 구독(Pro/Max) 로그인을 그대로 쓰는 경로.

Anthropic API 키가 없는 환경을 위한 것이다. Claude Pro 구독은 API 키를 주지
않으므로 `AnthropicLLM` 을 쓸 수 없고, 대신 이미 로그인된 `claude` CLI 를
헤드리스 모드(`claude -p`)로 호출한다.

    비용은 토큰이 아니라 **구독 사용량 한도**로 계산된다. 호출마다 CLI 의
    기본 시스템 프롬프트가 함께 올라가므로(약 2만 토큰) 토큰 수는 크게
    잡히지만 과금은 되지 않는다. 대신 호출 지연이 크다 — 프로세스 기동
    비용이 매 호출에 붙는다.

그래서 개발 중에는 `RecordingLLM` 으로 한 번 녹음해 두고 `ReplayLLM` 로
재생하는 편이 낫다. 이 어댑터는 녹음 대상이지 반복 실행 대상이 아니다.

도구는 전부 끈다. 이 자리의 모델은 **판단만** 하며, 제어와 도구 호출은
전부 agentcore 쪽 코드에 있다 — CLI 가 자체 판단으로 파일을 읽거나 웹을
뒤지면 그 결정이 궤적에 남지 않아 원칙이 조용히 깨진다.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from typing import Any

from ..trace import current_span
from .base import LLM, Completion, Message

# CLI 가 자체적으로 쓸 수 있는 도구. 전부 차단한다.
_ALL_TOOLS = (
    "Bash,Read,Write,Edit,Glob,Grep,WebFetch,WebSearch,Task,TodoWrite,"
    "NotebookEdit,SlashCommand,KillShell,BashOutput"
)


EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


class ClaudeCLIError(RuntimeError):
    """CLI 호출 실패. 조용한 빈 응답 대신 시끄럽게 터진다."""


class ClaudeCLILLM(LLM):
    """`claude -p` 를 통해 구독 계정으로 모델을 부르는 어댑터.

    model_id 는 CLI 가 받는 별칭("sonnet", "opus", "haiku") 또는 정식
    모델 ID 다. 재현성 요건상 별칭보다 정식 ID 를 권한다 — 별칭은 시간이
    지나면 다른 모델을 가리킨다.

    effort 는 추론에 쓰는 노력의 정도다. **모델 ID 와 함께 핀의 일부**이며,
    같은 모델도 effort 가 다르면 다른 답을 낸다. 그래서 궤적에 함께 남긴다 —
    남기지 않으면 "동일 입력·동일 핀 버전"이 재현을 보장하지 못한다.
    """

    name = "claude_cli"

    def __init__(
        self,
        model_id: str = "claude-opus-5",
        *,
        effort: str | None = "high",
        binary: str = "claude",
        timeout: float = 300.0,
        cwd: str | None = None,
        allow_tools: bool = False,
        size_hint_b: float | None = None,
    ) -> None:
        if effort is not None and effort not in EFFORT_LEVELS:
            raise ValueError(
                f"effort 는 {EFFORT_LEVELS} 중 하나여야 합니다: {effort!r}"
            )
        self.model_id = model_id
        self.effort = effort
        self.timeout = timeout
        self.allow_tools = allow_tools
        self.size_hint_b = size_hint_b
        resolved = shutil.which(binary)
        if resolved is None:
            raise ClaudeCLIError(
                f"'{binary}' 실행 파일을 찾을 수 없습니다. Claude Code CLI 가 "
                f"설치되어 있고 로그인되어 있어야 합니다."
            )
        self.binary = resolved
        # 프로젝트의 CLAUDE.md·설정을 끌어오지 않도록 중립 디렉터리에서 돈다.
        self.cwd = cwd or tempfile.gettempdir()

    # ----------------------------------------------------------------
    def _render(self, messages: list[Message]) -> tuple[str, str]:
        """system 과 대화를 분리한다. 헤드리스 CLI 는 단발 호출이므로
        assistant 턴은 프롬프트 안에 평문으로 접어 넣는다."""
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        turns = [m for m in messages if m.role in ("user", "assistant")]
        if len(turns) == 1:
            return system, turns[0].content
        label = {"user": "User", "assistant": "Assistant"}
        body = "\n\n".join(f"{label[m.role]}: {m.content}" for m in turns)
        return system, body

    def argv(self, system: str, **params: Any) -> list[str]:
        """호출 인자. 부작용이 없으므로 CLI 없이도 검증할 수 있다."""
        effort = params.pop("effort", self.effort)
        if effort is not None and effort not in EFFORT_LEVELS:
            raise ValueError(f"effort 는 {EFFORT_LEVELS} 중 하나여야 합니다: {effort!r}")

        argv = [
            self.binary,
            "-p",
            "--output-format", "json",
            "--model", params.pop("model", self.model_id),
            "--strict-mcp-config",
            "--mcp-config", '{"mcpServers":{}}',
        ]
        if effort:
            argv += ["--effort", effort]
        if system:
            argv += ["--system-prompt", system]
        if not self.allow_tools:
            argv += ["--disallowed-tools", _ALL_TOOLS]
        return argv

    def _complete(self, messages: list[Message], **params: Any) -> Completion:
        system, prompt = self._render(messages)
        effort = params.get("effort", self.effort)
        argv = self.argv(system, **params)
        params.pop("effort", None)
        params.pop("model", None)

        # 핀은 모델 ID 만이 아니다. effort 를 빼고 기록하면 같은 궤적이
        # 서로 다른 답을 낸 이유를 나중에 설명할 수 없다.
        span = current_span()
        if span is not None:
            span.attributes["effort"] = effort

        try:
            proc = subprocess.run(
                argv,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=params.pop("timeout", self.timeout),
                cwd=self.cwd,
            )
        except subprocess.TimeoutExpired as exc:
            raise ClaudeCLIError(f"CLI 호출이 {self.timeout}s 안에 끝나지 않았습니다") from exc

        if proc.returncode != 0:
            raise ClaudeCLIError(
                f"CLI 종료 코드 {proc.returncode}: {(proc.stderr or proc.stdout)[:500]}"
            )

        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise ClaudeCLIError(f"CLI 출력을 JSON 으로 읽을 수 없습니다: {proc.stdout[:300]}") from exc

        if payload.get("is_error"):
            raise ClaudeCLIError(f"CLI 오류 응답: {payload.get('result', '')[:300]}")

        usage = payload.get("usage") or {}
        # 캐시 생성분도 입력에 포함해 실제 상향 토큰을 정직하게 남긴다.
        input_tokens = (
            int(usage.get("input_tokens", 0))
            + int(usage.get("cache_creation_input_tokens", 0))
            + int(usage.get("cache_read_input_tokens", 0))
        )
        return Completion(
            text=payload.get("result", ""),
            input_tokens=input_tokens,
            output_tokens=int(usage.get("output_tokens", 0)),
            model=self.model_id,
            stop_reason=payload.get("stop_reason", "end_turn"),
            raw={"session_id": payload.get("session_id", ""),
                 "duration_api_ms": payload.get("duration_api_ms", 0),
                 "effort": effort},
        )
