"""Claude CLI 어댑터 — 호출 인자와 핀 기록.

CLI 를 실제로 부르지 않는다. 부작용 없는 부분(인자 구성, 메시지 렌더링,
검증)만 검증하며, 그래서 CI 에서도 돈다. 실제 호출은 이 파일의 관심사가
아니다 — 그것을 반복 검증하면 테스트가 구독 사용량을 태운다.
"""

from __future__ import annotations

import shutil

import pytest

from agentcore.models import Message
from agentcore.models.claude_cli import EFFORT_LEVELS, ClaudeCLIError, ClaudeCLILLM

pytestmark = pytest.mark.skipif(
    shutil.which("claude") is None, reason="claude CLI 없음"
)


def test_기본값은_opus5_high():
    m = ClaudeCLILLM()
    assert m.model_id == "claude-opus-5"
    assert m.effort == "high"


def test_effort_가_인자에_실린다():
    argv = ClaudeCLILLM(effort="xhigh").argv("")
    assert argv[argv.index("--effort") + 1] == "xhigh"


def test_effort_를_끌_수_있다():
    assert "--effort" not in ClaudeCLILLM(effort=None).argv("")


def test_알_수_없는_effort_는_생성_시점에_거절된다():
    """오타를 호출 시점까지 끌고 가면 실패가 비싸진다."""
    with pytest.raises(ValueError):
        ClaudeCLILLM(effort="ultra")
    with pytest.raises(ValueError):
        ClaudeCLILLM().argv("", effort="ultra")


@pytest.mark.parametrize("level", EFFORT_LEVELS)
def test_모든_레벨이_받아들여진다(level):
    argv = ClaudeCLILLM(effort=level).argv("")
    assert argv[argv.index("--effort") + 1] == level


def test_도구는_기본적으로_전부_차단된다():
    """CLI 가 자체 판단으로 도구를 쓰면 그 결정이 궤적에 남지 않는다."""
    argv = ClaudeCLILLM().argv("")
    blocked = argv[argv.index("--disallowed-tools") + 1]
    for tool in ("Bash", "Read", "WebSearch", "Task"):
        assert tool in blocked
    assert "--allowedTools" not in argv


def test_MCP_서버는_비워서_고정된다():
    argv = ClaudeCLILLM().argv("")
    assert "--strict-mcp-config" in argv
    assert argv[argv.index("--mcp-config") + 1] == '{"mcpServers":{}}'


def test_시스템_프롬프트가_없으면_인자도_없다():
    assert "--system-prompt" not in ClaudeCLILLM().argv("")
    assert "--system-prompt" in ClaudeCLILLM().argv("너는 계산기다")


def test_단발_호출은_프롬프트를_그대로_보낸다():
    m = ClaudeCLILLM()
    system, prompt = m._render([Message("system", "S"), Message("user", "질문")])
    assert system == "S" and prompt == "질문"


def test_여러_턴은_평문으로_접힌다():
    """헤드리스 CLI 는 단발이므로 이전 턴을 프롬프트에 실어야 이어진다."""
    m = ClaudeCLILLM()
    _, prompt = m._render([
        Message("system", "S"),
        Message("user", "안녕"),
        Message("assistant", "네"),
        Message("user", "내가 뭐라고 했지?"),
    ])
    assert prompt == "User: 안녕\n\nAssistant: 네\n\nUser: 내가 뭐라고 했지?"


def test_없는_실행_파일은_생성_시점에_터진다():
    with pytest.raises(ClaudeCLIError):
        ClaudeCLILLM(binary="claude-does-not-exist")
