"""챗봇 에이전트 — 원칙이 대화 루프에서도 지켜지는지 검증한다.

이 파일이 깨지면 "근거 없으면 내보내지 않는다"가 문서일 뿐이라는 뜻이다.
전 케이스가 결정론적이다 — 모델을 부르지 않는다.
"""

from __future__ import annotations

import pytest

from agentcore.apps import ChatAgent, is_abstention, lexical_entails, parse_claims
from agentcore.apps.chat import attach_support
from agentcore.memory import Document
from agentcore.models import EchoLLM
from agentcore.trace import MemorySink, SpanKind, Tracer

DOCS = [
    Document("vacation", "연차는 입사 1년 차에 11일 부여된다. 사용 3일 전에 신청한다."),
    Document("remote", "재택근무는 주 2회까지 가능하다. 전주 금요일까지 등록한다."),
]


def agent_with(script, **kw) -> ChatAgent:
    """응답을 고정한 에이전트. script 는 호출 순서대로 소비된다."""
    seq = list(script)

    def responder(messages):
        return seq.pop(0) if len(seq) > 1 else seq[0]

    return ChatAgent(DOCS, llm=EchoLLM(responder=responder), **kw)


# --------------------------------------------------------------------------
# 파싱과 함의 — 순수 함수
# --------------------------------------------------------------------------

def test_형식을_지킨_줄만_주장이_된다():
    claims, malformed = parse_claims(
        "안녕하세요!\n- 연차는 11일이다 [vacation]\n- 근거 없는 문장\n"
    )
    assert [c.text for c in claims] == ["연차는 11일이다"]
    assert len(malformed) == 2          # 인사말과 미인용 문장 — 조용히 버리지 않는다


def test_형식_위반은_버려지지_않고_되돌아온다():
    _, malformed = parse_claims("결론적으로 연차는 11일입니다.")
    assert malformed and "결론적으로" in malformed[0]


def test_존재하지_않는_출처는_원문으로_해석되지_않는다():
    claims, _ = parse_claims("- 아무 말 [ghost]")
    attached = attach_support(claims, {d.id: d for d in DOCS})
    assert attached[0].support == ()    # → GroundingVerifier 가 unsupported 로 잡는다


def test_라우터는_검색기와_같은_기준으로_판정한다():
    """조사가 붙어도 문서 질의로 인식되어야 한다. 어긋나면 질문이 샌다."""
    a = ChatAgent(DOCS, llm=EchoLLM(responder=lambda m: "- 연차는 입사 1년 차에 11일 부여된다 [vacation]"))
    assert a._overlap("연차 규정이 궁금해요") == {"연차"}
    assert a.ask("연차 이월 규정이 어떻게 되나요?").route == "grounded"
    assert a._overlap("우주선 발사 절차") == set()


def test_어휘_중첩_함의는_결정론적이다():
    entails = lexical_entails()
    assert entails("연차는 11일 부여된다", ("연차는 입사 1년 차에 11일 부여된다.",))
    assert not entails("연차는 30일 부여되고 이월된다", ("재택근무는 주 2회까지 가능하다.",))


def test_어휘_중첩은_의미_반전을_잡지_못한다():
    """알려진 한계를 테스트로 고정한다.

    한계를 문서에만 적으면 나중에 아무도 모른다. 이 테스트가 깨지는 날은
    함의 판정기를 NLI 모델로 교체한 날이며, 그때 이 테스트를 지운다.
    """
    entails = lexical_entails()
    assert entails("연차는 11일 신청한다", ("연차는 입사 1년 차에 11일 부여된다. 사용 3일 전에 신청한다.",))


# --------------------------------------------------------------------------
# 대화 경로
# --------------------------------------------------------------------------

def test_근거_있는_답은_출처와_함께_나간다():
    a = agent_with(["- 연차는 입사 1년 차에 11일 부여된다 [vacation]"])
    ans = a.ask("연차는 며칠인가요?")
    assert ans.route == "grounded"
    assert ans.grounded and not ans.blocked and not ans.degraded
    assert ans.source_ids == ("vacation",)


def test_허위_출처를_인용하면_차단된다():
    """검색되지 않은 출처를 대면 가드레일이 막는다. 이것이 이 시스템의 요지다."""
    a = agent_with(["- 연차는 60일이다 [payroll]"], max_rounds=1)
    ans = a.ask("연차는 며칠인가요?")
    assert ans.blocked
    assert "known_source" in ans.guardrails.counts_by_rail()
    assert ans.guardrails.breach_count >= 1


def test_근거에서_도출되지_않는_주장은_재시도_후에도_차단된다():
    a = agent_with(["- 연차는 무제한이며 해외 근무도 자유롭다 [vacation]"], max_rounds=2)
    ans = a.ask("연차는 며칠인가요?")
    assert ans.blocked or not ans.grounded
    assert ans.rounds == 2              # 유계 — 판정자가 종료 조건을 정하지 않는다


def test_실패한_주장만_되돌려주고_다음_라운드에서_고쳐진다():
    a = agent_with(
        [
            "- 연차는 무제한이며 해외 근무도 자유롭다 [vacation]",
            "- 연차는 입사 1년 차에 11일 부여된다 [vacation]",
        ],
        max_rounds=2,
    )
    ans = a.ask("연차는 며칠인가요?")
    assert ans.rounds == 2
    assert ans.grounded and not ans.blocked


def test_검색이_비면_모델을_부르지_않는다():
    """게이트는 비싼 작업 앞에 선다. 사후 게이트는 보호가 아니다."""
    calls = []

    def responder(messages):
        calls.append(1)
        return "- 아무 말 [vacation]"

    a = ChatAgent(DOCS, llm=EchoLLM(responder=responder))
    ans = a.ask("회사의 우주선 발사 절차를 알려주세요", mode="grounded")
    assert ans.missing == ("excerpts",)
    assert not ans.grounded
    assert calls == []                  # 한 번도 부르지 않았다


def test_기권은_실패가_아니다():
    a = agent_with(["- 제공된 문서로는 답할 수 없습니다 [none]"], max_rounds=2)
    ans = a.ask("연차 이월 규정이 어떻게 되나요?")
    assert not ans.blocked
    assert not ans.grounded             # 근거 없음을 명시 — 정상 답과 구분된다
    assert ans.rounds == 1              # 재시도 루프에 들어가지 않는다


# --------------------------------------------------------------------------
# 자유 대화 — 검증 불가 계층
# --------------------------------------------------------------------------

def test_코퍼스와_무관한_질문은_자유_대화로_간다():
    a = ChatAgent(DOCS, llm=EchoLLM(responder=lambda m: "안녕하세요."))
    ans = a.ask("오늘 기분이 어때?")
    assert ans.route == "free"
    assert ans.text == "안녕하세요."


def test_자유_대화는_저하가_아니라_검증_불가다():
    """검증 불가와 검증 실패를 같은 플래그로 묶으면 진짜 경고가 묻힌다."""
    a = ChatAgent(DOCS, llm=EchoLLM(responder=lambda m: "잘 모르겠습니다."))
    ans = a.ask("오늘 기분이 어때?")
    assert not ans.verifiable
    assert not ans.grounded
    assert not ans.degraded          # 저하가 아니다 — 애초에 검증 대상이 아니다
    assert ans.source_ids == ()
    assert ans.claims == ()


def test_문서가_없어도_대화가_된다():
    a = ChatAgent([], llm=EchoLLM(responder=lambda m: "무엇을 도와드릴까요?"))
    ans = a.ask("안녕하세요")
    assert ans.route == "free"
    assert ans.text == "무엇을 도와드릴까요?"


def test_자유_대화는_검증기와_가드레일을_지나지_않는다():
    sink = MemorySink()
    a = ChatAgent(DOCS, llm=EchoLLM(responder=lambda m: "아무 말"),
                  tracer=Tracer(sink=sink))
    a.ask("오늘 기분이 어때?")
    assert not any(s.kind is SpanKind.VERIFY for s in sink.spans)


def test_모드_지정이_자동_분류를_이긴다():
    a = agent_with(["- 연차는 입사 1년 차에 11일 부여된다 [vacation]"])
    # 코퍼스와 겹치는 질문이지만 자유 대화를 강제한다
    ans = a.ask("연차는 며칠인가요?", mode="free")
    assert ans.route == "free" and not ans.verifiable

    # 반대로 겹치지 않는 질문에 근거 경로를 강제하면 게이트에 걸린다
    ans = a.ask("오늘 날씨 어때?", mode="grounded")
    assert ans.route == "grounded"
    assert ans.missing == ("excerpts",)


def test_자유_대화를_끄면_근거_경로만_남는다():
    a = ChatAgent(DOCS, llm=EchoLLM(responder=lambda m: "아무 말"), allow_free=False)
    ans = a.ask("오늘 기분이 어때?")
    assert ans.route == "grounded"
    assert ans.missing == ("excerpts",)


def test_알_수_없는_모드는_거절된다():
    a = agent_with(["- 아무 말 [vacation]"])
    with pytest.raises(ValueError):
        a.ask("연차는?", mode="whatever")


def test_자유_대화도_이력을_이어받는다():
    seen = []

    def responder(messages):
        seen.append([(m.role, m.content) for m in messages])
        return "네."

    a = ChatAgent([], llm=EchoLLM(responder=responder))
    a.ask("내 이름은 윤빈이야", subject="u")
    a.ask("내 이름이 뭐라고?", subject="u")
    roles = [r for r, _ in seen[-1]]
    assert roles == ["system", "user", "assistant", "user"]


# --------------------------------------------------------------------------
# 기억 — 보존 규정이 적용되는 계층
# --------------------------------------------------------------------------

def test_기억은_주체_단위로_삭제된다():
    a = agent_with(["- 연차는 입사 1년 차에 11일 부여된다 [vacation]"])
    a.ask("연차는?", subject="u1")
    a.ask("연차는?", subject="u2")
    assert a.history("u1") and a.history("u2")

    removed = a.forget("u1")
    assert removed >= 1
    assert a.history("u1") == []
    assert a.history("u2")              # 다른 주체는 그대로
    assert set(a.corpus) == {"vacation", "remote"}   # 문서 인덱스는 안 건드린다


def test_이력은_상한_안에서만_쌓인다():
    a = agent_with(["- 연차는 입사 1년 차에 11일 부여된다 [vacation]"], history_turns=2)
    for _ in range(5):
        a.ask("연차는 며칠인가요?", subject="u")
    assert len(a.history("u")) == 2


# --------------------------------------------------------------------------
# 궤적 — 로그가 곧 평가 데이터
# --------------------------------------------------------------------------

def test_격리와_제어가_궤적에_남는다():
    from agentcore.evaluation import summarize_trajectory

    # 긴 문서라야 격리가 측정된다 — 짧은 문서는 원문이 곧 발췌라 흡수가 0이다.
    long_doc = Document(
        "vacation",
        "연차는 입사 1년 차에 11일 부여된다. 사용 3일 전에 신청한다. " + "부속 규정 조항. " * 200,
    )
    sink = MemorySink()
    a = ChatAgent(
        [long_doc, DOCS[1]],
        llm=EchoLLM(responder=lambda m: "- 연차는 입사 1년 차에 11일 부여된다 [vacation]"),
        tracer=Tracer(sink=sink),
    )
    a.ask("연차는 며칠인가요?")

    traj = summarize_trajectory(sink.spans)
    assert traj.model_calls == 1
    assert traj.control_ratio > 0.8     # 제어는 코드에 있다
    assert traj.absorbed_tokens > 0     # 원문이 코어로 그대로 올라가지 않았다
    assert any(s.kind is SpanKind.VERIFY for s in sink.spans)


def test_가드레일_리포트에는_점수가_없다():
    a = agent_with(["- 연차는 입사 1년 차에 11일 부여된다 [vacation]"])
    ans = a.ask("연차는 며칠인가요?")
    with pytest.raises(AttributeError):
        ans.guardrails.score


# --------------------------------------------------------------------------
# 온라인 갱신 — 비파라메트릭 적응 표면
# --------------------------------------------------------------------------

def test_문서_추가는_즉시_검색된다():
    a = agent_with(["- 경조사 휴가는 5일이다 [family]"])
    a.add_document(Document("family", "경조사 휴가는 5일 부여된다."))
    ans = a.ask("경조사 휴가는 며칠인가요?")
    assert ans.source_ids == ("family",)
    assert not ans.blocked


def test_문서_철회는_즉시_반영된다():
    """잘못 올린 문서가 계속 인용되면 업로드 기능이 위험해진다."""
    a = agent_with(["- 경조사 휴가는 5일이다 [family]"])
    a.add_document(Document("family", "경조사 휴가는 5일 부여된다."))
    assert a.ask("경조사 휴가는 며칠인가요?").source_ids == ("family",)

    assert a.remove_document("family") is True
    assert a.remove_document("family") is False
    ans = a.ask("경조사 휴가는 며칠인가요?")
    assert ans.route == "free" or ans.missing == ("excerpts",)
