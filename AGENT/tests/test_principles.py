"""원칙이 실제로 강제되는지 검증한다.

이 파일이 핵심이다. 원칙을 문서에만 적어두면 언젠가 조용히 위반된다.
여기 있는 테스트가 깨지면 아키텍처가 아니라 **논지**가 깨진 것이다.
"""

from __future__ import annotations

import pytest

from agentcore.broker import (
    Broker,
    BrokerMode,
    Candidate,
    ColdStart,
    QuorumPolicy,
    ServiceLevel,
    VerifierDownPolicy,
    error_correlation,
)
from agentcore.contracts import (
    ComponentContract,
    InterfaceSpec,
    OperatingEnvelope,
    PerformanceSpec,
    Role,
)
from agentcore.control import (
    Budget,
    BudgetExceeded,
    DeterministicSignal,
    NonExternalSignal,
    OpinionSignal,
    PatternWithoutJustification,
    SignalResult,
    Step,
    chain,
    refine,
)
from agentcore.evaluation import CalibratedJudge, JudgeVerdict, UncalibratedJudge
from agentcore.trace import MemorySink, Tracer
from agentcore.verify import Guardrail, GuardrailReport


def _contract(**over):
    base = dict(
        name="c",
        version="1.0",
        role=Role.SPECIALIST,
        interface=InterfaceSpec("in", "out"),
        performance=PerformanceSpec("es", {"acc": 0.9}),
        envelope=OperatingEnvelope("all"),
    )
    base.update(over)
    return ComponentContract(**base)


# -- 코어는 동결된다 --------------------------------------------------------

def test_core_finetuning_is_forbidden():
    with pytest.raises(ValueError, match="7단"):
        _contract(role=Role.CORE, ladder_rung=7)


def test_verifier_cannot_be_parametric():
    """적응하는 시스템에는 적응하지 않는 기준점이 필요하다."""
    with pytest.raises(ValueError, match="신뢰의 뿌리"):
        _contract(role=Role.VERIFIER, ladder_rung=6)


def test_verifier_at_decidable_rung_is_allowed():
    c = _contract(role=Role.VERIFIER, ladder_rung=2)
    assert not c.is_parametric


# -- 패턴은 정당화를 요구한다 ------------------------------------------------

def test_pattern_without_justification_is_rejected():
    with pytest.raises(PatternWithoutJustification):
        chain([Step("s", lambda x: x)], 1, prevents="")


def test_pattern_with_justification_runs():
    out = chain([Step("double", lambda x: x * 2)], 3, prevents="순차 의존성")
    assert out == 6


# -- E-O 는 외부 신호만 받는다 -----------------------------------------------

def test_refine_rejects_opinion_signal():
    """생성자와 같은 정보만 보는 판정자는 맹점을 공유한다."""
    with pytest.raises(NonExternalSignal):
        refine(
            lambda p, f: p,
            OpinionSignal(lambda c: SignalResult(True)),
            "x",
            prevents="환각 인용",
        )


def test_refine_is_bounded_and_targets_failures():
    seen: list[tuple[str, ...]] = []

    def gen(payload, failures):
        seen.append(failures)
        return f"{payload}-{len(failures)}"

    sig = DeterministicSignal(lambda c: SignalResult(passed=False, failures=("c1",)))
    _, result, rounds = refine(gen, sig, "draft", prevents="스키마 위반", max_rounds=3)
    assert rounds == 3 and not result.passed
    assert seen[0] == () and seen[1] == ("c1",)


# -- 동적 구간은 예산 안에서만 -----------------------------------------------

def test_budget_nests_and_propagates():
    root = Budget(max_tokens=100)
    child = root.child()
    child.spend(tokens=60)
    assert root.used_tokens == 60
    with pytest.raises(BudgetExceeded):
        child.spend(tokens=50)


def test_depth_is_bounded():
    b = Budget(max_depth=1)
    with pytest.raises(BudgetExceeded):
        b.child().child()


# -- 가드레일은 집계되지 않는다 ----------------------------------------------

def test_guardrail_report_has_no_score():
    """가드레일은 꼬리 지표다. 점수를 만들면 언젠가 평균된다."""
    r = GuardrailReport()
    assert not hasattr(r, "score")
    assert not hasattr(r, "rate")
    assert not hasattr(r, "pass_rate")


def test_guardrail_tolerance_requires_reason():
    with pytest.raises(ValueError, match="tolerance_reason"):
        Guardrail("g", lambda a, b: True, "m", tolerance=1)


# -- Judge 는 캘리브레이션 없이 못 쓴다 --------------------------------------

def test_uncalibrated_judge_is_blocked():
    j = CalibratedJudge("j@v1", lambda c, ctx: JudgeVerdict(True))
    with pytest.raises(UncalibratedJudge):
        j("anything")


# -- 가용성: 검증기 다운은 fail closed ---------------------------------------

def test_verifier_down_halts_output():
    q = QuorumPolicy(required_roles=frozenset({Role.VERIFIER}))
    d = q.decide({Role.CORE: {"a"}, Role.VERIFIER: set()})
    assert d.level is ServiceLevel.HALTED
    assert not d.can_proceed


def test_quorum_degrades_but_proceeds():
    q = QuorumPolicy(
        min_by_role={Role.CORE: 3}, required_roles=frozenset({Role.VERIFIER})
    )
    d = q.decide({Role.CORE: {"a", "b"}, Role.VERIFIER: {"v"}})
    assert d.level is ServiceLevel.DEGRADED and d.can_proceed


def test_verifier_down_can_degrade_only_with_explicit_policy():
    q = QuorumPolicy(
        required_roles=frozenset({Role.VERIFIER}),
        verifier_down=VerifierDownPolicy.DEGRADE_WITH_WARNING,
    )
    d = q.decide({Role.VERIFIER: set()})
    assert d.level is ServiceLevel.DEGRADED


# -- 브로커: 콜드 스타트와 다양성 --------------------------------------------

def test_dynamic_broker_refuses_cold_start():
    b = Broker(mode=BrokerMode.DYNAMIC)
    with pytest.raises(ColdStart):
        b.select([Candidate("a", "f", 1.0, value=0.9)], task_key="t")


def test_optimization_does_not_collapse_diversity():
    """순수 가치/비용 순이면 같은 계열 두 개를 뽑는다. 제약이 그것을 막는다."""
    cands = [
        Candidate("q1", "qwen", 1.0, 100, 0.95),
        Candidate("q2", "qwen", 1.0, 100, 0.94),
        Candidate("l1", "llama", 1.0, 100, 0.70),
    ]
    b = Broker(mode=BrokerMode.DYNAMIC, team_size=2, max_per_family=1, observations=99)
    sel = b.select(cands, task_key="t")
    assert set(sel.families) == {"qwen", "llama"}


def test_deterministic_mode_is_reproducible():
    cands = [Candidate(f"m{i}", "f", 1.0, 10, 0.5) for i in range(5)]
    b = Broker(mode=BrokerMode.DETERMINISTIC, fixed_team=("m1", "m3"))
    a = b.select(cands, task_key="t")
    c = b.select(cands, task_key="t")
    assert a.chosen == c.chosen == ("m1", "m3")


def test_replay_requires_recorded_decision():
    b = Broker(mode=BrokerMode.REPLAY)
    with pytest.raises(KeyError, match="기록"):
        b.select([Candidate("a", "f", 1.0)], task_key="unknown")


def test_selection_is_always_traced():
    """브로커 결정이 궤적에 남아야 재현이 가능하다."""
    sink = MemorySink()
    b = Broker(mode=BrokerMode.DETERMINISTIC, fixed_team=("a",))
    b.select([Candidate("a", "f", 1.0)], task_key="t1", tracer=Tracer(sink=sink))
    span = sink.spans[0]
    assert span.attributes["chosen"] == ["a"]
    assert span.attributes["task_key"] == "t1"


def test_error_correlation_detects_groupthink():
    out = {
        "a": [True, False, False, True],
        "b": [True, False, False, True],   # a 와 완전히 같은 실패
        "c": [False, True, True, False],   # 반대
    }
    corr = error_correlation(out)
    assert corr[("a", "b")] == 1.0
    assert corr[("a", "c")] == 0.0
