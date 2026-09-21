"""컴포넌트 층위 평가 — 모듈화 논지의 증명.

각 조각을 따로 측정하고 교체할 수 있어야 "조합형의 우위는 검증 가능성"이
참이 된다. 이 파일이 깨지면 그 주장이 근거를 잃는다.
"""

from __future__ import annotations

import pytest

from agentcore.contracts import (
    ComponentContract,
    FunctionComponent,
    InterfaceSpec,
    OperatingEnvelope,
    PerformanceSpec,
    Role,
)
from agentcore.evaluation import (
    ComponentBoard,
    ComponentEvalSet,
    EvalItem,
    Verdict,
    decide_swap,
    score_component,
)


def make(name: str, version: str, fn, limit: int = 5, rung: int = 6):
    contract = ComponentContract(
        name=name,
        version=version,
        role=Role.SPECIALIST,
        interface=InterfaceSpec("str", "str"),
        performance=PerformanceSpec("es", {}),
        envelope=OperatingEnvelope(f"{limit}자 이하", lambda p: len(p) <= limit),
        ladder_rung=rung,
    )
    return FunctionComponent(contract, fn)


@pytest.fixture
def eval_set():
    es = ComponentEvalSet("filter_v1", "f")
    for i, (payload, expected) in enumerate([("ab", "AB"), ("cd", "CD"), ("ef", "EF")]):
        es.add(EvalItem(f"i{i}", payload, expected))
    es.add(EvalItem("oob", "way_too_long_input", out_of_envelope=True))
    return es


# -- 측정 ------------------------------------------------------------------

def test_perfect_component_scores_one(eval_set):
    s = score_component(make("f", "1.0", str.upper), eval_set)
    assert s.accuracy == 1.0
    assert s.envelope_precision == 1.0
    assert s.n == 4


def test_envelope_rejection_counts_as_correct(eval_set):
    """범위 밖을 거절하는 것은 실패가 아니라 계약 준수다."""
    s = score_component(make("f", "1.0", str.upper), eval_set)
    assert s.envelope_precision == 1.0


def test_component_that_accepts_everything_fails_envelope():
    es = ComponentEvalSet("es", "f")
    es.add(EvalItem("oob", "way_too_long", out_of_envelope=True))
    permissive = make("f", "1.0", str.upper, limit=1000)
    assert score_component(permissive, es).envelope_precision == 0.0


def test_eval_set_reports_envelope_coverage(eval_set):
    """정상 경로만 테스트하면 유효 범위 계약이 검증되지 않는다."""
    assert eval_set.envelope_coverage == 0.25
    assert ComponentEvalSet("empty", "f").envelope_coverage == 0.0


def test_score_can_be_published_as_contract_spec(eval_set):
    """Stage 1(측정) → Stage 2(계약 발행)의 연결."""
    spec = score_component(make("f", "1.0", str.upper), eval_set).as_spec()
    assert spec.eval_set == "filter_v1"
    assert spec.meets({"accuracy": 0.9})


# -- 교체 판단 -------------------------------------------------------------

def test_t2_that_only_ties_t1_is_unjustified(eval_set):
    """기성품이 충분한데 맞춤형을 두면 관리 대상과 망각 위험만 늘어난다."""
    t1 = score_component(make("bm25", "1.0", str.upper), eval_set)
    t2 = score_component(make("f", "2.0", str.upper), eval_set)
    d = decide_swap(t2, incumbent=None, baseline=t1)
    assert d.verdict is Verdict.UNJUSTIFIED
    assert not d.should_deploy


def test_t2_beating_baseline_is_promoted():
    es = ComponentEvalSet("es", "f")
    es.add(EvalItem("a", "x", "X"))
    es.add(EvalItem("b", "y", "Y"))
    weak = score_component(make("bm25", "1.0", lambda s: s), es)          # 0.0
    strong = score_component(make("f", "2.0", str.upper), es)             # 1.0
    assert decide_swap(strong, baseline=weak).verdict is Verdict.PROMOTE


def test_regression_below_floor_rolls_back(eval_set):
    bad = score_component(make("f", "2.0", str.lower), eval_set)
    d = decide_swap(bad, floor={"accuracy": 0.9})
    assert d.verdict is Verdict.ROLLBACK
    assert "하한" in d.reason


def test_small_gain_keeps_incumbent():
    es = ComponentEvalSet("es", "f")
    for i in range(10):
        es.add(EvalItem(f"i{i}", f"a{i}", f"A{i}"))
    inc = score_component(make("f", "1.0", lambda s: s.upper() if s != "a0" else "wrong"), es)
    cand = score_component(make("f", "2.0", str.upper), es)
    assert decide_swap(cand, inc, min_gain=0.2).verdict is Verdict.KEEP


def test_accuracy_gain_does_not_justify_unbounded_cost():
    """성공률은 반드시 비용과 쌍으로 읽힌다."""
    from agentcore.evaluation import ComponentScore

    inc = ComponentScore("f", "1.0", "es", 10, 0.80, 1.0, 0, tokens=100, wall_seconds=0.1)
    cand = ComponentScore("f", "2.0", "es", 10, 0.85, 1.0, 0, tokens=500, wall_seconds=0.5)
    d = decide_swap(cand, inc, cost_ceiling_ratio=1.5)
    assert d.verdict is Verdict.KEEP
    assert "토큰" in d.reason


def test_deltas_are_reported_for_audit():
    from agentcore.evaluation import ComponentScore

    inc = ComponentScore("f", "1.0", "es", 10, 0.80, 1.0, 0, 100, 0.1)
    cand = ComponentScore("f", "2.0", "es", 10, 0.90, 1.0, 0, 110, 0.1)
    d = decide_swap(cand, inc)
    assert d.deltas["accuracy"] == pytest.approx(0.10)
    assert d.verdict is Verdict.PROMOTE


# -- 점수판 ----------------------------------------------------------------

def test_board_tracks_latest_version(eval_set):
    board = ComponentBoard()
    board.record(score_component(make("f", "1.0", str.upper), eval_set))
    board.record(score_component(make("f", "2.0", str.lower), eval_set))
    assert board.latest("f").version == "2.0"
    assert len(board.report()) == 2
