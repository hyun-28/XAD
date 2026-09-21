"""앙상블 심의 검증.

앙상블은 코어 자리를 채우는 구현 선택지이지 대안 아키텍처가 아니다.
그리고 오류가 독립일 때만 이긴다 — 그 조건이 코드로 강제되는지 본다.
"""

from __future__ import annotations

import pytest

from agentcore.broker import Candidate, error_correlation
from agentcore.ensemble import (
    Consensus,
    Deliberation,
    Draft,
    InsufficientDiversity,
    Review,
    SelfVoteRejected,
    quadratic_scores,
    vote_on_claims,
)
from agentcore.trace import MemorySink, Tracer

MEMBERS = [
    Candidate("m-a", "family-a", cost=1.0, vram_mb=16000, value=0.9),
    Candidate("m-b", "family-b", cost=1.0, vram_mb=16000, value=0.88),
    Candidate("m-c", "family-c", cost=1.0, vram_mb=16000, value=0.85),
]


def _deliberation(**kw):
    def propose(payload, author, state):
        return Draft(author=author, family="", content=f"{payload}:{author}", tokens=10)

    def review(reviewer, visible):
        # 사전순 첫 초안에 크레딧을 몰아준다 — 결정론적 심사.
        target = sorted(d.author for d in visible)[0]
        return [Review(reviewer=reviewer, target=target, credits=4.0)]

    kw.setdefault("members", MEMBERS)
    return Deliberation(propose=propose, review=review, **kw)


# -- 다양성은 가정이 아니라 강제된다 -----------------------------------------

def test_single_family_ensemble_is_rejected():
    with pytest.raises(InsufficientDiversity):
        Deliberation(
            propose=lambda *a: None,
            review=lambda *a: [],
            members=[Candidate("a", "qwen", 1.0), Candidate("b", "qwen", 1.0)],
        )


def test_min_families_is_configurable_but_defaults_to_two():
    d = _deliberation()
    assert d.min_families == 2
    assert d.run("q").diversity_ok


# -- 자기 투표 금지 ---------------------------------------------------------

def test_self_vote_is_rejected():
    with pytest.raises(SelfVoteRejected):
        quadratic_scores([Review("a", "a", 4.0)])


def test_reviewers_never_see_their_own_draft():
    seen: dict[str, set[str]] = {}

    def review(reviewer, visible):
        seen[reviewer] = {d.author for d in visible}
        return [Review(reviewer, sorted(d.author for d in visible)[0], 1.0)]

    d = _deliberation()
    d.review = review
    d.run("q")
    for reviewer, targets in seen.items():
        assert reviewer not in targets


# -- 이차 투표는 다수결이 아니다 --------------------------------------------

def test_quadratic_voting_dampens_concentrated_conviction():
    """한 명의 9크레딧 몰빵과 세 명의 1크레딧씩이 같은 무게가 된다."""
    scores = quadratic_scores(
        [
            Review("r1", "A", 9.0),
            Review("r2", "B", 1.0),
            Review("r3", "B", 1.0),
            Review("r4", "B", 1.0),
        ]
    )
    assert scores["A"] == pytest.approx(3.0)
    assert scores["B"] == pytest.approx(3.0)


def test_simple_majority_would_have_differed():
    """단순 다수결이면 B 가 3표로 이긴다. 이차 투표는 확신도를 반영한다."""
    reviews = [Review("r1", "A", 9.0), Review("r2", "B", 1.0), Review("r3", "B", 1.0)]
    scores = quadratic_scores(reviews)
    assert scores["A"] == pytest.approx(3.0)
    assert scores["B"] == pytest.approx(2.0)


# -- 심의 실행 -------------------------------------------------------------

def test_deliberation_converges_and_traces():
    sink = MemorySink()
    result = _deliberation().run("질문", tracer=Tracer(sink=sink))
    assert isinstance(result, Consensus)
    assert result.winner == "m-a"
    assert result.rounds >= 1
    span = next(s for s in sink.spans if s.name == "deliberate")
    assert span.attributes["winner"] == "m-a"
    assert set(span.attributes["families"]) == {"family-a", "family-b", "family-c"}


def test_rounds_are_bounded():
    d = _deliberation(max_rounds=1, convergence=2.0)   # 절대 수렴 못 하는 임계
    assert d.run("q").rounds == 1


def test_cost_and_vram_are_reported_together():
    """앙상블의 가치는 총 연산 절감이 아니라 최대 메모리 우회다.
    둘을 함께 봐야 판단이 선다."""
    r = _deliberation().run("q")
    assert r.total_tokens > 0            # 총 연산은 오히려 늘어난다
    assert r.peak_vram_mb == 16000       # 절약되는 것은 이쪽


# -- 열린 출력의 투표 단위 ---------------------------------------------------

def test_claim_level_voting_for_open_ended_output():
    """보고서 전체는 투표 단위가 없다. 주장 단위로 쪼개면 성립한다."""
    passed, votes = vote_on_claims(
        {
            "m1": {"c1": True, "c2": False},
            "m2": {"c1": True, "c2": False},
            "m3": {"c1": False, "c2": True},
        }
    )
    assert passed == ["c1"]
    assert votes["c1"].support_ratio == pytest.approx(2 / 3)
    assert votes["c2"].support_ratio == pytest.approx(1 / 3)


def test_error_correlation_would_flag_this_ensemble():
    """다양성 제약이 실제로 필요한지는 궤적에서 측정된다."""
    corr = error_correlation(
        {"m-a": [True, False, False], "m-b": [True, False, False], "m-c": [False, True, True]}
    )
    assert corr[("m-a", "m-b")] == 1.0     # 완전히 같은 실패 — groupthink
    assert corr[("m-a", "m-c")] == 0.0
