"""주식분석 예제 회귀 테스트.

이 도메인의 핵심 주장 — "데이터 계층은 완전히 검증 가능하고 판단 계층은
골드셋이 만들어지지 않는다" — 이 파이프라인에 반영되어 있는지 본다.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import equity  # noqa: E402

from agentcore.domains.finance import ClaimKind, split_by_verifiability  # noqa: E402
from agentcore.trace import MemorySink, Tracer  # noqa: E402

REQUEST = {"ticker": "EXMPL", "as_of": date(2024, 7, 1)}


def run():
    sink = MemorySink()
    return equity.analyze(REQUEST, Tracer(sink=sink)), sink


def test_pipeline_is_clean_and_deterministic():
    a, _ = run()
    b, _ = run()
    assert not a["blocked"] and not b["blocked"]
    assert [c.text for c in a["claims"]] == [c.text for c in b["claims"]]


def test_future_and_restated_data_are_withheld():
    """기준일 이후 공개된 값과 이후 정정된 값은 둘 다 보이면 안 된다."""
    out, _ = run()
    withheld = set(out["view"].withheld())
    assert "revenue_q2_2024" in withheld
    assert "net_income_2023_v2" in withheld


def test_no_lookahead_leaks():
    out, _ = run()
    assert out["integrity"].clean
    assert out["integrity"].leak_count == 0


def test_every_reported_metric_recomputes():
    """이 도메인에서 가장 싸고 확실한 검증 — LLM Judge 가 필요 없다."""
    out, _ = run()
    assert out["recomputed"].clean
    assert out["recomputed"].summary()["checked"] == 3


def test_verifiable_and_judgment_claims_are_separated():
    out, _ = run()
    verifiable, judgment = split_by_verifiability(out["claims"])
    assert all(c.kind.verifiable for c in verifiable)
    assert all(c.kind is ClaimKind.PROJECTED for c in judgment)
    assert len(judgment) == 1


def test_no_advisory_claims_are_produced():
    """권유형 출력은 이 도메인에 평가할 골드셋이 없으므로 만들지 않는다."""
    out, _ = run()
    assert not any(c.kind is ClaimKind.ADVISORY for c in out["claims"])


def test_projections_are_always_labeled():
    out, _ = run()
    projections = [c for c in out["claims"] if c.kind is ClaimKind.PROJECTED]
    assert projections and all(c.labeled for c in projections)


def test_every_verifiable_claim_carries_support_and_as_of():
    out, _ = run()
    for c in out["claims"]:
        if c.kind.verifiable:
            assert c.support, f"{c.id} 근거 없음"
            assert c.as_of, f"{c.id} 기준일 없음"


def test_guardrails_pass_on_clean_analysis():
    out, _ = run()
    assert out["guardrails"].breach_count == 0


def test_lookahead_is_structurally_impossible():
    out, _ = run()
    with pytest.raises(LookupError, match="룩어헤드"):
        out["view"].require("revenue_q2_2024")


def test_missing_slot_asks_back():
    out = equity.analyze({"ticker": "EXMPL"}, Tracer(sink=MemorySink()))
    assert out["asked_back"] and out["claims"] == []


def test_earlier_as_of_withholds_more():
    """공시 지연이 반영되는가 — 4월에는 Q1 실적도 아직 모른다."""
    early = equity.analyze(
        {"ticker": "EXMPL", "as_of": date(2024, 4, 1)}, Tracer(sink=MemorySink())
    )
    assert "revenue_q1_2024" in early["view"].withheld()


def test_missing_input_halts_instead_of_estimating():
    """없는 데이터를 추정으로 메우면 그것이 곧 룩어헤드다.
    채우지 않고 무엇이 없는지 말하고 멈춘다."""
    sink = MemorySink()
    early = equity.analyze({"ticker": "EXMPL", "as_of": date(2024, 4, 1)}, Tracer(sink=sink))
    assert early["claims"] == []
    assert "revenue_q1_2024" in early["insufficient"]
    root = next(s for s in sink.spans if s.name == "equity_analysis")
    assert root.attributes["service_level"] == "halted"
