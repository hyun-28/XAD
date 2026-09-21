"""도메인 골드셋 검증 — 연구·학습과 주식분석.

두 도메인은 검증 가능성에서 정반대로 갈린다. 그 갈림이 코드에 반영되는지 본다.
"""

from __future__ import annotations

from datetime import date

import pytest

from agentcore.domains.finance import (
    AsOfView,
    ClaimKind,
    DataPoint,
    FinancialClaim,
    LeakKind,
    MetricClaim,
    UndefinedMetric,
    audit_claims,
    check_integrity,
    classify_text,
    finance_guardrails,
    recompute,
    split_by_verifiability,
    verify_all,
)
from agentcore.domains.research import SourceGold, SourceGoldset, score_retrieval
from agentcore.trace import MemorySink, Tracer

AS_OF = date(2024, 7, 1)


# ==========================================================================
# 연구·학습 — 출처 재현율
# ==========================================================================

def test_goldset_rejects_questions_without_required_sources():
    """필수 출처를 못 적는 질문은 정답을 모른다는 뜻이다."""
    with pytest.raises(ValueError, match="required"):
        SourceGold("q", "무엇이 정답인지 모르는 질문", required=frozenset())


def test_required_and_forbidden_cannot_overlap():
    with pytest.raises(ValueError, match="겹칩니다"):
        SourceGold("q", "q", required=frozenset({"a"}), forbidden=frozenset({"a"}))


def test_recall_measures_required_sources_only():
    g = SourceGold("q", "q", required=frozenset({"a", "b"}), helpful=frozenset({"c"}))
    s = score_retrieval(g, ["a", "c"])
    assert s.recall == 0.5
    assert s.missed == ("b",)


def test_helpful_sources_do_not_hurt_precision():
    """넓게 찾는 것이 벌점이 되면 시스템은 좁게만 찾는다."""
    g = SourceGold("q", "q", required=frozenset({"a"}), helpful=frozenset({"c", "d"}))
    assert score_retrieval(g, ["a", "c", "d"]).precision == 1.0


def test_forbidden_source_is_a_count_not_a_rate():
    g = SourceGold("q", "q", required=frozenset({"a"}), forbidden=frozenset({"bad"}))
    s = score_retrieval(g, ["a", "bad"])
    assert s.recall == 1.0          # 재현율은 완벽해도
    assert not s.clean              # 실패다
    assert s.forbidden_hits == ("bad",)


def test_goldset_report_keeps_forbidden_separate():
    gs = SourceGoldset("t")
    gs.add(SourceGold("q1", "q", required=frozenset({"a"}), forbidden=frozenset({"x"})))
    rep = gs.evaluate({"q1": ["a", "x"]})
    assert rep["recall"] == 1.0
    assert rep["forbidden"]["total_hits"] == 1
    assert "recall" not in rep["forbidden"]


def test_goldset_roundtrip(tmp_path):
    gs = SourceGoldset("t")
    gs.add(SourceGold("q1", "질문", required=frozenset({"a"}), curator="me"))
    loaded = SourceGoldset.load(gs.write(tmp_path / "g.jsonl"))
    assert loaded.items[0].required == frozenset({"a"})
    assert loaded.items[0].curator == "me"


def test_worst_missed_points_at_actionable_gaps():
    gs = SourceGoldset("t")
    gs.add(SourceGold("q1", "q", required=frozenset({"a", "b", "c"})))
    rep = gs.evaluate({"q1": []})
    assert rep["worst_missed"][0][0] == "q1"
    assert set(rep["worst_missed"][0][1]) == {"a", "b", "c"}


# ==========================================================================
# 주식 — 시점 무결성
# ==========================================================================

def test_unpublished_data_is_a_leak():
    r = check_integrity([DataPoint("q2", 1, date(2024, 6, 30), date(2024, 8, 14))], AS_OF)
    assert not r.clean
    assert r.leaks[0].kind is LeakKind.UNPUBLISHED


def test_publication_lag_matters_not_period_end():
    """2024 Q1 은 3월에 확정되지만 5월에야 공개된다. 4월에 쓰면 룩어헤드다."""
    early = check_integrity(
        [DataPoint("q1", 1, date(2024, 3, 31), date(2024, 5, 15))], date(2024, 4, 15)
    )
    late = check_integrity(
        [DataPoint("q1", 1, date(2024, 3, 31), date(2024, 5, 15))], date(2024, 6, 1)
    )
    assert not early.clean and late.clean


def test_restatement_after_as_of_is_a_leak():
    r = check_integrity(
        [DataPoint("rev", 98, date(2024, 3, 31), date(2024, 5, 15), restated_at=date(2024, 11, 1))],
        AS_OF,
    )
    assert r.leaks[0].kind is LeakKind.RESTATED


def test_missing_timestamp_is_a_leak_not_a_pass():
    r = check_integrity([DataPoint("guess", 50)], AS_OF)
    assert r.leaks[0].kind is LeakKind.UNKNOWN_TIMESTAMP


def test_integrity_report_has_no_score():
    """한 건이라도 새면 무효다. '95% 깨끗함'은 의미 있는 상태가 아니다."""
    r = check_integrity([], AS_OF)
    assert not hasattr(r, "score")
    assert not hasattr(r, "rate")


def test_asof_view_makes_lookahead_structurally_impossible():
    v = AsOfView(as_of=AS_OF)
    v.ingest(
        [
            DataPoint("rev_q1", 100, date(2024, 3, 31), date(2024, 5, 15)),
            DataPoint("rev_q2", 120, date(2024, 6, 30), date(2024, 8, 14)),
        ]
    )
    assert v.available() == ["rev_q1"]
    assert v.withheld() == ["rev_q2"]
    assert v.get("rev_q2") is None
    with pytest.raises(LookupError, match="룩어헤드"):
        v.require("rev_q2")


def test_asof_view_returns_latest_known_revision():
    v = AsOfView(as_of=date(2024, 9, 1))
    v.ingest(
        [
            DataPoint("rev", 100, date(2024, 3, 31), date(2024, 5, 15)),
            DataPoint("rev", 105, date(2024, 3, 31), date(2024, 8, 1)),
        ]
    )
    assert v.require("rev").value == 105


# ==========================================================================
# 주식 — 지표 재계산
# ==========================================================================

def test_correct_metric_verifies():
    c = MetricClaim("m", "operating_margin", {"operating_income": 124, "revenue": 1000}, 0.124)
    assert recompute(c).ok


def test_wrong_metric_is_caught_with_both_values():
    c = MetricClaim("m", "roe", {"net_income": 80, "equity": 500}, 0.20)
    r = recompute(c)
    assert not r.ok and r.recomputed == pytest.approx(0.16)
    assert "0.16" in r.detail


def test_division_by_zero_is_loud_not_infinite():
    """0이나 inf 를 반환하면 그 값이 하류로 흘러가 조용히 결론을 망친다."""
    r = recompute(MetricClaim("m", "per", {"price": 5000, "eps": 0}, 15.0))
    assert not r.ok and "분모가 0" in r.detail


def test_growth_from_negative_base_is_refused():
    """적자에서의 증감률은 부호가 뒤집혀 오해를 만든다."""
    with pytest.raises(UndefinedMetric, match="음수"):
        from agentcore.domains.finance.metrics import yoy_growth

        yoy_growth(100, -50)


def test_unknown_metric_cannot_be_verified():
    r = recompute(MetricClaim("m", "made_up_ratio", {"a": 1}, 1.0))
    assert not r.ok and "재계산할 수 없으면" in r.detail


def test_report_counts_mismatches_not_accuracy():
    rep = verify_all(
        [
            MetricClaim("a", "roe", {"net_income": 80, "equity": 500}, 0.16),
            MetricClaim("b", "roe", {"net_income": 80, "equity": 500}, 0.99),
        ]
    )
    assert rep.summary()["mismatches"] == 1
    assert "accuracy" not in rep.summary()


# ==========================================================================
# 주식 — 주장 분류
# ==========================================================================

def test_verifiable_kinds_are_exactly_three():
    assert ClaimKind.MEASURED.verifiable
    assert ClaimKind.DERIVED.verifiable
    assert ClaimKind.SOURCED.verifiable
    assert not ClaimKind.PROJECTED.verifiable
    assert not ClaimKind.ADVISORY.verifiable


def test_unlabeled_projection_is_flagged():
    issues = audit_claims([FinancialClaim("c", "늘어날 것", ClaimKind.PROJECTED)])
    assert any(i.rule == "unlabeled_projection" for i in issues)


def test_labeled_projection_passes():
    issues = audit_claims(
        [FinancialClaim("c", "늘어날 것", ClaimKind.PROJECTED, labeled=True)]
    )
    assert not any(i.rule == "unlabeled_projection" for i in issues)


def test_advisory_is_blocked_by_default():
    issues = audit_claims([FinancialClaim("c", "매수 추천", ClaimKind.ADVISORY)])
    assert any(i.rule == "advisory_output" for i in issues)
    assert not audit_claims(
        [FinancialClaim("c", "매수 추천", ClaimKind.ADVISORY)], allow_advisory=True
    )


def test_projection_dressed_as_fact_is_caught():
    """검증된 숫자의 신뢰가 검증되지 않은 전망으로 번지는 것을 막는다."""
    issues = audit_claims(
        [FinancialClaim("c", "실적 개선이 예상된다", ClaimKind.MEASURED, ("x",), as_of="2024")]
    )
    assert any(i.rule == "misclassified_projection" for i in issues)


def test_advisory_dressed_as_sourced_is_caught():
    issues = audit_claims(
        [FinancialClaim("c", "저평가, 매수 의견", ClaimKind.SOURCED, ("r",), as_of="2024")]
    )
    assert any(i.rule == "misclassified_advisory" for i in issues)


def test_classifier_is_a_filter_not_a_judge():
    """규칙에만 의존하면 표현을 바꾼 권유가 통과한다 — 최종 분류는 작성자 몫."""
    assert classify_text("매수 추천") is ClaimKind.ADVISORY
    assert classify_text("전망이 밝다") is ClaimKind.PROJECTED
    assert classify_text("매출 1200억") is None


def test_split_separates_verifiable_from_not():
    v, rest = split_by_verifiability(
        [
            FinancialClaim("a", "매출 1200억", ClaimKind.MEASURED, ("x",), as_of="2024"),
            FinancialClaim("b", "늘 것", ClaimKind.PROJECTED, labeled=True),
        ]
    )
    assert [c.id for c in v] == ["a"]
    assert [c.id for c in rest] == ["b"]


# ==========================================================================
# 프레임워크 연결
# ==========================================================================

def test_domain_rails_plug_into_framework_guardrails():
    gs = finance_guardrails(as_of=AS_OF)
    rep = gs.evaluate(
        [
            DataPoint("q2", 1, date(2024, 6, 30), date(2024, 8, 14)),
            MetricClaim("m", "roe", {"net_income": 80, "equity": 500}, 0.20),
            FinancialClaim("c", "늘 것", ClaimKind.PROJECTED),
            FinancialClaim("d", "매수", ClaimKind.ADVISORY),
        ],
        tracer=Tracer(sink=MemorySink()),
    )
    assert rep.blocked
    rails = rep.counts_by_rail()
    assert rails["look_ahead"] == 1
    assert rails["metric_mismatch"] == 1
    assert rails["unlabeled_projection"] == 1
    assert rails["advisory_output"] == 1
    assert not hasattr(rep, "score")


def test_clean_analysis_passes_all_rails():
    gs = finance_guardrails(as_of=AS_OF)
    rep = gs.evaluate(
        [
            DataPoint("q1", 1, date(2024, 3, 31), date(2024, 5, 15)),
            MetricClaim("m", "roe", {"net_income": 80, "equity": 500}, 0.16),
            FinancialClaim("c", "매출 1200억", ClaimKind.MEASURED, ("10-Q",), as_of="2024-05-15"),
            FinancialClaim("d", "증가할 전망", ClaimKind.PROJECTED, labeled=True),
        ]
    )
    assert not rep.blocked
