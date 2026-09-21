"""주식분석 파이프라인 예제 — 시점 무결성이 구조로 강제되는 형태.

이 도메인의 설계 전제는 하나다.

    데이터 계층은 완전히 검증 가능하고, 판단 계층은 골드셋이 원리적으로
    만들어지지 않는다.

그래서 시스템은 **검증 가능한 것만 검증된 것으로 내보내고**, 전망은 라벨을
달아 분리하며, 권유형 출력은 차단한다. 성능 문제가 아니라 이 도메인에서
평가할 수 없는 출력을 검증된 것처럼 제시하지 않기 위한 설계다.

    [1] AsOfView 로 데이터 수용    — 룩어헤드가 구조적으로 불가능
    [2] 시점 무결성 검사           — 남은 누수를 개수로 확인
    [3] 지표 재계산                — LLM Judge 없이 산술로 검증
    [4] 주장 생성 + 종류 표시
    [5] 가드레일                   — 한 건이라도 걸리면 차단
    [6] 검증 가능 / 불가능 분리 출력
"""

from __future__ import annotations

from datetime import date

from agentcore.control import SlotSpec, SufficiencyGate
from agentcore.domains.finance import (
    AsOfView,
    ClaimKind,
    DataPoint,
    FinancialClaim,
    MetricClaim,
    check_integrity,
    finance_guardrails,
    split_by_verifiability,
    verify_all,
)
from agentcore.evaluation import summarize_trajectory
from agentcore.trace import JsonlSink, MemorySink, MultiSink, SpanKind, Tracer, record_usage

AS_OF = date(2024, 7, 1)

# 원자료. period_end 와 published_at 이 분리되어 있다 — 이 분리가 없으면
# 룩어헤드를 잡을 수 없다.
FILINGS = [
    DataPoint("revenue_2023", 4_000.0, date(2023, 12, 31), date(2024, 2, 14), source="10-K"),
    DataPoint("operating_income_2023", 480.0, date(2023, 12, 31), date(2024, 2, 14), source="10-K"),
    DataPoint("net_income_2023", 360.0, date(2023, 12, 31), date(2024, 2, 14), source="10-K"),
    DataPoint("equity_2023", 3_000.0, date(2023, 12, 31), date(2024, 2, 14), source="10-K"),
    DataPoint("revenue_q1_2024", 1_100.0, date(2024, 3, 31), date(2024, 5, 15), source="10-Q"),
    DataPoint("operating_income_q1_2024", 143.0, date(2024, 3, 31), date(2024, 5, 15), source="10-Q"),
    # 기준일 이후 공개 — 알 수 없어야 한다
    DataPoint("revenue_q2_2024", 1_250.0, date(2024, 6, 30), date(2024, 8, 14), source="10-Q"),
    # 기준일 이후 정정 — 그 시점에는 정정 전 값만 알 수 있었다
    DataPoint("net_income_2023_v2", 344.0, date(2023, 12, 31), date(2024, 2, 14),
              restated_at=date(2024, 11, 1), source="10-K/A"),
]

SUFFICIENCY = SufficiencyGate([SlotSpec("ticker"), SlotSpec("as_of")])

# 이 분석에 필요한 입력. 기준일에 하나라도 없으면 분석하지 않는다.
REQUIRED_INPUTS = (
    "revenue_2023",
    "operating_income_2023",
    "net_income_2023",
    "equity_2023",
    "revenue_q1_2024",
    "operating_income_q1_2024",
)


def analyze(request: dict, tracer: Tracer) -> dict:
    with tracer.span("equity_analysis", SpanKind.CONTROL, ticker=request.get("ticker")) as root:
        if not SUFFICIENCY.check(request, tracer=tracer):
            root.attributes["asked_back"] = True
            return {"claims": [], "blocked": False, "asked_back": True, "insufficient": []}

        as_of: date = request["as_of"]

        # [1] 접근 경로를 하나로 만들어 룩어헤드를 구조적으로 차단
        with tracer.span("ingest", SpanKind.TOOL) as span:
            view = AsOfView(as_of=as_of)
            view.ingest(FILINGS)
            span.attributes.update(available=view.available(), withheld=view.withheld())

        # [1.5] 필요한 입력이 기준일에 공개되어 있는가.
        #
        # 이것은 오류가 아니라 **정상 상황**이다. 4월에는 Q1 실적이 아직 없다.
        # 없는 데이터를 추정으로 메우면 그것이 곧 룩어헤드이므로, 채우지 않고
        # 무엇이 없는지 말하고 멈춘다.
        missing = [k for k in REQUIRED_INPUTS if view.get(k) is None]
        if missing:
            root.attributes.update(insufficient_data=missing, service_level="halted")
            return {
                "claims": [],
                "blocked": False,
                "asked_back": False,
                "insufficient": missing,
                "view": view,
            }

        # [2] 남은 누수 확인 — 개수로만 읽는다
        with tracer.span("integrity", SpanKind.VERIFY) as span:
            known = [DataPoint(k, view.require(k).value, view.require(k).period_end,
                               view.require(k).published_at, source=view.require(k).source)
                     for k in view.available()]
            integrity = check_integrity(known, as_of)
            span.attributes.update(checked=integrity.checked, leaks=integrity.leak_count)

        # [3] 지표 재계산 — 여기가 이 도메인에서 가장 싸고 확실한 검증
        with tracer.span("metrics", SpanKind.VERIFY) as span:
            metrics = [
                MetricClaim(
                    "om_2023", "operating_margin",
                    {"operating_income": view.require("operating_income_2023").value,
                     "revenue": view.require("revenue_2023").value},
                    reported=0.12, as_of=str(as_of), source="10-K",
                ),
                MetricClaim(
                    "roe_2023", "roe",
                    {"net_income": view.require("net_income_2023").value,
                     "equity": view.require("equity_2023").value},
                    reported=0.12, as_of=str(as_of), source="10-K",
                ),
                MetricClaim(
                    "om_q1_2024", "operating_margin",
                    {"operating_income": view.require("operating_income_q1_2024").value,
                     "revenue": view.require("revenue_q1_2024").value},
                    reported=0.13, as_of=str(as_of), source="10-Q",
                ),
            ]
            recomputed = verify_all(metrics)
            span.attributes.update(recomputed.summary())

        # [4] 주장 생성. 종류를 반드시 표시한다.
        with tracer.span("compose", SpanKind.MODEL) as span:
            record_usage(input_tokens=len(view.available()) * 20, output_tokens=80)
            claims = [
                FinancialClaim(
                    "c1", f"2023 매출 {view.require('revenue_2023').value:,.0f}",
                    ClaimKind.MEASURED, ("10-K",), as_of=str(as_of),
                ),
                FinancialClaim(
                    "c2", "2023 영업이익률 12.0%",
                    ClaimKind.DERIVED, ("om_2023",), as_of=str(as_of),
                ),
                FinancialClaim(
                    "c3", "2024 Q1 영업이익률은 13.0%로 전년 연간 대비 개선",
                    ClaimKind.DERIVED, ("om_q1_2024", "om_2023"), as_of=str(as_of),
                ),
                # 전망 — 라벨을 달아야 통과한다
                FinancialClaim(
                    "c4", "마진 개선 추세가 이어질 것으로 보인다",
                    ClaimKind.PROJECTED, labeled=True,
                ),
            ]
            span.attributes["claims"] = len(claims)

        # [5] 가드레일 — 한 건이라도 걸리면 차단
        rails = finance_guardrails(as_of=as_of)
        report = rails.evaluate([*known, *metrics, *claims], tracer=tracer)
        root.attributes["guardrail_breaches"] = report.breach_count

        return {
            "claims": claims,
            "integrity": integrity,
            "recomputed": recomputed,
            "guardrails": report,
            "view": view,
            "blocked": report.blocked,
            "asked_back": False,
            "insufficient": [],
        }


def main() -> None:
    mem = MemorySink()
    tracer = Tracer(sink=MultiSink(mem, JsonlSink("runs/equity.jsonl")))
    out = analyze({"ticker": "EXMPL", "as_of": AS_OF}, tracer)

    if out.get("insufficient"):
        print(f"{AS_OF} 시점에 필요한 입력이 공개되지 않았습니다: {out['insufficient']}")
        return

    view, integrity, recomputed = out["view"], out["integrity"], out["recomputed"]
    verifiable, judgment = split_by_verifiability(out["claims"])

    print("=" * 70)
    print(f"기준일 {AS_OF} · EXMPL")
    print("-" * 70)
    print(f"사용 가능 데이터   {len(view.available())}건")
    print(f"보류(미공개·정정)  {view.withheld()}")
    print(f"시점 누수          {integrity.leak_count}건  {'✓ 깨끗' if integrity.clean else '✗'}")
    print(f"지표 재계산        {recomputed.summary()['checked']}건 중 불일치 "
          f"{recomputed.summary()['mismatches']}건")
    for r in recomputed.mismatches:
        print(f"                   ✗ {r.claim_id}: {r.detail}")
    print("-" * 70)
    print("[검증된 사실]")
    for c in verifiable:
        print(f"  · {c.text}   ← {', '.join(c.support)}")
    print()
    print("[검증 불가 — 전망]  ※ 이 도메인에는 이를 평가할 골드셋이 없습니다")
    for c in judgment:
        print(f"  · {c.text}")
    print("-" * 70)
    g = out["guardrails"]
    print(f"가드레일 위반      {g.breach_count}건 {g.counts_by_rail() or ''}")
    traj = summarize_trajectory(mem.spans)
    # 이 파이프라인은 워커 격리를 쓰지 않는 직선 체인이므로 격리 효과가 아니라
    # 제어 비율이 의미 있는 지표다 — 검증이 전부 코드에 있는지를 본다.
    print(f"제어 비율          {traj.control_ratio:.2f}       (모델 호출 {traj.model_calls}회)")
    print("=" * 70)

    # 룩어헤드가 구조적으로 막히는지 확인
    print("\n[룩어헤드 차단 확인]")
    try:
        view.require("revenue_q2_2024")
    except LookupError as exc:
        print(f"  ✓ {exc}")


def demo_earlier_as_of() -> None:
    """기준일을 앞당기면 무엇이 달라지는가.

    4월에는 Q1 실적이 아직 공개되지 않았다. 시스템은 추정으로 메우지 않고
    무엇이 없는지 말하고 멈춘다.
    """
    out = analyze({"ticker": "EXMPL", "as_of": date(2024, 4, 1)}, Tracer(sink=MemorySink()))
    print(f"\n[기준일 2024-04-01]  분석 불가 — 미공개 입력: {out['insufficient']}")


if __name__ == "__main__":
    main()
    demo_earlier_as_of()
