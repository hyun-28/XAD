"""지표 재계산 검증 — 데이터 계층의 결정론적 골드셋.

주식 도메인은 검증 가능성에서 양극단으로 갈린다. 판단 계층은 골드셋이
원리적으로 만들어지지 않지만, **데이터 계층은 완전히 결정론적**이다.

지표는 입력에서 재계산된다. 그러면 "이 숫자가 맞나"는 취향 판정이 아니라
산술 대조가 된다 — LLM Judge 가 필요 없는 자리이고, 이 도메인에서 가장
싸고 가장 확실한 검증이다.

그리고 재계산은 **입력을 명시하도록 강제한다.** 어떤 값에서 나왔는지 못
적는 지표는 재계산할 수 없고, 재계산할 수 없으면 검증할 수 없다.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


class UndefinedMetric(ValueError):
    """정의되지 않는 지표. 0으로 나누기 등 — 조용히 0이나 inf 를 내지 않는다."""


def _div(numerator: float, denominator: float, what: str) -> float:
    if denominator == 0:
        raise UndefinedMetric(
            f"{what}: 분모가 0입니다. 0이나 무한대를 반환하면 그 값이 "
            f"하류로 흘러가 조용히 잘못된 결론을 만듭니다."
        )
    return numerator / denominator


# --------------------------------------------------------------------------
# 지표 정의 — 전부 순수 함수. 입력이 명시되고 재계산이 가능하다.
# --------------------------------------------------------------------------

def operating_margin(operating_income: float, revenue: float) -> float:
    return _div(operating_income, revenue, "영업이익률")


def net_margin(net_income: float, revenue: float) -> float:
    return _div(net_income, revenue, "순이익률")


def roe(net_income: float, equity: float) -> float:
    return _div(net_income, equity, "ROE")


def roa(net_income: float, assets: float) -> float:
    return _div(net_income, assets, "ROA")


def debt_to_equity(total_debt: float, equity: float) -> float:
    return _div(total_debt, equity, "부채비율")


def current_ratio(current_assets: float, current_liabilities: float) -> float:
    return _div(current_assets, current_liabilities, "유동비율")


def per(price: float, eps: float) -> float:
    return _div(price, eps, "PER")


def pbr(price: float, bps: float) -> float:
    return _div(price, bps, "PBR")


def yoy_growth(current: float, prior: float) -> float:
    if prior == 0:
        raise UndefinedMetric("전년 대비 증감률: 기준 기간 값이 0입니다")
    if prior < 0:
        raise UndefinedMetric(
            "전년 대비 증감률: 기준 기간이 음수입니다. 적자에서의 증감률은 "
            "부호가 뒤집혀 오해를 만듭니다 — 절대값 변화로 보고하십시오."
        )
    return (current - prior) / prior


def cagr(begin: float, end: float, years: float) -> float:
    if begin <= 0 or years <= 0:
        raise UndefinedMetric("CAGR: 시작값과 기간은 양수여야 합니다")
    return (end / begin) ** (1.0 / years) - 1.0


REGISTRY: dict[str, Callable[..., float]] = {
    "operating_margin": operating_margin,
    "net_margin": net_margin,
    "roe": roe,
    "roa": roa,
    "debt_to_equity": debt_to_equity,
    "current_ratio": current_ratio,
    "per": per,
    "pbr": pbr,
    "yoy_growth": yoy_growth,
    "cagr": cagr,
}


# --------------------------------------------------------------------------
# 재계산 검증
# --------------------------------------------------------------------------

@dataclass(slots=True)
class MetricClaim:
    """지표 주장. inputs 가 없으면 검증 자체가 불가능하다."""

    id: str
    metric: str
    inputs: dict[str, float]
    reported: float
    as_of: str = ""
    source: str = ""


@dataclass(slots=True)
class RecomputeResult:
    claim_id: str
    metric: str
    reported: float
    recomputed: float | None
    ok: bool
    detail: str = ""

    @property
    def error(self) -> float | None:
        if self.recomputed is None:
            return None
        return abs(self.reported - self.recomputed)


def recompute(
    claim: MetricClaim,
    *,
    rel_tol: float = 1e-6,
    registry: Mapping[str, Callable[..., float]] | None = None,
) -> RecomputeResult:
    """지표를 독립 재계산해 보고값과 대조한다."""
    reg = registry or REGISTRY
    fn = reg.get(claim.metric)
    if fn is None:
        return RecomputeResult(
            claim.id,
            claim.metric,
            claim.reported,
            None,
            ok=False,
            detail=f"알 수 없는 지표: {claim.metric}. 재계산할 수 없으면 검증할 수 없습니다",
        )
    try:
        value = fn(**claim.inputs)
    except TypeError as exc:
        return RecomputeResult(
            claim.id, claim.metric, claim.reported, None, ok=False,
            detail=f"입력이 정의와 맞지 않습니다: {exc}",
        )
    except UndefinedMetric as exc:
        return RecomputeResult(
            claim.id, claim.metric, claim.reported, None, ok=False, detail=str(exc)
        )

    ok = math.isclose(value, claim.reported, rel_tol=rel_tol, abs_tol=rel_tol)
    return RecomputeResult(
        claim.id,
        claim.metric,
        claim.reported,
        value,
        ok=ok,
        detail="" if ok else f"보고 {claim.reported:.6g} ≠ 재계산 {value:.6g}",
    )


@dataclass
class RecomputeReport:
    """재계산 결과.

    가드레일이므로 정확도 비율이 아니라 **불일치 개수**로 읽는다.
    지표 하나가 틀리면 그것을 근거로 한 모든 서술이 무효다.
    """

    results: list[RecomputeResult] = field(default_factory=list)

    def add(self, r: RecomputeResult) -> None:
        self.results.append(r)

    @property
    def mismatches(self) -> list[RecomputeResult]:
        return [r for r in self.results if not r.ok]

    @property
    def clean(self) -> bool:
        return not self.mismatches

    def summary(self) -> dict[str, Any]:
        return {
            "checked": len(self.results),
            "mismatches": len(self.mismatches),
            "by_metric": {
                m: sum(1 for r in self.mismatches if r.metric == m)
                for m in sorted({r.metric for r in self.mismatches})
            },
            "note": "재계산 불일치는 꼬리 지표다. 개수로 읽고 집계하지 않는다",
        }


def verify_all(
    claims: Sequence[MetricClaim], *, rel_tol: float = 1e-6
) -> RecomputeReport:
    report = RecomputeReport()
    for c in claims:
        report.add(recompute(c, rel_tol=rel_tol))
    return report
