"""도메인 검사를 프레임워크 가드레일로 연결한다.

전부 **개수**로 읽히는 통과/실패 관문이다. 비율이 아니다 —
룩어헤드 한 건, 재계산 불일치 한 건이면 그것을 근거로 한 서술 전체가 무효다.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from ...verify import Guardrail, GuardrailSet
from .claims import ClaimKind, FinancialClaim, classify_text
from .metrics import MetricClaim, recompute
from .pointintime import DataPoint


def _is_claim(c: Any) -> bool:
    return isinstance(c, FinancialClaim)


def point_in_time_rail(as_of: date) -> Guardrail:
    """as_of 이후 데이터가 한 건이라도 새면 실패."""

    def check(item: Any, _ctx: Any) -> bool:
        if not isinstance(item, DataPoint):
            return True
        return item.known_at(as_of)

    return Guardrail(
        name="look_ahead",
        check=check,
        message=f"{as_of} 시점에 알 수 없었던 데이터가 분석에 포함되었습니다",
    )


def recompute_rail(rel_tol: float = 1e-6) -> Guardrail:
    """보고된 지표가 입력에서 재계산되지 않으면 실패."""

    def check(item: Any, _ctx: Any) -> bool:
        if not isinstance(item, MetricClaim):
            return True
        return recompute(item, rel_tol=rel_tol).ok

    return Guardrail(
        name="metric_mismatch",
        check=check,
        message="보고된 지표가 명시된 입력에서 재계산되지 않습니다",
    )


UNSUPPORTED_FACT = Guardrail(
    name="unsupported_fact",
    check=lambda c, _: (not _is_claim(c)) or (not c.kind.verifiable) or bool(c.support),
    message="근거 없이 사실을 주장했습니다",
)

UNLABELED_PROJECTION = Guardrail(
    name="unlabeled_projection",
    check=lambda c, _: (not _is_claim(c)) or c.kind is not ClaimKind.PROJECTED or c.labeled,
    message="전망이 사실과 구분 없이 제시되었습니다",
)

ADVISORY_OUTPUT = Guardrail(
    name="advisory_output",
    check=lambda c, _: (not _is_claim(c)) or c.kind is not ClaimKind.ADVISORY,
    message="권유형 주장입니다 — 이 도메인에는 이를 평가할 골드셋이 없습니다",
)

MISCLASSIFIED = Guardrail(
    name="misclassified_claim",
    check=lambda c, _: (
        (not _is_claim(c))
        or classify_text(c.text) in (None, c.kind)
        or not (
            classify_text(c.text) is ClaimKind.ADVISORY
            or (classify_text(c.text) is ClaimKind.PROJECTED and c.kind.verifiable)
        )
    ),
    message="표시된 종류와 문장 표현이 어긋납니다",
)

MISSING_AS_OF = Guardrail(
    name="missing_as_of",
    check=lambda c, _: (not _is_claim(c)) or (not c.kind.verifiable) or bool(c.as_of),
    message="검증 가능한 주장에 기준일이 없습니다",
)


def finance_guardrails(as_of: date, *, rel_tol: float = 1e-6) -> GuardrailSet:
    """금융 분석 출력의 가드레일 묶음."""
    return GuardrailSet(
        rails=(
            point_in_time_rail(as_of),
            recompute_rail(rel_tol),
            UNSUPPORTED_FACT,
            UNLABELED_PROJECTION,
            ADVISORY_OUTPUT,
            MISCLASSIFIED,
            MISSING_AS_OF,
        ),
        name="finance_guardrails",
    )
