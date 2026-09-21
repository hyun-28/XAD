"""가드레일.

1부 원칙: **가드레일은 집계하지 않는다.**

목표 달성도와 품질은 평균으로 재고 92%면 훌륭하다.
가드레일은 꼬리로 재고 92%면 재앙이다.

같은 프레임워크에 넣어 종합 점수로 집계하면 품질 개선이 가드레일 악화를
가린다. 그래서 GuardrailReport 는 **점수를 만들지 않는다** — 개수만 센다.
이는 의도적 설계 제약이며, score() 메서드는 존재하지 않는다.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..trace import SpanKind, SpanStatus, Tracer


@dataclass(frozen=True, slots=True)
class Guardrail:
    """통과/실패 게이트 하나.

    지표는 비율이 아니라 **개수**다. 허용 위반 수는 기본 0이며,
    0이 아닌 값을 두려면 명시적으로 이유를 적어야 한다.
    """

    name: str
    check: Callable[[Any, Any], bool]
    message: str
    tolerance: int = 0
    tolerance_reason: str = ""

    def __post_init__(self) -> None:
        if self.tolerance > 0 and not self.tolerance_reason:
            raise ValueError(
                f"가드레일 '{self.name}' 의 tolerance 가 0이 아니면 "
                f"tolerance_reason 을 적어야 합니다. 가드레일 완화는 기록되어야 하는 결정입니다."
            )


@dataclass(slots=True)
class GuardrailBreach:
    guardrail: str
    message: str
    locus: str = ""


@dataclass(slots=True)
class GuardrailReport:
    """가드레일 결과.

    의도적으로 점수가 없다. breaches 는 개수로만 읽히며, 하나라도 있으면
    blocked 다. 다른 지표와 함께 평균되지 않도록 별도 타입으로 유지한다.
    """

    breaches: tuple[GuardrailBreach, ...] = ()
    evaluated: int = 0
    _tolerated: int = 0

    @property
    def blocked(self) -> bool:
        return bool(self.breaches)

    @property
    def breach_count(self) -> int:
        return len(self.breaches)

    def counts_by_rail(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for b in self.breaches:
            out[b.guardrail] = out.get(b.guardrail, 0) + 1
        return out

    def __bool__(self) -> bool:
        """리포트 자체는 '통과했는가'로 읽힌다."""
        return not self.blocked


@dataclass
class GuardrailSet:
    """가드레일 묶음. 출력 경로의 마지막 관문."""

    rails: Sequence[Guardrail] = field(default_factory=tuple)
    name: str = "guardrails"

    def evaluate(
        self,
        candidates: Sequence[Any],
        context: Any = None,
        *,
        tracer: Tracer | None = None,
    ) -> GuardrailReport:
        tracer = tracer or Tracer()
        with tracer.span(self.name, SpanKind.VERIFY, kind_detail="guardrail") as span:
            breaches: list[GuardrailBreach] = []
            tolerated = 0
            for rail in self.rails:
                hits: list[GuardrailBreach] = []
                for i, cand in enumerate(candidates):
                    try:
                        ok = bool(rail.check(cand, context))
                    except Exception as exc:
                        ok = False
                        hits.append(
                            GuardrailBreach(rail.name, f"{rail.message} (검사 오류: {exc})", str(i))
                        )
                        continue
                    if not ok:
                        hits.append(GuardrailBreach(rail.name, rail.message, locus=str(i)))
                if len(hits) <= rail.tolerance:
                    tolerated += len(hits)
                else:
                    breaches.extend(hits)

            report = GuardrailReport(
                breaches=tuple(breaches),
                evaluated=len(candidates),
                _tolerated=tolerated,
            )
            if report.blocked:
                span.status = SpanStatus.GUARDRAIL_BLOCKED
            span.attributes.update(
                breach_count=report.breach_count,
                by_rail=report.counts_by_rail(),
                tolerated=tolerated,
                evaluated=report.evaluated,
            )
            return report
