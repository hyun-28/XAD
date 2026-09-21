"""게이트.

두 종류가 있고 성격이 다르다.

  SufficiencyGate — 진행 전에 필요한 정보가 모였는가.
                    없으면 에이전트는 추측으로 진행한다.
  PreflightGate   — 비싸거나 되돌릴 수 없는 작업 앞에 선다.
                    **사후 게이트는 보호 장치가 아니다.**
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from ..trace import SpanKind, Tracer

T = TypeVar("T")


@dataclass(slots=True)
class SlotSpec:
    name: str
    required: bool = True
    description: str = ""


@dataclass(slots=True)
class Sufficiency:
    satisfied: bool
    missing: tuple[str, ...]
    filled: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.satisfied


class SufficiencyGate:
    """명시적 충분성 게이트.

    도메인마다 슬롯의 내용은 다르지만 게이트의 존재는 일반적이다.
    이 게이트가 있어야 "모르면 묻는" 동작이 설계로 보장된다.
    """

    def __init__(self, slots: Sequence[SlotSpec], name: str = "sufficiency") -> None:
        self.slots = tuple(slots)
        self.name = name

    def check(self, payload: dict[str, Any], *, tracer: Tracer | None = None) -> Sufficiency:
        tracer = tracer or Tracer()
        with tracer.span(self.name, SpanKind.CONTROL, gate="sufficiency") as span:
            missing = tuple(
                s.name
                for s in self.slots
                if s.required and payload.get(s.name) in (None, "", [], {})
            )
            filled = {s.name: payload.get(s.name) for s in self.slots if s.name in payload}
            span.attributes["missing"] = list(missing)
            span.attributes["satisfied"] = not missing
            return Sufficiency(satisfied=not missing, missing=missing, filled=filled)


@dataclass(slots=True)
class Estimate:
    """부작용 전 비용 추정. 도메인마다 추정 수단은 다르나 위치는 항상 부작용 앞이다."""

    cost: float
    unit: str = "rows"
    detail: str = ""


class PreflightGate(Generic[T]):
    """실행 **전** 게이트.

    estimator 는 실제 부작용 없이 비용을 추정해야 한다. 추정이 불가능한
    작업이라면 상한을 강제하는 쪽(LIMIT, timeout)으로 설계를 바꿔야 하며,
    실행 후에 재는 것은 보호가 아니다.
    """

    def __init__(
        self,
        estimator: Callable[[T], Estimate],
        ceiling: float,
        name: str = "preflight",
    ) -> None:
        self.estimator = estimator
        self.ceiling = ceiling
        self.name = name

    def admits(self, payload: T, *, tracer: Tracer | None = None) -> tuple[bool, Estimate]:
        tracer = tracer or Tracer()
        with tracer.span(self.name, SpanKind.CONTROL, gate="preflight") as span:
            est = self.estimator(payload)
            ok = est.cost <= self.ceiling
            span.attributes.update(
                estimated=est.cost, unit=est.unit, ceiling=self.ceiling, admitted=ok
            )
            return ok, est
