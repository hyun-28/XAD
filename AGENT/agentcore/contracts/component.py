"""컴포넌트 기반 클래스.

계약을 가진 모든 것은 Component 이다. 호출은 자동으로
  (1) 유효 범위를 검사하고
  (2) 궤적에 스팬을 남기고
  (3) 사용량을 기록한다.

이 셋이 자동이라는 게 중요하다. 수동이면 빠지고, 빠지면 평가할 수 없다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from ..trace import SpanKind, SpanStatus, Tracer, current_span
from .contract import ComponentContract, Role

I = TypeVar("I")
O = TypeVar("O")


class OutOfEnvelope(Exception):
    """유효 범위 밖 호출. 조용히 틀리는 대신 시끄럽게 거절한다."""

    def __init__(self, contract: ComponentContract, payload: Any) -> None:
        self.contract = contract
        self.payload = payload
        super().__init__(
            f"{contract.ref} 의 유효 범위를 벗어났습니다: {contract.envelope.description}"
        )


@dataclass(slots=True)
class Result(Generic[O]):
    """컴포넌트 실행 결과.

    value 가 None 이고 status 가 OK 가 아닐 수 있다 — 범위 이탈이나
    가드레일 차단은 예외가 아니라 값으로 표현되는 편이 상위에서 다루기 쉽다.
    """

    value: O | None
    status: SpanStatus = SpanStatus.OK
    component: str = ""
    version: str = ""
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status is SpanStatus.OK

    def unwrap(self) -> O:
        if not self.ok or self.value is None:
            raise RuntimeError(f"{self.component}@{self.version}: {self.status.value} {self.detail}")
        return self.value


class Component(ABC, Generic[I, O]):
    """계약을 가진 실행 단위."""

    def __init__(self, contract: ComponentContract) -> None:
        self.contract = contract

    @property
    def ref(self) -> str:
        return self.contract.ref

    @abstractmethod
    def run(self, payload: I) -> O:
        """실제 작업. 궤적·범위 검사는 __call__ 이 감싼다."""

    def __call__(self, payload: I, *, tracer: Tracer | None = None) -> Result[O]:
        c = self.contract
        tracer = tracer or Tracer()
        span_kind = SpanKind.VERIFY if c.role is Role.VERIFIER else SpanKind.COMPONENT

        with tracer.span(
            c.name,
            span_kind,
            component=c.name,
            component_version=c.version,
            role=c.role.value,
            ladder_rung=c.ladder_rung,
        ) as span:
            if not c.envelope.admits(payload):
                if c.envelope.on_violation == "reject":
                    span.status = SpanStatus.OUT_OF_ENVELOPE
                    span.attributes["envelope"] = c.envelope.description
                    return Result(
                        value=None,
                        status=SpanStatus.OUT_OF_ENVELOPE,
                        component=c.name,
                        version=c.version,
                        detail=c.envelope.description,
                    )
                span.attributes["envelope_warning"] = c.envelope.description

            value = self.run(payload)
            return Result(value=value, component=c.name, version=c.version)


class FunctionComponent(Component[I, O]):
    """함수를 컴포넌트로 감싸는 어댑터.

    기성 도구(T1)를 계약과 함께 등록할 때 쓴다. T1 은 버려지지 않고
    T2 의 비교 기준선이자 폴백으로 남는다.
    """

    def __init__(self, contract: ComponentContract, fn: Any) -> None:
        super().__init__(contract)
        self._fn = fn

    def run(self, payload: I) -> O:
        return self._fn(payload)
