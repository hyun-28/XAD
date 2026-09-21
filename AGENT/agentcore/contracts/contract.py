"""컴포넌트 계약.

1부 원칙: **동결이란 체크포인트를 저장하는 것이 아니라 계약을 발행하는 것이다.**

계약 없이 발행된 컴포넌트는 다음 시스템이 신뢰할 근거가 없고, 교체 판단의
기준도 없다. 계약은 세 부분으로 이루어진다.

  1. 인터페이스  — 입출력 스키마, 실패 모드, 지연 특성
  2. 성능 명세    — 어떤 평가셋에서 얼마
  3. 유효 범위    — 어떤 입력 분포에서 유효하며 밖에서 어떻게 실패하는가

세 번째가 가장 많이 빠지고 가장 비싸다. 동결된 도구를 학습 분포 밖에서 쓰면
조용히 틀리기 때문에, 여기서는 범위 이탈을 **명시적 결과**로 만든다.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any


class Role(str, Enum):
    """컴포넌트의 역할.

    코어는 크기가 아니라 **역할**이다. 단일 프론티어 모델이 채우든 앙상블이
    채우든 인터페이스는 같다 — 그래서 메모리 제약 환경에서 코어 박스만
    앙상블로 갈아끼울 수 있다. (§09 "코어의 위치" 해소)
    """

    CORE = "core"                # 계획, 모순 해소, 최종 합성. 동결. 파인튜닝 금지.
    SPECIALIST = "specialist"    # 좁은 판정 하나. T2 적응 대상.
    VERIFIER = "verifier"        # 신뢰의 뿌리. 적응하지 않는다.
    RETRIEVER = "retriever"      # 검색·메모리.
    ROUTER = "router"            # 라우팅·브로커.
    SKILL = "skill"              # 재사용 가능한 능력. 유일한 복리 계층.


class Amortization(str, Enum):
    """코어 교체 시 살아남는가. 투자 우선순위를 결정한다."""

    DURABLE = "durable"      # 생존, 오히려 강화됨 (검색·스킬·궤적)
    PARTIAL = "partial"      # 부분 상각 (파인튜닝된 소형 모델)
    REPLACED = "replaced"    # 교체 대상 그 자체 (코어)


@dataclass(frozen=True, slots=True)
class PerformanceSpec:
    """성능 명세. 평가셋 없이 발행된 컴포넌트는 채택 판단이 불가능하다."""

    eval_set: str
    metrics: dict[str, float]
    measured_on: date | None = None
    notes: str = ""

    def meets(self, floor: dict[str, float]) -> bool:
        """회귀 판정. 교체 후보가 기준선을 넘는지 본다."""
        return all(self.metrics.get(k, float("-inf")) >= v for k, v in floor.items())


@dataclass(frozen=True, slots=True)
class OperatingEnvelope:
    """유효 범위.

    predicate 가 False 를 반환하면 컴포넌트는 실행되지 않고
    OutOfEnvelope 로 명시적 실패한다. 조용히 틀리는 것보다 시끄럽게 거절하는 게 낫다.
    """

    description: str
    predicate: Callable[[Any], bool] = field(default=lambda _: True)
    on_violation: str = "reject"    # "reject" | "warn"

    def admits(self, payload: Any) -> bool:
        try:
            return bool(self.predicate(payload))
        except Exception:
            return False


@dataclass(frozen=True, slots=True)
class InterfaceSpec:
    """인터페이스 계약."""

    input_schema: str
    output_schema: str
    failure_modes: tuple[str, ...] = ()
    p50_latency_ms: float | None = None
    p99_latency_ms: float | None = None


@dataclass(frozen=True, slots=True)
class ComponentContract:
    """발행된 계약. 불변이며, 변경은 새 버전을 의미한다.

    v1 과 v2 는 교체가 아니라 **병존**한다. v1 은 사용량이 0이 될 때 은퇴한다.
    (1부 Stage 4: 사이클은 원이 아니라 나선)
    """

    name: str
    version: str
    role: Role
    interface: InterfaceSpec
    performance: PerformanceSpec
    envelope: OperatingEnvelope
    amortization: Amortization = Amortization.DURABLE

    # 적응 사다리에서 이 컴포넌트가 앉은 단(1-7). 6 이상은 망각 위험 구간.
    ladder_rung: int = 2
    # 이 컴포넌트를 대체할 수 있는 범용 기성품(T1). 폴백이자 비교 기준선.
    fallback: str | None = None
    deprecated: bool = False

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"

    @property
    def is_parametric(self) -> bool:
        """파라메트릭 적응 구간인가. 망각 위험과 회귀 검증 부채가 여기서 발생한다."""
        return self.ladder_rung >= 6

    def __post_init__(self) -> None:
        if not 1 <= self.ladder_rung <= 7:
            raise ValueError(f"ladder_rung 은 1..7 이어야 합니다: {self.ladder_rung}")
        if self.ladder_rung == 7:
            raise ValueError(
                "7단(코어 파인튜닝)은 설계 원칙상 금지됩니다. "
                "코어는 동결하고 적응은 주변부로 미십시오."
            )
        if self.role is Role.VERIFIER and self.is_parametric:
            raise ValueError(
                "검증기는 신뢰의 뿌리이므로 파라메트릭 적응 대상이 될 수 없습니다. "
                "적응하는 시스템에는 적응하지 않는 기준점이 필요합니다."
            )
