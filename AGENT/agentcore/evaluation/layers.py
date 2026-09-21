"""측정 4층위.

세 축(목표 달성도·품질·가드레일)은 **출력층**만 본다. 그런데 시스템은
출력이 못 보는 곳에서 실패한다. 그래서 네 층위로 나눈다.

  COMPONENT — 각 컴포넌트의 고정 테스트셋. 없으면 T2 교체 판단 불가
  TRAJECTORY— 경로 정확도, 중복률, 격리 효과. 운 좋은 정답을 걸러낸다
  OUTPUT    — 목표 달성도·품질·가드레일
  SYSTEM    — 비용, 지연, VRAM, 재현성, 코어 교체 생존율

그리고 가드레일은 다른 지표와 **함께 집계되지 않는다**.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from statistics import fmean
from typing import Any

from ..trace import Span, SpanKind, SpanStatus
from ..verify import GuardrailReport


class Layer(str, Enum):
    COMPONENT = "component"
    TRAJECTORY = "trajectory"
    OUTPUT = "output"
    SYSTEM = "system"


@dataclass(slots=True)
class Metric:
    name: str
    value: float
    layer: Layer
    unit: str = ""
    higher_is_better: bool = True


@dataclass
class TrajectoryMetrics:
    """궤적 층위 — 출력만 보면 안 보이는 것들."""

    steps: int = 0
    model_calls: int = 0
    control_decisions: int = 0
    duplicate_rate: float = 0.0
    out_of_envelope: int = 0
    retries: int = 0
    absorbed_tokens: int = 0
    emitted_tokens: int = 0

    @property
    def isolation_ratio(self) -> float:
        """격리 효과 = 워커가 흡수한 토큰 / 코어로 올라간 토큰.

        P4(컨텍스트 오염)를 직접 재는 유일한 지표. 모놀리식 구조에서는
        흡수가 0이므로 항상 0이 되어, 아키텍처를 판별한다.
        """
        # 분모를 1로 바닥 처리한다. emitted 가 0이면 격리가 **완전**했다는 뜻인데
        # 0을 반환하면 격리가 전혀 없었던 것처럼 읽혀 지표가 반대로 뒤집힌다.
        return self.absorbed_tokens / max(self.emitted_tokens, 1)

    @property
    def control_ratio(self) -> float:
        """제어 결정 / (제어 + 모델 호출).

        "LLM은 판단하고 제어는 코드가 한다"가 지켜지는지 궤적에서 검증한다.
        이 값이 낮으면 제어가 모델 안으로 흘러들어간 것이다.
        """
        total = self.control_decisions + self.model_calls
        return self.control_decisions / total if total else 0.0


@dataclass
class SystemMetrics:
    """시스템 층위 — P1 과 상각 논거가 검증되는 자리."""

    total_tokens: int = 0
    wall_seconds: float = 0.0
    peak_vram_mb: float = 0.0
    reproducible: bool = True
    component_versions: dict[str, str] = field(default_factory=dict)

    def core_swap_survival(self, after: "SystemMetrics") -> float:
        """코어 교체 생존율.

        코어를 업그레이드했을 때 그대로 살아남는 컴포넌트 비율.
        상각 논거·P3·s3 의 model-agnostic 결과가 전부 여기로 수렴하는 최종 지표.
        """
        before = set(self.component_versions.items())
        if not before:
            return 0.0
        survived = before & set(after.component_versions.items())
        return len(survived) / len(before)


@dataclass
class LayeredResult:
    """한 사례의 4층위 결과.

    guardrails 는 metrics 와 **별도 필드**다. 같은 딕셔너리에 넣으면
    언젠가 함께 평균된다.
    """

    case_id: str
    metrics: dict[Layer, dict[str, float]] = field(default_factory=dict)
    trajectory: TrajectoryMetrics = field(default_factory=TrajectoryMetrics)
    system: SystemMetrics = field(default_factory=SystemMetrics)
    guardrails: GuardrailReport = field(default_factory=GuardrailReport)
    passed: bool = True
    detail: str = ""

    def add(self, layer: Layer, name: str, value: float) -> None:
        self.metrics.setdefault(layer, {})[name] = value

    @property
    def blocked(self) -> bool:
        return self.guardrails.blocked


def summarize_trajectory(spans: list[Span]) -> TrajectoryMetrics:
    """궤적에서 지표를 뽑는다. 로그가 곧 평가 데이터라는 게 여기서 실현된다."""
    m = TrajectoryMetrics(steps=len(spans))
    for s in spans:
        if s.kind is SpanKind.MODEL:
            m.model_calls += 1
        elif s.kind in (SpanKind.CONTROL, SpanKind.SELECTION):
            # 브로커 선택도 코드가 내린 제어 결정이다. 동적 모드에서도
            # 텔레메트리를 읽어 코드가 고르는 것이지 모델이 고르는 게 아니다.
            m.control_decisions += 1
        if s.status is SpanStatus.OUT_OF_ENVELOPE:
            m.out_of_envelope += 1
        if s.name.startswith("refine:round"):
            m.retries += 1
        m.absorbed_tokens += s.absorbed_tokens
        m.emitted_tokens += s.emitted_tokens
        dup = s.attributes.get("duplicate_rate")
        if isinstance(dup, (int, float)):
            m.duplicate_rate = max(m.duplicate_rate, float(dup))
    return m


def summarize_system(spans: list[Span]) -> SystemMetrics:
    roots = [s for s in spans if s.parent_id is None]
    versions = {
        s.component: s.component_version
        for s in spans
        if s.component and s.component_version
    }
    return SystemMetrics(
        total_tokens=sum(s.usage.total_tokens for s in roots),
        wall_seconds=sum(s.usage.wall_seconds for s in roots),
        peak_vram_mb=max((s.usage.peak_vram_mb or 0.0 for s in spans), default=0.0),
        component_versions=versions,
    )


@dataclass
class Aggregate:
    """집계.

    성공률은 **반드시 비용과 쌍으로** 읽힌다. 비용 없는 성공률은
    "얼마를 태워서 얻었는가"를 감춘다.
    """

    results: list[LayeredResult] = field(default_factory=list)

    def add(self, r: LayeredResult) -> None:
        self.results.append(r)

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def success_rate(self) -> float:
        return fmean([1.0 if r.passed else 0.0 for r in self.results]) if self.results else 0.0

    @property
    def tokens_per_task(self) -> float:
        return fmean([r.system.total_tokens for r in self.results]) if self.results else 0.0

    @property
    def seconds_per_task(self) -> float:
        return fmean([r.system.wall_seconds for r in self.results]) if self.results else 0.0

    @property
    def isolation_ratio(self) -> float:
        return fmean([r.trajectory.isolation_ratio for r in self.results]) if self.results else 0.0

    def guardrail_breaches(self) -> dict[str, int]:
        """가드레일은 개수로만 보고된다. 비율도, 평균도, 종합 점수도 없다."""
        out: dict[str, int] = {}
        for r in self.results:
            for rail, count in r.guardrails.counts_by_rail().items():
                out[rail] = out.get(rail, 0) + count
        return out

    def report(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "output": {
                "success_rate": round(self.success_rate, 4),
            },
            "system": {
                "tokens_per_task": round(self.tokens_per_task, 1),
                "seconds_per_task": round(self.seconds_per_task, 4),
                "success_per_1k_tokens": round(
                    self.success_rate / (self.tokens_per_task / 1000), 4
                )
                if self.tokens_per_task
                else None,
            },
            "trajectory": {
                "isolation_ratio": round(self.isolation_ratio, 4),
            },
            # 별도 블록. 위 지표들과 절대 합산되지 않는다.
            "guardrails": {
                "breaches": self.guardrail_breaches(),
                "blocked_cases": sum(1 for r in self.results if r.blocked),
                "note": "가드레일은 꼬리 지표이므로 집계·평균하지 않는다",
            },
        }
