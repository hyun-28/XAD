"""컴포넌트 층위 평가 — 교체 판단의 근거.

이 파일이 모듈화 논지의 증명이다.

> 조합형 시스템의 우위는 성능이 아니라 검증 가능성·독립 버전 관리·실패 격리다.

그 주장은 **각 조각을 따로 측정하고 교체할 수 있을 때만** 참이다. 컴포넌트별
테스트셋이 없으면 T2 교체 판단이 불가능하고, 교체할 수 없으면 모듈화는
그냥 늘어난 표면적이다.

그리고 T1(범용 기성품)은 버려지지 않는다. **비교 기준선이자 폴백**으로
남으며, 그 비교가 매 배포마다 T2 의 존재 근거를 재확인한다.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from statistics import fmean
from typing import Any

from ..contracts import Component, PerformanceSpec
from ..trace import MemorySink, SpanStatus, Tracer


@dataclass(slots=True)
class EvalItem:
    """컴포넌트 테스트 항목 하나."""

    id: str
    payload: Any
    expected: Any = None
    equivalent: Callable[[Any, Any], bool] | None = None
    out_of_envelope: bool = False   # 범위 밖 입력. 거절해야 통과.

    def matches(self, actual: Any) -> bool:
        if self.equivalent is not None:
            return self.equivalent(actual, self.expected)
        return actual == self.expected


@dataclass
class ComponentEvalSet:
    """컴포넌트에 묶인 고정 테스트셋.

    **평가셋이 코드보다 먼저 존재한다.** 골드셋 없는 컴포넌트는 만들지 않는다 —
    Stage 2 에서 성능 명세가 필수였던 것이 이것이다.
    """

    name: str
    component_name: str
    items: list[EvalItem] = field(default_factory=list)
    version: str = "1.0"

    def add(self, item: EvalItem) -> None:
        self.items.append(item)

    @property
    def envelope_coverage(self) -> float:
        """범위 밖 입력이 테스트셋에 얼마나 있는가.

        정상 경로만 테스트하면 유효 범위 계약이 검증되지 않고, 그러면
        동결된 도구가 분포 밖에서 조용히 틀리는 것을 잡지 못한다.
        """
        if not self.items:
            return 0.0
        return sum(1 for i in self.items if i.out_of_envelope) / len(self.items)


@dataclass(slots=True)
class ComponentScore:
    component: str
    version: str
    eval_set: str
    n: int
    accuracy: float
    envelope_precision: float     # 범위 밖을 제대로 거절했는가
    errors: int
    tokens: int
    wall_seconds: float

    @property
    def metrics(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "envelope_precision": self.envelope_precision,
        }

    def as_spec(self, notes: str = "") -> PerformanceSpec:
        """측정 결과를 계약의 성능 명세로 발행한다.

        Stage 1(학습) → Stage 2(동결=계약 발행)의 연결 고리.
        """
        return PerformanceSpec(eval_set=self.eval_set, metrics=self.metrics, notes=notes)


def score_component(
    component: Component, eval_set: ComponentEvalSet
) -> ComponentScore:
    """컴포넌트를 자기 테스트셋에 돌린다."""
    correct = 0
    envelope_correct = 0
    envelope_total = 0
    errors = 0
    tokens = 0
    seconds = 0.0

    for item in eval_set.items:
        sink = MemorySink()
        tracer = Tracer(sink=sink)
        result = component(item.payload, tracer=tracer)

        for s in sink.spans:
            tokens += s.usage.total_tokens
            seconds += s.usage.wall_seconds

        if item.out_of_envelope:
            envelope_total += 1
            if result.status is SpanStatus.OUT_OF_ENVELOPE:
                envelope_correct += 1
                correct += 1
            continue

        if result.status is SpanStatus.ERROR:
            errors += 1
            continue
        if result.ok and item.matches(result.value):
            correct += 1

    n = len(eval_set.items) or 1
    return ComponentScore(
        component=component.contract.name,
        version=component.contract.version,
        eval_set=eval_set.name,
        n=len(eval_set.items),
        accuracy=correct / n,
        envelope_precision=envelope_correct / envelope_total if envelope_total else 1.0,
        errors=errors,
        tokens=tokens,
        wall_seconds=seconds,
    )


# --------------------------------------------------------------------------
# 교체 판단
# --------------------------------------------------------------------------

class Verdict(str, Enum):
    PROMOTE = "promote"          # 후보가 낫다. 승격
    KEEP = "keep"                # 현행 유지
    ROLLBACK = "rollback"        # 후보가 회귀. 폴백으로 되돌림
    UNJUSTIFIED = "unjustified"  # T1 기준선을 못 이김 — T2 는 순손실


@dataclass(slots=True)
class SwapDecision:
    verdict: Verdict
    reason: str
    incumbent: ComponentScore | None
    candidate: ComponentScore
    baseline: ComponentScore | None = None
    deltas: dict[str, float] = field(default_factory=dict)

    @property
    def should_deploy(self) -> bool:
        return self.verdict is Verdict.PROMOTE


def decide_swap(
    candidate: ComponentScore,
    incumbent: ComponentScore | None = None,
    baseline: ComponentScore | None = None,
    *,
    floor: dict[str, float] | None = None,
    min_gain: float = 0.0,
    cost_ceiling_ratio: float = 1.5,
) -> SwapDecision:
    """T2 후보를 배포할지 판정한다.

    세 비교를 순서대로 한다.

      1. 절대 하한(floor) — 넘지 못하면 무조건 거절
      2. T1 기준선 대비   — 못 이기면 T2 는 순손실이다
      3. 현행 대비        — 개선이 min_gain 을 넘고 비용이 과하지 않은가

    2번이 핵심이다. 범용 기성품이 이미 충분하면 맞춤형은 관리 대상만 늘린다.
    """
    deltas: dict[str, float] = {}
    if incumbent is not None:
        deltas = {
            k: round(candidate.metrics[k] - incumbent.metrics.get(k, 0.0), 4)
            for k in candidate.metrics
        }

    if floor:
        below = {k: v for k, v in floor.items() if candidate.metrics.get(k, -1.0) < v}
        if below:
            return SwapDecision(
                Verdict.ROLLBACK,
                f"절대 하한 미달: {below}",
                incumbent,
                candidate,
                baseline,
                deltas,
            )

    if baseline is not None and candidate.accuracy <= baseline.accuracy:
        return SwapDecision(
            Verdict.UNJUSTIFIED,
            (
                f"T1 기준선({baseline.component}@{baseline.version}, "
                f"{baseline.accuracy:.3f})을 이기지 못했습니다. 맞춤형 컴포넌트는 "
                f"관리 대상과 망각 위험만 늘립니다 — 기성품을 쓰십시오."
            ),
            incumbent,
            candidate,
            baseline,
            deltas,
        )

    if incumbent is None:
        return SwapDecision(
            Verdict.PROMOTE, "현행 없음 — 신규 배포", None, candidate, baseline, deltas
        )

    gain = candidate.accuracy - incumbent.accuracy
    if gain < min_gain:
        return SwapDecision(
            Verdict.KEEP,
            f"개선 폭 {gain:+.4f} < 요구 {min_gain}. 현행 유지",
            incumbent,
            candidate,
            baseline,
            deltas,
        )

    if incumbent.tokens and candidate.tokens > incumbent.tokens * cost_ceiling_ratio:
        return SwapDecision(
            Verdict.KEEP,
            (
                f"정확도는 {gain:+.4f} 올랐으나 토큰이 "
                f"{candidate.tokens / incumbent.tokens:.2f}배 — 상한 {cost_ceiling_ratio}배 초과. "
                f"성공률은 반드시 비용과 쌍으로 읽습니다."
            ),
            incumbent,
            candidate,
            baseline,
            deltas,
        )

    return SwapDecision(
        Verdict.PROMOTE, f"개선 {gain:+.4f}, 비용 허용 범위", incumbent, candidate, baseline, deltas
    )


@dataclass
class ComponentBoard:
    """컴포넌트별 최신 점수판.

    `Registry.parametric_components()` 와 함께 보면 "망각 위험을 지면서
    실제로 값을 하는가"가 한눈에 나온다.
    """

    scores: dict[str, ComponentScore] = field(default_factory=dict)

    def record(self, score: ComponentScore) -> None:
        self.scores[f"{score.component}@{score.version}"] = score

    def latest(self, component: str) -> ComponentScore | None:
        matches = [s for k, s in self.scores.items() if s.component == component]
        return sorted(matches, key=lambda s: s.version)[-1] if matches else None

    def report(self) -> list[dict[str, Any]]:
        return [
            {
                "ref": ref,
                "eval_set": s.eval_set,
                "n": s.n,
                "accuracy": round(s.accuracy, 4),
                "envelope_precision": round(s.envelope_precision, 4),
                "tokens": s.tokens,
            }
            for ref, s in sorted(self.scores.items())
        ]
