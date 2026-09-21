"""LLM Judge 와 그 검증.

1부 원칙 두 개가 여기서 코드가 된다.

  (1) 결정론 우선 — "이 출력이 맞는지 프로그램으로 확인할 수 있는 하위 성질이
      무엇인가?" 그것을 최대한 뽑고 남은 것만 Judge 에 맡긴다.
  (2) Judge 검증 — Judge 는 리더보드 전체의 신뢰 뿌리다. 사람 라벨 대비
      일치율을 측정하고 버전을 고정하지 않으면, 리더보드는 모델이 아니라
      Judge 의 변화를 측정한다.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from statistics import fmean
from typing import Any

from ..trace import SpanKind, Tracer


class UncalibratedJudge(RuntimeError):
    """캘리브레이션 없이 Judge 를 운영에 쓰려 했다."""


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    passed: bool
    score: float = 0.0
    rationale: str = ""


@dataclass(frozen=True, slots=True)
class HumanLabel:
    case_id: str
    passed: bool
    note: str = ""


@dataclass(slots=True)
class Calibration:
    """사람 라벨 대비 Judge 일치율."""

    n: int
    agreement: float
    false_pass: int      # 사람은 실패라 했는데 Judge 가 통과시킨 것 — 가장 위험
    false_fail: int

    @property
    def usable(self) -> bool:
        return self.n >= 30 and self.agreement >= 0.8


@dataclass
class CalibratedJudge:
    """버전이 고정되고 캘리브레이션된 Judge.

    judge_id 는 모델과 프롬프트를 함께 식별한다. 둘 중 하나만 바뀌어도
    다른 Judge 이며, 이전 리더보드와 비교할 수 없다.
    """

    judge_id: str
    fn: Callable[[Any, Any], JudgeVerdict]
    calibration: Calibration | None = None
    require_calibration: bool = True

    def calibrate(
        self,
        cases: Sequence[tuple[str, Any, Any]],
        labels: Sequence[HumanLabel],
    ) -> Calibration:
        by_id = {l.case_id: l for l in labels}
        agree = 0
        fp = fn_ = 0
        n = 0
        for case_id, candidate, context in cases:
            label = by_id.get(case_id)
            if label is None:
                continue
            n += 1
            verdict = self.fn(candidate, context)
            if verdict.passed == label.passed:
                agree += 1
            elif verdict.passed and not label.passed:
                fp += 1
            else:
                fn_ += 1
        self.calibration = Calibration(
            n=n, agreement=agree / n if n else 0.0, false_pass=fp, false_fail=fn_
        )
        return self.calibration

    def __call__(
        self, candidate: Any, context: Any = None, *, tracer: Tracer | None = None
    ) -> JudgeVerdict:
        if self.require_calibration and (
            self.calibration is None or not self.calibration.usable
        ):
            got = (
                f"n={self.calibration.n}, agreement={self.calibration.agreement:.2f}"
                if self.calibration
                else "미측정"
            )
            raise UncalibratedJudge(
                f"Judge '{self.judge_id}' 가 캘리브레이션되지 않았습니다 ({got}). "
                f"사람 라벨 30건 이상, 일치율 0.8 이상이 필요합니다. "
                f"캘리브레이션 없는 Judge 를 쓰면 리더보드가 모델이 아니라 "
                f"Judge 의 변화를 측정합니다."
            )
        tracer = tracer or Tracer()
        with tracer.span("judge", SpanKind.VERIFY, judge_id=self.judge_id) as span:
            v = self.fn(candidate, context)
            span.attributes.update(passed=v.passed, score=v.score)
            return v


@dataclass
class DeterministicFirst:
    """결정론 검사를 먼저 돌리고, 통과한 것만 Judge 로 보낸다.

    비용·분산·불확실성을 동시에 줄인다. 그리고 결정론 검사에서 걸린 항목에
    대한 판정은 애초에 의미가 없다 — 형식이 깨진 출력의 품질을 논할 수 없다.
    """

    deterministic: Callable[[Any, Any], bool]
    judge: CalibratedJudge | None = None
    stats: dict[str, int] = field(default_factory=lambda: {"deterministic": 0, "judged": 0})

    def evaluate(
        self, candidate: Any, context: Any = None, *, tracer: Tracer | None = None
    ) -> JudgeVerdict:
        if not self.deterministic(candidate, context):
            self.stats["deterministic"] += 1
            return JudgeVerdict(passed=False, rationale="결정론 검사 실패 — 판정 생략")
        if self.judge is None:
            return JudgeVerdict(passed=True, score=1.0, rationale="결정론 검사 통과")
        self.stats["judged"] += 1
        return self.judge(candidate, context, tracer=tracer)

    @property
    def judge_load(self) -> float:
        """Judge 로 넘어간 비율. 낮을수록 좋다 — 결정론 영역을 더 넓혔다는 뜻."""
        total = sum(self.stats.values())
        return self.stats["judged"] / total if total else 0.0
