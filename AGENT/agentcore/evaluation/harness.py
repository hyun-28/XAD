"""평가 하네스.

두 그림을 잇는 화살표가 여기 있다.

    런타임 궤적 로그  ──→  벤치마크 데이터셋

이 연결이 없으면 벤치마크는 정적이고, 실사용에서 드러난 실패가 평가에
반영되지 않는다. 그리고 **사용자 개입 지점은 공짜 고품질 라벨**이다 —
거부·수정·재시도가 일어난 지점이 최고 품질의 학습·평가 데이터다.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..trace import MemorySink, SpanKind, Tracer, read_trace
from ..verify import GuardrailReport, GuardrailSet
from .layers import Aggregate, Layer, LayeredResult, summarize_system, summarize_trajectory


@dataclass(slots=True)
class GoldenAction:
    """정답을 최종 출력이 아니라 **행동 경로**로 정의한다.

    경로가 여럿일 수 있으므로 문자열 일치가 아니라 등가성으로 판정한다.
    equivalence 가 주어지면 그것이 우선한다.
    """

    steps: tuple[str, ...]
    equivalence: Callable[[Sequence[str]], bool] | None = None

    def matches(self, actual: Sequence[str]) -> bool:
        if self.equivalence is not None:
            return self.equivalence(actual)
        return tuple(actual) == self.steps


@dataclass(slots=True)
class Case:
    """평가 사례.

    level 은 능력 사다리의 단(L1~L7)이며, 역할별로 필요한 단이 다르다 —
    Specialist 는 L1만, 코어는 전부. 시스템 전체를 한 점수로 매기면
    "어디를 고쳐야 하는가"가 사라진다.
    """

    id: str
    payload: Any
    level: int = 1
    golden: GoldenAction | None = None
    expected: Any = None
    equivalent: Callable[[Any, Any], bool] | None = None
    failure_path: bool = False   # 실패 경로 변형. 오류 처리는 단계가 아니라 차원이다.
    tags: tuple[str, ...] = ()

    def output_matches(self, actual: Any) -> bool:
        if self.equivalent is not None:
            return self.equivalent(actual, self.expected)
        return actual == self.expected


@dataclass
class Suite:
    cases: list[Case] = field(default_factory=list)
    name: str = "suite"
    environment: dict[str, str] = field(default_factory=dict)

    def add(self, case: Case) -> None:
        self.cases.append(case)

    def by_level(self) -> dict[int, list[Case]]:
        out: dict[int, list[Case]] = {}
        for c in self.cases:
            out.setdefault(c.level, []).append(c)
        return out

    def coverage(self) -> dict[str, float]:
        """정상 경로와 실패 경로 양쪽을 덮고 있는가.

        오류 처리를 별도 단계로 두면 L1~L3 가 해피패스만 테스트하게 되고,
        실무 실패의 대부분이 빠진다.
        """
        total = len(self.cases) or 1
        fails = sum(1 for c in self.cases if c.failure_path)
        return {
            "failure_path_ratio": fails / total,
            "levels_covered": len(self.by_level()) / 7,
        }


@dataclass
class Harness:
    """사례를 시스템에 태우고 4층위 결과를 모은다."""

    suite: Suite
    guardrails: GuardrailSet | None = None

    def run(
        self,
        system: Callable[[Any, Tracer], Any],
        *,
        extract_steps: Callable[[list], list[str]] | None = None,
    ) -> Aggregate:
        agg = Aggregate()
        for case in self.suite.cases:
            sink = MemorySink()
            tracer = Tracer(sink=sink)
            result = LayeredResult(case_id=case.id)
            output: Any = None
            try:
                output = system(case.payload, tracer)
            except Exception as exc:
                result.passed = False
                result.detail = f"{type(exc).__name__}: {exc}"

            spans = sink.spans
            result.trajectory = summarize_trajectory(spans)
            result.system = summarize_system(spans)

            if result.detail == "":
                if case.expected is not None:
                    result.passed = case.output_matches(output)
                if case.golden is not None:
                    steps = (
                        extract_steps(spans)
                        if extract_steps
                        else [s.name for s in spans if s.kind is SpanKind.CONTROL]
                    )
                    path_ok = case.golden.matches(steps)
                    result.add(Layer.TRAJECTORY, "path_match", 1.0 if path_ok else 0.0)
                    result.passed = result.passed and path_ok

            if self.guardrails is not None:
                candidates = output if isinstance(output, list) else [output]
                result.guardrails = self.guardrails.evaluate(candidates, tracer=tracer)
                if result.guardrails.blocked:
                    result.passed = False

            result.add(Layer.OUTPUT, "passed", 1.0 if result.passed else 0.0)
            result.add(Layer.SYSTEM, "tokens", float(result.system.total_tokens))
            result.add(Layer.TRAJECTORY, "isolation", result.trajectory.isolation_ratio)
            agg.add(result)
        return agg


# --------------------------------------------------------------------------
# 궤적 → 벤치마크
# --------------------------------------------------------------------------

def cases_from_trace(
    path: str | Path,
    *,
    only_interventions: bool = True,
) -> list[Case]:
    """운영 궤적에서 평가 사례를 뽑는다.

    only_interventions=True 면 **사용자가 개입한 지점만** 가져온다.
    거부·수정·재시도는 사람이 붙여준 실패 라벨이며, 손라벨링 없이 얻는
    최고 품질의 데이터다. 대부분의 시스템이 이것을 그냥 버린다.
    """
    cases: list[Case] = []
    for row in read_trace(path):
        attrs = row.get("attributes", {}) or {}
        is_intervention = row.get("kind") == SpanKind.HUMAN.value or attrs.get("rejected")
        if only_interventions and not is_intervention:
            continue
        cases.append(
            Case(
                id=f"trace:{row['span_id']}",
                payload=attrs.get("payload"),
                expected=attrs.get("corrected"),
                level=int(attrs.get("level", 1)),
                failure_path=bool(attrs.get("rejected")),
                tags=("from_trace",),
            )
        )
    return cases


def selections_from_trace(path: str | Path) -> dict[str, tuple[str, ...]]:
    """궤적에서 브로커 선택을 뽑아 재현 모드의 입력으로 만든다.

    이것이 있어야 "동적 선택은 재현 불가"라는 약점이 국소화된다 —
    선택 자체를 기록하면 과거 판단을 정확히 재생할 수 있다.
    """
    out: dict[str, tuple[str, ...]] = {}
    for row in read_trace(path):
        if row.get("kind") != SpanKind.SELECTION.value:
            continue
        attrs = row.get("attributes", {}) or {}
        key = attrs.get("task_key")
        chosen = attrs.get("chosen")
        if key and chosen:
            out[key] = tuple(chosen)
    return out


def write_report(agg: Aggregate, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(agg.report(), ensure_ascii=False, indent=2), encoding="utf-8")
    return p
