"""증류 파이프라인 — 코어를 동결한 채 전용 모델을 얻는 경로.

1부 적응 사다리에서 5단(교체)과 6단(파인튜닝) 사이를 잇는다.
**코어가 교사, 소형 모델이 학생, 코어의 가중치는 영원히 안 건드린다.**

파라메트릭 진입 조건 세 개를 코드가 검사한다. 하나라도 빠지면 내려가지 않는다.

  1. 작업이 좁고 안정적   — 스펙이 바뀌면 재학습이 반복되고 P3 가 재현된다
  2. 호출 빈도가 높음     — 학습 + 검증 비용을 회수할 물량이 있어야 한다
  3. 라벨 자동 생성 가능  — 손라벨링이 필요하면 대개 경제성이 없다

세 조건을 사람의 판단에 맡기면 "이번엔 예외"가 반복된다. 그래서 게이트로 만든다.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..contracts import (
    Amortization,
    ComponentContract,
    InterfaceSpec,
    OperatingEnvelope,
    PerformanceSpec,
    Role,
)
from ..evaluation import ComponentEvalSet, ComponentScore, EvalItem
from ..trace import SpanKind, Tracer, read_trace


class EntryConditionsUnmet(RuntimeError):
    """파라메트릭 진입 조건 미충족. 사다리 6단으로 내려가지 않는다."""


@dataclass(slots=True)
class EntryCheck:
    stable_spec: bool
    high_frequency: bool
    auto_labelable: bool
    observed_calls: int = 0
    spec_changes: int = 0
    detail: str = ""

    @property
    def satisfied(self) -> bool:
        return self.stable_spec and self.high_frequency and self.auto_labelable

    def failures(self) -> list[str]:
        out = []
        if not self.stable_spec:
            out.append(f"스펙이 불안정합니다 (변경 {self.spec_changes}회) — 재학습이 반복됩니다")
        if not self.high_frequency:
            out.append(f"호출 빈도 부족 ({self.observed_calls}회) — 학습·검증 비용을 회수할 수 없습니다")
        if not self.auto_labelable:
            out.append("라벨 자동 생성 불가 — 손라벨링이 필요하면 대개 경제성이 없습니다")
        return out


def check_entry(
    trace_path: str | Path,
    task_name: str,
    *,
    min_calls: int = 500,
    max_spec_changes: int = 0,
    auto_labelable: bool = True,
) -> EntryCheck:
    """궤적에서 진입 조건을 실측한다.

    빈도와 스펙 안정성은 의견이 아니라 로그에서 나오는 숫자다.
    """
    calls = 0
    signatures: set[str] = set()
    for row in read_trace(trace_path):
        if row.get("component") != task_name and row.get("name") != task_name:
            continue
        calls += 1
        attrs = row.get("attributes") or {}
        sig = attrs.get("io_signature") or row.get("component_version") or ""
        if sig:
            signatures.add(str(sig))

    changes = max(0, len(signatures) - 1)
    return EntryCheck(
        stable_spec=changes <= max_spec_changes,
        high_frequency=calls >= min_calls,
        auto_labelable=auto_labelable,
        observed_calls=calls,
        spec_changes=changes,
        detail=f"{task_name}: 호출 {calls}회, 시그니처 {len(signatures)}종",
    )


# --------------------------------------------------------------------------
# 라벨 생성 — 코어가 교사
# --------------------------------------------------------------------------

@dataclass(slots=True)
class LabeledExample:
    payload: Any
    label: Any
    teacher: str
    source_span: str = ""
    verified: bool = False       # 검증기를 통과한 라벨만 학습에 쓴다

    @property
    def key(self) -> str:
        blob = json.dumps({"p": self.payload}, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass
class DistillationSet:
    """증류 학습셋.

    verified 가 아닌 예제는 기본적으로 배제된다. 교사가 틀린 라벨을 학생이
    배우면, 그 오류는 이제 가중치에 들어가 되돌리기 어려워진다 —
    사다리 6단의 비용이 정확히 그것이다.
    """

    task: str
    teacher: str
    examples: list[LabeledExample] = field(default_factory=list)

    def add(self, example: LabeledExample) -> None:
        self.examples.append(example)

    def verified_only(self) -> list[LabeledExample]:
        return [e for e in self.examples if e.verified]

    def dedup(self) -> list[LabeledExample]:
        seen: set[str] = set()
        out: list[LabeledExample] = []
        for e in self.verified_only():
            if e.key in seen:
                continue
            seen.add(e.key)
            out.append(e)
        return out

    @property
    def verified_ratio(self) -> float:
        return len(self.verified_only()) / len(self.examples) if self.examples else 0.0

    def split(self, holdout: float = 0.2) -> tuple[list[LabeledExample], ComponentEvalSet]:
        """학습셋과 **고정 평가셋**으로 나눈다.

        평가셋이 코드보다 먼저 존재한다 — 여기서는 학습 전에 잘라둔다.
        나중에 자르면 학습에 쓴 것으로 평가하게 된다.
        """
        items = self.dedup()
        cut = max(1, int(len(items) * holdout)) if items else 0
        eval_items, train = items[:cut], items[cut:]
        es = ComponentEvalSet(name=f"{self.task}_holdout", component_name=self.task)
        for i, e in enumerate(eval_items):
            es.add(EvalItem(id=f"h{i}", payload=e.payload, expected=e.label))
        return train, es

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            for e in self.dedup():
                fh.write(
                    json.dumps(
                        {"input": e.payload, "output": e.label, "teacher": e.teacher},
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n"
                )
        return p


def label_with_core(
    payloads: Iterable[Any],
    teacher: Callable[[Any], Any],
    *,
    task: str,
    teacher_id: str,
    verify: Callable[[Any, Any], bool] | None = None,
    tracer: Tracer | None = None,
) -> DistillationSet:
    """코어로 라벨을 생성한다.

    손라벨링 없이 전용 모델의 학습 데이터를 얻고, **코어는 동결 상태를 유지한다.**
    verify 가 주어지면 검증을 통과한 라벨만 verified 로 표시된다.
    """
    tracer = tracer or Tracer()
    ds = DistillationSet(task=task, teacher=teacher_id)
    with tracer.span("distill_label", SpanKind.CONTROL, task=task, teacher=teacher_id) as span:
        for payload in payloads:
            label = teacher(payload)
            ok = verify(payload, label) if verify is not None else False
            ds.add(
                LabeledExample(
                    payload=payload, label=label, teacher=teacher_id, verified=ok
                )
            )
        span.attributes.update(
            examples=len(ds.examples),
            verified=len(ds.verified_only()),
            verified_ratio=round(ds.verified_ratio, 4),
        )
    return ds


# --------------------------------------------------------------------------
# 계약 발행 — Stage 2
# --------------------------------------------------------------------------

def publish_contract(
    *,
    name: str,
    version: str,
    score: ComponentScore,
    envelope: OperatingEnvelope,
    interface: InterfaceSpec,
    fallback: str,
    teacher: str,
    entry: EntryCheck,
) -> ComponentContract:
    """증류 결과를 계약으로 발행한다.

    fallback 은 필수다 — T1 기준선 없이 발행된 T2 는 회귀했을 때
    되돌아갈 곳이 없다.
    """
    if not entry.satisfied:
        raise EntryConditionsUnmet(
            "파라메트릭 진입 조건 미충족:\n  - " + "\n  - ".join(entry.failures())
        )
    if not fallback:
        raise ValueError(
            "fallback 없이 T2 를 발행할 수 없습니다. T1 은 버려지지 않고 "
            "비교 기준선이자 폴백으로 남습니다."
        )
    return ComponentContract(
        name=name,
        version=version,
        role=Role.SPECIALIST,
        interface=interface,
        performance=score.as_spec(notes=f"교사 {teacher} 증류; {entry.detail}"),
        envelope=envelope,
        amortization=Amortization.PARTIAL,   # 코어 교체 시 부분 상각된다
        ladder_rung=6,
        fallback=fallback,
    )
