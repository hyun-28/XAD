"""궤적 스팬.

1부 원칙: "평가를 설계 초기에 넣는다"의 실체는 "궤적 로그를 1일차에 깐다"이다.
평가셋·지표·판정기는 나중에도 만들 수 있으나 과거 궤적은 만들 수 없다.

스팬 하나는 시스템이 내린 판단 하나에 대응한다. 스팬은 세 곳에서 소비된다.

  1. 컴포넌트 평가의 데이터 소스     (evaluation/)
  2. 브로커 기여도 추정의 근거        (broker/)
  3. 규제 환경의 감사 증거 그 자체     (재현 모드)
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SpanKind(str, Enum):
    """스팬이 기록하는 판단의 종류.

    제어(CONTROL)와 판단(MODEL/COMPONENT)을 구분하는 것이 핵심이다.
    1부 원칙 "LLM은 판단하고, 제어는 코드가 한다"가 궤적에서 검증 가능하려면,
    두 종류가 분리되어 기록되어야 한다.
    """

    CONTROL = "control"          # 코드가 내린 제어 결정 (분기, 재시도, 게이트)
    MODEL = "model"              # LLM 호출
    COMPONENT = "component"      # 계약을 가진 컴포넌트 호출
    TOOL = "tool"                # 외부 도구 호출
    VERIFY = "verify"            # 검증기
    SELECTION = "selection"      # 브로커의 팀/모델 선택 — 재현 모드의 입력
    HUMAN = "human"              # 사용자 개입 — 공짜 고품질 라벨


class SpanStatus(str, Enum):
    OK = "ok"
    ERROR = "error"
    OUT_OF_ENVELOPE = "out_of_envelope"   # 유효 범위 밖 — 조용히 틀리지 않게
    BUDGET_EXCEEDED = "budget_exceeded"
    GUARDRAIL_BLOCKED = "guardrail_blocked"


@dataclass(slots=True)
class Usage:
    """자원 사용량. P1(비용·지연)이 측정되는 유일한 자리."""

    input_tokens: int = 0
    output_tokens: int = 0
    wall_seconds: float = 0.0
    peak_vram_mb: float | None = None   # 앙상블의 "메모리 제약 우회" 주장 검증용

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def merge(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            wall_seconds=self.wall_seconds + other.wall_seconds,
            peak_vram_mb=max(
                (v for v in (self.peak_vram_mb, other.peak_vram_mb) if v is not None),
                default=None,
            ),
        )


@dataclass(slots=True)
class Span:
    name: str
    kind: SpanKind
    trace_id: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_id: str | None = None

    component: str | None = None
    component_version: str | None = None

    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None

    status: SpanStatus = SpanStatus.OK
    error: str | None = None

    usage: Usage = field(default_factory=Usage)
    attributes: dict[str, Any] = field(default_factory=dict)

    # 격리 효과 측정용: 이 스팬이 흡수하고 상위로 올려보내지 않은 토큰.
    # P4(컨텍스트 오염)를 직접 재는 유일한 지표의 원자료.
    absorbed_tokens: int = 0
    emitted_tokens: int = 0

    @property
    def duration(self) -> float:
        return (self.ended_at or time.time()) - self.started_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "kind": self.kind.value,
            "component": self.component,
            "component_version": self.component_version,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration": round(self.duration, 6),
            "status": self.status.value,
            "error": self.error,
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
                "wall_seconds": round(self.usage.wall_seconds, 6),
                "peak_vram_mb": self.usage.peak_vram_mb,
            },
            "absorbed_tokens": self.absorbed_tokens,
            "emitted_tokens": self.emitted_tokens,
            "attributes": self.attributes,
        }
