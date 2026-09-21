"""프로바이더 중립 모델 인터페이스.

코어는 크기가 아니라 **역할**이다. 같은 인터페이스를 단일 프론티어 모델이
채우든 앙상블이 채우든 상위 코드는 바뀌지 않는다 — 그래서 메모리 제약
환경에서 코어 박스만 앙상블로 갈아끼울 수 있다.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..trace import SpanKind, Tracer, record_usage


@dataclass(slots=True)
class Message:
    role: str      # "system" | "user" | "assistant"
    content: str


@dataclass(slots=True)
class Completion:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    stop_reason: str = "end_turn"
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def cassette_key(model: str, messages: list[Message], params: dict[str, Any]) -> str:
    """호출을 결정론적으로 식별하는 키.

    record-replay 의 기반. 같은 입력은 같은 키를 갖고, 같은 응답을 재생한다.
    """
    payload = json.dumps(
        {
            "model": model,
            "messages": [(m.role, m.content) for m in messages],
            "params": {k: v for k, v in sorted(params.items())},
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


class LLM(ABC):
    """모델 어댑터."""

    name: str = "llm"
    model_id: str = ""
    # 이 모델이 앉는 자리. Role.CORE 자리는 판단·합성 전용.
    size_hint_b: float | None = None
    peak_vram_mb: float | None = None

    @abstractmethod
    def _complete(self, messages: list[Message], **params: Any) -> Completion: ...

    def complete(
        self, messages: list[Message], *, tracer: Tracer | None = None, **params: Any
    ) -> Completion:
        tracer = tracer or Tracer()
        with tracer.span(
            self.name, SpanKind.MODEL, component=self.model_id or self.name
        ) as span:
            out = self._complete(messages, **params)
            record_usage(
                input_tokens=out.input_tokens,
                output_tokens=out.output_tokens,
                peak_vram_mb=self.peak_vram_mb,
            )
            span.attributes.update(
                model=out.model or self.model_id,
                stop_reason=out.stop_reason,
                size_b=self.size_hint_b,
            )
            return out


class EchoLLM(LLM):
    """테스트용 결정론 모델. 외부 호출 없이 파이프라인을 돌린다."""

    name = "echo"
    model_id = "echo-1"

    def __init__(self, responder: Any = None, size_hint_b: float | None = None) -> None:
        self._responder = responder or (lambda msgs: msgs[-1].content[::-1])
        self.size_hint_b = size_hint_b

    def _complete(self, messages: list[Message], **params: Any) -> Completion:
        text = self._responder(messages)
        return Completion(
            text=text,
            input_tokens=sum(len(m.content) // 4 for m in messages),
            output_tokens=len(text) // 4,
            model=self.model_id,
        )
