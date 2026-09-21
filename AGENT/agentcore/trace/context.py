"""현재 궤적 컨텍스트.

contextvars 기반이므로 asyncio 태스크와 스레드 사이에서 부모-자식 관계가
자동으로 유지된다. 병렬 워커가 각자의 하위 스팬을 남겨도 트리가 깨지지 않는다.
"""

from __future__ import annotations

import contextvars
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from .recorder import MemorySink, TraceSink
from .span import Span, SpanKind, SpanStatus, Usage

_current_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
    "agentcore_current_span", default=None
)
_current_tracer: contextvars.ContextVar["Tracer | None"] = contextvars.ContextVar(
    "agentcore_current_tracer", default=None
)


@dataclass
class Tracer:
    """궤적 기록의 진입점.

    sink 를 주입받으므로 테스트에서는 MemorySink, 운영에서는 JsonlSink 를 쓴다.
    """

    sink: TraceSink = field(default_factory=MemorySink)
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])

    @contextmanager
    def span(
        self,
        name: str,
        kind: SpanKind = SpanKind.CONTROL,
        *,
        component: str | None = None,
        component_version: str | None = None,
        **attributes: Any,
    ) -> Iterator[Span]:
        parent = _current_span.get()
        span = Span(
            name=name,
            kind=kind,
            trace_id=self.trace_id,
            parent_id=parent.span_id if parent else None,
            component=component,
            component_version=component_version,
            attributes=dict(attributes),
        )
        span_token = _current_span.set(span)
        tracer_token = _current_tracer.set(self)
        start = time.perf_counter()
        try:
            yield span
        except Exception as exc:
            span.status = SpanStatus.ERROR
            span.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span.usage.wall_seconds = time.perf_counter() - start
            span.ended_at = time.time()
            _current_span.reset(span_token)
            _current_tracer.reset(tracer_token)
            # 자식의 사용량은 부모로 합산된다 — 시스템 층위 집계가 공짜로 된다.
            if parent is not None:
                parent.usage = parent.usage.merge(span.usage)
            self.sink.emit(span)


def current_span() -> Span | None:
    return _current_span.get()


def current_tracer() -> Tracer | None:
    return _current_tracer.get()


def record_usage(
    input_tokens: int = 0,
    output_tokens: int = 0,
    peak_vram_mb: float | None = None,
) -> None:
    """현재 스팬에 사용량을 더한다."""
    span = _current_span.get()
    if span is None:
        return
    span.usage = span.usage.merge(
        Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            peak_vram_mb=peak_vram_mb,
        )
    )


def record_isolation(absorbed: int, emitted: int) -> None:
    """격리 효과를 기록한다.

    absorbed = 이 스팬이 처리하고 상위로 보내지 않은 토큰
    emitted  = 상위 컨텍스트로 실제로 올라간 토큰

    격리 효과 = absorbed / emitted. P4(컨텍스트 오염)를 직접 재는 지표이며,
    모놀리식 구조에서는 absorbed 가 항상 0이 되므로 아키텍처를 판별한다.
    """
    span = _current_span.get()
    if span is None:
        return
    span.absorbed_tokens += absorbed
    span.emitted_tokens += emitted
