"""궤적 기록기.

JSONL 싱크. 한 줄이 스팬 하나이며, 스팬은 종료 시점에 기록된다.
파일은 append-only이므로 감사 증거로 쓸 수 있고, 재현 모드의 입력이 된다.
"""

from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .span import Span


class TraceSink(ABC):
    """스팬 싱크. 구현을 갈아끼워도 상위 코드는 바뀌지 않는다."""

    @abstractmethod
    def emit(self, span: Span) -> None: ...

    def close(self) -> None:  # pragma: no cover - 기본 구현은 no-op
        return None


class MemorySink(TraceSink):
    """테스트와 인메모리 분석용."""

    def __init__(self) -> None:
        self.spans: list[Span] = []
        self._lock = threading.Lock()

    def emit(self, span: Span) -> None:
        with self._lock:
            self.spans.append(span)

    def by_trace(self, trace_id: str) -> list[Span]:
        return [s for s in self.spans if s.trace_id == trace_id]

    def clear(self) -> None:
        with self._lock:
            self.spans.clear()


class JsonlSink(TraceSink):
    """append-only JSONL. 운영 기본값."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fh = self.path.open("a", encoding="utf-8")

    def emit(self, span: Span) -> None:
        line = json.dumps(span.to_dict(), ensure_ascii=False)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()

    def __enter__(self) -> JsonlSink:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class MultiSink(TraceSink):
    def __init__(self, *sinks: TraceSink) -> None:
        self.sinks = sinks

    def emit(self, span: Span) -> None:
        for sink in self.sinks:
            sink.emit(span)

    def close(self) -> None:
        for sink in self.sinks:
            sink.close()


def read_trace(path: str | Path) -> Iterator[dict[str, Any]]:
    """기록된 궤적을 다시 읽는다. 재현 모드와 오프라인 평가의 입력."""
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
