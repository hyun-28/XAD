"""검색과 기억 — 분리된 두 계층.

1부 지적: 이 둘을 한 박스에 그리면 컴플라이언스 리뷰에서 걸린다.

  Retrieval — 외부 문서 지식. 정적 인덱스, 갱신 주기가 길다.
              보존 규정 대상이 아니다.
  Memory    — 세션 상태, 과거 상호작용. **사용자 데이터를 포함하므로**
              보존 기간·삭제 요구·주체별 격리가 적용된다.

둘 다 비파라메트릭이므로 망각이 없고 코어 교체를 그대로 살아남는다 —
장기 자산이다.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..trace import SpanKind, Tracer, record_isolation


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def approx_tokens(self) -> int:
        return len(self.text) // 4


@dataclass(frozen=True, slots=True)
class Hit:
    document: Document
    score: float


class Retriever(ABC):
    """검색기. 비파라메트릭 적응 표면 — 온라인 갱신이 허용된다."""

    name = "retriever"

    @abstractmethod
    def _search(self, query: str, k: int) -> list[Hit]: ...

    def search(self, query: str, k: int = 5, *, tracer: Tracer | None = None) -> list[Hit]:
        tracer = tracer or Tracer()
        with tracer.span(self.name, SpanKind.TOOL, query=query[:120], k=k) as span:
            hits = self._search(query, k)
            span.attributes.update(
                returned=len(hits),
                doc_ids=[h.document.id for h in hits],
                total_tokens=sum(h.document.approx_tokens for h in hits),
            )
            return hits


class KeywordRetriever(Retriever):
    """의존성 없는 기본 검색기. T1 기준선 역할.

    T2(맞춤형)를 도입하기 전에 이것으로 먼저 측정한다 — 기준선 없이
    맞춤형을 만들면 개선을 증명할 수 없다.
    """

    name = "keyword_retriever"

    def __init__(self, documents: Sequence[Document]) -> None:
        self.documents = list(documents)

    def add(self, doc: Document) -> None:
        self.documents.append(doc)

    def _search(self, query: str, k: int) -> list[Hit]:
        terms = [t for t in query.lower().split() if t]
        scored: list[Hit] = []
        for doc in self.documents:
            body = doc.text.lower()
            score = sum(body.count(t) for t in terms)
            if score:
                scored.append(Hit(doc, float(score)))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]


@dataclass
class IsolatingRetriever:
    """검색 결과를 워커 안에서 압축해 요약만 올려보내는 래퍼.

    P4 대응의 재사용 가능한 형태. 원문은 이 경계를 넘지 않으며,
    흡수·상향 토큰이 궤적에 기록되어 격리 효과로 측정된다.
    """

    inner: Retriever
    summarize: Any  # Callable[[list[Hit]], str]
    name: str = "isolating_retriever"

    def search_summary(
        self, query: str, k: int = 5, *, tracer: Tracer | None = None
    ) -> tuple[str, frozenset[str]]:
        tracer = tracer or Tracer()
        with tracer.span(self.name, SpanKind.COMPONENT) as span:
            hits = self.inner.search(query, k, tracer=tracer)
            raw = sum(h.document.approx_tokens for h in hits)
            summary = self.summarize(hits)
            emitted = len(summary) // 4
            record_isolation(absorbed=max(0, raw - emitted), emitted=emitted)
            span.attributes.update(absorbed=raw - emitted, emitted=emitted)
            return summary, frozenset(h.document.id for h in hits)


# --------------------------------------------------------------------------
# 기억 — 보존 규정이 적용되는 계층
# --------------------------------------------------------------------------

@dataclass(slots=True)
class MemoryRecord:
    subject: str          # 데이터 주체. 삭제 요구의 단위.
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    ttl_seconds: float | None = None

    @property
    def expired(self) -> bool:
        return (
            self.ttl_seconds is not None
            and (time.time() - self.created_at) > self.ttl_seconds
        )


@dataclass
class Memory:
    """세션·사용자 기억.

    검색기와 달리 **보존 기간과 주체별 삭제**를 지원한다. 규제 환경에서
    이 둘을 한 저장소에 두면 문서 인덱스에까지 삭제 요구가 번진다.
    """

    default_ttl_seconds: float | None = None
    _records: list[MemoryRecord] = field(default_factory=list)

    def put(self, subject: str, key: str, value: Any, ttl_seconds: float | None = None) -> None:
        self._records.append(
            MemoryRecord(
                subject=subject,
                key=key,
                value=value,
                ttl_seconds=ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds,
            )
        )

    def get(self, subject: str, key: str) -> Any | None:
        for r in reversed(self._records):
            if r.subject == subject and r.key == key and not r.expired:
                return r.value
        return None

    def forget_subject(self, subject: str) -> int:
        """데이터 주체 단위 삭제. 삭제 요구 대응의 최소 단위."""
        before = len(self._records)
        self._records = [r for r in self._records if r.subject != subject]
        return before - len(self._records)

    def purge_expired(self) -> int:
        before = len(self._records)
        self._records = [r for r in self._records if not r.expired]
        return before - len(self._records)

    def __len__(self) -> int:
        return len([r for r in self._records if not r.expired])
