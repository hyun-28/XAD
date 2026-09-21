"""스킬 라이브러리 — 유일한 복리 계층.

나머지 계층은 포화한다. 검색 품질은 정체되고, 소형 모델은 천장에 닿고,
라우터는 수렴한다. 스킬만 복리로 쌓인다 — 새 스킬이 기존 스킬과 조합되고,
코어가 강해질수록 같은 스킬을 더 잘 쓴다.

그래서 두 가지 실패 양상을 **처음부터** 막는다.

  자기 검색 문제 — 스킬이 200개가 되면 코어가 무엇이 있는지 모른다.
                   인덱싱·네임스페이스·지연 로딩이 나중에 붙을 수 없다.
  무덤 문제     — API 가 바뀌어 조용히 죽은 스킬은 없는 스킬보다 나쁘다.
                   스킬마다 자기 헬스체크를 가진다.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..trace import SpanKind, Tracer


class Health(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    DEAD = "dead"          # 조용히 죽은 스킬. 없는 스킬보다 나쁘다.
    UNKNOWN = "unknown"    # 헬스체크를 한 번도 돌리지 않음


@dataclass(slots=True)
class HealthReport:
    status: Health
    checked_at: float
    detail: str = ""

    @property
    def usable(self) -> bool:
        return self.status in (Health.OK, Health.DEGRADED)


@dataclass
class Skill:
    """재사용 가능한 능력 하나.

    namespace 는 자기 검색 문제에 대한 첫 번째 대응이다. 평평한 이름 공간은
    수십 개에서 무너진다.

    body 는 **지연 로딩**된다 — 스킬 본문(프롬프트, 코드, 예시)을 전부
    메모리에 올리면 라이브러리 자체가 컨텍스트를 오염시킨다.
    """

    name: str
    namespace: str
    summary: str                       # 인덱스에 올라가는 한 줄. 이것만 코어가 항상 본다.
    keywords: tuple[str, ...] = ()
    version: str = "1.0"
    loader: Callable[[], Any] | None = None      # 지연 로딩
    healthcheck: Callable[[], HealthReport] | None = None
    deprecated: bool = False
    superseded_by: str | None = None

    _body: Any = field(default=None, repr=False)
    _health: HealthReport | None = field(default=None, repr=False)
    _uses: int = field(default=0, repr=False)

    @property
    def ref(self) -> str:
        return f"{self.namespace}/{self.name}@{self.version}"

    @property
    def index_entry(self) -> str:
        """코어가 항상 보는 것. 본문이 아니라 이 한 줄이다."""
        return f"{self.namespace}/{self.name} — {self.summary}"

    def load(self) -> Any:
        """본문을 그때 가져온다. 안 쓰는 스킬은 컨텍스트를 차지하지 않는다."""
        if self._body is None and self.loader is not None:
            self._body = self.loader()
        return self._body

    def check(self, *, force: bool = False) -> HealthReport:
        if self._health is not None and not force:
            return self._health
        if self.healthcheck is None:
            self._health = HealthReport(Health.UNKNOWN, time.time(), "헬스체크 미정의")
            return self._health
        try:
            self._health = self.healthcheck()
        except Exception as exc:
            self._health = HealthReport(Health.DEAD, time.time(), f"{type(exc).__name__}: {exc}")
        return self._health


class SkillLibrary:
    """스킬 저장소 + 인덱스.

    라이브러리가 커질수록 **검색 계층에 의존**하게 된다 — 계층들은 독립이
    아니다. retriever 를 주입하면 그것을 쓰고, 없으면 키워드 매칭으로
    떨어진다(기준선).
    """

    def __init__(self, retriever: Any = None, index_budget: int = 40) -> None:
        self._skills: dict[str, Skill] = {}
        self.retriever = retriever
        self.index_budget = index_budget   # 코어에 한 번에 노출할 최대 스킬 수

    # -- 등록 ------------------------------------------------------------

    def register(self, skill: Skill) -> Skill:
        if skill.ref in self._skills:
            raise ValueError(f"{skill.ref} 이 이미 등록되어 있습니다. 변경은 새 버전입니다.")
        self._skills[skill.ref] = skill
        return skill

    def deprecate(self, ref: str, superseded_by: str | None = None) -> None:
        """폐기 경로. 이것이 없으면 라이브러리는 무덤이 된다."""
        skill = self._skills[ref]
        skill.deprecated = True
        skill.superseded_by = superseded_by

    def __len__(self) -> int:
        return len(self._skills)

    # -- 검색 ------------------------------------------------------------

    def live(self) -> list[Skill]:
        return [s for s in self._skills.values() if not s.deprecated]

    def index(self, namespace: str | None = None) -> list[str]:
        """코어에 노출할 인덱스. 본문이 아니라 한 줄 요약만.

        index_budget 을 넘으면 자르고, 자른 사실을 명시한다 — 조용히
        누락시키면 코어는 없는 스킬을 없다고 판단한다.
        """
        pool = [s for s in self.live() if namespace is None or s.namespace == namespace]
        entries = [s.index_entry for s in pool[: self.index_budget]]
        if len(pool) > self.index_budget:
            entries.append(
                f"... 외 {len(pool) - self.index_budget}건 — 검색으로 좁히십시오"
            )
        return entries

    def find(
        self, query: str, k: int = 5, *, tracer: Tracer | None = None
    ) -> list[Skill]:
        """스킬 검색. 라이브러리가 자기 검색 문제가 되는 지점."""
        tracer = tracer or Tracer()
        with tracer.span("skill_find", SpanKind.TOOL, query=query[:120]) as span:
            pool = [s for s in self.live() if s.check().usable]
            if self.retriever is not None:
                hits = self.retriever.search(query, k, tracer=tracer)
                ids = {h.document.id for h in hits}
                found = [s for s in pool if s.ref in ids][:k]
            else:
                terms = {t for t in query.lower().split() if t}

                def score(s: Skill) -> int:
                    hay = f"{s.name} {s.summary} {' '.join(s.keywords)}".lower()
                    return sum(1 for t in terms if t in hay)

                found = sorted(
                    (s for s in pool if score(s)), key=score, reverse=True
                )[:k]
            span.attributes.update(
                candidates=len(pool), found=[s.ref for s in found]
            )
            return found

    def get(self, namespace: str, name: str, version: str | None = None) -> Skill:
        matches = [
            s
            for s in self._skills.values()
            if s.namespace == namespace and s.name == name
        ]
        if not matches:
            raise KeyError(f"스킬 없음: {namespace}/{name}")
        if version is not None:
            for s in matches:
                if s.version == version:
                    return s
            raise KeyError(f"{namespace}/{name}@{version} 없음")
        live = [s for s in matches if not s.deprecated]
        return sorted(live or matches, key=lambda s: s.version)[-1]

    # -- 사용과 은퇴 ------------------------------------------------------

    def use(self, skill: Skill, *, tracer: Tracer | None = None) -> Any:
        tracer = tracer or Tracer()
        with tracer.span(
            f"skill:{skill.name}",
            SpanKind.TOOL,
            component=skill.ref,
            namespace=skill.namespace,
        ) as span:
            health = skill.check()
            span.attributes["health"] = health.status.value
            if not health.usable:
                raise RuntimeError(
                    f"{skill.ref} 상태 {health.status.value}: {health.detail}. "
                    f"조용히 죽은 스킬은 없는 스킬보다 나쁩니다."
                )
            skill._uses += 1
            return skill.load()

    def sweep(self, *, force: bool = False) -> dict[str, list[str]]:
        """전수 헬스체크. 정기적으로 돌려 무덤을 치운다."""
        buckets: dict[str, list[str]] = {h.value: [] for h in Health}
        for s in self._skills.values():
            buckets[s.check(force=force).status.value].append(s.ref)
        return buckets

    def retirement_candidates(self) -> list[str]:
        """폐기됐고 사용량이 0인 스킬. v1 은 사용량이 0일 때 은퇴한다."""
        return [s.ref for s in self._skills.values() if s.deprecated and s._uses == 0]

    def usage(self) -> dict[str, int]:
        return {s.ref: s._uses for s in self._skills.values()}


def index_documents(library: SkillLibrary) -> list[Any]:
    """스킬 인덱스를 검색기에 넣을 문서로 변환한다.

    라이브러리가 커지면 검색 계층에 의존하게 되는데, 그 연결이 여기다.
    """
    from ..memory import Document

    return [
        Document(
            id=s.ref,
            text=f"{s.name} {s.summary} {' '.join(s.keywords)}",
            metadata={"namespace": s.namespace, "version": s.version},
        )
        for s in library.live()
    ]
