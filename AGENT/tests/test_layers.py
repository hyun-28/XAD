"""검색·기억·스킬 계층 검증."""

from __future__ import annotations

import time

import pytest

from agentcore.evaluation import summarize_trajectory
from agentcore.memory import (
    Document,
    IsolatingRetriever,
    KeywordRetriever,
    Memory,
)
from agentcore.skills import Health, HealthReport, Skill, SkillLibrary, index_documents
from agentcore.trace import MemorySink, Tracer


# -- 검색 ------------------------------------------------------------------

def test_keyword_retriever_ranks_by_frequency():
    r = KeywordRetriever(
        [Document("a", "매출 매출 매출"), Document("b", "매출"), Document("c", "비용")]
    )
    hits = r.search("매출")
    assert [h.document.id for h in hits] == ["a", "b"]


def test_isolating_retriever_absorbs_raw_text():
    """원문은 경계를 넘지 않는다 — P4 대응이 재사용 가능한 형태로."""
    sink = MemorySink()
    r = KeywordRetriever([Document("d1", "매출 " * 4000)])
    iso = IsolatingRetriever(r, summarize=lambda hits: f"{len(hits)}건 요약" * 10)
    summary, ids = iso.search_summary("매출", tracer=Tracer(sink=sink))
    assert ids == {"d1"}
    assert summarize_trajectory(sink.spans).isolation_ratio > 100


def test_total_isolation_does_not_read_as_zero():
    """상향 토큰이 0이면 격리가 완전했다는 뜻이다. 0을 반환하면 지표가 뒤집힌다."""
    from agentcore.evaluation import TrajectoryMetrics

    assert TrajectoryMetrics(absorbed_tokens=5000, emitted_tokens=0).isolation_ratio == 5000.0
    assert TrajectoryMetrics(absorbed_tokens=0, emitted_tokens=0).isolation_ratio == 0.0


def test_retrieval_is_traced_with_doc_ids():
    sink = MemorySink()
    KeywordRetriever([Document("d1", "x")]).search("x", tracer=Tracer(sink=sink))
    assert sink.spans[0].attributes["doc_ids"] == ["d1"]


# -- 기억: 보존 규정이 적용되는 계층 -----------------------------------------

def test_memory_supports_subject_level_deletion():
    """문서 인덱스와 사용자 기억을 한 저장소에 두면 삭제 요구가 번진다."""
    m = Memory()
    m.put("u1", "k", 1)
    m.put("u1", "k2", 2)
    m.put("u2", "k", 3)
    assert m.forget_subject("u1") == 2
    assert m.get("u1", "k") is None
    assert m.get("u2", "k") == 3


def test_memory_ttl_expires():
    m = Memory()
    m.put("u1", "k", 1, ttl_seconds=-1)
    assert m.get("u1", "k") is None
    assert m.purge_expired() == 1


def test_memory_returns_latest_value():
    m = Memory()
    m.put("u1", "k", "old")
    m.put("u1", "k", "new")
    assert m.get("u1", "k") == "new"


# -- 스킬 ------------------------------------------------------------------

def _skill(name, ns="sql", health=Health.OK, **kw):
    return Skill(
        name=name,
        namespace=ns,
        summary=kw.pop("summary", f"{name} 설명"),
        keywords=kw.pop("keywords", (name,)),
        loader=lambda: f"<{name}>",
        healthcheck=lambda: HealthReport(health, time.time()),
        **kw,
    )


def test_index_exposes_summaries_not_bodies():
    """라이브러리 자체가 컨텍스트를 오염시키면 안 된다."""
    lib = SkillLibrary()
    lib.register(_skill("explain"))
    entries = lib.index()
    assert entries == ["sql/explain — explain 설명"]
    assert "<explain>" not in " ".join(entries)


def test_index_truncation_is_explicit_not_silent():
    lib = SkillLibrary(index_budget=1)
    lib.register(_skill("a"))
    lib.register(_skill("b"))
    entries = lib.index()
    assert "외 1건" in entries[-1]


def test_body_is_lazily_loaded():
    loads: list[int] = []
    s = Skill(
        name="x",
        namespace="ns",
        summary="s",
        loader=lambda: loads.append(1) or "body",
        healthcheck=lambda: HealthReport(Health.OK, time.time()),
    )
    lib = SkillLibrary()
    lib.register(s)
    lib.index()
    assert loads == []          # 인덱싱만으로는 안 불린다
    assert lib.use(s) == "body"
    assert loads == [1]


def test_dead_skill_is_blocked_not_silently_used():
    lib = SkillLibrary()
    dead = Skill(
        name="broken",
        namespace="sql",
        summary="s",
        healthcheck=lambda: (_ for _ in ()).throw(ConnectionError("404")),
    )
    lib.register(dead)
    with pytest.raises(RuntimeError, match="dead"):
        lib.use(dead)


def test_sweep_buckets_by_health():
    lib = SkillLibrary()
    lib.register(_skill("ok1"))
    lib.register(_skill("bad", health=Health.DEAD))
    lib.register(Skill(name="unknown", namespace="ns", summary="s"))
    buckets = lib.sweep()
    assert buckets["ok"] == ["sql/ok1@1.0"]
    assert buckets["dead"] == ["sql/bad@1.0"]
    assert buckets["unknown"] == ["ns/unknown@1.0"]


def test_find_excludes_unusable_skills():
    lib = SkillLibrary()
    lib.register(_skill("alive", keywords=("sql",)))
    lib.register(_skill("zombie", keywords=("sql",), health=Health.DEAD))
    assert [s.name for s in lib.find("sql")] == ["alive"]


def test_versions_coexist_and_latest_wins():
    lib = SkillLibrary()
    lib.register(_skill("explain"))
    lib.register(_skill("explain", version="2.0"))
    assert lib.get("sql", "explain").version == "2.0"
    assert lib.get("sql", "explain", "1.0").version == "1.0"


def test_deprecated_skill_leaves_index_but_stays_gettable():
    lib = SkillLibrary()
    lib.register(_skill("old"))
    lib.deprecate("sql/old@1.0", superseded_by="sql/new@1.0")
    assert lib.index() == []
    assert lib.get("sql", "old").superseded_by == "sql/new@1.0"


def test_retirement_needs_zero_usage():
    lib = SkillLibrary()
    s = _skill("old")
    lib.register(s)
    lib.deprecate("sql/old@1.0")
    assert lib.retirement_candidates() == ["sql/old@1.0"]
    lib.use(s)
    assert lib.retirement_candidates() == []


def test_library_can_delegate_to_retriever():
    """라이브러리가 커지면 검색 계층에 의존하게 된다 — 계층은 독립이 아니다."""
    lib = SkillLibrary()
    lib.register(_skill("explain", keywords=("plan",)))
    lib.register(_skill("profile", keywords=("stats",)))
    lib.retriever = KeywordRetriever(index_documents(lib))
    assert [s.name for s in lib.find("stats")] == ["profile"]
