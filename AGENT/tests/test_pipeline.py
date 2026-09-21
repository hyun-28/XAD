"""예제 파이프라인 회귀 테스트.

외부 호출이 없으므로 전 과정이 결정론적이고, 그대로 회귀 테스트가 된다.
이것이 record-replay 를 깐 이유다 — 회귀 테스트가 가능한 시스템만
컴포넌트를 교체할 수 있고, 교체할 수 없으면 모듈화 논지가 증명되지 않는다.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import pipeline  # noqa: E402

from agentcore.contracts import Role  # noqa: E402
from agentcore.evaluation import (  # noqa: E402
    Case,
    Harness,
    Suite,
    summarize_system,
    summarize_trajectory,
)
from agentcore.trace import MemorySink, Tracer  # noqa: E402

PAYLOAD = {"question": "2021-2023 실적 요약", "period": "2021-2023"}


def run_once():
    sink = MemorySink()
    tracer = Tracer(sink=sink)
    claims = pipeline.run(PAYLOAD, tracer)
    return claims, sink


def test_pipeline_is_deterministic():
    a, _ = run_once()
    b, _ = run_once()
    assert [(c.id, c.text, sorted(c.source_ids)) for c in a] == [
        (c.id, c.text, sorted(c.source_ids)) for c in b
    ]


def test_every_claim_carries_sources():
    claims, _ = run_once()
    assert claims
    assert all(c.source_ids for c in claims)
    assert all(c.support for c in claims)


def test_guardrails_pass_on_happy_path():
    claims, sink = run_once()
    report = pipeline.GUARDRAILS.evaluate(claims, tracer=Tracer(sink=sink))
    assert report.breach_count == 0


def test_context_isolation_actually_happens():
    """워커가 원문을 흡수하지 않으면 이 값이 무너진다."""
    _, sink = run_once()
    assert summarize_trajectory(sink.spans).isolation_ratio > 50


def test_control_stays_in_code_not_in_model():
    """LLM은 판단하고 제어는 코드가 한다 — 궤적에서 검증된다."""
    _, sink = run_once()
    assert summarize_trajectory(sink.spans).control_ratio > 0.8


def test_verifier_down_halts_output():
    sink = MemorySink()
    claims = pipeline.run(
        PAYLOAD,
        Tracer(sink=sink),
        healthy={Role.CORE: {"core-a"}, Role.VERIFIER: set()},
    )
    assert claims == []
    root = next(s for s in sink.spans if s.name == "pipeline")
    assert root.attributes["service_level"] == "halted"


def test_missing_slot_asks_back_instead_of_guessing():
    sink = MemorySink()
    claims = pipeline.run({"question": "2021-2023 실적 요약"}, Tracer(sink=sink))
    assert claims == []
    root = next(s for s in sink.spans if s.name == "pipeline")
    assert root.attributes["asked_back"] is True


def test_broker_selection_is_recorded_for_replay():
    _, sink = run_once()
    sel = next(s for s in sink.spans if s.name == "broker")
    assert sel.attributes["task_key"] == PAYLOAD["question"]
    assert sel.attributes["chosen"] == ["core-a"]


def test_duplicate_sources_do_not_inflate_corroboration():
    """doc1 과 doc3 은 같은 사실을 담는다. 출처 단위 집계가 이를 드러낸다."""
    _, sink = run_once()
    syn = next(s for s in sink.spans if s.name == "synthesize")
    assert syn.attributes["independent_sources"] == 4


def test_harness_runs_the_pipeline():
    suite = Suite(name="pipeline")
    suite.add(Case(id="deep", payload=PAYLOAD, level=4))
    suite.add(Case(id="short", payload={"question": "매출", "period": "2021"}, level=1))
    agg = Harness(suite).run(lambda p, t: pipeline.run(p, t))
    rep = agg.report()
    assert rep["n"] == 2
    assert rep["system"]["tokens_per_task"] > 0
    assert rep["guardrails"]["blocked_cases"] == 0
