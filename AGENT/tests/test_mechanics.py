"""구성요소 동작 검증."""

from __future__ import annotations

import pytest

from agentcore.contracts import (
    Amortization,
    ComponentContract,
    FunctionComponent,
    InterfaceSpec,
    OperatingEnvelope,
    PerformanceSpec,
    Registry,
    Role,
)
from agentcore.control import (
    Budget,
    Classification,
    Finding,
    PreflightGate,
    SlotSpec,
    Estimate,
    SufficiencyGate,
    Subtask,
    join_findings,
    orchestrate,
    route,
)
from agentcore.evaluation import (
    Case,
    GoldenAction,
    Harness,
    Suite,
    summarize_system,
    summarize_trajectory,
)
from agentcore.models import Cassette, CassetteMiss, EchoLLM, Message, RecordingLLM, ReplayLLM
from agentcore.trace import MemorySink, SpanKind, SpanStatus, Tracer, record_isolation, record_usage
from agentcore.verify import Claim, GroundingVerifier, Guardrail, GuardrailSet, VerifierStack, RuleVerifier


def make_component(name="f", version="1.0", limit=10, rung=2):
    ct = ComponentContract(
        name=name,
        version=version,
        role=Role.SPECIALIST,
        interface=InterfaceSpec("list", "list"),
        performance=PerformanceSpec("es", {"acc": 0.9}),
        envelope=OperatingEnvelope(f"{limit}건 이하", lambda p: len(p) <= limit),
        ladder_rung=rung,
        amortization=Amortization.PARTIAL,
    )
    return FunctionComponent(ct, lambda xs: [x for x in xs if x])


# -- 궤적 ------------------------------------------------------------------

def test_child_usage_merges_into_parent():
    sink = MemorySink()
    t = Tracer(sink=sink)
    with t.span("root"):
        with t.span("child", SpanKind.MODEL):
            record_usage(input_tokens=100, output_tokens=50)
    root = next(s for s in sink.spans if s.name == "root")
    assert root.usage.total_tokens == 150


def test_parent_child_links_survive_nesting():
    sink = MemorySink()
    t = Tracer(sink=sink)
    with t.span("a"):
        with t.span("b"):
            with t.span("c"):
                pass
    by_name = {s.name: s for s in sink.spans}
    assert by_name["c"].parent_id == by_name["b"].span_id
    assert by_name["b"].parent_id == by_name["a"].span_id
    assert by_name["a"].parent_id is None


def test_exception_marks_span_error():
    sink = MemorySink()
    t = Tracer(sink=sink)
    with pytest.raises(ValueError):
        with t.span("boom"):
            raise ValueError("nope")
    assert sink.spans[0].status is SpanStatus.ERROR
    assert "ValueError" in (sink.spans[0].error or "")


def test_isolation_ratio_discriminates_architecture():
    """모놀리식은 흡수가 0이므로 격리 효과도 0이다."""
    sink = MemorySink()
    t = Tracer(sink=sink)
    with t.span("root"):
        with t.span("worker", SpanKind.COMPONENT):
            record_isolation(absorbed=9000, emitted=150)
    assert summarize_trajectory(sink.spans).isolation_ratio == 60.0

    mono = MemorySink()
    t2 = Tracer(sink=mono)
    with t2.span("root"):
        record_isolation(absorbed=0, emitted=9150)
    assert summarize_trajectory(mono.spans).isolation_ratio == 0.0


# -- 계약 ------------------------------------------------------------------

def test_out_of_envelope_is_explicit_not_silent():
    c = make_component(limit=2)
    r = c([1, 2, 3])
    assert r.status is SpanStatus.OUT_OF_ENVELOPE
    assert not r.ok
    with pytest.raises(RuntimeError):
        r.unwrap()


def test_registry_versions_coexist():
    reg = Registry()
    reg.register(make_component(version="1.0"))
    reg.register(make_component(version="2.0"))
    assert reg.versions("f") == ["1.0", "2.0"]
    assert reg.get("f").contract.version == "2.0"   # 기본은 최신
    assert reg.get("f", "1.0").contract.version == "1.0"


def test_registry_rejects_duplicate_version():
    reg = Registry()
    reg.register(make_component(version="1.0"))
    with pytest.raises(ValueError, match="불변"):
        reg.register(make_component(version="1.0"))


def test_retirement_candidates_are_unused_versions():
    reg = Registry()
    reg.register(make_component(version="1.0"))
    reg.register(make_component(version="2.0"))
    reg.get("f", "2.0")
    assert reg.retirement_candidates() == ["f@1.0"]


def test_parametric_components_are_listed():
    reg = Registry()
    reg.register(make_component(name="small", rung=6))
    reg.register(make_component(name="index", rung=4))
    assert reg.parametric_components() == ["small@1.0"]


def test_performance_spec_regression_floor():
    spec = PerformanceSpec("es", {"acc": 0.91, "recall": 0.80})
    assert spec.meets({"acc": 0.90})
    assert not spec.meets({"acc": 0.95})


# -- 패턴 ------------------------------------------------------------------

def test_route_falls_back_on_low_confidence():
    calls: list[str] = []
    routes = {
        "cheap": lambda p: calls.append("cheap") or "c",
        "deep": lambda p: calls.append("deep") or "d",
    }
    route(
        lambda p: Classification("cheap", confidence=0.2),
        routes,
        "q",
        prevents="쉬운 질의가 심층 경로를 탐",
        fallback="deep",
        min_confidence=0.6,
    )
    assert calls == ["deep"]


def test_route_requires_fallback_in_routes():
    with pytest.raises(ValueError, match="fallback"):
        route(
            lambda p: Classification("a"),
            {"a": lambda p: p},
            "q",
            prevents="x",
            fallback="missing",
        )


def test_join_dedups_by_source_not_by_text():
    """워커 넷이 같은 원출처를 찾아오면 독립 출처는 넷이 아니라 하나다."""
    j = join_findings(
        [
            Finding("매출 증가", frozenset({"doc1"}), "w1"),
            Finding("매출이 늘었다", frozenset({"doc1"}), "w2"),
            Finding("비용 감소", frozenset({"doc2"}), "w3"),
        ]
    )
    assert j.independent_sources == 2
    assert len(j.findings) == 2


def test_orchestrate_respects_worker_ceiling():
    budget = Budget(max_workers=2, max_depth=3)
    seen: list[str] = []

    def worker(st: Subtask) -> Finding[str]:
        seen.append(st.name)
        return Finding(st.name, frozenset({st.name}))

    orchestrate(
        lambda p: [Subtask(f"s{i}", i, scope=f"영역{i}") for i in range(5)],
        worker,
        "q",
        prevents="하위작업 개수를 사전에 알 수 없음",
        budget=budget,
    )
    assert len(seen) == 2


def test_orchestrate_records_exclusivity_scopes():
    sink = MemorySink()
    orchestrate(
        lambda p: [Subtask("a", 1, scope="2020-2021"), Subtask("b", 2, scope="2022-2023")],
        lambda st: Finding(st.name, frozenset({st.name})),
        "q",
        prevents="동적 하위작업",
        budget=Budget(max_workers=4),
        tracer=Tracer(sink=sink),
    )
    par = next(s for s in sink.spans if s.name == "parallel")
    assert par.attributes["exclusivity"] == {"a": "2020-2021", "b": "2022-2023"}


# -- 게이트 ----------------------------------------------------------------

def test_sufficiency_gate_lists_missing_slots():
    g = SufficiencyGate([SlotSpec("table"), SlotSpec("metric"), SlotSpec("period", required=False)])
    s = g.check({"table": "sales"})
    assert not s and s.missing == ("metric",)


def test_preflight_gate_blocks_before_side_effect():
    executed: list[int] = []
    gate = PreflightGate(lambda p: Estimate(cost=p, unit="rows"), ceiling=100)
    for payload in (50, 500):
        ok, est = gate.admits(payload)
        if ok:
            executed.append(payload)
    assert executed == [50]


# -- 검증 ------------------------------------------------------------------

def test_verifier_stack_short_circuits_before_judgment():
    """형식이 깨진 출력의 품질을 논하는 것은 의미가 없고 비싸다."""
    judged: list[int] = []
    from agentcore.verify import JudgmentVerifier, VerificationResult, VerificationTier

    stack = VerifierStack(
        decidable=[RuleVerifier([("nonempty", lambda c, _: bool(c), "비어 있음")])],
        judgment=[
            JudgmentVerifier(
                lambda c, ctx: judged.append(1)
                or VerificationResult(True, VerificationTier.JUDGMENT)
            )
        ],
    )
    r = stack.verify([])
    assert not r.passed and judged == []
    stack.verify([1])
    assert judged == [1]


def test_grounding_flags_unsupported_and_unentailed():
    gv = GroundingVerifier(entails=lambda t, sup: "yes" in " ".join(sup))
    r = gv([Claim("c1", "a", ("yes",)), Claim("c2", "b", ()), Claim("c3", "c", ("no",))])
    assert r.failure_loci == ("c2", "c3")


def test_guardrails_report_counts_not_rates():
    gs = GuardrailSet([Guardrail("v", lambda c, _: c.get("ok", False), "미검증")])
    rep = gs.evaluate([{"ok": True}, {"ok": False}, {"ok": False}])
    assert rep.breach_count == 2
    assert rep.counts_by_rail() == {"v": 2}
    assert rep.blocked


# -- record-replay ---------------------------------------------------------

def test_cassette_roundtrip_is_deterministic(tmp_path):
    cas = Cassette(path=tmp_path / "c.json", environment={"schema": "v1"})
    rec = RecordingLLM(EchoLLM(), cas)
    out = rec.complete([Message("user", "hello")])
    cas.save()
    rp = ReplayLLM(Cassette.load(tmp_path / "c.json"), model_id="echo-1")
    assert rp.complete([Message("user", "hello")]).text == out.text


def test_cassette_miss_is_loud(tmp_path):
    cas = Cassette(path=tmp_path / "c.json")
    cas.save()
    rp = ReplayLLM(Cassette.load(tmp_path / "c.json"), model_id="echo-1")
    with pytest.raises(CassetteMiss):
        rp.complete([Message("user", "unseen")])


def test_environment_drift_is_detected(tmp_path):
    cas = Cassette(path=tmp_path / "c.json", environment={"schema": "v1"})
    cas.save()
    loaded = Cassette.load(tmp_path / "c.json")
    loaded.assert_environment({"schema": "v1"})
    with pytest.raises(RuntimeError, match="환경"):
        loaded.assert_environment({"schema": "v2"})


# -- 하네스 ----------------------------------------------------------------

def test_harness_scores_output_and_path():
    suite = Suite(name="s")
    suite.add(
        Case(
            id="ok",
            payload=2,
            expected=4,
            golden=GoldenAction(steps=("double",)),
        )
    )

    def system(payload, tracer):
        with tracer.span("double"):
            return payload * 2

    agg = Harness(suite).run(system)
    assert agg.success_rate == 1.0
    assert agg.results[0].metrics[__import__("agentcore.evaluation", fromlist=["Layer"]).Layer.TRAJECTORY]["path_match"] == 1.0


def test_harness_blocks_on_guardrail():
    suite = Suite()
    suite.add(Case(id="g", payload=1, expected={"ok": False}))
    gs = GuardrailSet([Guardrail("ok", lambda c, _: c.get("ok", False), "위반")])
    agg = Harness(suite, guardrails=gs).run(lambda p, t: {"ok": False})
    assert not agg.results[0].passed
    assert agg.report()["guardrails"]["blocked_cases"] == 1


def test_suite_coverage_reports_failure_paths():
    s = Suite()
    s.add(Case(id="a", payload=1, level=1))
    s.add(Case(id="b", payload=1, level=1, failure_path=True))
    assert s.coverage()["failure_path_ratio"] == 0.5


def test_report_keeps_guardrails_separate_from_metrics():
    """가드레일이 성공률과 같은 블록에 들어가면 언젠가 함께 평균된다."""
    suite = Suite()
    suite.add(Case(id="a", payload=1, expected=1))
    agg = Harness(suite).run(lambda p, t: 1)
    rep = agg.report()
    assert "guardrails" in rep and "success_rate" not in rep["guardrails"]
    assert "breaches" in rep["guardrails"]
