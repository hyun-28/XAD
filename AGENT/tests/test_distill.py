"""증류 파이프라인 검증.

코어를 동결한 채 전용 모델을 얻는 경로. 사다리 6단으로 내려가는 것은
망각 위험과 회귀 검증 부채를 지는 일이므로, 진입 조건이 게이트여야 한다.
"""

from __future__ import annotations

import pytest

from agentcore.contracts import InterfaceSpec, OperatingEnvelope
from agentcore.distill import (
    DistillationSet,
    EntryCheck,
    EntryConditionsUnmet,
    LabeledExample,
    check_entry,
    label_with_core,
    publish_contract,
)
from agentcore.evaluation import ComponentScore
from agentcore.trace import JsonlSink, MemorySink, SpanKind, Tracer


SCORE = ComponentScore("f", "1.0", "es", 20, 0.93, 1.0, 0, tokens=200, wall_seconds=0.2)
GOOD_ENTRY = EntryCheck(True, True, True, observed_calls=900, spec_changes=0, detail="ok")


# -- 진입 조건 -------------------------------------------------------------

def test_all_three_conditions_required():
    assert GOOD_ENTRY.satisfied
    assert not EntryCheck(False, True, True).satisfied
    assert not EntryCheck(True, False, True).satisfied
    assert not EntryCheck(True, True, False).satisfied


def test_failures_name_the_specific_reason():
    e = EntryCheck(False, False, False, observed_calls=12, spec_changes=3)
    reasons = " ".join(e.failures())
    assert "재학습이 반복" in reasons
    assert "회수할 수 없습니다" in reasons
    assert "손라벨링" in reasons


def test_publish_is_blocked_when_conditions_unmet():
    with pytest.raises(EntryConditionsUnmet):
        publish_contract(
            name="f",
            version="1.0",
            score=SCORE,
            envelope=OperatingEnvelope("all"),
            interface=InterfaceSpec("a", "b"),
            fallback="bm25",
            teacher="core@v1",
            entry=EntryCheck(True, False, True, observed_calls=10),
        )


def test_t2_cannot_be_published_without_a_fallback():
    """T1 없이 발행된 T2 는 회귀했을 때 되돌아갈 곳이 없다."""
    with pytest.raises(ValueError, match="fallback"):
        publish_contract(
            name="f",
            version="1.0",
            score=SCORE,
            envelope=OperatingEnvelope("all"),
            interface=InterfaceSpec("a", "b"),
            fallback="",
            teacher="core@v1",
            entry=GOOD_ENTRY,
        )


def test_published_contract_is_parametric_and_partially_amortized():
    from agentcore.contracts import Amortization

    c = publish_contract(
        name="f",
        version="1.0",
        score=SCORE,
        envelope=OperatingEnvelope("all"),
        interface=InterfaceSpec("a", "b"),
        fallback="bm25",
        teacher="core@v1",
        entry=GOOD_ENTRY,
    )
    assert c.is_parametric
    assert c.ladder_rung == 6
    assert c.amortization is Amortization.PARTIAL
    assert c.fallback == "bm25"
    assert "core@v1" in c.performance.notes


def test_entry_check_reads_frequency_from_trace(tmp_path):
    """빈도와 스펙 안정성은 의견이 아니라 로그에서 나오는 숫자다."""
    path = tmp_path / "t.jsonl"
    sink = JsonlSink(path)
    tracer = Tracer(sink=sink)
    for _ in range(5):
        with tracer.span("filter", SpanKind.COMPONENT, component="filter", component_version="1.0"):
            pass
    sink.close()

    lax = check_entry(path, "filter", min_calls=3)
    assert lax.high_frequency and lax.stable_spec and lax.observed_calls == 5

    strict = check_entry(path, "filter", min_calls=100)
    assert not strict.high_frequency


def test_spec_drift_is_detected(tmp_path):
    path = tmp_path / "t.jsonl"
    sink = JsonlSink(path)
    tracer = Tracer(sink=sink)
    for version in ("1.0", "2.0", "3.0"):
        with tracer.span("filter", SpanKind.COMPONENT, component="filter", component_version=version):
            pass
    sink.close()
    check = check_entry(path, "filter", min_calls=1)
    assert check.spec_changes == 2
    assert not check.stable_spec


# -- 라벨 생성 -------------------------------------------------------------

def test_core_labels_without_touching_core_weights():
    ds = label_with_core(
        ["ab", "cd"],
        teacher=str.upper,
        task="filter",
        teacher_id="core@v1",
        verify=lambda p, l: l.isupper(),
        tracer=Tracer(sink=MemorySink()),
    )
    assert ds.teacher == "core@v1"
    assert ds.verified_ratio == 1.0


def test_unverified_labels_are_excluded_from_training():
    """교사가 틀린 라벨을 학생이 배우면 그 오류는 가중치에 들어간다."""
    ds = label_with_core(
        ["ab", "cd"],
        teacher=lambda s: s if s == "ab" else s.upper(),
        task="f",
        teacher_id="core@v1",
        verify=lambda p, l: l.isupper(),
    )
    assert len(ds.examples) == 2
    assert len(ds.verified_only()) == 1
    assert ds.verified_ratio == 0.5


def test_no_verifier_means_nothing_is_trusted():
    ds = label_with_core(["ab"], teacher=str.upper, task="f", teacher_id="c")
    assert ds.verified_only() == []


def test_duplicates_are_removed():
    ds = label_with_core(
        ["ab", "ab", "cd"],
        teacher=str.upper,
        task="f",
        teacher_id="c",
        verify=lambda p, l: True,
    )
    assert len(ds.dedup()) == 2


def test_holdout_is_cut_before_training_not_after():
    """나중에 자르면 학습에 쓴 것으로 평가하게 된다."""
    ds = DistillationSet(task="f", teacher="c")
    for i in range(10):
        ds.add(LabeledExample(payload=f"p{i}", label=f"L{i}", teacher="c", verified=True))
    train, holdout = ds.split(holdout=0.3)
    assert len(holdout.items) == 3
    assert len(train) == 7
    train_payloads = {e.payload for e in train}
    holdout_payloads = {i.payload for i in holdout.items}
    assert train_payloads.isdisjoint(holdout_payloads)


def test_written_jsonl_contains_only_verified_examples(tmp_path):
    ds = label_with_core(
        ["ab", "cd"],
        teacher=lambda s: s if s == "ab" else s.upper(),
        task="f",
        teacher_id="c",
        verify=lambda p, l: l.isupper(),
    )
    path = ds.write(tmp_path / "train.jsonl")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
