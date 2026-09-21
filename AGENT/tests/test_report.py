"""궤적 분석 CLI 검증.

별도 계측 없이 로그만으로 4층위 지표가 나와야 한다. 안 나오면
"궤적 로그가 곧 평가 데이터"는 말뿐이다.
"""

from __future__ import annotations

import json

import pytest

from agentcore.report import analyze, main, selections
from agentcore.trace import JsonlSink, SpanKind, Tracer, read_trace, record_isolation, record_usage
from agentcore.broker import Broker, BrokerMode, Candidate


@pytest.fixture
def trace_file(tmp_path):
    path = tmp_path / "trace.jsonl"
    sink = JsonlSink(path)
    tracer = Tracer(sink=sink)
    with tracer.span("pipeline", SpanKind.CONTROL, pattern="route", prevents="비싼 경로"):
        with tracer.span("worker", SpanKind.COMPONENT, component="filter", component_version="1.0"):
            record_usage(input_tokens=8000, output_tokens=100)
            record_isolation(absorbed=7900, emitted=100)
        with tracer.span("core", SpanKind.MODEL):
            record_usage(input_tokens=100, output_tokens=50)
        Broker(mode=BrokerMode.DETERMINISTIC, fixed_team=("core-a",)).select(
            [Candidate("core-a", "fam", 1.0)], task_key="q1", tracer=tracer
        )
    sink.close()
    return path


def test_report_derives_all_four_layers(trace_file):
    rep = analyze(list(read_trace(trace_file)))
    assert rep["trajectory"]["isolation_ratio"] == 79.0
    # 제어 결정 2건(pipeline, broker) / 모델 1건
    assert rep["trajectory"]["control_ratio"] == pytest.approx(2 / 3, abs=1e-3)
    assert rep["trajectory"]["control_decisions"] == 2
    assert rep["system"]["total_tokens"] == 8250   # 워커 8,100 + 코어 150
    assert rep["component"]["invocations"] == {"filter@1.0": 1}
    assert "guardrails" in rep


def test_guardrails_stay_in_their_own_block(trace_file):
    rep = analyze(list(read_trace(trace_file)))
    assert "isolation_ratio" not in rep["guardrails"]
    assert "success_rate" not in rep["guardrails"]


def test_pattern_justifications_survive_to_operations(trace_file):
    """궤적이 '왜 이 패턴이 있는가'를 스스로 문서화한다."""
    rep = analyze(list(read_trace(trace_file)))
    assert rep["pattern_justifications"]["route"] == "비싼 경로"


def test_selections_extracted_for_replay(trace_file):
    assert selections(list(read_trace(trace_file))) == {"q1": ["core-a"]}


def test_cli_json_output(trace_file, capsys):
    assert main([str(trace_file), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["spans"] > 0


def test_cli_human_output(trace_file, capsys):
    assert main([str(trace_file)]) == 0
    out = capsys.readouterr().out
    assert "격리 효과" in out and "집계하지 않음" in out


def test_cli_missing_file_is_an_error(tmp_path, capsys):
    assert main([str(tmp_path / "nope.jsonl")]) == 1
