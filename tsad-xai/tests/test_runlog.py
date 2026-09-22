"""BRIEF B5: calls.jsonl / failures.jsonl and the env header."""
import json

from src.runlog import CallLog, env_header


def test_header_has_every_required_field():
    h = env_header()
    for key in ("python=", "numpy=", "sklearn=", "TSB-AD=", "OS=", "hostname="):
        assert key in h


def test_calllog_records_and_counts(tmp_path):
    with CallLog(tmp_path / "run") as log:
        meta = dict(series_num=1, detector="IForest", config={"a": 1}, seed=0, fit_on="train_prefix")
        out, err = log.timed(lambda: 42, phase="fit", n=10, **meta)
        assert out == 42 and err is None
        out, err = log.timed(lambda: 1 / 0, phase="score", n=10, **meta)
        assert out is None and isinstance(err, ZeroDivisionError)
        assert log.summary() == {"n_calls": 2, "n_failed": 1, "failure_rate": 0.5}
    calls = [json.loads(l) for l in (tmp_path / "run" / "calls.jsonl").read_text().splitlines()]
    fails = [json.loads(l) for l in (tmp_path / "run" / "failures.jsonl").read_text().splitlines()]
    assert len(calls) == 2 and len(fails) == 1
    assert calls[0]["ok"] is True and fails[0]["error"].startswith("ZeroDivisionError")
    assert (tmp_path / "run" / "env.txt").read_text().startswith("python=")
