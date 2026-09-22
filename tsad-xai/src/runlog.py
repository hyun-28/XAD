"""Run-log header (BRIEF §2.2) and per-call logging (BRIEF B5).

env_header(): first line of every script's output —
    python / numpy / sklearn / TSB-AD (+ source SHA256, no git hash exists for
    the PyPI 1.5 release: DECISIONS D-A0-2, D-B1-1) / OS / hostname.

CallLog: one JSON line per fit/score call in runs/<run_id>/calls.jsonl, failed
calls additionally in runs/<run_id>/failures.jsonl. Nothing is skipped
silently: the caller records the failure and moves on, and `summary()` gives
the failure rate so the script can exit != 0 above its tolerance.
"""
from __future__ import annotations

import hashlib
import json
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# The TSB-AD source files whose behaviour the wrapper depends on
# (docs/TSBAD_INTERNALS.md). Their hashes identify the installed code exactly.
TSB_AD_KEY_FILES = ("models/IForest.py", "model_wrapper.py", "models/feature.py", "utils/utility.py")


def _ver(pkg: str) -> str:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return "not-installed"


def tsb_ad_source_hashes() -> dict[str, str]:
    try:
        import TSB_AD
    except Exception:
        return {}
    root = Path(TSB_AD.__file__).parent
    out = {}
    for rel in TSB_AD_KEY_FILES:
        p = root / rel
        out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()[:12] if p.is_file() else "missing"
    return out


def env_header() -> str:
    h = tsb_ad_source_hashes()
    src = ",".join(f"{k.split('/')[-1]}:{v}" for k, v in h.items()) if h else "not-importable"
    return (
        f"python={sys.version.split()[0]} numpy={_ver('numpy')} "
        f"sklearn={_ver('scikit-learn')} TSB-AD={_ver('TSB-AD')}(PyPI; src {src}) "
        f"OS={platform.system()}-{platform.release()}-{platform.machine()} "
        f"hostname={socket.gethostname()}"
    )


class CallLog:
    """runs/<run_id>/calls.jsonl + failures.jsonl (BRIEF B5)."""

    FIELDS = ("series_num", "detector", "config", "seed", "fit_on", "phase", "n", "runtime_s", "ok", "error")

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._calls = (self.run_dir / "calls.jsonl").open("a")
        self._fails = (self.run_dir / "failures.jsonl").open("a")
        self.n_calls = 0
        self.n_failed = 0
        (self.run_dir / "env.txt").write_text(env_header() + "\n")

    def record(self, *, series_num, detector, config, seed, fit_on, phase, n, runtime_s, ok, error=None,
               **extra) -> None:
        if phase not in ("fit", "score"):
            raise ValueError(f"phase must be fit|score, got {phase!r}")
        row = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
               "series_num": series_num, "detector": detector, "config": config, "seed": seed,
               "fit_on": fit_on, "phase": phase, "n": int(n), "runtime_s": float(runtime_s),
               "ok": bool(ok), "error": None if error is None else str(error), **extra}
        line = json.dumps(row, default=str) + "\n"
        self._calls.write(line)
        self._calls.flush()
        self.n_calls += 1
        if not ok:
            self._fails.write(line)
            self._fails.flush()
            self.n_failed += 1

    def timed(self, fn, **meta):
        """Run fn(), record it, re-raise nothing: returns (result_or_None, error_or_None)."""
        t0 = time.perf_counter()
        try:
            out = fn()
            self.record(runtime_s=time.perf_counter() - t0, ok=True, **meta)
            return out, None
        except Exception as e:
            self.record(runtime_s=time.perf_counter() - t0, ok=False, error=f"{type(e).__name__}: {e}", **meta)
            return None, e

    def summary(self) -> dict:
        rate = self.n_failed / self.n_calls if self.n_calls else 0.0
        return {"n_calls": self.n_calls, "n_failed": self.n_failed, "failure_rate": rate}

    def close(self) -> None:
        self._calls.close()
        self._fails.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
