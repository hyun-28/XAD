"""궤적 분석 CLI.

    python -m agentcore.report runs/trace.jsonl
    python -m agentcore.report runs/trace.jsonl --json
    python -m agentcore.report runs/trace.jsonl --selections

궤적 로그가 곧 평가 데이터라는 주장의 실현. 별도 계측 없이 로그만으로
4층위 지표가 나오지 않으면, 그 주장은 말뿐이다.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .trace import read_trace
from .trace.span import SpanKind, SpanStatus


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_trace: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_trace[r["trace_id"]].append(r)

    kinds = Counter(r["kind"] for r in rows)
    statuses = Counter(r["status"] for r in rows)
    components = Counter(
        f"{r['component']}@{r['component_version']}"
        for r in rows
        if r.get("component") and r.get("component_version")
    )

    roots = [r for r in rows if r.get("parent_id") is None]
    absorbed = sum(r.get("absorbed_tokens", 0) for r in rows)
    emitted = sum(r.get("emitted_tokens", 0) for r in rows)

    # 브로커 선택도 코드가 내린 제어 결정이므로 함께 센다.
    control = kinds.get(SpanKind.CONTROL.value, 0) + kinds.get(SpanKind.SELECTION.value, 0)
    model = kinds.get(SpanKind.MODEL.value, 0)

    patterns = Counter(
        r["attributes"]["pattern"] for r in rows if "pattern" in (r.get("attributes") or {})
    )
    prevents = {
        r["attributes"]["pattern"]: r["attributes"].get("prevents", "")
        for r in rows
        if "pattern" in (r.get("attributes") or {})
    }

    return {
        "traces": len(by_trace),
        "spans": len(rows),
        "component": {
            "invocations": dict(components.most_common()),
            "out_of_envelope": statuses.get(SpanStatus.OUT_OF_ENVELOPE.value, 0),
            "errors": statuses.get(SpanStatus.ERROR.value, 0),
        },
        "trajectory": {
            "isolation_ratio": round(absorbed / max(emitted, 1), 2),
            "absorbed_tokens": absorbed,
            "emitted_tokens": emitted,
            "control_ratio": round(control / (control + model), 3) if (control + model) else 0.0,
            "model_calls": model,
            "control_decisions": control,
            "patterns": dict(patterns),
            "retries": sum(1 for r in rows if r["name"].startswith("refine:round")),
        },
        "system": {
            "total_tokens": sum(r["usage"]["input_tokens"] + r["usage"]["output_tokens"] for r in roots),
            "wall_seconds": round(sum(r["usage"]["wall_seconds"] for r in roots), 4),
            "peak_vram_mb": max(
                (r["usage"].get("peak_vram_mb") or 0.0 for r in rows), default=0.0
            ),
            "component_versions": sorted(components),
        },
        "guardrails": {
            "blocked_spans": statuses.get(SpanStatus.GUARDRAIL_BLOCKED.value, 0),
            "note": "가드레일은 꼬리 지표이므로 다른 지표와 집계하지 않는다",
        },
        "pattern_justifications": prevents,
    }


def selections(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    """브로커 선택 추출 — 재현 모드의 입력."""
    out: dict[str, list[str]] = {}
    for r in rows:
        if r["kind"] != SpanKind.SELECTION.value:
            continue
        attrs = r.get("attributes") or {}
        if attrs.get("task_key") and attrs.get("chosen"):
            out[attrs["task_key"]] = list(attrs["chosen"])
    return out


def _fmt(report: dict[str, Any]) -> str:
    t, s, c = report["trajectory"], report["system"], report["component"]
    lines = [
        "=" * 62,
        f"궤적 {report['traces']}건 · 스팬 {report['spans']}개",
        "-" * 62,
        "[궤적]",
        f"  격리 효과        {t['isolation_ratio']:>10} : 1   (흡수 {t['absorbed_tokens']:,} / 상향 {t['emitted_tokens']:,})",
        f"  제어 비율        {t['control_ratio']:>10}       (코드 결정 {t['control_decisions']} / 모델 {t['model_calls']})",
        f"  재시도           {t['retries']:>10}",
        f"  패턴             {t['patterns'] or '없음'}",
        "",
        "[시스템]",
        f"  총 토큰          {s['total_tokens']:>10,}",
        f"  벽시계           {s['wall_seconds']:>10} s",
        f"  최대 VRAM        {s['peak_vram_mb']:>10} MB",
        "",
        "[컴포넌트]",
        f"  범위 이탈        {c['out_of_envelope']:>10}",
        f"  오류             {c['errors']:>10}",
    ]
    for ref, n in c["invocations"].items():
        lines.append(f"    {ref:<34} {n:>6}")
    lines += [
        "",
        "[가드레일]  ※ 위 지표와 집계하지 않음",
        f"  차단             {report['guardrails']['blocked_spans']:>10}",
    ]
    if report["pattern_justifications"]:
        lines += ["", "[패턴 정당화]"]
        for pat, why in report["pattern_justifications"].items():
            lines.append(f"  {pat}: {why}")
    lines.append("=" * 62)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agentcore.report", description="궤적 분석")
    ap.add_argument("trace", type=Path, help="JSONL 궤적 파일")
    ap.add_argument("--json", action="store_true", help="JSON 출력")
    ap.add_argument("--selections", action="store_true", help="브로커 선택만 추출")
    args = ap.parse_args(argv)

    if not args.trace.exists():
        print(f"파일 없음: {args.trace}", file=sys.stderr)
        return 1

    rows = list(read_trace(args.trace))
    if not rows:
        print("빈 궤적입니다.", file=sys.stderr)
        return 1

    if args.selections:
        print(json.dumps(selections(rows), ensure_ascii=False, indent=2))
        return 0

    report = analyze(rows)
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else _fmt(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
