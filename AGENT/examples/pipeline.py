"""엔드투엔드 예제 — 도메인 중립 조사 파이프라인.

1부 토폴로지를 그대로 구현한다.

    [1] 가용성 확인      정족수 · 검증기 다운이면 출력 차단
    [2] 충분성 게이트    필요한 정보가 모였는가. 없으면 되묻는다
    [3] 라우팅          쉬운 질의가 비싼 경로를 타지 않게. 저확신은 폴백
    [4] 오케스트레이션   예산 안에서 워커 생성. 워커 = 컨텍스트 격리 장치
    [5] 합성            코어. 출처 단위 중복 제거 후 모순 해소
    [6] 근거 검증 루프   외부 신호(원문 대조) 기반 E-O. 실패 항목만 재작성
    [7] 가드레일        집계되지 않는 통과/실패 관문

외부 호출은 없다 — 전 과정이 결정론적이므로 그대로 회귀 테스트가 된다.
"""

from __future__ import annotations

import json
from pathlib import Path

from agentcore.broker import Broker, BrokerMode, Candidate, QuorumPolicy, ServiceLevel
from agentcore.contracts import Role
from agentcore.control import (
    Budget,
    Classification,
    DeterministicSignal,
    Finding,
    SignalResult,
    SlotSpec,
    Subtask,
    SufficiencyGate,
    orchestrate,
    refine,
    route,
)
from agentcore.evaluation import (
    Aggregate,
    Case,
    Harness,
    Suite,
    summarize_system,
    summarize_trajectory,
)
from agentcore.trace import JsonlSink, MemorySink, MultiSink, SpanKind, Tracer, record_isolation, record_usage
from agentcore.verify import Claim, GroundingVerifier, Guardrail, GuardrailSet

# --------------------------------------------------------------------------
# 모사 코퍼스. 각 문서는 토큰이 크다 — 격리 효과를 보이기 위한 것.
# --------------------------------------------------------------------------

CORPUS = {
    "doc1": ("2021 매출은 전년 대비 4% 증가했다. " * 120, "revenue"),
    "doc2": ("2022 비용은 전년 대비 7% 감소했다. " * 120, "cost"),
    "doc3": ("2021 매출 증가는 4%로 집계되었다. " * 120, "revenue"),   # doc1 과 같은 사실
    "doc4": ("2023 인력은 12% 늘었다. " * 120, "headcount"),
}


def search(topic: str) -> list[tuple[str, str]]:
    return [(doc_id, text) for doc_id, (text, tag) in CORPUS.items() if tag == topic]


# --------------------------------------------------------------------------
# [4] 워커 — 컨텍스트 격리 장치
# --------------------------------------------------------------------------

def worker(subtask: Subtask) -> Finding[str]:
    """원문을 흡수하고 요약만 올려보낸다.

    이것이 P4(컨텍스트 오염)에 대한 구조적 대응이다. 원문 수만 토큰이
    코어 컨텍스트로 올라가지 않는다.
    """
    hits = search(subtask.payload)
    raw_tokens = sum(len(t) // 4 for _, t in hits)
    summary = f"{subtask.payload}: {len(hits)}건 확인"
    emitted = len(summary) // 4

    record_usage(input_tokens=raw_tokens, output_tokens=emitted)
    record_isolation(absorbed=raw_tokens - emitted, emitted=emitted)

    return Finding(
        value=summary,
        source_ids=frozenset(doc_id for doc_id, _ in hits),
        worker=subtask.name,
    )


def plan(question: str) -> list[Subtask]:
    """하위 질문 분해. 각 워커에 배타적 영역을 명시한다."""
    return [
        Subtask("revenue", "revenue", scope="매출 관련 문서만"),
        Subtask("cost", "cost", scope="비용 관련 문서만"),
        Subtask("headcount", "headcount", scope="인력 관련 문서만"),
    ]


# --------------------------------------------------------------------------
# [6] 외부 검증 신호 — 원문 대조
# --------------------------------------------------------------------------

def entails(claim_text: str, support: tuple[str, ...]) -> bool:
    """주장이 근거 구절에서 도출되는가. 객관적이고, 조밀하고, 싸다."""
    key = claim_text.split(":")[0].strip()
    return any(key in s for s in support)


GROUNDING = GroundingVerifier(entails=entails)


def grounding_signal(claims: list[Claim]) -> SignalResult:
    result = GROUNDING.check(claims)
    return SignalResult(
        passed=result.passed,
        failures=result.failure_loci,
        detail=f"{len(result.violations)}건 미검증",
    )


# --------------------------------------------------------------------------
# 파이프라인
# --------------------------------------------------------------------------

SUFFICIENCY = SufficiencyGate([SlotSpec("question"), SlotSpec("period")])

GUARDRAILS = GuardrailSet(
    [
        Guardrail(
            "unverified_claim",
            lambda c, _: bool(getattr(c, "support", ())),
            "근거 없는 사실 주장",
        ),
        Guardrail(
            "unattributed",
            lambda c, _: bool(getattr(c, "source_ids", frozenset())),
            "출처가 없는 주장",
        ),
    ]
)

QUORUM = QuorumPolicy(
    min_by_role={Role.CORE: 1},
    required_roles=frozenset({Role.VERIFIER, Role.CORE}),
)

BROKER = Broker(mode=BrokerMode.DETERMINISTIC, fixed_team=("core-a",), team_size=1)

CANDIDATES = [
    Candidate("core-a", "family-a", cost=1.0, vram_mb=16000, value=0.9),
    Candidate("core-b", "family-b", cost=0.8, vram_mb=9000, value=0.82),
]


def run(payload: dict, tracer: Tracer, *, healthy: dict | None = None) -> list[Claim]:
    healthy = healthy or {Role.CORE: {"core-a"}, Role.VERIFIER: {"grounding"}}

    with tracer.span("pipeline", SpanKind.CONTROL) as root:
        # [1] 가용성
        decision = QUORUM.decide(healthy)
        root.attributes["service_level"] = decision.level.value
        if not decision.can_proceed:
            root.attributes["halt_reason"] = decision.reason
            return []

        # [2] 충분성
        if not SUFFICIENCY.check(payload, tracer=tracer):
            root.attributes["asked_back"] = True
            return []

        # 코어 선택 — 결정론 모드. 선택은 궤적에 남는다.
        BROKER.select(CANDIDATES, task_key=payload["question"], tracer=tracer)

        # [3] 라우팅
        def classify(p: dict) -> Classification:
            deep = len(p["question"]) > 12
            return Classification(
                route="deep" if deep else "shallow",
                confidence=0.9,
                reason="질문 길이 기반",
            )

        def shallow(p: dict) -> list[Claim]:
            with tracer.span("shallow_lookup", SpanKind.CONTROL):
                return [Claim("s1", "revenue: 1건 확인", ("revenue: 1건 확인",), frozenset({"doc1"}))]

        def deep(p: dict) -> list[Claim]:
            # [4] 오케스트레이션 — 반드시 예산 안에서
            budget = Budget(max_tokens=200_000, max_workers=3, max_depth=2)
            joined = orchestrate(
                plan,
                worker,
                p["question"],
                prevents="하위작업 개수를 사전에 알 수 없고, 원문이 코어 컨텍스트를 오염시킴",
                budget=budget,
                tracer=tracer,
            )

            # [5] 합성 — 코어. 독립 출처 수를 함께 전달한다.
            with tracer.span("synthesize", SpanKind.MODEL) as span:
                span.attributes["independent_sources"] = joined.independent_sources
                span.attributes["duplicate_rate"] = joined.duplicate_rate
                record_usage(input_tokens=sum(len(f.value) // 4 for f in joined.findings), output_tokens=40)
                claims = [
                    Claim(
                        id=f"c{i}",
                        text=f.value,
                        support=(f.value,),
                        source_ids=f.source_ids,
                    )
                    for i, f in enumerate(joined.findings)
                ]

            # [6] 근거 검증 루프 — 외부 신호만 받는다
            def regenerate(base: list[Claim], failures: tuple[str, ...]) -> list[Claim]:
                if not failures:
                    return base
                return [c for c in base if c.id not in failures]

            refined, result, rounds = refine(
                regenerate,
                DeterministicSignal(grounding_signal),
                claims,
                prevents="검증되지 않은 인용이 출력됨",
                max_rounds=2,
                tracer=tracer,
            )
            return refined

        return route(
            classify,
            {"shallow": shallow, "deep": deep},
            payload,
            prevents="쉬운 질의가 심층 경로를 타서 비용이 수십 배가 됨",
            fallback="deep",
            tracer=tracer,
        )


def main() -> None:
    out_dir = Path("runs")
    out_dir.mkdir(exist_ok=True)
    mem = MemorySink()
    tracer = Tracer(sink=MultiSink(mem, JsonlSink(out_dir / "trace.jsonl")))

    payload = {"question": "2021-2023 실적 요약", "period": "2021-2023"}
    claims = run(payload, tracer)

    traj = summarize_trajectory(mem.spans)
    sys_m = summarize_system(mem.spans)
    guard = GUARDRAILS.evaluate(claims, tracer=tracer)

    print("=" * 66)
    print("주장", len(claims), "건")
    for c in claims:
        print(f"  · {c.text}  ← {sorted(c.source_ids)}")
    print("-" * 66)
    print(f"총 토큰            {sys_m.total_tokens:,}")
    print(f"격리 효과          {traj.isolation_ratio:.1f} : 1   (흡수 {traj.absorbed_tokens:,} / 상향 {traj.emitted_tokens:,})")
    print(f"제어 비율          {traj.control_ratio:.2f}       (코드 제어 결정 / 전체 결정)")
    print(f"중복 탐색률        {traj.duplicate_rate:.2f}")
    print(f"모델 호출          {traj.model_calls}")
    print(f"가드레일 위반      {guard.breach_count}건 {guard.counts_by_rail() or ''}")
    print("=" * 66)

    # 검증기 다운 → 출력 차단
    halted = run(payload, Tracer(sink=MemorySink()), healthy={Role.CORE: {"core-a"}, Role.VERIFIER: set()})
    print(f"검증기 다운 시 출력: {len(halted)}건 (fail closed)")

    # 슬롯 미충족 → 되묻기
    asked = run({"question": "2021-2023 실적 요약"}, Tracer(sink=MemorySink()))
    print(f"슬롯 미충족 시 출력: {len(asked)}건 (되묻기)")


if __name__ == "__main__":
    main()
