"""agentcore 튜토리얼 — 10단계로 에이전트 하나 세우기.

    python examples/tutorial.py            전체 실행
    python examples/tutorial.py 6          6단계만 실행

각 단계는 **먼저 무엇이 깨지는지 보여주고, 그 다음 그것을 막는 계층을
붙인다.** 이 순서가 이 시스템의 설계 원칙 그 자체다 —

    각 계층은 자기가 막는 실패를 이름으로 댈 수 있을 때만 들어간다.

이름을 못 대는 계층은 넣지 않는다. 그래서 이 튜토리얼에는 "있으면 좋은 것"이
하나도 없다.
"""

from __future__ import annotations

import sys
from datetime import date

# ══════════════════════════════════════════════════════════════════════════
# 출력 보조
# ══════════════════════════════════════════════════════════════════════════

W = 74


def _width(text: str) -> int:
    """한글·한자는 터미널에서 두 칸을 차지한다. 정렬에 반영한다."""
    import unicodedata

    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _width(text))


def step(n: int, title: str, prevents: str) -> None:
    print(f"\n{'═' * W}")
    print(f" {n}. {title}")
    print(f"    막는 실패: {prevents}")
    print("═" * W)


def say(label: str, value: object = "") -> None:
    print(f"  {_pad(label, 30)} {value}")


def bad(msg: str) -> None:
    print(f"  ✗ {msg}")


def good(msg: str) -> None:
    print(f"  ✓ {msg}")


# ══════════════════════════════════════════════════════════════════════════
# 1. 궤적 — 가장 먼저 까는 것
# ══════════════════════════════════════════════════════════════════════════

def step1() -> None:
    step(1, "궤적 로그", "나중에 평가하려는데 과거 데이터가 없음")

    from agentcore.trace import MemorySink, SpanKind, Tracer, record_usage

    print("\n  평가셋·지표·판정기는 나중에도 만들 수 있다. 과거 궤적은 못 만든다.")
    print("  그래서 이것이 1일차 인프라이고, 나머지 전부가 여기 의존한다.\n")

    sink = MemorySink()
    tracer = Tracer(sink=sink)

    with tracer.span("답변_생성"):                       # 루트
        with tracer.span("검색", SpanKind.TOOL):
            record_usage(input_tokens=0, output_tokens=0)
        with tracer.span("모델_호출", SpanKind.MODEL):
            record_usage(input_tokens=1200, output_tokens=300)

    for s in sink.spans:
        depth = "  " if s.parent_id else ""
        say(f"{depth}{s.name}", f"{s.kind.value:<9} {s.usage.total_tokens:>6} 토큰")

    print()
    good("자식 사용량이 부모로 자동 합산된다 — 시스템 층위 집계가 공짜")
    print("     계측을 수동으로 두면 빠지고, 빠지면 평가할 수 없다.")


# ══════════════════════════════════════════════════════════════════════════
# 2. 계약 — 동결은 체크포인트 저장이 아니라 계약 발행
# ══════════════════════════════════════════════════════════════════════════

def make_filter(version: str = "1.0", limit: int = 5):
    from agentcore.contracts import (
        Amortization,
        ComponentContract,
        FunctionComponent,
        InterfaceSpec,
        OperatingEnvelope,
        PerformanceSpec,
        Role,
    )

    contract = ComponentContract(
        name="relevance_filter",
        version=version,
        role=Role.SPECIALIST,
        interface=InterfaceSpec("list[str]", "list[str]", failure_modes=("empty_input",)),
        performance=PerformanceSpec("filter_v1", {"precision": 0.91, "recall": 0.88}),
        envelope=OperatingEnvelope(f"문서 {limit}건 이하", lambda docs: len(docs) <= limit),
        ladder_rung=6,                       # 파라메트릭 — 망각 위험 구간
        amortization=Amortization.PARTIAL,
        fallback="bm25_rerank",              # T1 기준선이자 폴백
    )
    return FunctionComponent(contract, lambda docs: [d for d in docs if "관련" in d])


def step2() -> None:
    step(2, "계약과 컴포넌트", "동결된 도구가 학습 분포 밖에서 조용히 틀림")

    from agentcore.contracts import ComponentContract, InterfaceSpec, OperatingEnvelope, PerformanceSpec, Role, Registry
    from agentcore.trace import MemorySink, Tracer

    comp = make_filter()
    tracer = Tracer(sink=MemorySink())

    ok = comp(["관련 문서", "무관"], tracer=tracer)
    say("정상 입력", f"{ok.status.value} → {ok.value}")

    over = comp(["문서"] * 99, tracer=tracer)
    say("범위 밖 입력", f"{over.status.value} — {over.detail}")

    print()
    good("범위 이탈이 예외가 아니라 **값**으로 나온다")
    print("     예외는 try/except 로 삼켜지지만, 값은 궤적에 남고 집계된다.\n")

    # 계약이 원칙을 강제한다
    def try_contract(**kw):
        base = dict(
            name="x", version="1", role=Role.SPECIALIST,
            interface=InterfaceSpec("a", "b"), performance=PerformanceSpec("es", {}),
            envelope=OperatingEnvelope("all"),
        )
        base.update(kw)
        return ComponentContract(**base)

    for label, kw in [
        ("코어 파인튜닝(7단)", dict(role=Role.CORE, ladder_rung=7)),
        ("검증기 파라메트릭화", dict(role=Role.VERIFIER, ladder_rung=6)),
    ]:
        try:
            try_contract(**kw)
        except ValueError as exc:
            bad(f"{label} → {str(exc)[:52]}")

    reg = Registry()
    reg.register(make_filter("1.0"))
    reg.register(make_filter("2.0"))
    print()
    say("버전 병존", reg.versions("relevance_filter"))
    say("망각 위험 컴포넌트", reg.parametric_components())
    print("\n     이 목록이 길어지면 파국적 망각을 스스로 재도입하고 있다는 신호다.")


# ══════════════════════════════════════════════════════════════════════════
# 3. 예산 — 동적 구간의 상한
# ══════════════════════════════════════════════════════════════════════════

def step3() -> None:
    step(3, "예산", "하위작업 개수를 모르는 패턴이 비용에 상한 없이 돌아감")

    from agentcore.control import Budget, BudgetExceeded

    root = Budget(max_tokens=1000, max_depth=2, max_workers=3)
    child = root.child()
    child.spend(tokens=400)

    say("자식이 쓴 토큰", 400)
    say("부모에 반영된 값", root.used_tokens)
    say("자식 잔여", child.remaining_tokens)

    print()
    try:
        child.spend(tokens=700)
    except BudgetExceeded as exc:
        bad(str(exc))

    try:
        child.child().child()
    except BudgetExceeded as exc:
        bad(str(exc))

    say("워커 12개 요청 →", f"{root.workers_allowed(12)}개로 절삭")
    print("\n     예산은 중첩된다. 하위 워커가 전체 예산을 넘을 수 없다.")


# ══════════════════════════════════════════════════════════════════════════
# 4. 게이트 — 추측 방지와 부작용 방지
# ══════════════════════════════════════════════════════════════════════════

def step4() -> None:
    step(4, "게이트", "정보가 부족한데 추측으로 진행 / 비싼 작업이 이미 실행됨")

    from agentcore.control import Estimate, PreflightGate, SlotSpec, SufficiencyGate

    gate = SufficiencyGate([SlotSpec("종목"), SlotSpec("기간"), SlotSpec("지표")])

    partial = gate.check({"종목": "EXMPL"})
    say("슬롯 미충족", f"부족: {partial.missing} → 되묻는다")
    full = gate.check({"종목": "EXMPL", "기간": "2024", "지표": "ROE"})
    say("슬롯 충족", f"{bool(full)} → 진행")

    print("\n     이 게이트가 없으면 에이전트는 추측으로 진행한다.")
    print("     '모르면 묻는' 동작이 설계로 보장되는 자리.\n")

    executed: list[int] = []
    preflight = PreflightGate(lambda rows: Estimate(cost=rows, unit="rows"), ceiling=1000)
    for rows in (200, 50_000):
        admitted, est = preflight.admits(rows)
        if admitted:
            executed.append(rows)
            good(f"{est.cost:,} rows — 실행")
        else:
            bad(f"{est.cost:,} rows — 사전 차단 (상한 1,000)")

    print("\n     사후 게이트는 보호 장치가 아니다. 이미 실행된 뒤니까.")


# ══════════════════════════════════════════════════════════════════════════
# 5. 패턴 — 제어는 코드가 한다
# ══════════════════════════════════════════════════════════════════════════

def step5() -> None:
    step(5, "워크플로우 패턴", "쉬운 질의가 비싼 경로를 타서 비용이 수십 배")

    from agentcore.control import (
        Classification,
        PatternWithoutJustification,
        Step,
        chain,
        route,
    )
    from agentcore.trace import MemorySink, Tracer

    try:
        chain([Step("s", lambda x: x)], 1, prevents="")
    except PatternWithoutJustification as exc:
        bad(str(exc)[:66])
    print("     막는 실패를 이름 못 대면 그 패턴은 들어가지 않는다.\n")

    out = chain(
        [Step("정규화", str.strip), Step("소문자", str.lower), Step("토큰화", str.split)],
        "  Hello World  ",
        prevents="단계 의존성이 명확한 전처리",
    )
    say("chain 결과", out)

    calls: list[str] = []
    tracer = Tracer(sink=MemorySink())
    routes = {
        "간단": lambda p: calls.append("간단") or "캐시 조회",
        "심층": lambda p: calls.append("심층") or "전체 파이프라인",
    }

    for q, conf in [("주가?", 0.95), ("애매한 질문", 0.30)]:
        route(
            lambda p, c=conf: Classification("간단", confidence=c, reason="길이"),
            routes, q,
            prevents="쉬운 질의가 심층 경로를 타서 비용 폭증",
            fallback="심층", min_confidence=0.6, tracer=tracer,
        )
    say("라우팅 결과", calls)
    print("\n     확신도가 낮으면 폴백으로 간다. 폴백은 **더 비싸고 더 안전한** 쪽이다.")
    print("     오분류의 비용은 잘못된 답이지 느린 답이 아니므로.")


# ══════════════════════════════════════════════════════════════════════════
# 6. 컨텍스트 격리 — 이 아키텍처의 핵심 지표
# ══════════════════════════════════════════════════════════════════════════

def step6() -> None:
    step(6, "컨텍스트 격리", "원문 수만 토큰이 코어 컨텍스트를 오염시켜 합성 품질 저하")

    from agentcore.control import Budget, Finding, Subtask, orchestrate
    from agentcore.evaluation import summarize_trajectory
    from agentcore.trace import MemorySink, SpanKind, Tracer, record_isolation, record_usage

    CORPUS = {"매출": "원문 " * 2000, "비용": "원문 " * 2000, "인력": "원문 " * 2000}

    # (a) 모놀리식 — 전부 코어로
    mono = MemorySink()
    t1 = Tracer(sink=mono)
    with t1.span("모놀리식"):
        with t1.span("코어", SpanKind.MODEL):
            raw = sum(len(v) // 4 for v in CORPUS.values())
            record_usage(input_tokens=raw)
            record_isolation(absorbed=0, emitted=raw)

    # (b) 워커 격리 — 원문은 워커 안에서 흡수
    iso = MemorySink()
    t2 = Tracer(sink=iso)

    def worker(st: Subtask) -> Finding[str]:
        raw = len(CORPUS[st.payload]) // 4
        summary = f"{st.payload}: 확인됨"
        record_usage(input_tokens=raw, output_tokens=len(summary) // 4)
        record_isolation(absorbed=raw - len(summary) // 4, emitted=len(summary) // 4)
        return Finding(summary, frozenset({st.payload}), st.name)

    orchestrate(
        lambda p: [Subtask(k, k, scope=f"{k}만") for k in CORPUS],
        worker, "실적 요약",
        prevents="하위작업 개수를 모르고, 원문이 코어 컨텍스트를 오염시킴",
        budget=Budget(max_tokens=200_000, max_workers=3),
        tracer=t2,
    )

    m1, m2 = summarize_trajectory(mono.spans), summarize_trajectory(iso.spans)
    say("모놀리식 격리 효과", f"{m1.isolation_ratio:.1f} : 1   (코어로 {m1.emitted_tokens:,} 토큰)")
    say("워커 격리 효과", f"{m2.isolation_ratio:.1f} : 1   (코어로 {m2.emitted_tokens:,} 토큰)")

    print()
    good("이 숫자 하나가 아키텍처를 판별한다")
    print("     모놀리식은 흡수가 0이라 항상 0이 나온다. 다른 지표로는 안 보인다.")


# ══════════════════════════════════════════════════════════════════════════
# 7. 검증 — 신뢰의 뿌리
# ══════════════════════════════════════════════════════════════════════════

def step7() -> None:
    step(7, "검증기와 가드레일", "검증되지 않은 인용이 출력됨 / 품질 점수가 안전 실패를 가림")

    from agentcore.verify import (
        Claim,
        GroundingVerifier,
        Guardrail,
        GuardrailSet,
        RuleVerifier,
        VerifierStack,
        VerificationResult,
        VerificationTier,
        JudgmentVerifier,
    )
    from agentcore.trace import MemorySink, Tracer

    tracer = Tracer(sink=MemorySink())

    grounding = GroundingVerifier(entails=lambda text, sup: text.split(":")[0] in " ".join(sup))
    claims = [
        Claim("c1", "매출: 4% 증가", ("매출 4% 증가 확인",)),
        Claim("c2", "비용: 감소", ()),                       # 근거 없음
        Claim("c3", "인력: 증가", ("무관한 구절",)),          # 도출 안 됨
    ]
    r = grounding(claims, tracer=tracer)
    say("검사한 주장", r.checked)
    say("실패 위치", r.failure_loci)
    for v in r.violations:
        bad(f"{v.locus}: {v.message}")

    print("\n     주장 단위로 쪼개면 (1) 어디가 틀렸는지 알고 (2) 실패한 것만 고칠 수 있다.\n")

    judged: list[int] = []
    stack = VerifierStack(
        decidable=[RuleVerifier([("비어있지_않음", lambda c, _: bool(c), "출력이 비었음")])],
        judgment=[JudgmentVerifier(lambda c, ctx: judged.append(1) or VerificationResult(True, VerificationTier.JUDGMENT))],
    )
    stack.verify([], tracer=tracer)
    say("빈 출력 → 판단 검증", f"{len(judged)}회 호출 (단락)")
    stack.verify(["내용"], tracer=tracer)
    say("정상 출력 → 판단 검증", f"{len(judged)}회 호출")
    print("\n     형식이 깨진 출력의 품질을 논하는 것은 의미가 없고 비싸다.\n")

    rails = GuardrailSet([
        Guardrail("미검증_인용", lambda c, _: c.get("verified", False), "검증되지 않은 인용"),
        Guardrail("경계밖_유출", lambda c, _: "주민번호" not in c.get("text", ""), "데이터 유출"),
    ])
    rep = rails.evaluate(
        [{"verified": True, "text": "정상"}, {"verified": False, "text": "x"}, {"verified": True, "text": "주민번호 1"}],
        tracer=tracer,
    )
    say("가드레일 위반", f"{rep.breach_count}건 {rep.counts_by_rail()}")
    say("score 속성 존재?", hasattr(rep, "score"))

    print("\n     품질 92%는 훌륭하고 가드레일 92%는 재앙이다.")
    print("     점수를 만들면 언젠가 평균되므로, 이 타입에는 점수가 **없다**.")


# ══════════════════════════════════════════════════════════════════════════
# 8. 브로커 — 재현성과 효율의 교환
# ══════════════════════════════════════════════════════════════════════════

def step8() -> None:
    step(8, "브로커", "동적 선택이 과거 판단의 재현을 불가능하게 만듦")

    from agentcore.broker import Broker, BrokerMode, Candidate, ColdStart, error_correlation
    from agentcore.trace import MemorySink, Tracer

    cands = [
        Candidate("qwen-a", "qwen", cost=1.0, vram_mb=8000, value=0.95),
        Candidate("qwen-b", "qwen", cost=1.0, vram_mb=8000, value=0.94),
        Candidate("llama-a", "llama", cost=1.1, vram_mb=9000, value=0.80),
    ]
    tracer = Tracer(sink=MemorySink())

    b = Broker(mode=BrokerMode.DYNAMIC, team_size=2, max_per_family=1)
    try:
        b.select(cands, task_key="q1", tracer=tracer)
    except ColdStart as exc:
        bad(str(exc)[:64])
    print("     기여도는 순환적이다 — 돌려보기 전에는 모른다.\n")

    b.observations = 50
    sel = b.select(cands, task_key="q1", tracer=tracer)
    say("동적 선택", f"{sel.chosen}  계열 {sel.families}")
    print("     순수 가치/비용 순이면 qwen 둘을 뽑는다. 다양성 제약이 그것을 막는다.")
    print("     최적화는 가장 좋은 모델들을 뽑고, 그것들은 대개 서로 비슷하다.\n")

    det = Broker(mode=BrokerMode.DETERMINISTIC, fixed_team=("llama-a",))
    say("결정론 모드", f"{det.select(cands, task_key='q1', tracer=tracer).chosen}  (규제 배포용)")

    rep = Broker(mode=BrokerMode.REPLAY)
    rep.load_replay({"q1": ("qwen-a", "llama-a")})
    say("재현 모드", rep.select(cands, task_key="q1", tracer=tracer).chosen)

    corr = error_correlation({
        "qwen-a": [True, False, False, True],
        "qwen-b": [True, False, False, True],
        "llama-a": [False, True, True, False],
    })
    print()
    say("오류 상관도 qwen쌍", f"{corr[('qwen-a','qwen-b')]:.2f}  ← 측정 가능한 groupthink")
    say("오류 상관도 교차쌍", f"{corr[('llama-a','qwen-a')]:.2f}")
    print("\n     선택은 모드와 무관하게 항상 궤적에 남는다 — 그것이 재현의 입력이다.")


# ══════════════════════════════════════════════════════════════════════════
# 9. 평가 — 모듈화 논지의 증명
# ══════════════════════════════════════════════════════════════════════════

def step9() -> None:
    step(9, "컴포넌트 평가와 교체", "측정할 수 없는 것은 교체할 수 없다")

    from agentcore.contracts import (
        ComponentContract, FunctionComponent, InterfaceSpec,
        OperatingEnvelope, PerformanceSpec, Role,
    )
    from agentcore.evaluation import ComponentEvalSet, EvalItem, decide_swap, score_component

    def mk(name: str, version: str, fn):
        return FunctionComponent(
            ComponentContract(
                name=name, version=version, role=Role.SPECIALIST,
                interface=InterfaceSpec("str", "str"),
                performance=PerformanceSpec("es", {}),
                envelope=OperatingEnvelope("8자 이하", lambda p: len(p) <= 8),
                ladder_rung=6, fallback="bm25",
            ),
            fn,
        )

    es = ComponentEvalSet("filter_v1", "f")
    for i, (p, e) in enumerate([("ab", "AB"), ("cd", "CD"), ("ef", "EF")]):
        es.add(EvalItem(f"i{i}", p, e))
    es.add(EvalItem("oob", "너무나도긴입력값입니다", out_of_envelope=True))

    t1 = score_component(mk("bm25", "1.0", str.upper), es)          # 기성품
    tie = score_component(mk("f", "2.0", str.upper), es)            # 동점
    weak = score_component(mk("f", "3.0", str.lower), es)           # 회귀

    say("T1 기성품 정확도", f"{t1.accuracy:.2f}")
    say("범위 정밀도", f"{t1.envelope_precision:.2f}  (범위 밖을 제대로 거절했는가)")
    say("평가셋 범위 커버리지", f"{es.envelope_coverage:.2f}")

    print()
    for label, cand, kw in [
        ("동점 후보", tie, {"baseline": t1}),
        ("회귀 후보", weak, {"floor": {"accuracy": 0.9}}),
    ]:
        d = decide_swap(cand, **kw)
        bad(f"{label} → {d.verdict.value.upper()}: {d.reason[:46]}")

    print("\n     **동점도 거절된다.** 기성품이 충분하다는 뜻이므로 맞춤형은 순손실이다.")
    print("     T1 은 버려지지 않고 비교 기준선이자 폴백으로 남는다.")


# ══════════════════════════════════════════════════════════════════════════
# 10. 도메인 — 시점 무결성
# ══════════════════════════════════════════════════════════════════════════

def step10() -> None:
    step(10, "도메인 골드셋", "룩어헤드 편향 — 결과가 더 좋아 보여서 아무도 의심하지 않음")

    from agentcore.domains.finance import (
        AsOfView, ClaimKind, DataPoint, FinancialClaim, MetricClaim,
        finance_guardrails, recompute, split_by_verifiability,
    )
    from agentcore.domains.research import SourceGold, SourceGoldset

    AS_OF = date(2024, 7, 1)
    view = AsOfView(as_of=AS_OF)
    view.ingest([
        DataPoint("매출_2023", 4000, date(2023, 12, 31), date(2024, 2, 14)),
        DataPoint("매출_Q1", 1100, date(2024, 3, 31), date(2024, 5, 15)),
        DataPoint("매출_Q2", 1250, date(2024, 6, 30), date(2024, 8, 14)),   # 아직 미공개
        DataPoint("순익_정정", 344, date(2023, 12, 31), date(2024, 2, 14),
                  restated_at=date(2024, 11, 1)),                            # 나중에 정정
    ])
    say("사용 가능", view.available())
    say("보류", view.withheld())
    try:
        view.require("매출_Q2")
    except LookupError as exc:
        bad(str(exc)[:64])
    print("     접근 경로가 하나면 룩어헤드를 **구조적으로** 못 빠뜨린다.\n")

    r = recompute(MetricClaim("m", "roe", {"net_income": 80, "equity": 500}, reported=0.20))
    bad(f"지표 재계산 — {r.detail}   (LLM Judge 불필요)")

    claims = [
        FinancialClaim("c1", "2023 매출 4,000", ClaimKind.MEASURED, ("10-K",), as_of=str(AS_OF)),
        FinancialClaim("c2", "마진 개선이 이어질 것", ClaimKind.PROJECTED, labeled=True),
        FinancialClaim("c3", "저평가, 매수 의견", ClaimKind.ADVISORY),
    ]
    v, j = split_by_verifiability(claims)
    print()
    say("검증 가능", [c.id for c in v])
    say("검증 불가", [c.id for c in j])
    rep = finance_guardrails(as_of=AS_OF).evaluate(claims)
    say("가드레일", f"{rep.breach_count}건 {rep.counts_by_rail()}")
    print("     권유형 출력은 이 도메인에 평가할 골드셋이 없으므로 차단된다.\n")

    gs = SourceGoldset("연구_v1")
    gs.add(SourceGold("q1", "어텐션 원논문?", required=frozenset({"vaswani2017"}),
                      forbidden=frozenset({"철회논문42"})))
    out = gs.evaluate({"q1": ["vaswani2017", "철회논문42"]})
    say("출처 재현율", out["recall"])
    say("금지 출처 적발", f"{out['forbidden']['total_hits']}건 → 재현율 100%여도 실패")


# ══════════════════════════════════════════════════════════════════════════

STEPS = [step1, step2, step3, step4, step5, step6, step7, step8, step9, step10]


def main() -> None:
    args = sys.argv[1:]
    chosen = [STEPS[int(a) - 1] for a in args] if args else STEPS
    for fn in chosen:
        fn()

    if not args:
        print(f"\n{'═' * W}")
        print(" 다음")
        print("═" * W)
        print("  python examples/pipeline.py              조사 파이프라인 전체")
        print("  python examples/equity.py                주식분석 파이프라인")
        print("  agentcore-report runs/trace.jsonl        궤적 → 4층위 지표")
        print("  python -m pytest tests/test_principles.py  원칙이 강제되는지 검증")
        print()


if __name__ == "__main__":
    main()
