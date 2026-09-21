"""워크플로우 패턴.

1부 원칙: **각 패턴은 자기가 막는 실패를 이름으로 댈 수 있을 때만 들어간다.**
그래서 모든 패턴이 `prevents=` 를 요구한다. 이름을 못 대면 코드가 거절한다.

그리고 **LLM은 판단하고, 제어는 코드가 한다.** 분기·재시도·게이트·루프는
전부 여기에 있고, LLM 은 각 단계 안에서 판단만 한다.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar

from ..trace import SpanKind, Tracer, record_isolation
from .budget import Budget

I = TypeVar("I")
O = TypeVar("O")
T = TypeVar("T")


class PatternWithoutJustification(ValueError):
    """막는 실패를 이름대지 못한 패턴은 들어갈 수 없다."""


def _require_justification(pattern: str, prevents: str) -> None:
    if not prevents or not prevents.strip():
        raise PatternWithoutJustification(
            f"{pattern} 패턴에 prevents= 가 없습니다. "
            f"이 패턴이 막는 구체적 실패를 이름대지 못하면 넣지 마십시오."
        )


# --------------------------------------------------------------------------
# 1. Prompt Chaining — 단계 의존성이 명확할 때
# --------------------------------------------------------------------------

@dataclass(slots=True)
class Step(Generic[I, O]):
    name: str
    fn: Callable[[I], O]


def chain(
    steps: Sequence[Step],
    payload: Any,
    *,
    prevents: str,
    tracer: Tracer | None = None,
    budget: Budget | None = None,
) -> Any:
    """순차 실행. 이전 단계의 출력이 다음 단계의 입력."""
    _require_justification("chain", prevents)
    tracer = tracer or Tracer()
    with tracer.span("chain", SpanKind.CONTROL, pattern="chain", prevents=prevents) as span:
        span.attributes["steps"] = [s.name for s in steps]
        value = payload
        for step in steps:
            if budget:
                budget.check()
            with tracer.span(step.name, SpanKind.CONTROL, step=step.name):
                value = step.fn(value)
        return value


# --------------------------------------------------------------------------
# 2. Routing — 입력 유형별 전략이 다를 때
# --------------------------------------------------------------------------

@dataclass(slots=True)
class Classification:
    """분류 결과. confidence 가 낮으면 폴백으로 넘어간다.

    라우터의 오분류는 조용하다. 그래서 확신도가 없는 분류기는 받지 않는다.
    """

    route: str
    confidence: float = 1.0
    reason: str = ""


def route(
    classifier: Callable[[Any], Classification],
    routes: dict[str, Callable[[Any], Any]],
    payload: Any,
    *,
    prevents: str,
    fallback: str,
    min_confidence: float = 0.6,
    tracer: Tracer | None = None,
) -> Any:
    """분기 처리.

    확신도가 낮으면 fallback 으로 간다. 규약상 fallback 은 **더 비싸고 더 안전한**
    경로여야 한다 — 오분류의 비용은 잘못된 답이지 느린 답이 아니다.
    """
    _require_justification("route", prevents)
    if fallback not in routes:
        raise ValueError(f"fallback '{fallback}' 이 routes 에 없습니다")

    tracer = tracer or Tracer()
    with tracer.span("route", SpanKind.CONTROL, pattern="route", prevents=prevents) as span:
        cls = classifier(payload)
        chosen = cls.route if (cls.route in routes and cls.confidence >= min_confidence) else fallback
        span.attributes.update(
            classified=cls.route,
            confidence=cls.confidence,
            chosen=chosen,
            fell_back=chosen != cls.route,
            reason=cls.reason,
        )
        with tracer.span(f"route:{chosen}", SpanKind.CONTROL):
            return routes[chosen](payload)


# --------------------------------------------------------------------------
# 3. Parallelization — 독립 작업. 거짓 합의를 막는 조인이 진짜 일이다.
# --------------------------------------------------------------------------

@dataclass(slots=True)
class Finding(Generic[T]):
    """워커 하나의 발견.

    source_ids 가 핵심이다. 조인 시점의 중복 제거는 요약문이 아니라
    **출처 단위**로 이루어져야 한다 — 그래야 상관된 증거가 독립 확증으로
    둔갑하지 않는다.
    """

    value: T
    source_ids: frozenset[str] = frozenset()
    worker: str = ""


@dataclass(slots=True)
class Joined(Generic[T]):
    findings: list[Finding[T]]
    independent_sources: int
    duplicate_rate: float

    @property
    def corroboration(self) -> int:
        """독립 출처 수. "몇 명이 말했나"가 아니라 "몇 개의 독립 출처인가"."""
        return self.independent_sources


def join_findings(findings: Iterable[Finding[T]]) -> Joined[T]:
    """출처 단위 중복 제거.

    워커 넷이 같은 원출처를 각자 찾아왔다면 독립 출처는 넷이 아니라 하나다.
    이 구분이 없으면 코어가 신뢰도를 과대평가한다.
    """
    items = list(findings)
    all_ids: set[str] = set()
    seen_signatures: set[frozenset[str]] = set()
    unique: list[Finding[T]] = []
    duplicates = 0
    for f in items:
        all_ids |= set(f.source_ids)
        if f.source_ids and f.source_ids in seen_signatures:
            duplicates += 1
            continue
        if f.source_ids:
            seen_signatures.add(f.source_ids)
        unique.append(f)
    rate = duplicates / len(items) if items else 0.0
    return Joined(findings=unique, independent_sources=len(all_ids), duplicate_rate=rate)


def parallel(
    tasks: Sequence[tuple[str, Callable[[], Finding[T]]]],
    *,
    prevents: str,
    budget: Budget | None = None,
    tracer: Tracer | None = None,
    exclusivity: dict[str, str] | None = None,
) -> Joined[T]:
    """독립 작업 동시 실행 + 출처 단위 조인.

    exclusivity 는 각 워커에게 "다른 워커가 무엇을 맡았는지" 알려주는 힌트다.
    이것이 없으면 워커들이 같은 영역을 파고, 중복 탐색이 거짓 합의를 만든다.
    """
    _require_justification("parallel", prevents)
    tracer = tracer or Tracer()
    n = budget.workers_allowed(len(tasks)) if budget else len(tasks)

    with tracer.span("parallel", SpanKind.CONTROL, pattern="parallel", prevents=prevents) as span:
        span.attributes.update(
            requested=len(tasks), running=n, exclusivity=exclusivity or {}
        )
        results: list[Finding[T]] = []
        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = {pool.submit(fn): name for name, fn in tasks[:n]}
            for fut, name in futures.items():
                f = fut.result()
                if not f.worker:
                    f.worker = name
                results.append(f)
        joined = join_findings(results)
        span.attributes.update(
            independent_sources=joined.independent_sources,
            duplicate_rate=round(joined.duplicate_rate, 4),
        )
        return joined


# --------------------------------------------------------------------------
# 4. Orchestrator-Workers — 하위작업 개수를 미리 알 수 없을 때
# --------------------------------------------------------------------------

@dataclass(slots=True)
class Subtask:
    name: str
    payload: Any
    scope: str = ""     # 이 워커가 맡은 영역. 배타성 명시가 중복 탐색을 줄인다.


def orchestrate(
    planner: Callable[[Any], list[Subtask]],
    worker: Callable[[Subtask], Finding[T]],
    payload: Any,
    *,
    prevents: str,
    budget: Budget,
    tracer: Tracer | None = None,
) -> Joined[T]:
    """동적 하위작업 생성.

    유일하게 하위작업 개수를 사전에 모르는 패턴이므로, **반드시 예산 안**에서
    돈다. budget 은 선택 인자가 아니다.

    워커는 컨텍스트 격리 장치다 — 원문은 워커 안에서 흡수되고 요약만 올라온다.
    """
    _require_justification("orchestrate", prevents)
    tracer = tracer or Tracer()

    with tracer.span(
        "orchestrate", SpanKind.CONTROL, pattern="orchestrate", prevents=prevents
    ) as span:
        subtasks = planner(payload)
        allowed = budget.workers_allowed(len(subtasks))
        span.attributes.update(planned=len(subtasks), running=allowed)

        scopes = {s.name: s.scope for s in subtasks[:allowed]}
        child = budget.child()

        def run_one(st: Subtask) -> Finding[T]:
            with tracer.span(f"worker:{st.name}", SpanKind.CONTROL, scope=st.scope):
                f = worker(st)
                if not f.worker:
                    f.worker = st.name
                return f

        tasks = [(st.name, (lambda s=st: run_one(s))) for st in subtasks[:allowed]]
        return parallel(
            tasks,
            prevents=prevents,
            budget=child,
            tracer=tracer,
            exclusivity=scopes,
        )


# --------------------------------------------------------------------------
# 5. Evaluator-Optimizer — 외부 신호가 있을 때만
# --------------------------------------------------------------------------

class ExternalSignal(Protocol):
    """개선 신호.

    1부 정정: E-O 의 트리거는 "반복 개선이 중요하다"가 아니라
    **"외부에 검증 가능한 개선 신호가 있다"** 이다.

    is_external 이 True 여야 refine() 이 받아들인다. 생성자와 같은 정보만
    보는 판정자는 같은 맹점을 공유하므로 루프가 수렴하지 않는다.
    """

    is_external: bool

    def __call__(self, candidate: Any) -> "SignalResult": ...


@dataclass(slots=True)
class SignalResult:
    passed: bool
    failures: tuple[str, ...] = ()
    detail: str = ""


@dataclass
class DeterministicSignal:
    """프로그램으로 판정되는 신호. 컴파일, 테스트, 스키마, 원문 대조.

    이것이 E-O 에 넣을 수 있는 유일한 종류의 신호다.
    """

    fn: Callable[[Any], SignalResult]
    name: str = "deterministic"
    is_external: bool = True

    def __call__(self, candidate: Any) -> SignalResult:
        return self.fn(candidate)


@dataclass
class OpinionSignal:
    """생성자와 같은 정보만 보는 판정. **E-O 에 쓰면 안 된다.**

    이 클래스는 거절되기 위해 존재한다. 오프라인 평가에서는 쓸 수 있으나
    런타임 개선 루프의 종료 조건으로는 쓸 수 없다.
    """

    fn: Callable[[Any], SignalResult]
    name: str = "opinion"
    is_external: bool = False

    def __call__(self, candidate: Any) -> SignalResult:
        return self.fn(candidate)


class NonExternalSignal(ValueError):
    pass


def refine(
    generate: Callable[[Any, tuple[str, ...]], T],
    signal: ExternalSignal,
    payload: Any,
    *,
    prevents: str,
    max_rounds: int = 3,
    tracer: Tracer | None = None,
    budget: Budget | None = None,
) -> tuple[T, SignalResult, int]:
    """반복 개선 루프. 유계이며, 실패한 항목만 재작성한다.

    max_rounds 는 선택이 아니다 — 종료 조건을 판정자가 정하면 비용에 상한이 없다.
    """
    _require_justification("refine", prevents)
    if not getattr(signal, "is_external", False):
        raise NonExternalSignal(
            "refine() 은 외부 검증 신호만 받습니다. 생성자와 같은 정보만 보는 "
            "판정자는 맹점을 공유하므로 표현만 매끄러워지고 사실성은 개선되지 "
            "않습니다. 그런 판정은 오프라인 평가 계층에 두십시오."
        )

    tracer = tracer or Tracer()
    with tracer.span("refine", SpanKind.CONTROL, pattern="refine", prevents=prevents) as span:
        failures: tuple[str, ...] = ()
        candidate: T | None = None
        result = SignalResult(passed=False)
        rounds = 0
        for rounds in range(1, max_rounds + 1):
            if budget:
                budget.check()
            with tracer.span(f"refine:round{rounds}", SpanKind.CONTROL, round=rounds):
                candidate = generate(payload, failures)
                result = signal(candidate)
                failures = result.failures
            if result.passed:
                break
        span.attributes.update(
            rounds=rounds, passed=result.passed, residual_failures=list(failures)
        )
        return candidate, result, rounds  # type: ignore[return-value]
