"""검증기 — 신뢰의 뿌리.

1부 원칙: **적응하는 시스템에는 적응하지 않는 기준점이 필요하다.**

나머지 전 계층이 검증기의 신호로 적응하므로, 검증기가 틀리면 시스템 전체가
확신을 갖고 잘못된 방향으로 최적화된다. 그것도 조용히.

그래서 검증기는 셋으로 쪼개진다. "검증은 생성보다 쉽다"는 직관은
**증명서가 존재할 때만** 참이기 때문이다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..trace import SpanKind, Tracer


class VerificationTier(str, Enum):
    """검증의 종류. 티어가 적정 모델 크기를 결정한다."""

    DECIDABLE = "decidable"
    """결정 가능한 검증 — 스키마, 형식, 제약, 원문 대조, 함의 판정.
    증명서가 존재하므로 1B-3B 소형 모델 또는 순수 코드로 충분하다.
    소형 모델이 진짜 강한 자리."""

    JUDGMENT = "judgment"
    """판단적 검증 — 추론 사슬의 타당성, 출처 충돌 판정, 근거 충분성.
    증명서가 없으므로 코어가 맡는다. 여기에 소형 모델을 쓰면
    확신에 찬 틀린 신호가 모든 적응 루프로 전파된다."""

    HUMAN = "human"
    """골드셋 기반 주기 검증 — 사람. 적응하지 않는 진짜 기준점.
    다른 두 티어를 캘리브레이션하는 근거."""


@dataclass(slots=True)
class Violation:
    rule: str
    message: str
    locus: str = ""          # 어느 주장·필드·라인인가. credit assignment 의 근거.
    evidence: str = ""       # 원문 대조 결과 등 객관적 증거


@dataclass(slots=True)
class VerificationResult:
    passed: bool
    tier: VerificationTier
    violations: tuple[Violation, ...] = ()
    checked: int = 0
    detail: str = ""

    @property
    def failure_loci(self) -> tuple[str, ...]:
        """실패한 항목만 재작성하면 refine 루프가 유계가 된다."""
        return tuple(v.locus for v in self.violations if v.locus)

    def __bool__(self) -> bool:
        return self.passed


class Verifier(ABC):
    """검증기 기반 클래스."""

    tier: VerificationTier = VerificationTier.DECIDABLE
    name: str = "verifier"

    @abstractmethod
    def check(self, candidate: Any, context: Any = None) -> VerificationResult: ...

    def __call__(
        self, candidate: Any, context: Any = None, *, tracer: Tracer | None = None
    ) -> VerificationResult:
        tracer = tracer or Tracer()
        with tracer.span(self.name, SpanKind.VERIFY, tier=self.tier.value) as span:
            result = self.check(candidate, context)
            span.attributes.update(
                passed=result.passed,
                checked=result.checked,
                violations=len(result.violations),
                loci=list(result.failure_loci),
            )
            return result


class RuleVerifier(Verifier):
    """순수 코드 검증. 가장 싸고 가장 신뢰할 수 있다.

    결정론 우선 원칙: "이 출력이 맞는지 프로그램으로 확인할 수 있는 하위 성질이
    무엇인가?" — 그것을 최대한 뽑아내고 남은 것만 판정자에게 맡긴다.
    """

    tier = VerificationTier.DECIDABLE

    def __init__(
        self,
        rules: Sequence[tuple[str, Callable[[Any, Any], bool], str]],
        name: str = "rules",
    ) -> None:
        self.rules = tuple(rules)
        self.name = name

    def check(self, candidate: Any, context: Any = None) -> VerificationResult:
        violations = []
        for rule_name, predicate, message in self.rules:
            try:
                ok = bool(predicate(candidate, context))
            except Exception as exc:
                ok = False
                message = f"{message} (평가 중 오류: {exc})"
            if not ok:
                violations.append(Violation(rule=rule_name, message=message, locus=rule_name))
        return VerificationResult(
            passed=not violations,
            tier=self.tier,
            violations=tuple(violations),
            checked=len(self.rules),
        )


@dataclass
class Claim:
    """검증 단위.

    출력을 통째로 평가하지 않고 주장 단위로 쪼갠다. 이 분해가
    (1) credit assignment 를 가능하게 하고
    (2) 실패한 항목만 재작성하게 하고
    (3) 주장 단위 투표·집계의 단위가 된다.
    """

    id: str
    text: str
    support: tuple[str, ...] = ()      # 근거 구절/행/출처 ID
    source_ids: frozenset[str] = frozenset()


class GroundingVerifier(Verifier):
    """근거 대조 검증.

    "이 주장이 제시된 근거에서 실제로 도출되는가"는 취향 판정이 아니라
    검증 가능한 신호다. 네 성질을 동시에 만족하는 드문 신호이며,
    그래서 적응 신호로도 평가 지표로도 쓰인다.

      목표 정렬 ✓  조밀(주장 단위) ✓  객관(원문 대조) ✓  저비용 ✓
    """

    tier = VerificationTier.DECIDABLE
    name = "grounding"

    def __init__(self, entails: Callable[[str, tuple[str, ...]], bool]) -> None:
        self.entails = entails

    def check(self, candidate: Any, context: Any = None) -> VerificationResult:
        claims: Sequence[Claim] = candidate
        violations = []
        for claim in claims:
            if not claim.support:
                violations.append(
                    Violation(
                        rule="unsupported",
                        message="근거가 제시되지 않은 사실 주장",
                        locus=claim.id,
                    )
                )
                continue
            if not self.entails(claim.text, claim.support):
                violations.append(
                    Violation(
                        rule="not_entailed",
                        message="제시된 근거에서 도출되지 않음",
                        locus=claim.id,
                        evidence=" / ".join(claim.support[:2]),
                    )
                )
        return VerificationResult(
            passed=not violations,
            tier=self.tier,
            violations=tuple(violations),
            checked=len(claims),
        )


class JudgmentVerifier(Verifier):
    """판단적 검증. **코어가 맡는다.**

    증명서가 없는 판정 — 추론 타당성, 출처 충돌 해소, 근거 충분성.
    소형 모델을 여기 쓰는 것은 가장 나쁜 절약이다.
    """

    tier = VerificationTier.JUDGMENT
    name = "judgment"

    def __init__(self, judge: Callable[[Any, Any], VerificationResult], name: str = "judgment") -> None:
        self._judge = judge
        self.name = name

    def check(self, candidate: Any, context: Any = None) -> VerificationResult:
        return self._judge(candidate, context)


@dataclass
class VerifierStack:
    """티어 순서대로 검증한다.

    싼 것부터 돌리고, 결정 가능한 검증에서 걸리면 판단적 검증까지 가지 않는다.
    비용 절감이자 신호 품질 향상이다 — 형식이 깨진 출력에 대한 판단은 의미가 없다.
    """

    decidable: list[Verifier] = field(default_factory=list)
    judgment: list[Verifier] = field(default_factory=list)

    def verify(
        self, candidate: Any, context: Any = None, *, tracer: Tracer | None = None
    ) -> VerificationResult:
        tracer = tracer or Tracer()
        with tracer.span("verify_stack", SpanKind.VERIFY) as span:
            all_violations: list[Violation] = []
            checked = 0
            for v in self.decidable:
                r = v(candidate, context, tracer=tracer)
                checked += r.checked
                all_violations.extend(r.violations)
            if all_violations:
                span.attributes.update(short_circuited=True, tier="decidable")
                return VerificationResult(
                    passed=False,
                    tier=VerificationTier.DECIDABLE,
                    violations=tuple(all_violations),
                    checked=checked,
                    detail="결정 가능한 검증에서 실패 — 판단적 검증 생략",
                )
            for v in self.judgment:
                r = v(candidate, context, tracer=tracer)
                checked += r.checked
                all_violations.extend(r.violations)
            return VerificationResult(
                passed=not all_violations,
                tier=VerificationTier.JUDGMENT if self.judgment else VerificationTier.DECIDABLE,
                violations=tuple(all_violations),
                checked=checked,
            )
