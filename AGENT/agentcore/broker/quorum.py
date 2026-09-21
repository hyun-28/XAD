"""정족수와 저하 모드.

§09 해소: **가용성에 담당 컴포넌트가 없었다.**

앙상블은 원리적으로 가용성에 유리하다 — 하나가 죽어도 나머지로 진행할 수
있으니까. 그러나 그것을 설계로 만들지 않으면 실패 지점이 N개로 늘어난 것뿐이다.

핵심 규칙: **검증기가 죽으면 출력을 차단한다(fail closed).**
검증기는 신뢰의 뿌리이므로, 그것 없이 나온 출력은 근거 없는 출력이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..contracts import Role


class ServiceLevel(str, Enum):
    FULL = "full"                # 전 컴포넌트 정상
    DEGRADED = "degraded"        # 정족수는 충족. 품질 저하를 명시하고 진행
    HALTED = "halted"            # 진행 불가. 조용히 나쁜 답을 내는 것보다 낫다


class VerifierDownPolicy(str, Enum):
    HALT = "halt"                # 기본값. 검증기 없이는 출력하지 않는다
    DEGRADE_WITH_WARNING = "degrade_with_warning"   # 명시적 승인이 있을 때만


@dataclass(slots=True)
class QuorumDecision:
    level: ServiceLevel
    available: frozenset[str]
    missing: frozenset[str]
    reason: str = ""
    degradations: tuple[str, ...] = ()

    @property
    def can_proceed(self) -> bool:
        return self.level is not ServiceLevel.HALTED


@dataclass
class QuorumPolicy:
    """정족수 규칙.

    min_by_role 은 역할별 최소 생존 수다. 코어 자리를 앙상블이 채우는 구성에서
    "5개 중 3개 응답하면 진행"이 여기서 표현된다.
    """

    min_by_role: dict[Role, int] = field(default_factory=dict)
    required_roles: frozenset[Role] = frozenset({Role.VERIFIER})
    verifier_down: VerifierDownPolicy = VerifierDownPolicy.HALT

    def decide(self, healthy: dict[Role, set[str]]) -> QuorumDecision:
        available = frozenset(r for names in healthy.values() for r in names)
        degradations: list[str] = []
        missing_roles: list[str] = []

        for role in self.required_roles:
            if not healthy.get(role):
                if role is Role.VERIFIER and self.verifier_down is VerifierDownPolicy.HALT:
                    return QuorumDecision(
                        level=ServiceLevel.HALTED,
                        available=available,
                        missing=frozenset({role.value}),
                        reason=(
                            "검증기 없음 — 출력 차단. 검증기는 신뢰의 뿌리이므로 "
                            "그것 없이 나온 출력은 근거 없는 출력입니다."
                        ),
                    )
                missing_roles.append(role.value)
                degradations.append(f"{role.value} 부재")

        for role, minimum in self.min_by_role.items():
            have = len(healthy.get(role, ()))
            if have < minimum:
                if have == 0 and role in self.required_roles:
                    missing_roles.append(role.value)
                    continue
                if have == 0:
                    return QuorumDecision(
                        level=ServiceLevel.HALTED,
                        available=available,
                        missing=frozenset({role.value}),
                        reason=f"{role.value} 정족수 미달: {have}/{minimum}, 대체 불가",
                    )
                degradations.append(f"{role.value} 정족수 미달 ({have}/{minimum})")

        if degradations:
            return QuorumDecision(
                level=ServiceLevel.DEGRADED,
                available=available,
                missing=frozenset(missing_roles),
                reason="; ".join(degradations),
                degradations=tuple(degradations),
            )
        return QuorumDecision(
            level=ServiceLevel.FULL, available=available, missing=frozenset()
        )
