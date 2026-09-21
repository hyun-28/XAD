"""주장 분류 — 검증 가능한 것과 아닌 것을 분리한다.

금융 분석 출력에는 성격이 전혀 다른 문장이 섞인다.

  MEASURED   "2024 Q1 매출 1,200억"          → 원자료에서 재계산 가능. 검증됨
  DERIVED    "영업이익률 12.4%"               → 명시된 입력으로 재계산 가능
  SOURCED    "회사는 신규 라인 증설을 발표"    → 출처 대조 가능
  PROJECTED  "내년 매출은 늘어날 것"          → **검증 불가**
  ADVISORY   "매수할 만하다"                  → **검증 불가이며 권유에 해당**

앞의 셋과 뒤의 둘은 근본적으로 다르다. 그런데 같은 문단에 섞여 나오면
독자는 구분하지 못하고, 검증된 숫자의 신뢰가 검증되지 않은 전망으로 번진다.

**이 시스템은 골드셋을 만들 수 없는 것에 대해 골드셋이 있는 척하지 않는다.**
PROJECTED 는 라벨을 달아 통과시키고, ADVISORY 는 기본적으로 차단한다 —
성능 문제가 아니라 이 도메인에서 평가 불가능한 출력을 검증된 것처럼
내보내지 않기 위한 설계다.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum


class ClaimKind(str, Enum):
    MEASURED = "measured"
    DERIVED = "derived"
    SOURCED = "sourced"
    PROJECTED = "projected"
    ADVISORY = "advisory"

    @property
    def verifiable(self) -> bool:
        return self in (ClaimKind.MEASURED, ClaimKind.DERIVED, ClaimKind.SOURCED)


@dataclass(slots=True)
class FinancialClaim:
    """금융 주장 하나.

    kind 가 verifiable 이면 support 가 필수다 — 근거 없이 사실을 주장할 수 없다.
    kind 가 PROJECTED 면 labeled 가 필수다 — 라벨 없는 전망은 사실로 읽힌다.
    """

    id: str
    text: str
    kind: ClaimKind
    support: tuple[str, ...] = ()
    as_of: str = ""
    labeled: bool = False           # 전망임이 출력에 명시되었는가
    inputs: dict[str, float] = field(default_factory=dict)   # DERIVED 재계산용
    expected: float | None = None


# 권유형 표현. 완전하지 않으며, 규칙 기반 1차 필터로만 쓴다.
_ADVISORY_PATTERNS = [
    r"매수|매도|사세요|파세요|사야|팔아야|비중\s*확대|비중\s*축소",
    r"\b(buy|sell|hold|overweight|underweight)\b",
    r"목표\s*주가|target\s*price",
    r"추천(합니다|한다|드립니다)?",
]
_PROJECTION_PATTERNS = [
    r"할\s*것(이다|으로|입니다)|예상|전망|기대|겠(다|습니다)|될\s*것",
    r"\b(will|expect|forecast|project|likely|outlook)\b",
]


def classify_text(text: str) -> ClaimKind | None:
    """규칙 기반 1차 분류.

    **판정기가 아니라 필터다.** 여기서 걸리지 않는다고 안전한 것이 아니며,
    최종 분류는 작성자(코어)가 명시적으로 달아야 한다. 규칙에만 의존하면
    표현을 바꾼 권유가 그대로 통과한다.
    """
    for pat in _ADVISORY_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return ClaimKind.ADVISORY
    for pat in _PROJECTION_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return ClaimKind.PROJECTED
    return None


@dataclass(slots=True)
class ClaimIssue:
    claim_id: str
    rule: str
    message: str


def audit_claims(
    claims: Sequence[FinancialClaim],
    *,
    allow_advisory: bool = False,
) -> tuple[ClaimIssue, ...]:
    """주장 집합을 검사한다. 개수로 보고되며 비율로 집계되지 않는다."""
    issues: list[ClaimIssue] = []

    for c in claims:
        if c.kind.verifiable and not c.support:
            issues.append(
                ClaimIssue(
                    c.id,
                    "unsupported_fact",
                    f"[{c.kind.value}] 근거 없이 사실을 주장합니다",
                )
            )

        if c.kind is ClaimKind.PROJECTED and not c.labeled:
            issues.append(
                ClaimIssue(
                    c.id,
                    "unlabeled_projection",
                    "전망이 사실과 구분 없이 제시되었습니다. "
                    "검증된 숫자의 신뢰가 검증되지 않은 전망으로 번집니다",
                )
            )

        if c.kind is ClaimKind.ADVISORY and not allow_advisory:
            issues.append(
                ClaimIssue(
                    c.id,
                    "advisory_output",
                    "권유형 주장입니다. 이 도메인에는 이를 평가할 골드셋이 없으므로 "
                    "검증된 출력과 함께 내보내지 않습니다",
                )
            )

        # 작성자가 단 라벨과 표현이 어긋나는 경우 — 규칙이 더 강한 쪽을 채택
        detected = classify_text(c.text)
        if detected is ClaimKind.ADVISORY and c.kind is not ClaimKind.ADVISORY:
            issues.append(
                ClaimIssue(
                    c.id,
                    "misclassified_advisory",
                    f"[{c.kind.value}] 로 표시됐으나 권유형 표현이 감지되었습니다",
                )
            )
        elif (
            detected is ClaimKind.PROJECTED
            and c.kind.verifiable
        ):
            issues.append(
                ClaimIssue(
                    c.id,
                    "misclassified_projection",
                    f"[{c.kind.value}] 로 표시됐으나 전망형 표현이 감지되었습니다",
                )
            )

        if c.kind.verifiable and not c.as_of:
            issues.append(
                ClaimIssue(
                    c.id, "missing_as_of", "검증 가능한 주장에 기준일이 없습니다"
                )
            )

    return tuple(issues)


def split_by_verifiability(
    claims: Iterable[FinancialClaim],
) -> tuple[list[FinancialClaim], list[FinancialClaim]]:
    """검증 가능한 것과 아닌 것을 나눈다.

    출력을 구성할 때 이 둘을 **시각적으로도 분리**해야 한다.
    같은 문단에 섞이면 분류의 의미가 사라진다.
    """
    verifiable = [c for c in claims if c.kind.verifiable]
    rest = [c for c in claims if not c.kind.verifiable]
    return verifiable, rest
