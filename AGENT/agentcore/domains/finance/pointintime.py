"""시점 무결성 — 금융 분석에서 가장 흔하고 가장 조용한 오류.

룩어헤드 편향(look-ahead bias)은 분석 시점에 알 수 없었던 정보가 분석에
새어 들어가는 것이다. 이것이 무서운 이유는 세 가지다.

  · 결과가 **더 좋아 보인다** — 그래서 아무도 의심하지 않는다
  · 완전히 조용하다 — 오류 메시지도, 이상한 숫자도 없다
  · 한 건만 새도 전체 결론이 무효다

그래서 비율이 아니라 **개수**로 재는 가드레일이다. 그리고 완전히 결정론적으로
검증된다 — 타임스탬프 비교일 뿐이다.

세 가지를 모두 본다.

  1. 관측 시점  — 데이터가 만들어진 때가 as_of 이후인가
  2. 공시 지연  — 재무제표는 분기 종료 후 수십 일 뒤에 공개된다.
                  기간 종료일이 아니라 **공개일**을 기준으로 해야 한다
  3. 소급 정정  — 나중에 수정된 값을 그때 알았던 것처럼 쓰면 안 된다
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Any


class LeakKind(str, Enum):
    FUTURE_OBSERVATION = "future_observation"     # as_of 이후에 관측된 값
    UNPUBLISHED = "unpublished"                   # 기간은 지났으나 아직 공개 전
    RESTATED = "restated"                         # 나중에 정정된 값을 소급 사용
    UNKNOWN_TIMESTAMP = "unknown_timestamp"       # 시점을 모르는 데이터


@dataclass(frozen=True, slots=True)
class DataPoint:
    """시점 정보를 가진 데이터.

    period_end 와 published_at 을 **분리해서** 갖는 것이 핵심이다.
    2024 Q1 실적은 2024-03-31 에 확정되지만 5월에야 공개된다.
    4월에 그 값을 쓰면 룩어헤드다.
    """

    key: str
    value: Any
    period_end: date | None = None
    published_at: date | None = None
    restated_at: date | None = None
    source: str = ""

    def known_at(self, as_of: date) -> bool:
        if self.published_at is None:
            return False
        if self.published_at > as_of:
            return False
        if self.restated_at is not None and self.restated_at > as_of:
            # 정정 전 값을 써야 하는데 정정 후 값을 들고 있다.
            return False
        return True


@dataclass(frozen=True, slots=True)
class Leak:
    key: str
    kind: LeakKind
    detail: str
    source: str = ""


@dataclass(slots=True)
class IntegrityReport:
    """시점 무결성 결과.

    의도적으로 점수가 없다. 한 건이라도 새면 분석 전체가 무효이므로
    "95% 깨끗함"은 의미 있는 상태가 아니다.
    """

    as_of: date
    checked: int
    leaks: tuple[Leak, ...] = ()

    @property
    def clean(self) -> bool:
        return not self.leaks

    @property
    def leak_count(self) -> int:
        return len(self.leaks)

    def counts_by_kind(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for leak in self.leaks:
            out[leak.kind.value] = out.get(leak.kind.value, 0) + 1
        return out

    def __bool__(self) -> bool:
        return self.clean


def check_integrity(
    points: Iterable[DataPoint],
    as_of: date,
    *,
    max_publication_lag_days: int | None = None,
) -> IntegrityReport:
    """as_of 시점에 알 수 있었던 데이터만 들어왔는지 검사한다.

    max_publication_lag_days 를 주면 공시 지연이 비정상적으로 짧은 항목도
    잡는다 — 공개일 자체가 잘못 기록된 경우다.
    """
    leaks: list[Leak] = []
    checked = 0

    for p in points:
        checked += 1
        if p.published_at is None:
            leaks.append(
                Leak(
                    p.key,
                    LeakKind.UNKNOWN_TIMESTAMP,
                    "공개일이 없습니다. 시점을 모르는 데이터는 시점 검증을 통과할 수 없습니다",
                    p.source,
                )
            )
            continue

        if p.published_at > as_of:
            leaks.append(
                Leak(
                    p.key,
                    LeakKind.UNPUBLISHED,
                    f"공개일 {p.published_at} 이 기준일 {as_of} 이후입니다",
                    p.source,
                )
            )
            continue

        if p.restated_at is not None and p.restated_at > as_of:
            leaks.append(
                Leak(
                    p.key,
                    LeakKind.RESTATED,
                    f"정정일 {p.restated_at} 이 기준일 {as_of} 이후입니다 — "
                    f"그 시점에는 정정 전 값만 알 수 있었습니다",
                    p.source,
                )
            )
            continue

        if p.period_end is not None and p.period_end > as_of:
            leaks.append(
                Leak(
                    p.key,
                    LeakKind.FUTURE_OBSERVATION,
                    f"대상 기간 종료일 {p.period_end} 이 기준일 이후입니다",
                    p.source,
                )
            )
            continue

        if (
            max_publication_lag_days is not None
            and p.period_end is not None
            and (p.published_at - p.period_end) < timedelta(days=0)
        ):
            leaks.append(
                Leak(
                    p.key,
                    LeakKind.FUTURE_OBSERVATION,
                    f"공개일 {p.published_at} 이 기간 종료 {p.period_end} 보다 앞섭니다 — "
                    f"공개일 기록 오류로 보입니다",
                    p.source,
                )
            )

    return IntegrityReport(as_of=as_of, checked=checked, leaks=tuple(leaks))


@dataclass
class AsOfView:
    """기준일 스냅샷.

    데이터 접근을 이 객체로 강제하면 룩어헤드가 **구조적으로 불가능**해진다.
    검사보다 낫다 — 검사는 빠뜨릴 수 있지만 접근 경로가 하나면 못 빠뜨린다.
    """

    as_of: date
    _points: dict[str, list[DataPoint]] = field(default_factory=dict)

    def ingest(self, points: Iterable[DataPoint]) -> None:
        for p in points:
            self._points.setdefault(p.key, []).append(p)

    def get(self, key: str) -> DataPoint | None:
        """기준일에 알 수 있었던 것 중 가장 최근 값."""
        candidates = [p for p in self._points.get(key, ()) if p.known_at(self.as_of)]
        if not candidates:
            return None
        return max(candidates, key=lambda p: (p.published_at or date.min))

    def require(self, key: str) -> DataPoint:
        p = self.get(key)
        if p is None:
            raise LookupError(
                f"{key} 는 {self.as_of} 시점에 알 수 없었습니다. "
                f"없는 데이터를 추정으로 메우면 그것이 곧 룩어헤드입니다."
            )
        return p

    def available(self) -> list[str]:
        return sorted(k for k in self._points if self.get(k) is not None)

    def withheld(self) -> list[str]:
        """기준일에는 아직 몰랐던 항목. 무엇이 빠졌는지 알아야 한다."""
        return sorted(k for k in self._points if self.get(k) is None)
