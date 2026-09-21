"""Dynamic Expertise Broker.

§09 해소: **브로커 결정론 모드.**

동적 선택은 효율성·적응성을 올리고 재현성·이의제기성을 내린다.
규제 환경은 과거 판단의 재현을 요구하므로 세 모드를 제공한다.

  DYNAMIC        — 텔레메트리 기반 선택. 효율 최대, 재현 불가
  DETERMINISTIC  — 고정 팀. 규제 배포용. 효율을 감사 가능성과 맞바꾼다
  REPLAY         — 기록된 선택을 재생. 과거 판단의 정확한 재현

그리고 두 가지 함정을 코드로 막는다.

  콜드 스타트 — 기여도(value)는 순환적이다. 돌려보기 전에는 모른다.
                궤적이 충분히 쌓이기 전에는 동적 모드를 켜지 않는다.
  다양성 붕괴 — 순수 비용·성능 최적화는 가장 좋은 모델들을 뽑고,
                가장 좋은 모델들은 대개 서로 비슷하다. 최적화가 다양성을
                체계적으로 제거하므로 다양성을 제약으로 명시한다.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

from ..trace import SpanKind, Tracer


class BrokerMode(str, Enum):
    DYNAMIC = "dynamic"
    DETERMINISTIC = "deterministic"
    REPLAY = "replay"


@dataclass(frozen=True, slots=True)
class Candidate:
    """선택 후보.

    family 가 다양성 제약의 단위다. 같은 계열 모델은 상관된 오류를 내므로
    peer review 가 틀린 답에 확신 있는 합의를 만든다.
    """

    ref: str
    family: str
    cost: float           # weight — 토큰 단가, 지연, VRAM 등의 합성
    vram_mb: float = 0.0
    value: float | None = None   # 기여도 추정. None 이면 미측정 = 콜드 스타트


@dataclass(slots=True)
class Selection:
    chosen: tuple[str, ...]
    mode: BrokerMode
    reason: str
    total_cost: float = 0.0
    peak_vram_mb: float = 0.0
    families: tuple[str, ...] = ()
    cold_start: bool = False


class ColdStart(RuntimeError):
    """기여도 데이터 없이 동적 선택을 시도했다."""


@dataclass
class Broker:
    """비용 제약 아래 다양성을 지키며 팀을 고른다.

    capacity 는 VRAM 예산이다 — 앙상블의 실제 가치는 총 연산 절감이 아니라
    **단일 장치 최대 메모리 우회**이므로, 여기가 진짜 제약이다.
    """

    mode: BrokerMode = BrokerMode.DETERMINISTIC
    capacity_vram_mb: float = float("inf")
    cost_ceiling: float = float("inf")
    team_size: int = 3
    max_per_family: int = 1          # 다양성 제약. 1이면 계열 중복 금지.
    fixed_team: tuple[str, ...] = ()  # DETERMINISTIC 모드의 고정 팀
    min_observations: int = 30        # 콜드 스타트 임계
    observations: int = 0
    _replay_log: dict[str, tuple[str, ...]] = field(default_factory=dict)

    # -- 재현 지원 -------------------------------------------------------

    def load_replay(self, decisions: dict[str, Sequence[str]]) -> None:
        """궤적에서 추출한 과거 선택을 적재한다."""
        self._replay_log = {k: tuple(v) for k, v in decisions.items()}

    # -- 선택 -------------------------------------------------------------

    def select(
        self,
        candidates: Sequence[Candidate],
        *,
        task_key: str = "",
        tracer: Tracer | None = None,
    ) -> Selection:
        tracer = tracer or Tracer()
        with tracer.span("broker", SpanKind.SELECTION, mode=self.mode.value) as span:
            sel = self._select_inner(candidates, task_key)
            # 선택은 반드시 궤적에 남는다 — 이것이 재현 모드의 입력이다.
            span.attributes.update(
                chosen=list(sel.chosen),
                reason=sel.reason,
                total_cost=sel.total_cost,
                peak_vram_mb=sel.peak_vram_mb,
                families=list(sel.families),
                task_key=task_key,
                cold_start=sel.cold_start,
            )
            return sel

    def _select_inner(self, candidates: Sequence[Candidate], task_key: str) -> Selection:
        by_ref = {c.ref: c for c in candidates}

        if self.mode is BrokerMode.REPLAY:
            recorded = self._replay_log.get(task_key)
            if recorded is None:
                raise KeyError(
                    f"재현 모드인데 task_key='{task_key}' 의 기록이 없습니다. "
                    f"재현은 기록된 선택에 대해서만 가능합니다."
                )
            return self._finish(recorded, by_ref, BrokerMode.REPLAY, "기록된 선택 재생")

        if self.mode is BrokerMode.DETERMINISTIC:
            team = self.fixed_team or tuple(c.ref for c in candidates[: self.team_size])
            return self._finish(
                team, by_ref, BrokerMode.DETERMINISTIC, "고정 팀 — 재현성 우선"
            )

        # DYNAMIC
        if self.observations < self.min_observations:
            raise ColdStart(
                f"기여도 관측이 부족합니다 ({self.observations}/{self.min_observations}). "
                f"value 는 순환적이라 돌려보기 전에는 모릅니다. 궤적이 쌓일 때까지 "
                f"DETERMINISTIC 모드로 운영하십시오."
            )
        if any(c.value is None for c in candidates):
            raise ColdStart("기여도(value)가 없는 후보가 있습니다. 동적 선택 불가.")

        chosen = self._knapsack_with_diversity(candidates)
        return self._finish(
            tuple(c.ref for c in chosen), by_ref, BrokerMode.DYNAMIC, "동적 선택 (다양성 제약 적용)"
        )

    def _knapsack_with_diversity(self, candidates: Sequence[Candidate]) -> list[Candidate]:
        """가치/비용 비로 탐욕 선택하되 계열 중복을 제한한다.

        다양성 제약이 없으면 브로커는 항상 가장 좋은 모델들을 뽑고,
        그것들은 대개 서로 비슷해 groupthink 가 오히려 심해진다.
        """
        ranked = sorted(
            candidates,
            key=lambda c: ((c.value or 0.0) / c.cost if c.cost else float("inf")),
            reverse=True,
        )
        chosen: list[Candidate] = []
        family_count: dict[str, int] = {}
        vram = 0.0
        cost = 0.0
        for c in ranked:
            if len(chosen) >= self.team_size:
                break
            if family_count.get(c.family, 0) >= self.max_per_family:
                continue
            if vram + c.vram_mb > self.capacity_vram_mb:
                continue
            if cost + c.cost > self.cost_ceiling:
                continue
            chosen.append(c)
            family_count[c.family] = family_count.get(c.family, 0) + 1
            vram += c.vram_mb
            cost += c.cost
        return chosen

    def _finish(
        self,
        refs: Sequence[str],
        by_ref: dict[str, Candidate],
        mode: BrokerMode,
        reason: str,
    ) -> Selection:
        picked = [by_ref[r] for r in refs if r in by_ref]
        return Selection(
            chosen=tuple(p.ref for p in picked),
            mode=mode,
            reason=reason,
            total_cost=sum(p.cost for p in picked),
            peak_vram_mb=max((p.vram_mb for p in picked), default=0.0),
            families=tuple(sorted({p.family for p in picked})),
            cold_start=self.observations < self.min_observations,
        )


def error_correlation(outcomes: dict[str, Sequence[bool]]) -> dict[tuple[str, str], float]:
    """모델 쌍의 오류 상관도.

    "어떤 모델 쌍이 같은 문제에서 함께 틀렸는가" — 측정 가능한 groupthink 지표다.
    궤적 로그에서 직접 계산되며, 브로커의 다양성 제약을 데이터로 뒷받침한다.
    """
    refs = sorted(outcomes)
    out: dict[tuple[str, str], float] = {}
    for i, a in enumerate(refs):
        for b in refs[i + 1 :]:
            xs, ys = outcomes[a], outcomes[b]
            n = min(len(xs), len(ys))
            if n == 0:
                continue
            both_wrong = sum(1 for k in range(n) if not xs[k] and not ys[k])
            either_wrong = sum(1 for k in range(n) if not xs[k] or not ys[k])
            out[(a, b)] = both_wrong / either_wrong if either_wrong else 0.0
    return out
