"""N-way 심의 — 코어 자리를 채우는 구현 선택지.

**대안 아키텍처가 아니다.** 강한 단일 코어를 쓸 수 없는 환경(온프레미스,
에어갭, VRAM 제약)에서 `Role.CORE` 박스를 앙상블로 구현한 것이다.
인터페이스가 같으므로 상위 코드는 바뀌지 않는다.

정직하게 적어둘 것: **이것은 비용 최적화가 아니다.**
N개 모델 × R 라운드는 단일 대형 모델 1회보다 총 연산이 많다. 절약되는 것은
총 연산이 아니라 **단일 장치 최대 메모리**다. 100B 를 못 올리는 장비에
20B 다섯 개는 올라간다 — 그것이 전부이고, 그것으로 충분한 상황이 있다.

그리고 앙상블은 **오류가 독립일 때만** 이긴다. 비슷하게 학습된 모델들은
상관된 오류를 내고, 같은 맹점을 가진 모델끼리의 peer review 는 틀린 답에
확신 있는 합의를 만든다. 다양성은 가정이 아니라 설계 대상이다.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..broker import Candidate
from ..trace import SpanKind, Tracer, record_usage


@dataclass(slots=True)
class Draft:
    """모델 하나의 초안. 익명 심사를 위해 author 는 심사 시 가려진다."""

    author: str
    family: str
    content: Any
    tokens: int = 0


@dataclass(slots=True)
class Review:
    """익명 peer review.

    reviewer 는 자기 초안에 투표할 수 없다 — LLM 은 자기 출력을 선호하는
    경향이 문서화되어 있고, 자기 투표를 허용하면 그 편향이 그대로 결과가 된다.
    """

    reviewer: str
    target: str
    credits: float          # 이차 투표에 쓰이는 배분 크레딧
    rationale: str = ""


@dataclass(slots=True)
class Consensus:
    winner: str | None
    scores: dict[str, float]
    rounds: int
    unanimous: bool
    families: tuple[str, ...]
    diversity_ok: bool
    total_tokens: int = 0
    peak_vram_mb: float = 0.0


class SelfVoteRejected(ValueError):
    pass


class InsufficientDiversity(ValueError):
    """같은 계열 모델만으로 구성된 앙상블은 앙상블이 아니다."""


def quadratic_scores(reviews: Sequence[Review]) -> dict[str, float]:
    """이차 투표 집계.

    단순 다수결이 아니다. 한 심사자가 한 대상에 몰아준 크레딧은 제곱근으로
    환산되므로, 강한 확신 하나가 약한 선호 여럿을 무한정 이기지 못한다.
    확신도를 반영하되 독점은 막는 절충이다.
    """
    per_reviewer: dict[str, dict[str, float]] = {}
    for r in reviews:
        if r.reviewer == r.target:
            raise SelfVoteRejected(
                f"{r.reviewer} 가 자기 초안에 투표했습니다. 자기 선호 편향이 "
                f"그대로 결과가 됩니다."
            )
        per_reviewer.setdefault(r.reviewer, {})
        per_reviewer[r.reviewer][r.target] = (
            per_reviewer[r.reviewer].get(r.target, 0.0) + max(0.0, r.credits)
        )

    scores: dict[str, float] = {}
    for allocations in per_reviewer.values():
        for target, credits in allocations.items():
            scores[target] = scores.get(target, 0.0) + math.sqrt(credits)
    return scores


@dataclass
class Deliberation:
    """N-way 자기평가 심의.

    propose → 익명 review → 합의 상태 되먹임 → 종료 판정.
    라운드 수와 참여 모델 수가 곧 비용이므로 둘 다 상한이 있다.
    """

    propose: Callable[[Any, str, dict[str, float]], Draft]
    review: Callable[[str, Sequence[Draft]], list[Review]]
    members: Sequence[Candidate]
    max_rounds: int = 2
    min_families: int = 2
    convergence: float = 0.6      # 최고 득점이 총점의 이 비율을 넘으면 종료

    def __post_init__(self) -> None:
        families = {m.family for m in self.members}
        if len(families) < self.min_families:
            raise InsufficientDiversity(
                f"계열이 {len(families)}종뿐입니다(최소 {self.min_families}). "
                f"비슷하게 학습된 모델은 상관된 오류를 내므로, 같은 맹점을 가진 "
                f"모델끼리의 심사는 틀린 답에 확신 있는 합의를 만듭니다."
            )

    def run(self, payload: Any, *, tracer: Tracer | None = None) -> Consensus:
        tracer = tracer or Tracer()
        with tracer.span(
            "deliberate",
            SpanKind.MODEL,
            members=[m.ref for m in self.members],
            families=sorted({m.family for m in self.members}),
        ) as span:
            state: dict[str, float] = {}
            drafts: list[Draft] = []
            total_tokens = 0
            rounds = 0

            for rounds in range(1, self.max_rounds + 1):
                with tracer.span(f"deliberate:round{rounds}", SpanKind.CONTROL, round=rounds):
                    drafts = []
                    for m in self.members:
                        d = self.propose(payload, m.ref, state)
                        d.family = m.family
                        drafts.append(d)
                        total_tokens += d.tokens

                    reviews: list[Review] = []
                    for m in self.members:
                        # 심사자는 자기 초안을 볼 수 없다 — 익명성과 자기투표 금지.
                        visible = [d for d in drafts if d.author != m.ref]
                        reviews.extend(self.review(m.ref, visible))

                    state = quadratic_scores(reviews)

                total = sum(state.values()) or 1.0
                top = max(state.values(), default=0.0)
                if top / total >= self.convergence:
                    break

            record_usage(output_tokens=total_tokens)
            winner = max(state, key=state.__getitem__) if state else None
            families = tuple(sorted({m.family for m in self.members}))
            result = Consensus(
                winner=winner,
                scores=state,
                rounds=rounds,
                unanimous=len(state) == 1,
                families=families,
                diversity_ok=len(families) >= self.min_families,
                total_tokens=total_tokens,
                peak_vram_mb=max((m.vram_mb for m in self.members), default=0.0),
            )
            span.attributes.update(
                winner=result.winner,
                rounds=result.rounds,
                scores={k: round(v, 3) for k, v in result.scores.items()},
                peak_vram_mb=result.peak_vram_mb,
                # 이 둘을 함께 남기는 이유: 앙상블의 가치는 총 연산 절감이 아니라
                # 최대 메모리 우회이므로, 둘을 같이 봐야 판단이 선다.
                total_tokens=result.total_tokens,
            )
            return result


# --------------------------------------------------------------------------
# 열린 출력의 투표 단위
# --------------------------------------------------------------------------

@dataclass
class ClaimVote:
    """주장 단위 투표.

    AIME·코드처럼 검증 가능한 단답은 투표할 "같은 답"이 존재하지만,
    열린 서술은 그렇지 않다. 보고서 전체가 아니라 **주장 단위**로 쪼개면
    투표가 성립하고, 근거 검증 인프라와 단위가 공유된다.
    """

    claim_id: str
    supporters: set[str] = field(default_factory=set)
    dissenters: set[str] = field(default_factory=set)

    @property
    def support_ratio(self) -> float:
        total = len(self.supporters) + len(self.dissenters)
        return len(self.supporters) / total if total else 0.0


def vote_on_claims(
    ballots: dict[str, dict[str, bool]],
    *,
    threshold: float = 0.5,
) -> tuple[list[str], dict[str, ClaimVote]]:
    """모델별 주장 찬반을 모아 통과 주장을 고른다.

    ballots: {model_ref: {claim_id: supports?}}
    """
    votes: dict[str, ClaimVote] = {}
    for model, opinions in ballots.items():
        for claim_id, supports in opinions.items():
            v = votes.setdefault(claim_id, ClaimVote(claim_id))
            (v.supporters if supports else v.dissenters).add(model)
    passed = [cid for cid, v in votes.items() if v.support_ratio > threshold]
    return sorted(passed), votes
