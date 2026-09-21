"""연구·학습 도메인의 골드셋 — 출처 재현율.

1부 권고: **투자 대비 효과가 가장 큰 골드셋은 출처 재현율이다.**

전문가가 "이 질문에는 이 출처들이 반드시 나와야 한다"만 나열하면 된다.
보고서를 쓰게 하는 것보다 훨씬 싸고, 가장 중요한 실패를 잡는다.

  · 답이 맞아도 근거를 못 찾고 나왔으면 운이었고, 다음엔 안 된다
  · 궤적 층위를 직접 측정하는 유일한 방법이다
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any


@dataclass(slots=True)
class SourceGold:
    """한 질문의 정답 출처 집합.

    required  — 없으면 답이 성립하지 않는 출처. 재현율의 분모.
    helpful   — 있으면 좋으나 필수는 아님. 재현율에 안 들어간다.
    forbidden — 나오면 안 되는 출처. 철회된 논문, 신뢰할 수 없는 매체 등.
    """

    question_id: str
    question: str
    required: frozenset[str] = frozenset()
    helpful: frozenset[str] = frozenset()
    forbidden: frozenset[str] = frozenset()
    curator: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if not self.required:
            raise ValueError(
                f"{self.question_id}: required 출처가 비었습니다. "
                f"필수 출처를 못 적는 질문은 골드셋에 넣을 수 없습니다 — "
                f"무엇이 정답인지 모른다는 뜻입니다."
            )
        overlap = self.required & self.forbidden
        if overlap:
            raise ValueError(f"{self.question_id}: required 와 forbidden 이 겹칩니다: {overlap}")


@dataclass(slots=True)
class RetrievalScore:
    question_id: str
    recall: float
    precision: float
    forbidden_hits: tuple[str, ...]
    missed: tuple[str, ...]
    retrieved: int

    @property
    def clean(self) -> bool:
        """금지 출처가 하나도 없는가. 비율이 아니라 개수로 판정한다."""
        return not self.forbidden_hits


def score_retrieval(gold: SourceGold, retrieved: Iterable[str]) -> RetrievalScore:
    got = set(retrieved)
    hit = got & gold.required
    missed = gold.required - got
    forbidden = got & gold.forbidden
    # 정밀도의 분모에서 helpful 은 제외한다 — 유용한 출처를 가져온 것을
    # 오답으로 세면 넓게 찾는 것이 벌점이 된다.
    counted = got - gold.helpful
    useful = hit
    return RetrievalScore(
        question_id=gold.question_id,
        recall=len(hit) / len(gold.required),
        precision=len(useful) / len(counted) if counted else 0.0,
        forbidden_hits=tuple(sorted(forbidden)),
        missed=tuple(sorted(missed)),
        retrieved=len(got),
    )


@dataclass
class SourceGoldset:
    """질문별 정답 출처 모음. JSONL 로 오간다 — 전문가가 손으로 채운다."""

    name: str
    items: list[SourceGold] = field(default_factory=list)

    def add(self, gold: SourceGold) -> None:
        self.items.append(gold)

    @classmethod
    def load(cls, path: str | Path, name: str = "") -> SourceGoldset:
        gs = cls(name=name or Path(path).stem)
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                row = json.loads(line)
                gs.add(
                    SourceGold(
                        question_id=row["question_id"],
                        question=row["question"],
                        required=frozenset(row.get("required", ())),
                        helpful=frozenset(row.get("helpful", ())),
                        forbidden=frozenset(row.get("forbidden", ())),
                        curator=row.get("curator", ""),
                        note=row.get("note", ""),
                    )
                )
        return gs

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            for g in self.items:
                fh.write(
                    json.dumps(
                        {
                            "question_id": g.question_id,
                            "question": g.question,
                            "required": sorted(g.required),
                            "helpful": sorted(g.helpful),
                            "forbidden": sorted(g.forbidden),
                            "curator": g.curator,
                            "note": g.note,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return p

    def evaluate(self, results: dict[str, Sequence[str]]) -> dict[str, Any]:
        """질문별 검색 결과를 채점한다.

        forbidden_total 은 **개수**로 보고된다 — 가드레일이므로 평균하지 않는다.
        """
        scores = [
            score_retrieval(g, results.get(g.question_id, ()))
            for g in self.items
            if g.question_id in results
        ]
        if not scores:
            return {"n": 0}
        return {
            "n": len(scores),
            "recall": round(fmean(s.recall for s in scores), 4),
            "precision": round(fmean(s.precision for s in scores), 4),
            "perfect_recall": sum(1 for s in scores if s.recall == 1.0),
            # 가드레일 — 집계하지 않고 개수와 위치만 보고한다
            "forbidden": {
                "total_hits": sum(len(s.forbidden_hits) for s in scores),
                "questions": [s.question_id for s in scores if not s.clean],
                "note": "금지 출처는 꼬리 지표다. 한 건이라도 있으면 실패다",
            },
            "worst_missed": sorted(
                ((s.question_id, s.missed) for s in scores if s.missed),
                key=lambda x: -len(x[1]),
            )[:5],
        }
