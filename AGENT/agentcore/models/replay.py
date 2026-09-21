"""record–replay 카세트.

1부 평가 인프라 규칙 1: **도구 응답 고정.**
외부 응답을 녹화해 재생하지 않으면 회귀 테스트가 불가능하다 — 외부 세계가
바뀌고, 돈이 들고, 속도 제한에 걸린다.

카세트에는 환경 버전을 함께 고정한다. 스키마나 인덱스가 바뀌면 캐시된
벤치마크가 조용히 현실과 괴리되기 때문이다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import LLM, Completion, Message, cassette_key


class CassetteMiss(KeyError):
    """재생 중 없는 호출. 조용히 실호출로 넘어가지 않는다 —
    그러면 결정론이 깨지고 그 사실을 아무도 모른다."""


@dataclass
class Cassette:
    """녹화된 호출 모음."""

    path: Path
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> Cassette:
        p = Path(path)
        if not p.exists():
            return cls(path=p)
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls(
            path=p,
            entries=data.get("entries", {}),
            environment=data.get("environment", {}),
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {"environment": self.environment, "entries": self.entries},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def assert_environment(self, expected: dict[str, str]) -> None:
        """환경 버전 대조.

        스키마·인덱스·도구 버전이 녹화 시점과 다르면 벤치마크는 이미
        현실과 괴리되어 있다. 조용히 통과시키지 않는다.
        """
        drift = {
            k: (self.environment.get(k), v)
            for k, v in expected.items()
            if self.environment.get(k) != v
        }
        if drift:
            raise RuntimeError(
                f"카세트 환경이 현재와 다릅니다 (녹화, 현재): {drift}. "
                f"벤치마크를 재생성하거나 환경을 고정하십시오."
            )


class RecordingLLM(LLM):
    """실제 모델을 감싸 호출을 녹화한다."""

    def __init__(self, inner: LLM, cassette: Cassette) -> None:
        self.inner = inner
        self.cassette = cassette
        self.name = f"record:{inner.name}"
        self.model_id = inner.model_id
        self.size_hint_b = inner.size_hint_b
        self.peak_vram_mb = inner.peak_vram_mb

    def _complete(self, messages: list[Message], **params: Any) -> Completion:
        key = cassette_key(self.model_id, messages, params)
        out = self.inner._complete(messages, **params)
        self.cassette.entries[key] = {
            "text": out.text,
            "input_tokens": out.input_tokens,
            "output_tokens": out.output_tokens,
            "model": out.model,
            "stop_reason": out.stop_reason,
        }
        return out


class ReplayLLM(LLM):
    """카세트에서 재생한다. 완전 결정론적이며 외부 호출이 없다."""

    def __init__(self, cassette: Cassette, model_id: str = "", strict: bool = True) -> None:
        self.cassette = cassette
        self.model_id = model_id
        self.name = f"replay:{model_id or 'any'}"
        self.strict = strict
        self.misses: list[str] = []

    def _complete(self, messages: list[Message], **params: Any) -> Completion:
        key = cassette_key(self.model_id, messages, params)
        entry = self.cassette.entries.get(key)
        if entry is None:
            self.misses.append(key)
            if self.strict:
                preview = messages[-1].content[:80] if messages else ""
                raise CassetteMiss(
                    f"카세트에 없는 호출: {key} ({preview!r}). "
                    f"strict=False 로 두면 결정론이 조용히 깨집니다."
                )
            return Completion(text="", model=self.model_id, stop_reason="cassette_miss")
        return Completion(
            text=entry["text"],
            input_tokens=entry.get("input_tokens", 0),
            output_tokens=entry.get("output_tokens", 0),
            model=entry.get("model", self.model_id),
            stop_reason=entry.get("stop_reason", "end_turn"),
        )
