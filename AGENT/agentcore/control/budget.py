"""예산.

1부 원칙: **가장자리는 결정론적, 가운데만 동적.**
동적 구간(Orchestrator-Workers)은 반드시 코드가 감싼 예산 안에 있어야 한다.
예산 없는 동적 생성은 비용에 상한이 없다는 뜻이다.

예산은 중첩된다 — 자식이 쓴 만큼 부모에서도 차감되므로, 하위 워커가
전체 예산을 초과할 수 없다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


class BudgetExceeded(Exception):
    def __init__(self, kind: str, limit: float, used: float) -> None:
        self.kind = kind
        self.limit = limit
        self.used = used
        super().__init__(f"예산 초과 [{kind}]: 상한 {limit}, 사용 {used}")


@dataclass
class Budget:
    """토큰·시간·깊이·호출·워커 수의 상한.

    max_depth 는 재귀적 하위작업 생성의 깊이를 막는다. 이것이 없으면
    오케스트레이터가 자기 자신을 무한히 위임할 수 있다.
    """

    max_tokens: int | None = None
    max_wall_seconds: float | None = None
    max_depth: int = 3
    max_calls: int | None = None
    max_workers: int = 8

    used_tokens: int = 0
    used_calls: int = 0
    depth: int = 0
    _started: float = field(default_factory=time.perf_counter)
    _parent: "Budget | None" = field(default=None, repr=False)

    def child(self, **overrides: object) -> Budget:
        """하위 예산. 부모의 남은 예산을 넘을 수 없다."""
        remaining_tokens = self.remaining_tokens
        kwargs: dict[str, object] = {
            "max_tokens": remaining_tokens,
            "max_wall_seconds": self.remaining_seconds,
            "max_depth": self.max_depth,
            "max_calls": None if self.max_calls is None else self.max_calls - self.used_calls,
            "max_workers": self.max_workers,
        }
        kwargs.update(overrides)
        b = Budget(**kwargs)  # type: ignore[arg-type]
        b.depth = self.depth + 1
        b._parent = self
        if b.depth > self.max_depth:
            raise BudgetExceeded("depth", self.max_depth, b.depth)
        return b

    @property
    def remaining_tokens(self) -> int | None:
        return None if self.max_tokens is None else max(0, self.max_tokens - self.used_tokens)

    @property
    def remaining_seconds(self) -> float | None:
        if self.max_wall_seconds is None:
            return None
        return max(0.0, self.max_wall_seconds - (time.perf_counter() - self._started))

    def spend(self, tokens: int = 0, calls: int = 1) -> None:
        """사용량을 차감하고 부모로 전파한다."""
        self.used_tokens += tokens
        self.used_calls += calls
        if self._parent is not None:
            self._parent.spend(tokens=tokens, calls=calls)
        self.check()

    def check(self) -> None:
        if self.max_tokens is not None and self.used_tokens > self.max_tokens:
            raise BudgetExceeded("tokens", self.max_tokens, self.used_tokens)
        if self.max_calls is not None and self.used_calls > self.max_calls:
            raise BudgetExceeded("calls", self.max_calls, self.used_calls)
        if self.max_wall_seconds is not None:
            elapsed = time.perf_counter() - self._started
            if elapsed > self.max_wall_seconds:
                raise BudgetExceeded("wall_seconds", self.max_wall_seconds, elapsed)

    def workers_allowed(self, requested: int) -> int:
        """요청한 워커 수를 상한으로 자른다. 초과는 오류가 아니라 절삭."""
        return max(1, min(requested, self.max_workers))


UNLIMITED = Budget(max_depth=8)
