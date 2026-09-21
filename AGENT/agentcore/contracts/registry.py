"""컴포넌트 레지스트리.

버전은 교체가 아니라 **병존**한다. v1 은 사용량이 0이 될 때 은퇴한다.
레지스트리는 그 판단에 필요한 사용량을 세고, 폴백 경로를 보장한다.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .component import Component
from .contract import Role


class ComponentNotFound(KeyError):
    pass


@dataclass
class Registry:
    """이름 → 버전 → 컴포넌트.

    resolve() 가 폴백을 자동으로 따라가므로, T2 가 회귀하면 T1 으로
    되돌아가는 경로가 코드 수정 없이 확보된다.
    """

    _by_name: dict[str, dict[str, Component]] = field(default_factory=lambda: defaultdict(dict))
    _usage: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def register(self, component: Component) -> Component:
        c = component.contract
        existing = self._by_name[c.name].get(c.version)
        if existing is not None:
            raise ValueError(
                f"{c.ref} 은 이미 등록되어 있습니다. 계약은 불변이며 "
                f"변경은 새 버전을 의미합니다."
            )
        self._by_name[c.name][c.version] = component
        return component

    def versions(self, name: str) -> list[str]:
        return sorted(self._by_name.get(name, {}))

    def get(self, name: str, version: str | None = None) -> Component:
        table = self._by_name.get(name)
        if not table:
            raise ComponentNotFound(f"등록되지 않은 컴포넌트: {name}")
        if version is None:
            live = [v for v, c in table.items() if not c.contract.deprecated]
            if not live:
                raise ComponentNotFound(f"{name} 의 모든 버전이 폐기되었습니다")
            version = sorted(live)[-1]
        if version not in table:
            raise ComponentNotFound(f"{name}@{version} 없음. 사용 가능: {self.versions(name)}")
        self._usage[f"{name}@{version}"] += 1
        return table[version]

    def resolve(self, name: str, version: str | None = None) -> Component:
        """폴백을 따라가며 해석한다. T2 실패 시 T1 으로 자동 강등."""
        try:
            return self.get(name, version)
        except ComponentNotFound:
            for table in self._by_name.values():
                for comp in table.values():
                    if comp.contract.fallback == name:
                        return comp
            raise

    def by_role(self, role: Role) -> list[Component]:
        return [
            c
            for table in self._by_name.values()
            for c in table.values()
            if c.contract.role is role and not c.contract.deprecated
        ]

    def usage(self) -> dict[str, int]:
        """버전별 호출 횟수. 은퇴 판단의 근거 — 사용량 0이면 은퇴 후보."""
        return dict(self._usage)

    def retirement_candidates(self) -> list[str]:
        refs = [
            f"{name}@{ver}"
            for name, table in self._by_name.items()
            for ver in table
            if len(table) > 1
        ]
        return [r for r in refs if self._usage.get(r, 0) == 0]

    def parametric_components(self) -> list[str]:
        """망각 위험과 회귀 검증 부채를 지는 컴포넌트 목록.

        이 목록이 길어지면 P3 를 스스로 재도입하고 있다는 신호다.
        """
        return [
            c.contract.ref
            for table in self._by_name.values()
            for c in table.values()
            if c.contract.is_parametric
        ]
