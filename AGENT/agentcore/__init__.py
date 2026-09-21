"""agentcore — 고정 코어와 적응하는 주변부.

설계 기준 1부를 코드로 옮긴 것. 원칙은 문서가 아니라 타입과 런타임 검사로
강제된다 — 조용히 위반될 수 있는 원칙은 문서일 뿐이기 때문이다.

강제되는 원칙
  · 코어 파인튜닝(사다리 7단) 금지            contracts.ComponentContract
  · 검증기는 파라메트릭 적응 대상이 아님       contracts.ComponentContract
  · 막는 실패를 이름대지 못한 패턴은 거절      control.patterns
  · E-O 는 외부 검증 신호만 받음               control.refine
  · 동적 구간은 예산 안에서만                  control.orchestrate
  · 가드레일은 점수를 만들지 않음              verify.GuardrailReport
  · 캘리브레이션 없는 Judge 는 운영 불가       evaluation.CalibratedJudge
  · 검증기 다운 시 출력 차단(fail closed)      broker.QuorumPolicy
  · 관측 없는 동적 브로커는 콜드 스타트 거부   broker.Broker
  · 단일 계열 앙상블은 앙상블이 아님           ensemble.Deliberation
  · 자기 초안에 투표할 수 없음                 ensemble.quadratic_scores
  · 진입 조건 없이 파인튜닝 금지               distill.publish_contract
  · 폴백 없는 T2 는 발행 불가                  distill.publish_contract
  · T1 을 못 이긴 T2 는 배포되지 않음          evaluation.decide_swap
"""

from . import (
    broker,
    contracts,
    control,
    distill,
    ensemble,
    evaluation,
    memory,
    models,
    skills,
    trace,
    verify,
)

__version__ = "0.1.0"

__all__ = [
    "broker",
    "contracts",
    "control",
    "distill",
    "ensemble",
    "evaluation",
    "memory",
    "models",
    "skills",
    "trace",
    "verify",
]
