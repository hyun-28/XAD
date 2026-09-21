# agentcore

**고정 코어와 적응하는 주변부** — 범용 멀티에이전트 시스템 골격.

추론 코어를 동결하고 모든 적응을 교체 가능한 주변 계층으로 밀어내는 아키텍처의
실행 가능한 구현. 설계 기준 1부를 코드로 옮긴 것이다.

핵심 설계 결정 하나: **원칙은 문서가 아니라 타입과 런타임 검사로 강제된다.**
조용히 위반될 수 있는 원칙은 문서일 뿐이다.

```bash
pip install -e ".[dev]"
python -m pytest                     # 216 passed
python examples/tutorial.py          # 10단계 튜토리얼 — 여기서 시작
python examples/tutorial.py 6        # 6단계만
python examples/webchat.py           # 웹 챗봇 (pip install -e ".[web]")
```

`tutorial.py` 는 각 단계에서 **먼저 무엇이 깨지는지 보여주고, 그 다음 그것을
막는 계층을 붙인다.** 이 순서가 설계 원칙 그 자체다 — 막는 실패를 이름댈 수
없는 계층은 넣지 않는다.

---

## 강제되는 원칙

| 원칙 | 강제 지점 | 위반 시 |
|---|---|---|
| 코어는 파인튜닝하지 않는다 | `ComponentContract(ladder_rung=7)` | `ValueError` |
| 검증기는 적응하지 않는다 | `ComponentContract(role=VERIFIER, rung>=6)` | `ValueError` |
| 막는 실패를 이름댄 패턴만 | `control.patterns(prevents=...)` | `PatternWithoutJustification` |
| E-O 는 외부 신호만 받는다 | `control.refine(signal)` | `NonExternalSignal` |
| 동적 구간은 예산 안에서만 | `control.orchestrate(budget=...)` | 필수 인자 |
| 가드레일은 집계되지 않는다 | `GuardrailReport` | `score` 속성 부재 |
| Judge 는 캘리브레이션 필수 | `CalibratedJudge.__call__` | `UncalibratedJudge` |
| 검증기 다운 시 출력 차단 | `QuorumPolicy.decide` | `ServiceLevel.HALTED` |
| 관측 없는 동적 선택 금지 | `Broker(mode=DYNAMIC)` | `ColdStart` |
| 유효 범위 밖은 명시적 실패 | `Component.__call__` | `OUT_OF_ENVELOPE` |
| 카세트 미스는 시끄럽게 | `ReplayLLM` | `CassetteMiss` |
| 죽은 스킬은 쓰이지 않는다 | `SkillLibrary.use` | `RuntimeError` |
| 단일 계열 앙상블은 앙상블이 아니다 | `Deliberation.__post_init__` | `InsufficientDiversity` |
| 자기 초안에 투표할 수 없다 | `quadratic_scores` | `SelfVoteRejected` |
| 진입 조건 없이 파인튜닝 금지 | `distill.publish_contract` | `EntryConditionsUnmet` |
| 폴백 없는 T2 는 발행 불가 | 같은 곳 | `ValueError` |

`tests/test_principles.py` 가 이 표를 검증한다. 이 파일이 깨지면 아키텍처가
아니라 **논지**가 깨진 것이다.

---

## 모듈

| 모듈 | 역할 | 상각 |
|---|---|---|
| `trace/` | 궤적 로그. 1일차 인프라 — 나머지 전부가 여기 의존한다 | 누적 |
| `contracts/` | 계약·컴포넌트·레지스트리. 동결 = 계약 발행 | 누적 |
| `control/` | 예산, 게이트, 5패턴. **제어는 전부 여기 있다** | 누적 |
| `verify/` | 검증기 3분할, 근거 대조, 가드레일 | 누적 |
| `memory/` | 검색기와 기억을 **분리**. 주체 단위 삭제 | 누적 |
| `skills/` | 스킬 라이브러리. 유일한 복리 계층 | 누적 |
| `models/` | 프로바이더 중립 + record–replay | 누적 |
| `broker/` | 3모드 선택, 다양성 제약, 정족수 | 혼합 |
| `ensemble/` | N-way 심의. 코어 자리의 구현 선택지 | 혼합 |
| `distill/` | 코어→라벨→계약. 사다리 5단→6단 경로 | 부분 |
| `evaluation/` | 4층위, 컴포넌트 교체 판단, Judge 캘리브레이션 | 누적 |
| `domains/` | 도메인 골드셋 — 연구·학습, 주식분석 | 누적 |
| `apps/` | 조립된 에이전트. Q&A 챗봇과 로컬 웹 UI | 누적 |
| `report.py` | 궤적 분석 CLI | 누적 |

**넷 중 하나만 상각된다** — 파인튜닝된 소형 모델. 나머지는 코어를 교체해도
살아남고, 더 강한 코어가 같은 도구를 더 잘 쓰므로 오히려 강화된다.

---

## 궤적이 먼저다

```python
from agentcore.trace import Tracer, JsonlSink, record_usage, record_isolation

tracer = Tracer(sink=JsonlSink("runs/trace.jsonl"))
with tracer.span("pipeline"):
    with tracer.span("worker"):
        record_usage(input_tokens=9000, output_tokens=150)
        record_isolation(absorbed=8850, emitted=150)   # 원문은 코어로 안 올라간다
```

자식 사용량이 부모로 자동 합산되므로 시스템 층위 집계가 공짜로 나온다.
그리고 `record_isolation` 이 **격리 효과**를 남긴다 — 모놀리식 구조에서는
흡수가 0이라 항상 0이 되므로, 이 하나가 아키텍처를 판별한다.

---

## 계약 없이 발행하지 않는다

```python
from agentcore.contracts import *

contract = ComponentContract(
    name="relevance_filter", version="1.0", role=Role.SPECIALIST,
    interface=InterfaceSpec("list[Doc]", "list[Doc]", failure_modes=("empty_input",)),
    performance=PerformanceSpec("filter_v1", {"precision": 0.91, "recall": 0.88}),
    envelope=OperatingEnvelope("문서 200건 이하", lambda docs: len(docs) <= 200),
    ladder_rung=6,                    # 파라메트릭 — 망각 위험 구간
    amortization=Amortization.PARTIAL,
    fallback="bm25_rerank",           # T1 기준선이자 폴백
)
```

`envelope` 이 가장 많이 빠지고 가장 비싸다. 동결된 도구를 학습 분포 밖에서
쓰면 조용히 틀리기 때문에, 범위 이탈을 예외가 아니라 **명시적 결과**로 만든다.

```python
registry.parametric_components()   # 망각 위험을 지는 컴포넌트 목록
registry.retirement_candidates()   # 사용량 0인 구버전 — 은퇴 후보
```

`parametric_components()` 가 길어지면 P3(파국적 망각)를 스스로 재도입하고
있다는 신호다.

---

## 패턴은 정당화를 요구한다

```python
orchestrate(
    plan, worker, question,
    prevents="하위작업 개수를 사전에 알 수 없고, 원문이 코어 컨텍스트를 오염시킴",
    budget=Budget(max_tokens=200_000, max_workers=3, max_depth=2),
)
```

`prevents=` 는 선택 인자가 아니다. 이름을 못 대면 그 패턴은 들어가지 않는다.
`orchestrate` 의 `budget` 도 마찬가지 — 하위작업 개수를 모르는 유일한
패턴이므로 상한 없이는 비용에 상한이 없다.

### Evaluator-Optimizer 는 외부 신호만

```python
refine(regenerate, DeterministicSignal(grounding_check), claims,
       prevents="검증되지 않은 인용이 출력됨", max_rounds=2)

refine(regenerate, OpinionSignal(llm_taste), claims, prevents="...")
# NonExternalSignal: 생성자와 같은 정보만 보는 판정자는 맹점을 공유합니다
```

E-O 의 트리거는 "반복 개선이 중요하다"가 아니라 **"외부에 검증 가능한
개선 신호가 있다"** 이다. 신호가 없으면 표현만 매끄러워지고 사실성은 그대로다.

---

## 검증기는 셋으로 쪼갠다

"검증은 생성보다 쉽다"는 직관은 **증명서가 존재할 때만** 참이다.

| 티어 | 대상 | 적정 |
|---|---|---|
| `DECIDABLE` | 스키마, 형식, 원문 대조, 함의 | 1B–3B 또는 순수 코드 |
| `JUDGMENT` | 추론 타당성, 출처 충돌, 근거 충분성 | **코어** |
| `HUMAN` | 골드셋 주기 검증 | 사람 |

`VerifierStack` 은 싼 것부터 돌리고 결정 가능한 검증에서 걸리면 판단적 검증을
생략한다. 형식이 깨진 출력의 품질을 논하는 것은 의미가 없고 비싸다.

가드레일은 별개다 — **집계되지 않는다.**

```python
report.breach_count       # 개수
report.counts_by_rail()   # 레일별 개수
report.score              # AttributeError — 의도적
```

품질 92%는 훌륭하지만 가드레일 92%는 재앙이다. 점수를 만들면 언젠가 평균된다.

---

## 브로커: 재현성과 효율의 교환

```python
Broker(mode=BrokerMode.DETERMINISTIC, fixed_team=("core-a",))   # 규제 배포
Broker(mode=BrokerMode.DYNAMIC, max_per_family=1, observations=n)
Broker(mode=BrokerMode.REPLAY)   # 기록된 선택 재생
```

동적 선택은 효율성·적응성을 올리고 재현성·이의제기성을 내린다. 세 가지로 대응한다.

1. 선택은 **항상 궤적에 남는다** (`SpanKind.SELECTION`)
2. `selections_from_trace()` 로 재현 모드에 적재
3. 규제 배포는 `DETERMINISTIC` — 효율을 감사 가능성과 맞바꾼다

두 함정도 코드가 막는다.

- **콜드 스타트** — 기여도는 순환적이다. 관측이 쌓이기 전 동적 모드는 `ColdStart`
- **다양성 붕괴** — 순수 최적화는 가장 좋은 모델들을 뽑고, 그것들은 대개 서로
  비슷하다. `max_per_family` 가 그것을 막고, `error_correlation()` 이 측정한다

---

## 가용성: fail closed

```python
QuorumPolicy(min_by_role={Role.CORE: 3}, required_roles={Role.VERIFIER})
```

- 코어 3개 중 2개 생존 → `DEGRADED`, 진행하되 저하를 명시
- **검증기 부재 → `HALTED`, 출력 차단**

검증기는 신뢰의 뿌리이므로 그것 없이 나온 출력은 근거 없는 출력이다.
조용히 나쁜 답을 내는 것보다 멈추는 편이 낫다.

---

## 평가: 4층위

| 층위 | 없으면 |
|---|---|
| 컴포넌트 | T2 교체 판단 불가 — 모듈화 논지가 증명되지 않음 |
| 궤적 | 운 좋은 정답을 걸러내지 못함 |
| 출력 | 사용자 가치 미확인 |
| 시스템 | P1 과 상각 논거가 검증되지 않음 |

```python
agg.report()
# {"output": {"success_rate": ...},
#  "system": {"tokens_per_task": ..., "success_per_1k_tokens": ...},
#  "trajectory": {"isolation_ratio": ...},
#  "guardrails": {"breaches": {...}, "note": "꼬리 지표이므로 집계하지 않는다"}}
```

성공률은 **반드시 비용과 쌍으로** 읽힌다. 그리고 가드레일은 별도 블록이다.

### 결정론 우선

> 이 출력이 맞는지 프로그램으로 확인할 수 있는 하위 성질이 무엇인가?

그것을 최대한 뽑고 남은 것만 Judge 에 맡긴다. `DeterministicFirst.judge_load`
가 Judge 로 넘어간 비율을 재며, 낮을수록 좋다.

Judge 자체는 캘리브레이션 없이 못 쓴다 — 사람 라벨 30건 이상, 일치율 0.8 이상.
안 하면 리더보드가 모델이 아니라 Judge 의 변화를 측정한다.

### 궤적 → 벤치마크

```python
cases_from_trace("runs/trace.jsonl", only_interventions=True)
```

사용자가 개입한 지점 — 거부·수정·재시도 — 은 **사람이 붙여준 실패 라벨**이다.
손라벨링 없이 얻는 최고 품질의 데이터이며, 대부분의 시스템이 그냥 버린다.

---

## 챗봇: 골격을 대화 루프로 조립하면

```bash
python examples/chatbot.py                  # 내장 샘플 문서 + Claude 구독
python examples/chatbot.py --docs ./notes   # 디렉터리의 .md/.txt 를 코퍼스로
python examples/chatbot.py --llm echo       # 모델 없이 구조만 (무료·결정론)
```

일반 RAG 챗봇과 다른 점은 하나다 — **근거가 없으면 문장을 내보내지 않는다.**

```python
from agentcore.apps import ChatAgent
from agentcore.memory import Document
from agentcore.models import ClaudeCLILLM

agent = ChatAgent(documents, llm=ClaudeCLILLM())   # opus-5 · effort=high
ans = agent.ask("연차는 며칠인가요?", subject="user-42")

ans.text          # "· 연차는 입사 1년 차에 11일 부여된다"
ans.source_ids    # ('vacation',)
ans.degraded      # False — True 면 정상 답과 구분해 표시할 책임이 호출자에게 있다
agent.forget("user-42")   # 주체 단위 삭제. 문서 인덱스는 건드리지 않는다
```

모델 출력은 자유 텍스트로 받지 않는다. `- 문장 [출처ID]` 형식을 강제하고
파싱은 순수 코드가 한다 — 형식 준수는 프로그램이 판정할 수 있고, 판정
가능한 것을 판정자에게 맡기는 것은 낭비다.

| 경로 | 조건 | 결과 |
|---|---|---|
| 잡담 | 짧고 코퍼스와 무관 | 검색·모델 호출 없음 |
| 검색 실패 | 발췌 슬롯 미충족 | **모델을 부르지 않고** 무엇이 없는지 말한다 |
| 기권 | `[none]` | 실패가 아님. 재시도 루프에 넣지 않는다 |
| 허위 출처 | 검색되지 않은 ID 인용 | 가드레일 차단 |
| 근거 미달 | 원문에서 도출 안 됨 | 실패 항목만 재작성, 최대 2회 후 차단 |

`refine` 의 신호는 형식 검사와 원문 대조 — 둘 다 결정론적이므로
`DeterministicSignal` 이고, LLM 판정자를 넣으면 `NonExternalSignal` 로 거절된다.

### 두 경로는 검증 가능성에서 갈린다

문서를 안 올려도 대화는 된다. 다만 그 경로는 **검증 불가**로 표시된다.

| 경로 | 지나는 계층 | 표시 |
|---|---|---|
| `grounded` | 검색·격리 → 게이트 → 코어 → 근거 검증 → 가드레일 | 근거 확인됨 / 차단됨 |
| `free` | 코어만 | **검증 불가** |

자유 대화를 "저하된 grounded" 로 취급하지 않는다. 그것은 원리적으로 검증할
수 없는 출력이지 실패한 검증이 아니며, 둘을 한 플래그로 묶으면 "근거 미확인"
경고가 일상이 되어 진짜 경고가 묻힌다. `Answer.verifiable` 로 분리하고
`degraded` 는 **검증 가능한 경로가 검증을 통과하지 못한 경우에만** 참이다 —
`domains/finance` 가 판단 계층을 다루는 방식과 같다.

경로는 코드가 고른다. 라우터는 검색기와 같은 기준으로 판정하며(조사가 붙은
한국어에서 이 어긋남은 예외가 아니라 기본값이다), `mode=` 로 덮어쓸 수 있다.

```python
agent.ask(q)                      # 자동 — 코퍼스와 겹치면 grounded
agent.ask(q, mode="grounded")     # 문서 근거만. 없으면 게이트에서 멈춘다
agent.ask(q, mode="free")         # 자유 대화. verifiable=False
ChatAgent(docs, allow_free=False) # 자유 대화 자체를 끈다
```

---

## 웹 UI: 검증 과정을 눈으로

```bash
pip install -e ".[web]"
python examples/webchat.py                  # 빈 코퍼스 — 브라우저에서 올린다
python examples/webchat.py --docs ./notes   # 시작 시 적재
python examples/webchat.py --llm echo       # 모델 없이 구조만
```

왼쪽은 대화, 오른쪽은 **그 답이 어떻게 검증되었는지**다 — 주장별 인용 출처와
근거 원문, 어느 문장이 어느 가드레일에 걸렸는지, 재작성 라운드, 그리고
토큰·제어 비율·격리 효과. 답변을 클릭하면 그 턴의 궤적만 잘라서 보여준다.

문서는 브라우저에서 끌어다 놓으면 즉시 색인되고(`add_document`), × 로 철회하면
즉시 인용에서 빠진다(`remove_document`). "기억 삭제"는 대화 기억만 지우고
문서 인덱스는 건드리지 않는다 — 그래서 검색기와 기억을 분리해 둔 것이다.

`agentcore/apps/web.py` 에도 판단은 없다. `ChatAgent` 가 전부 하고 웹 계층은
HTTP 로 번역만 하며, 그 계층의 유일한 책임은 **저하 상태를 숨기지 않는 것**이다.
`blocked` 를 빼먹고 답변 텍스트만 내려보내면 앞의 여섯 계층이 무의미해진다.
`tests/test_web.py` 가 그것을 검증한다.

**로컬 전용이다.** 기본 바인딩은 127.0.0.1 이고, 구독 경로는 이 머신에
로그인된 CLI 를 쓰므로 서버에 올려도 남의 요청은 처리하지 못한다. 공개
배포하려면 코어를 `AnthropicLLM`(API 키)으로 바꿔야 한다.

### Claude Pro/Max 구독으로 돌리기

Pro 구독은 API 키를 주지 않으므로 `AnthropicLLM` 을 쓸 수 없다. 대신
로그인된 Claude Code CLI 를 헤드리스로 호출한다.

```python
from agentcore.models import ClaudeCLILLM

llm = ClaudeCLILLM()                                  # 기본값: opus-5 · effort=high
llm = ClaudeCLILLM("claude-sonnet-5", effort="medium")  # 빠르고 싸게
llm = ClaudeCLILLM(effort=None)                       # CLI 기본값에 맡긴다
```

`effort` 는 `low·medium·high·xhigh·max` 중 하나이며 **모델 ID 와 함께 핀의
일부**다. 같은 모델도 effort 가 다르면 다른 답을 내므로 궤적에 함께 남긴다 —
빼고 기록하면 "동일 입력·동일 핀 버전"이 재현을 보장하지 못한다.

```bash
agentcore-report runs/chat.jsonl --json | grep effort
# {"component": "claude-opus-5", "model": "claude-opus-5", "effort": "high"}
```

CLI 의 도구는 전부 끈다. 이 자리의 모델은 **판단만** 하며, 도구 호출과
제어는 agentcore 쪽 코드에 있다 — CLI 가 자체 판단으로 파일을 읽으면 그
결정이 궤적에 남지 않아 원칙이 조용히 깨진다.

과금은 토큰이 아니라 구독 사용량으로 계산되지만 호출마다 프로세스가 뜨므로
느리다. 반복 실행은 `RecordingLLM` 으로 녹음해 `ReplayLLM` 로 재생한다.

---

## 예제

```
$ python examples/pipeline.py
주장 3 건
  · revenue: 2건 확인  ← ['doc1', 'doc3']
  · cost: 1건 확인  ← ['doc2']
  · headcount: 1건 확인  ← ['doc4']
------------------------------------------------------------------
총 토큰            2,728
격리 효과          295.7 : 1   (흡수 2,661 / 상향 9)
제어 비율          0.92       (코드 제어 결정 / 전체 결정)
가드레일 위반      0건
==================================================================
검증기 다운 시 출력: 0건 (fail closed)
슬롯 미충족 시 출력: 0건 (되묻기)
```

**제어 비율 0.92** 가 "LLM은 판단하고 제어는 코드가 한다"의 측정값이다.
분기·재시도·게이트·루프가 전부 코드에 있으면 이 값이 높게 나온다.

---

## 궤적 분석

```bash
agentcore-report runs/trace.jsonl              # 4층위 지표
agentcore-report runs/trace.jsonl --json
agentcore-report runs/trace.jsonl --selections # 재현 모드 입력 추출
```

로그만으로 4층위 지표가 나오지 않으면 "궤적 로그가 곧 평가 데이터"는 말뿐이다.
리포트는 **패턴 정당화**도 함께 출력한다 — 궤적이 "왜 이 패턴이 있는가"를
스스로 문서화한다.

---

## 컴포넌트 교체 판단

```python
t1   = score_component(bm25_reranker, eval_set)     # 기성품 기준선
inc  = score_component(current, eval_set)
cand = score_component(candidate, eval_set)

decide_swap(cand, incumbent=inc, baseline=t1,
            floor={"accuracy": 0.85}, min_gain=0.02, cost_ceiling_ratio=1.5)
# Verdict.UNJUSTIFIED — T1 기준선을 이기지 못했습니다.
#   맞춤형 컴포넌트는 관리 대상과 망각 위험만 늘립니다 — 기성품을 쓰십시오.
```

**동점도 거절된다.** 기성품이 충분하다는 뜻이므로 T2 는 순손실이다.

---

## 증류: 코어를 동결한 채 전용 모델 얻기

```python
entry = check_entry("runs/trace.jsonl", "relevance_filter", min_calls=500)
# EntryCheck(stable_spec=True, high_frequency=False, ...)

ds = label_with_core(payloads, teacher=core_call, task="relevance_filter",
                     teacher_id="claude-sonnet-5", verify=grounding_check)
train, holdout = ds.split(holdout=0.2)     # 학습 **전에** 자른다
ds.write("data/train.jsonl")

publish_contract(name="relevance_filter", version="2.0", score=score,
                 fallback="bm25_rerank", teacher="claude-sonnet-5", entry=entry)
# EntryConditionsUnmet: 호출 빈도 부족 (12회) — 학습·검증 비용을 회수할 수 없습니다
```

학습 루프 자체는 포함하지 않는다 — 프레임워크가 학습 스택을 고르면 그 선택이
상각된다. JSONL 을 내보내고 결과를 계약으로 다시 받는다.

---

## 도메인: 연구·학습과 주식분석

두 도메인은 검증 가능성에서 갈린다. 골드셋을 **검증 가능한 것에만** 만든다.

### 연구·학습 — 출처 재현율

전문가가 "이 질문에는 이 출처들이 반드시 나와야 한다"만 나열하면 된다.
보고서를 쓰게 하는 것보다 훨씬 싸고, 가장 중요한 실패를 잡는다.

```python
gold = SourceGold("q1", "트랜스포머 어텐션 원논문은?",
                  required=frozenset({"vaswani2017"}),
                  helpful=frozenset({"blogpost"}),
                  forbidden=frozenset({"retracted42"}))

goldset.evaluate({"q1": ["vaswani2017", "retracted42"]})
# recall 1.0 — 그런데 forbidden.total_hits 1 → 실패
```

**재현율이 완벽해도 금지 출처가 한 건 있으면 실패다.** 금지 출처는 꼬리
지표이므로 재현율과 함께 평균되지 않는다. `helpful` 은 정밀도 분모에서
빠진다 — 넓게 찾는 것이 벌점이 되면 시스템은 좁게만 찾는다.

### 주식분석 — 데이터 계층만 검증 가능하다

이 도메인의 설계 전제:

> **데이터 계층은 완전히 검증 가능하고, 판단 계층은 골드셋이 원리적으로
> 만들어지지 않는다.**

"종목이 올랐으니 분석이 맞았다"는 표본 1에 노이즈가 지배적이고, 백테스트로
평가하면 룩어헤드 편향을 학습한다. 그래서 검증 가능한 세 가지에만 골드셋을 둔다.

**① 시점 무결성** — 이 도메인 최대의 조용한 오류

```python
view = AsOfView(as_of=date(2024, 7, 1))
view.ingest(filings)
view.require("revenue_q2_2024")
# LookupError: 2024-07-01 시점에 알 수 없었습니다.
#   없는 데이터를 추정으로 메우면 그것이 곧 룩어헤드입니다.
```

접근 경로를 하나로 만들면 룩어헤드가 **구조적으로 불가능**해진다 — 검사보다
낫다. 검사는 빠뜨릴 수 있지만 접근 경로가 하나면 못 빠뜨린다.

`period_end` 와 `published_at` 을 분리해서 갖는 것이 핵심이다. 2024 Q1
실적은 3월에 확정되지만 5월에야 공개된다. 4월에 쓰면 룩어헤드다.
소급 정정(`restated_at`)도 같은 방식으로 잡는다.

**② 지표 재계산** — LLM Judge 가 필요 없는 자리

```python
recompute(MetricClaim("m", "roe", {"net_income": 80, "equity": 500}, reported=0.20))
# ok=False, "보고 0.2 ≠ 재계산 0.16"
```

재계산은 **입력을 명시하도록 강제한다.** 어떤 값에서 나왔는지 못 적는 지표는
검증할 수 없다. 0으로 나누기와 적자 기준 증감률은 값을 내지 않고 거절한다 —
`inf` 나 뒤집힌 부호가 하류로 흘러가면 조용히 결론을 망친다.

**③ 주장 분류** — 검증된 신뢰가 검증되지 않은 것으로 번지지 않게

| 종류 | 예 | 검증 |
|---|---|---|
| `MEASURED` | "2024 Q1 매출 1,200억" | 재계산 |
| `DERIVED` | "영업이익률 12.4%" | 재계산 |
| `SOURCED` | "회사는 증설을 발표" | 출처 대조 |
| `PROJECTED` | "내년 매출은 늘 것" | **불가** — 라벨 필수 |
| `ADVISORY` | "매수할 만하다" | **불가** — 기본 차단 |

라벨 없는 전망, 그리고 `MEASURED` 로 표시됐지만 전망 표현이 든 문장은
가드레일에 걸린다. 권유형 출력은 기본적으로 차단된다 — 성능 문제가 아니라
**이 도메인에서 평가할 수 없는 출력을 검증된 것처럼 내보내지 않기 위한
설계**다. 이 시스템은 골드셋이 없는 것에 대해 골드셋이 있는 척하지 않는다.

```bash
python examples/equity.py
# 시점 누수 0건 ✓ · 지표 재계산 3건 중 불일치 0건 · 가드레일 위반 0건
# [기준일 2024-04-01] 분석 불가 — 미공개 입력: ['revenue_q1_2024', ...]
```

기준일에 필요한 입력이 없으면 **추정으로 메우지 않고 무엇이 없는지 말하고
멈춘다.** 없는 데이터를 채우는 것이 곧 룩어헤드다.

---

## 아직 열려 있는 것

| 항목 | 상태 |
|---|---|
| 실데이터 어댑터 | 공시·시세 소스에서 `DataPoint` 로 넣는 커넥터 |
| 출처 재현율 골드셋 채우기 | 구조는 있고 항목이 비었다 — 전문가 작업 |
| 판단 계층 평가 | **원리적으로 불가.** 대신 검증 가능한 것과 분리해 표시한다 |
| 상관 증거 탐지 | 출처 단위 중복 제거는 "같은 문서 두 번"은 잡지만 "서로 베낀 두 문서"는 못 잡는다 |
| 학습 루프 | 의도적 제외. JSONL 내보내기까지가 프레임워크의 몫 |
| 인용 검증 NLI 모델 | 현재는 주입식 `entails`. 실제 모델 연결 필요 |
| 저하 모드 표시 | 저하 상태로 나온 답을 정상 답과 구분해 보여줄 책임이 호출자에게 있다 |

### 상관 증거에 대한 정직한 한계

`join_findings` 는 출처 ID 로 중복을 제거하므로 **같은 문서를 두 번 찾아온
경우**는 잡는다. 그러나 서로 베낀 두 문서가 같은 사실을 담고 있으면 독립
출처 2건으로 계산된다. 이는 알려진 한계이며, 해결하려면 출처 간 파생 관계를
모델링해야 한다. 현재 코드는 이를 하지 않는다.
