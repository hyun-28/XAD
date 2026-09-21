"""문서 기반 Q&A 챗봇 — 골격을 대화 루프로 조립한 것.

일반 RAG 챗봇과 다른 점은 하나다. **근거가 없으면 문장을 내보내지 않는다.**

    [1] 라우팅        잡담이 검색·검증 경로를 타지 않게. 저확신은 근거 경로로
    [2] 검색 + 격리   원문은 워커 경계 안에서 흡수되고 요약만 코어로 올라간다
    [3] 충분성 게이트 검색 결과가 비면 추측하지 않고 무엇이 없는지 말한다
    [4] 코어 응답     주장 단위 + 출처 ID 형식을 강제
    [5] 근거 검증 루프 형식·근거 판정은 전부 코드. 실패한 주장만 재작성
    [6] 가드레일      미인용·허위 출처는 개수로 세고, 하나라도 있으면 차단
    [7] 기억          주체 단위 저장. forget() 으로 삭제 요구에 대응

제어는 전부 이 파일의 파이썬 코드에 있고, LLM 은 [4] 안에서 판단만 한다.
그래서 궤적의 제어 비율이 높게 나온다 — 그것이 이 구조의 측정값이다.

경로는 둘이고, **검증 가능성에서 갈린다.**

    grounded  문서 근거 + 형식 강제 + 원문 대조 + 가드레일.  검증 가능
    free      자유 대화. 검증기도 가드레일도 지나지 않는다.  검증 불가

자유 대화를 저하된 grounded 로 취급하지 않는다. 그것은 원리적으로 검증할 수
없는 출력이지 실패한 검증이 아니며, 둘을 섞으면 "근거 미확인" 경고가 일상이
되어 진짜 경고가 묻힌다. 대신 `verifiable=False` 로 **분리해서 표시한다** —
domains/finance 가 판단 계층을 다루는 방식과 같다.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..control import (
    Classification,
    DeterministicSignal,
    SignalResult,
    SlotSpec,
    SufficiencyGate,
    route,
)
from ..memory import Document, Hit, IsolatingRetriever, KeywordRetriever, Memory, Retriever
from ..models import LLM, EchoLLM, Message
from ..trace import SpanKind, Tracer
from ..verify import (
    Claim,
    GroundingVerifier,
    Guardrail,
    GuardrailReport,
    GuardrailSet,
    VerificationResult,
)

# --------------------------------------------------------------------------
# 출력 형식 — 검증 가능해야 하므로 자유 텍스트를 받지 않는다
# --------------------------------------------------------------------------

# "- 문장입니다 [doc1, doc3]"
_LINE = re.compile(r"^\s*[-*•]\s*(?P<text>.+?)\s*\[(?P<ids>[^\[\]]*)\]\s*$")
_STOPWORDS = frozenset(
    "은 는 이 가 을 를 의 에 에서 와 과 도 로 으로 만 및 그리고 하지만 "
    "the a an of to in on for and or is are was were be been that this it".split()
)


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[0-9A-Za-z가-힣]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def parse_claims(text: str, prefix: str = "c") -> tuple[tuple[Claim, ...], tuple[str, ...]]:
    """모델 출력을 주장 단위로 쪼갠다.

    파싱은 순수 코드다 — 출력이 형식을 지켰는지는 프로그램으로 판정 가능하고,
    판정 가능한 것을 판정자에게 맡기는 것은 낭비다. 형식을 어긴 줄은 버리지
    않고 **실패 항목으로 되돌려준다**. 조용히 버리면 답이 짧아진 이유를
    아무도 모른다.
    """
    claims: list[Claim] = []
    malformed: list[str] = []
    for i, raw in enumerate(text.splitlines()):
        line = raw.strip()
        if not line:
            continue
        m = _LINE.match(line)
        if not m:
            malformed.append(line[:120])
            continue
        ids = frozenset(s.strip() for s in m.group("ids").split(",") if s.strip())
        claims.append(
            Claim(
                id=f"{prefix}{i}",
                text=m.group("text").strip(),
                support=(),          # 원문 구절은 아래에서 붙인다
                source_ids=ids,
            )
        )
    return tuple(claims), tuple(malformed)


def attach_support(claims: Sequence[Claim], corpus: dict[str, Document]) -> tuple[Claim, ...]:
    """인용된 출처 ID 를 실제 원문으로 해석한다.

    존재하지 않는 ID 를 인용하면 support 가 비고, 그 상태로 GroundingVerifier
    에 걸린다 — 허위 출처가 조용히 통과하지 않는다.
    """
    out = []
    for c in claims:
        support = tuple(corpus[i].text for i in sorted(c.source_ids) if i in corpus)
        out.append(Claim(id=c.id, text=c.text, support=support, source_ids=c.source_ids))
    return tuple(out)


NO_ANSWER = "none"


def is_abstention(claims: Sequence[Claim]) -> bool:
    """모델이 "답할 수 없다"고 말한 경우.

    이것은 실패가 아니라 **정상적인 출력**이다. 근거가 없을 때 답을 만들어
    내지 않는 것이 이 시스템의 목적이므로, 기권을 형식 위반으로 처리해
    재시도 루프에 밀어넣으면 안 된다.
    """
    return len(claims) == 1 and claims[0].source_ids == frozenset({NO_ANSWER})


def lexical_entails(threshold: float = 0.6):
    """어휘 중첩 기반 함의 판정 — T1 기준선.

    주장의 내용어가 근거 원문에 실제로 나타나는 비율을 본다. NLI 모델보다
    약하지만 **결정론적이고 공짜이며 재현 가능하다.** 결정론 우선 원칙에
    따라 여기서 먼저 재고, 이 기준선을 이기는 것이 증명된 뒤에 모델로
    바꾼다. 바꾸는 판단은 `evaluation.decide_swap` 이 한다.

    알려진 한계: 어휘가 겹치면서 의미가 뒤집힌 주장("연차는 무제한이다")은
    통과할 수 있다. 부정·수량 왜곡은 어휘 중첩으로 잡히지 않는다. 이것이
    NLI 모델로 교체할 이유이며, 교체 전까지는 **잡히지 않는다는 사실을
    아는 상태**로 쓰는 것이 모르고 쓰는 것보다 낫다.
    """

    def entails(claim_text: str, support: tuple[str, ...]) -> bool:
        need = _tokens(claim_text)
        if not need:
            return False
        have: set[str] = set()
        for s in support:
            have |= _tokens(s)
        return len(need & have) / len(need) >= threshold

    return entails


# --------------------------------------------------------------------------
# 결과 타입
# --------------------------------------------------------------------------

@dataclass(slots=True)
class Turn:
    question: str
    answer: str


@dataclass(slots=True)
class Answer:
    """한 턴의 결과.

    `text` 만 보고 쓰면 안 된다. `blocked` 와 `grounded` 를 함께 읽어야
    저하 상태로 나온 답을 정상 답과 구분할 수 있다 — 그 구분의 책임은
    호출자에게 있다.
    """

    text: str
    route: str
    claims: tuple[Claim, ...] = ()
    source_ids: tuple[str, ...] = ()
    grounded: bool = True
    verifiable: bool = True
    blocked: bool = False
    missing: tuple[str, ...] = ()
    rounds: int = 0
    residual: tuple[str, ...] = ()
    guardrails: GuardrailReport = field(default_factory=GuardrailReport)

    @property
    def degraded(self) -> bool:
        """검증 가능한 경로가 검증을 통과하지 못한 상태.

        검증 불가 경로(자유 대화)는 저하가 아니다 — 애초에 검증 대상이
        아니기 때문이다. 둘을 같은 플래그로 묶으면 경고가 일상이 되고,
        일상이 된 경고는 아무도 읽지 않는다.
        """
        if not self.verifiable:
            return False
        return self.blocked or not self.grounded or bool(self.missing)


# --------------------------------------------------------------------------
# 에이전트
# --------------------------------------------------------------------------

_SYSTEM = """당신은 문서 기반 질의응답 시스템의 합성 단계다.

규칙:
1. 제공된 발췌에 실제로 적힌 내용만 말한다. 추측·일반 상식·배경 지식 금지.
2. 출력은 주장 한 줄씩, 반드시 다음 형식만 쓴다.
   - 문장 [출처ID]
   출처 ID 는 발췌에 붙은 ID 중에서만 고른다. 여러 개면 쉼표로 나열한다.
3. 형식을 벗어난 문장, 인사말, 머리말, 맺음말을 쓰지 않는다.
4. 발췌로 답할 수 없으면 다음 한 줄만 출력한다.
   - 제공된 문서로는 답할 수 없습니다 [none]
"""


_FREE_SYSTEM = """당신은 일반 대화 상대다.

이 경로에는 검증이 없다. 문서 근거가 없으므로 당신의 출력은 사용자에게
**검증 불가**로 표시되어 전달된다. 그러니 아는 것과 모르는 것을 분명히
구분해 말하고, 확실하지 않으면 확실하지 않다고 말한다.

사용자가 특정 문서·자료의 내용을 물으면 그 문서가 이 대화에 제공되지
않았다고 말한다. 내용을 지어내지 않는다."""


class ChatAgent:
    """문서 기반 Q&A 챗봇.

    llm 은 코어 자리다. `EchoLLM`(테스트), `ClaudeCLILLM`(구독),
    `AnthropicLLM`(API), `ReplayLLM`(회귀) 중 무엇을 꽂아도 이 클래스는
    바뀌지 않는다 — 코어는 크기가 아니라 역할이기 때문이다.
    """

    def __init__(
        self,
        documents: Sequence[Document],
        *,
        llm: LLM | None = None,
        tracer: Tracer | None = None,
        retriever: Retriever | None = None,
        k: int = 4,
        max_rounds: int = 2,
        entails: Any = None,
        history_turns: int = 4,
        memory_ttl_seconds: float | None = None,
        allow_free: bool = True,
    ) -> None:
        self.corpus = {d.id: d for d in documents}
        self.llm = llm or EchoLLM()
        self.tracer = tracer or Tracer()
        self.k = k
        self.max_rounds = max_rounds
        self.history_turns = history_turns
        self.allow_free = allow_free

        base = retriever or KeywordRetriever(list(documents))
        self.retriever = IsolatingRetriever(inner=base, summarize=self._summarize)
        self.memory = Memory(default_ttl_seconds=memory_ttl_seconds)

        self.gate = SufficiencyGate(
            [
                SlotSpec("question", description="사용자 질문"),
                SlotSpec("excerpts", description="검색된 근거 발췌"),
            ],
            name="chat_sufficiency",
        )
        self.verifier = GroundingVerifier(entails or lexical_entails())
        self.guardrails = GuardrailSet(
            rails=(
                Guardrail(
                    "cited",
                    lambda c, ctx: bool(c.source_ids),
                    "출처 없이 나간 주장",
                ),
                Guardrail(
                    "known_source",
                    lambda c, ctx: c.source_ids <= set(ctx or ()),
                    "검색되지 않은 출처를 인용한 주장",
                ),
                Guardrail(
                    "grounded",
                    lambda c, ctx: bool(c.support),
                    "원문으로 해석되지 않는 출처",
                ),
            ),
            name="chat_guardrails",
        )
        # 검색 결과의 원문 ↔ 발췌 ID 매핑. 프롬프트에 넣는 것과 검증에 쓰는
        # 것이 같은 출처여야 하므로 한 곳에서 만든다.
        self._last_hits: tuple[Hit, ...] = ()

    # ---------------------------------------------------------------- 검색
    def _summarize(self, hits: Sequence[Hit]) -> str:
        """워커 경계. 원문 전체가 아니라 발췌만 코어로 올라간다."""
        self._last_hits = tuple(hits)
        return "\n".join(
            f"[{h.document.id}] {h.document.text.strip()[:400]}" for h in hits
        )

    # ---------------------------------------------------------------- 분류
    def _classify(self, payload: dict[str, Any]) -> Classification:
        """라우터는 코드다. LLM 라우터는 오분류가 조용하고 비용도 더 든다.

        사용자가 mode 를 지정하면 그것이 이긴다 — 자동 분류는 편의이지
        사용자 의도를 덮어쓸 근거가 아니다. 지정 여부도 궤적에 남는다.
        """
        mode = payload.get("mode", "auto")
        if mode in ("free", "grounded"):
            return Classification(mode, confidence=1.0, reason="사용자 지정")

        if not self.corpus:
            return Classification("free", confidence=1.0, reason="코퍼스 없음")

        overlap = self._overlap(payload["question"])
        if overlap:
            return Classification(
                "grounded", confidence=1.0, reason=f"코퍼스 어휘 {len(overlap)}건 일치"
            )
        return Classification("free", confidence=0.7, reason="코퍼스와 겹치는 어휘 없음")

    def _overlap(self, question: str) -> set[str]:
        """라우터는 **검색기와 같은 기준**으로 판정해야 한다.

        KeywordRetriever 는 부분 문자열로 세므로("연차" 가 "연차는" 에 걸린다)
        라우터가 정확 일치만 보면 검색되었을 질문이 자유 대화로 샌다. 조사가
        붙는 한국어에서 이 어긋남은 예외적 상황이 아니라 기본값이다.

        두 곳이 같은 판정을 쓰는 것이 규칙이 아니라 **같은 코드를 보게 하는
        것**이 규칙이다 — 검색기를 바꾸면 이 함수도 같이 바뀌어야 한다.
        """
        return {t for t in _tokens(question) if t in self._corpus_text}

    @property
    def _corpus_text(self) -> str:
        if not hasattr(self, "_ct"):
            self._ct = "\n".join(d.text for d in self.corpus.values()).lower()
        return self._ct

    # ---------------------------------------------------------------- 생성
    def _prompt(
        self, question: str, excerpts: str, history: Sequence[Turn], failures: tuple[str, ...]
    ) -> list[Message]:
        parts = []
        if history:
            prior = "\n".join(f"Q: {t.question}\nA: {t.answer}" for t in history)
            parts.append(f"## 이전 대화\n{prior}")
        parts.append(f"## 발췌\n{excerpts}")
        parts.append(f"## 질문\n{question}")
        if failures:
            # 실패한 항목만 지목한다 — 전체 재작성은 통과한 주장까지 흔든다.
            parts.append(
                "## 직전 시도에서 실패한 항목\n"
                + "\n".join(f"- {f}" for f in failures)
                + "\n위 항목만 고쳐서 전체 목록을 다시 출력한다."
            )
        return [Message("system", _SYSTEM), Message("user", "\n\n".join(parts))]

    def _signal(self, valid_ids: frozenset[str]):
        """외부 검증 신호 — 형식과 원문 대조. 둘 다 프로그램이 판정한다.

        생성자와 같은 정보만 보는 판정이 아니므로 refine() 이 받는다.
        """

        def check(candidate: tuple[tuple[Claim, ...], tuple[str, ...]]) -> SignalResult:
            claims, malformed = candidate
            if is_abstention(claims) and not malformed:
                return SignalResult(passed=True, detail="기권 — 근거 없음을 명시")
            failures = [f"형식 위반: {m}" for m in malformed]
            if not claims:
                failures.append("주장이 하나도 파싱되지 않음")
            result: VerificationResult = self.verifier(claims, tracer=self.tracer)
            by_id = {c.id: c for c in claims}
            for v in result.violations:
                c = by_id.get(v.locus)
                shown = c.text[:80] if c else v.locus
                failures.append(f"{v.message}: {shown}")
            unknown = [
                f"존재하지 않는 출처 인용: {sorted(c.source_ids - valid_ids)}"
                for c in claims
                if not c.source_ids <= valid_ids
            ]
            failures.extend(unknown)
            return SignalResult(
                passed=not failures,
                failures=tuple(failures),
                detail=f"주장 {len(claims)}건 / 위반 {len(failures)}건",
            )

        return DeterministicSignal(check, name="format_and_grounding")

    # ---------------------------------------------------------------- 경로
    def _free(self, payload: dict[str, Any]) -> Answer:
        """자유 대화. 검색도 검증도 가드레일도 지나지 않는다.

        이 경로의 출력은 **검증 불가**다. 그것을 숨기지 않고 표시하는 것이
        이 경로를 두는 조건이다 — 검증되지 않은 것을 검증된 것과 같은
        모양으로 내보내면 앞의 여섯 계층이 전부 무의미해진다.
        """
        msgs = [Message("system", _FREE_SYSTEM)]
        for t in payload["history"]:
            msgs.append(Message("user", t.question))
            msgs.append(Message("assistant", t.answer))
        msgs.append(Message("user", payload["question"]))

        out = self.llm.complete(msgs, tracer=self.tracer)
        return Answer(
            text=out.text.strip(),
            route="free",
            grounded=False,
            verifiable=False,
        )

    def _grounded(self, payload: dict[str, Any]) -> Answer:
        from ..control.patterns import refine

        question: str = payload["question"]
        history: Sequence[Turn] = payload["history"]

        excerpts, hit_ids = self.retriever.search_summary(
            question, self.k, tracer=self.tracer
        )

        # 게이트는 비싼 작업 **앞**에 선다. 검색이 비면 모델을 부르지 않는다.
        suff = self.gate.check(
            {"question": question, "excerpts": excerpts}, tracer=self.tracer
        )
        if not suff:
            return Answer(
                text="관련 문서를 찾지 못했습니다. 추측으로 답하지 않습니다.",
                route="grounded",
                grounded=False,
                missing=suff.missing,
            )

        valid_ids = frozenset(hit_ids)

        def generate(p: dict[str, Any], failures: tuple[str, ...]):
            msgs = self._prompt(question, excerpts, history, failures)
            out = self.llm.complete(msgs, tracer=self.tracer)
            claims, malformed = parse_claims(out.text)
            return attach_support(claims, self.corpus), malformed

        (claims, malformed), signal, rounds = refine(
            generate,
            self._signal(valid_ids),
            payload,
            prevents="근거 없는 문장과 형식 위반이 그대로 사용자에게 나감",
            max_rounds=self.max_rounds,
            tracer=self.tracer,
        )

        if is_abstention(claims):
            return Answer(
                text=claims[0].text,
                route="grounded",
                grounded=False,
                rounds=rounds,
            )

        report = self.guardrails.evaluate(claims, valid_ids, tracer=self.tracer)
        cited = tuple(sorted({i for c in claims for i in c.source_ids} & valid_ids))

        if report.blocked:
            # 가드레일은 점수가 아니라 관문이다. 하나라도 걸리면 안 내보낸다.
            return Answer(
                text="검증을 통과하지 못해 답변을 보류했습니다.",
                route="grounded",
                claims=claims,
                source_ids=cited,
                grounded=False,
                blocked=True,
                rounds=rounds,
                residual=signal.failures,
                guardrails=report,
            )

        text = "\n".join(f"· {c.text}" for c in claims)
        return Answer(
            text=text,
            route="grounded",
            claims=claims,
            source_ids=cited,
            grounded=signal.passed,
            rounds=rounds,
            residual=signal.failures,
            guardrails=report,
        )

    # ---------------------------------------------------------------- 진입점
    def ask(
        self, question: str, *, subject: str = "default", mode: str = "auto"
    ) -> Answer:
        """한 턴 처리.

        subject 는 데이터 주체 — 삭제 요구의 단위다.
        mode 는 "auto" | "grounded" | "free". auto 는 코드 분류기에 맡긴다.
        """
        if mode not in ("auto", "grounded", "free"):
            raise ValueError(f"알 수 없는 mode: {mode}")

        with self.tracer.span("turn", SpanKind.CONTROL, subject=subject, mode=mode) as span:
            history = self.history(subject)
            payload = {
                "question": question,
                "history": history,
                "subject": subject,
                "mode": mode,
            }

            routes: dict[str, Any] = {"grounded": self._grounded}
            if self.allow_free:
                routes["free"] = self._free
            # 폴백은 더 비싸고 더 안전한 쪽. 문서가 있으면 근거 경로가 안전하고,
            # 문서가 없으면 근거 경로는 안전한 게 아니라 그냥 불가능하다.
            fallback = "grounded" if (self.corpus or not self.allow_free) else "free"

            answer: Answer = route(
                self._classify,
                routes,
                payload,
                prevents="자유 대화가 검색·검증·재시도 경로를 타서 비용이 수십 배 / "
                         "문서 질의가 검증 없이 나감",
                fallback=fallback,
                tracer=self.tracer,
            )

            self.memory.put(
                subject, "history", list(history) + [Turn(question, answer.text)]
            )
            span.attributes.update(
                route=answer.route,
                verifiable=answer.verifiable,
                grounded=answer.grounded,
                blocked=answer.blocked,
                rounds=answer.rounds,
                sources=list(answer.source_ids),
            )
            return answer

    def history(self, subject: str = "default") -> list[Turn]:
        stored = self.memory.get(subject, "history") or []
        return list(stored)[-self.history_turns :]

    def forget(self, subject: str) -> int:
        """주체 단위 삭제. 문서 인덱스는 건드리지 않는다 — 그래서 분리했다."""
        return self.memory.forget_subject(subject)

    def add_document(self, doc: Document) -> None:
        """비파라메트릭 적응 표면. 온라인 갱신이 허용되는 유일한 자리다."""
        self.corpus[doc.id] = doc
        inner = self.retriever.inner
        if isinstance(inner, KeywordRetriever):
            inner.add(doc)
        else:
            raise TypeError("add_document 는 KeywordRetriever 기반일 때만 가능합니다")
        self._invalidate()

    def remove_document(self, doc_id: str) -> bool:
        """문서 철회. 잘못 올린 문서가 계속 인용되는 것을 막는다."""
        if doc_id not in self.corpus:
            return False
        del self.corpus[doc_id]
        inner = self.retriever.inner
        if isinstance(inner, KeywordRetriever):
            inner.documents = [d for d in inner.documents if d.id != doc_id]
        else:
            raise TypeError("remove_document 는 KeywordRetriever 기반일 때만 가능합니다")
        self._invalidate()
        return True

    def _invalidate(self) -> None:
        if hasattr(self, "_ct"):
            del self._ct
