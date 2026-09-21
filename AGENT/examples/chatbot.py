"""문서 기반 Q&A 챗봇 CLI — ChatAgent 의 얇은 래퍼.

    python examples/chatbot.py                     내장 샘플 문서 + Claude 구독
    python examples/chatbot.py --docs ./notes      디렉터리의 .md/.txt 를 코퍼스로
    python examples/chatbot.py --llm echo          모델 없이 구조만 확인 (무료)
    python examples/chatbot.py --model claude-sonnet-5 --effort medium

    /sources   직전 답변의 근거 출처
    /trace     이번 세션 지표
    /forget    내 기억 삭제 (주체 단위)
    /quit

이 파일에는 로직이 없다. 로직은 전부 `agentcore.apps.chat.ChatAgent` 에
있고 여기는 입출력만 한다 — 그래서 웹·슬랙 어디에 붙여도 같은 동작이 나온다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agentcore.apps import ChatAgent
from agentcore.evaluation import summarize_system, summarize_trajectory
from agentcore.memory import Document
from agentcore.models import ClaudeCLILLM, EchoLLM
from agentcore.trace import JsonlSink, MemorySink, MultiSink, Tracer

# --------------------------------------------------------------------------
# 내장 샘플 코퍼스 — --docs 를 안 줬을 때
# --------------------------------------------------------------------------

SAMPLE = [
    Document(
        "onboarding",
        "신규 입사자는 첫 주에 보안 교육을 이수해야 한다. 교육은 온라인으로 "
        "진행되며 소요 시간은 약 2시간이다. 미이수 시 사내 시스템 접근 권한이 "
        "부여되지 않는다.",
    ),
    Document(
        "vacation",
        "연차는 입사 1년 차에 11일, 2년 차부터 15일이 부여된다. 연차 사용은 "
        "최소 3일 전에 신청해야 하며, 팀장 승인이 필요하다. 미사용 연차는 "
        "다음 해로 이월되지 않는다.",
    ),
    Document(
        "equipment",
        "업무용 노트북은 입사일에 지급된다. 개인 장비 사용을 원하는 경우 "
        "보안팀의 사전 승인이 필요하다. 장비 분실 시 24시간 내에 보안팀에 "
        "신고해야 한다.",
    ),
    Document(
        "remote",
        "재택근무는 주 2회까지 가능하다. 재택근무일은 전주 금요일까지 팀 "
        "캘린더에 등록한다. 신규 입사자는 첫 3개월간 재택근무가 제한된다.",
    ),
]


def load_documents(path: Path) -> list[Document]:
    docs = []
    for f in sorted(path.rglob("*")):
        if f.suffix.lower() in (".md", ".txt") and f.is_file():
            docs.append(Document(f.stem, f.read_text(encoding="utf-8"), {"path": str(f)}))
    if not docs:
        raise SystemExit(f"{path} 에서 .md/.txt 를 찾지 못했습니다")
    return docs


# --------------------------------------------------------------------------
# 스크립트된 Echo — 모델 없이 형식과 검증 경로를 확인하는 용도
# --------------------------------------------------------------------------

def echo_responder(messages):
    """발췌 첫 줄을 그대로 인용해 되돌려주는 결정론 응답자.

    실제 합성은 아니지만 **형식·검증·가드레일 경로 전체를 돌린다.**
    모델 없이 파이프라인이 살아 있는지 보는 것이 목적이다.
    """
    body = messages[-1].content
    lines = [l for l in body.splitlines() if l.startswith("[")]
    if not lines:
        return "- 제공된 문서로는 답할 수 없습니다 [none]"
    out = []
    for line in lines[:2]:
        doc_id = line[1 : line.index("]")]
        sentence = line[line.index("]") + 1 :].strip().split(".")[0]
        out.append(f"- {sentence} [{doc_id}]")
    return "\n".join(out)


# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="문서 기반 Q&A 챗봇")
    ap.add_argument("--docs", type=Path, help=".md/.txt 가 들어 있는 디렉터리")
    ap.add_argument("--llm", choices=("cli", "echo"), default="cli",
                    help="cli=Claude 구독(claude CLI), echo=모델 없음")
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--effort", default="high",
                    choices=("low", "medium", "high", "xhigh", "max"),
                    help="추론 노력. 모델 ID 와 함께 재현성의 핀이다")
    ap.add_argument("--trace", type=Path, default=Path("runs/chat.jsonl"))
    ap.add_argument("--subject", default="local-user", help="데이터 주체 ID")
    ap.add_argument("--ask", help="한 번만 묻고 종료 (비대화형)")
    args = ap.parse_args(argv)

    documents = load_documents(args.docs) if args.docs else SAMPLE

    if args.llm == "cli":
        llm = ClaudeCLILLM(model_id=args.model, effort=args.effort)
        note = f"Claude 구독 경로 ({args.model} · effort={args.effort})"
    else:
        llm = EchoLLM(responder=echo_responder)
        note = "Echo 모드 — 모델을 부르지 않습니다"

    args.trace.parent.mkdir(parents=True, exist_ok=True)
    mem = MemorySink()
    tracer = Tracer(sink=MultiSink(mem, JsonlSink(str(args.trace))))
    agent = ChatAgent(documents, llm=llm, tracer=tracer)

    def respond(question: str) -> None:
        answer = agent.ask(question, subject=args.subject)
        print()
        print(answer.text)
        if answer.source_ids:
            print(f"  근거: {', '.join(answer.source_ids)}")
        if answer.degraded:
            # 저하 상태를 정상 답과 구분해 표시하는 것은 호출자의 책임이다.
            flags = []
            if answer.blocked:
                flags.append("가드레일 차단")
            if not answer.grounded:
                flags.append("근거 미확인")
            if answer.missing:
                flags.append(f"미충족 슬롯 {list(answer.missing)}")
            print(f"  ⚠ {' · '.join(flags)}")
        if answer.residual:
            print(f"  잔여 위반: {len(answer.residual)}건")
        print()

    if args.ask:
        respond(args.ask)
        return 0

    print(f"문서 {len(documents)}건 · {note}")
    print("질문을 입력하세요. /sources /trace /forget /quit\n")

    while True:
        try:
            line = input("› ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/quit", "/exit"):
            break
        if line == "/forget":
            n = agent.forget(args.subject)
            print(f"  기억 {n}건 삭제 (문서 인덱스는 그대로)\n")
            continue
        if line == "/sources":
            print(f"  코퍼스: {', '.join(sorted(agent.corpus))}\n")
            continue
        if line == "/trace":
            spans = list(mem.spans)
            turns = max(1, sum(1 for s in spans if s.name == "turn"))
            sysm = summarize_system(spans)
            traj = summarize_trajectory(spans)
            print(f"  턴 {turns} · 총 토큰 {sysm.total_tokens:,} "
                  f"(턴당 {sysm.total_tokens / turns:,.0f})")
            print(f"  격리 효과 {traj.isolation_ratio:.1f} : 1 · "
                  f"제어 비율 {traj.control_ratio:.2f} · "
                  f"모델 호출 {traj.model_calls} · 재시도 {traj.retries}\n")
            continue
        respond(line)

    return 0


if __name__ == "__main__":
    sys.exit(main())
