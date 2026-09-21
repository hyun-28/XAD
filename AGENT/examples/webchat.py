"""로컬 웹 챗봇 실행기.

    pip install -e ".[web]"

    python examples/webchat.py                   # 빈 코퍼스 — 자유 대화부터
    python examples/webchat.py --docs ./notes    # 시작 시 문서 적재
    python examples/webchat.py --llm echo        # 모델 없이 구조만 (무료)
    python examples/webchat.py --port 8123

브라우저에서 문서를 올릴 수 있으므로 `--docs` 는 선택이다.

**로컬 전용이다.** 기본 바인딩은 127.0.0.1 이며, Claude 구독 경로는 이
머신에 로그인된 CLI 를 쓰므로 외부에 공개해도 남의 요청을 처리할 수 없다.
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from agentcore.apps import ChatAgent
from agentcore.apps.web import create_app
from agentcore.models import ClaudeCLILLM, EchoLLM
from agentcore.trace import JsonlSink, MemorySink, MultiSink, Tracer

sys.path.insert(0, str(Path(__file__).parent))
from chatbot import SAMPLE, echo_responder, load_documents  # noqa: E402


def build(args) -> tuple:
    if args.docs:
        documents = load_documents(args.docs)
    elif args.sample:
        documents = SAMPLE
    else:
        documents = []          # 브라우저에서 올리거나, 그냥 대화하거나

    llm = (
        EchoLLM(responder=echo_responder)
        if args.llm == "echo"
        else ClaudeCLILLM(model_id=args.model, effort=args.effort)
    )

    args.trace.parent.mkdir(parents=True, exist_ok=True)
    sink = MemorySink()
    tracer = Tracer(sink=MultiSink(sink, JsonlSink(str(args.trace))))
    agent = ChatAgent(documents, llm=llm, tracer=tracer)
    return create_app(agent, sink, subject=args.subject), agent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="문서 기반 Q&A 챗봇 — 로컬 웹 UI")
    ap.add_argument("--docs", type=Path, help=".md/.txt 디렉터리")
    ap.add_argument("--sample", action="store_true", help="내장 샘플 문서로 시작")
    ap.add_argument("--llm", choices=("cli", "echo"), default="cli")
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--effort", default="high",
                    choices=("low", "medium", "high", "xhigh", "max"),
                    help="추론 노력. 모델 ID 와 함께 재현성의 핀이다")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--trace", type=Path, default=Path("runs/webchat.jsonl"))
    ap.add_argument("--subject", default="local-user")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args(argv)

    try:
        import uvicorn
    except ImportError:
        raise SystemExit('웹 의존성이 없습니다:  pip install -e ".[web]"')

    app, agent = build(args)
    url = f"http://{args.host}:{args.port}"
    effort = getattr(agent.llm, "effort", None)
    pin = agent.llm.model_id + (f" · effort={effort}" if effort else "")
    print(f"문서 {len(agent.corpus)}건 · 코어 {agent.llm.name}:{pin}")
    print(f"{url}  (Ctrl+C 종료)")
    if not args.no_open:
        webbrowser.open(url)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
