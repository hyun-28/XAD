"""로컬 웹 UI — ChatAgent 를 브라우저에 붙인다.

이 파일에도 판단은 없다. 전부 `ChatAgent` 가 하고 여기는 HTTP 로 번역만
한다 — 그래서 CLI 와 웹이 같은 답을 내고, 같은 테스트가 둘 다를 덮는다.

의존성은 선택 사항이다.

    pip install -e ".[web]"

`agentcore` 본체는 여전히 의존성이 없다. 프레임워크가 웹 스택을 고르면
그 선택이 상각되므로, FastAPI 는 **이 파일에서만** 임포트되고
`agentcore.apps.__init__` 은 이 모듈을 건드리지 않는다. 웹을 안 쓰는
환경에서는 이 파일이 로드되지 않으므로 의존성도 필요 없다.

**로컬 전용이다.** 기본 바인딩은 127.0.0.1 이고, `ClaudeCLILLM` 경로는
이 머신에 로그인된 CLI 를 쓰므로 서버에 올려도 남의 요청을 처리할 수 없다.
공개 배포하려면 코어를 `AnthropicLLM`(API 키)으로 바꿔야 한다.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

try:
    from fastapi import Body, FastAPI, HTTPException, UploadFile
    from fastapi.responses import FileResponse
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        '웹 UI 에는 추가 의존성이 필요합니다:  pip install -e ".[web]"'
    ) from exc

from ..evaluation import summarize_trajectory
from ..memory import Document
from ..trace import MemorySink, Span
from .chat import Answer, ChatAgent

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 2_000_000
ALLOWED_SUFFIXES = (".md", ".txt", ".markdown")


# --------------------------------------------------------------------------
# 직렬화 — 저하 상태를 숨기지 않는 것이 이 계층의 유일한 책임이다
# --------------------------------------------------------------------------

def answer_to_dict(ans: Answer, agent: ChatAgent, spans: list[Span]) -> dict[str, Any]:
    traj = summarize_trajectory(spans)
    roots = [s for s in spans if s.name == "turn"]
    tokens = sum(s.usage.total_tokens for s in roots)

    # 가드레일 위반의 locus 는 주장 목록의 인덱스다. 어느 문장이 걸렸는지
    # 보여주지 않으면 "차단됨"은 사용자에게 아무 정보도 주지 않는다.
    breach_by_index: dict[int, list[str]] = {}
    for b in ans.guardrails.breaches:
        if b.locus.isdigit():
            breach_by_index.setdefault(int(b.locus), []).append(f"{b.guardrail}: {b.message}")

    claims = [
        {
            "text": c.text,
            "sources": sorted(c.source_ids),
            "support": list(c.support),
            "breaches": breach_by_index.get(i, []),
        }
        for i, c in enumerate(ans.claims)
    ]

    return {
        "text": ans.text,
        "route": ans.route,
        "verifiable": ans.verifiable,
        "grounded": ans.grounded,
        "blocked": ans.blocked,
        "degraded": ans.degraded,
        "missing": list(ans.missing),
        "rounds": ans.rounds,
        "residual": list(ans.residual),
        "sources": list(ans.source_ids),
        "claims": claims,
        "guardrails": {
            "breach_count": ans.guardrails.breach_count,
            "by_rail": ans.guardrails.counts_by_rail(),
            "evaluated": ans.guardrails.evaluated,
        },
        "metrics": {
            "tokens": tokens,
            "model_calls": traj.model_calls,
            "control_decisions": traj.control_decisions,
            "control_ratio": round(traj.control_ratio, 3),
            "isolation_ratio": round(traj.isolation_ratio, 1),
            "retries": traj.retries,
        },
    }


def _pin(agent: ChatAgent) -> str:
    """화면에 보이는 핀. 모델 ID 만으로는 무엇이 답했는지 다 말하지 못한다."""
    effort = getattr(agent.llm, "effort", None)
    return f"{agent.llm.model_id}" + (f" · {effort}" if effort else "")


def _doc_summary(agent: ChatAgent) -> list[dict[str, Any]]:
    return [
        {
            "id": d.id,
            "chars": len(d.text),
            "preview": d.text.strip()[:120],
            "source": d.metadata.get("path", "업로드"),
        }
        for d in sorted(agent.corpus.values(), key=lambda x: x.id)
    ]


# --------------------------------------------------------------------------

def create_app(agent: ChatAgent, sink: MemorySink, *, subject: str = "local-user"):
    """FastAPI 앱을 만든다. agent 와 sink 는 호출자가 소유한다."""
    app = FastAPI(title="agentcore chat", docs_url=None, redoc_url=None)
    # ChatAgent 는 상태(기억·코퍼스)를 갖고 FastAPI 는 동기 핸들러를 스레드풀에
    # 올린다. 한 사람이 쓰는 로컬 앱이라도 요청이 겹칠 수 있으므로 직렬화한다.
    lock = threading.Lock()

    @app.get("/")
    def index() -> Any:
        return FileResponse(STATIC / "index.html")

    @app.post("/api/chat")
    def chat(
        question: str = Body(..., embed=True),
        mode: str = Body("auto", embed=True),
    ) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise HTTPException(400, "빈 질문")
        if mode not in ("auto", "grounded", "free"):
            raise HTTPException(400, f"알 수 없는 mode: {mode}")

        with lock:
            # 이 턴이 만든 스팬만 잘라낸다 — 지표가 세션 전체로 번지면
            # "이 답이 얼마나 들었나"를 알 수 없다.
            start = len(sink.spans)
            answer = agent.ask(question, subject=subject, mode=mode)
            spans = list(sink.spans[start:])
        return answer_to_dict(answer, agent, spans)

    @app.get("/api/documents")
    def documents() -> dict[str, Any]:
        return {"documents": _doc_summary(agent)}

    @app.post("/api/documents")
    async def upload(files: list[UploadFile]) -> dict[str, Any]:
        added, skipped = [], []
        for f in files:
            name = Path(f.filename or "").name
            if not name.lower().endswith(ALLOWED_SUFFIXES):
                skipped.append({"name": name, "why": ".md/.txt 만 받습니다"})
                continue
            raw = await f.read()
            if len(raw) > MAX_UPLOAD_BYTES:
                skipped.append({"name": name, "why": f"{MAX_UPLOAD_BYTES:,}바이트 초과"})
                continue
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                skipped.append({"name": name, "why": "UTF-8 로 읽을 수 없습니다"})
                continue
            if not text.strip():
                skipped.append({"name": name, "why": "빈 파일"})
                continue

            doc_id = Path(name).stem
            with lock:
                # 같은 이름은 교체한다. 조용히 둘 다 남기면 어느 쪽이 인용된
                # 건지 알 수 없다.
                agent.remove_document(doc_id)
                agent.add_document(Document(doc_id, text, {"path": name}))
            added.append(doc_id)
        return {"added": added, "skipped": skipped, "documents": _doc_summary(agent)}

    @app.delete("/api/documents/{doc_id}")
    def remove(doc_id: str) -> dict[str, Any]:
        with lock:
            ok = agent.remove_document(doc_id)
        if not ok:
            raise HTTPException(404, f"문서 없음: {doc_id}")
        return {"removed": doc_id, "documents": _doc_summary(agent)}

    @app.delete("/api/memory")
    def forget() -> dict[str, Any]:
        """주체 단위 삭제. 문서 인덱스는 건드리지 않는다 — 그래서 분리했다."""
        with lock:
            removed = agent.forget(subject)
        return {"removed": removed}

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        traj = summarize_trajectory(list(sink.spans))
        turns = sum(1 for s in sink.spans if s.name == "turn")
        return {
            "documents": _doc_summary(agent),
            "model": _pin(agent),
            "history": [
                {"question": t.question, "answer": t.answer}
                for t in agent.history(subject)
            ],
            "session": {
                "turns": turns,
                "model_calls": traj.model_calls,
                "control_ratio": round(traj.control_ratio, 3),
                "isolation_ratio": round(traj.isolation_ratio, 1),
                "retries": traj.retries,
            },
        }

    return app
