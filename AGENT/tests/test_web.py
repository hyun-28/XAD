"""웹 계층 — HTTP 로 번역하면서 상태를 숨기지 않는지 검증한다.

이 계층의 유일한 책임은 **저하 상태를 숨기지 않는 것**이다. 답변 텍스트만
내려보내고 blocked·verifiable 을 빼먹으면 앞의 여섯 계층이 무의미해진다.
"""

from __future__ import annotations

import io

import pytest

fastapi = pytest.importorskip("fastapi", reason='pip install -e ".[web]"')
from fastapi.testclient import TestClient  # noqa: E402

from agentcore.apps import ChatAgent  # noqa: E402
from agentcore.apps.web import create_app  # noqa: E402
from agentcore.memory import Document  # noqa: E402
from agentcore.models import EchoLLM  # noqa: E402
from agentcore.trace import MemorySink, Tracer  # noqa: E402

DOCS = [Document("vacation", "연차는 입사 1년 차에 11일 부여된다. 사용 3일 전에 신청한다.")]


def client(script, documents=DOCS, **kw):
    seq = list(script)
    sink = MemorySink()
    agent = ChatAgent(
        documents,
        llm=EchoLLM(responder=lambda m: seq.pop(0) if len(seq) > 1 else seq[0]),
        tracer=Tracer(sink=sink),
        **kw,
    )
    return TestClient(create_app(agent, sink, subject="t")), agent


def upload(c, name, text):
    return c.post(
        "/api/documents",
        files={"files": (name, io.BytesIO(text.encode("utf-8")), "text/plain")},
    )


# --------------------------------------------------------------------------

def test_페이지가_뜬다():
    c, _ = client(["- 아무 말 [vacation]"])
    r = c.get("/")
    assert r.status_code == 200
    assert "agentcore chat" in r.text


def test_근거_있는_답은_주장과_원문을_함께_내려준다():
    c, _ = client(["- 연차는 입사 1년 차에 11일 부여된다 [vacation]"])
    j = c.post("/api/chat", json={"question": "연차는 며칠인가요?"}).json()

    assert j["route"] == "grounded"
    assert j["verifiable"] and j["grounded"] and not j["degraded"]
    assert j["sources"] == ["vacation"]
    assert len(j["claims"]) == 1
    assert j["claims"][0]["sources"] == ["vacation"]
    assert j["claims"][0]["support"]          # 원문이 실려야 패널에서 대조가 된다
    assert j["claims"][0]["breaches"] == []


def test_차단된_답은_차단되었다고_내려준다():
    """이 계층이 blocked 를 빼먹으면 앞의 검증이 전부 무의미해진다."""
    c, _ = client(["- 연차는 60일이다 [payroll]"], max_rounds=1)
    j = c.post("/api/chat", json={"question": "연차는 며칠인가요?"}).json()

    assert j["blocked"] and j["degraded"]
    assert j["guardrails"]["breach_count"] >= 1
    assert "known_source" in j["guardrails"]["by_rail"]
    # 어느 문장이 걸렸는지 없으면 "차단됨"은 아무 정보도 주지 않는다
    assert j["claims"][0]["breaches"]


def test_자유_대화는_검증_불가로_표시된다():
    c, _ = client(["안녕하세요."], documents=[])
    j = c.post("/api/chat", json={"question": "안녕"}).json()

    assert j["route"] == "free"
    assert j["verifiable"] is False
    assert j["degraded"] is False          # 저하가 아니라 검증 대상이 아님
    assert j["claims"] == [] and j["sources"] == []


def test_게이트에_걸리면_미충족_슬롯을_내려준다():
    c, _ = client(["- 아무 말 [vacation]"])
    j = c.post("/api/chat", json={"question": "우주선 발사 절차", "mode": "grounded"}).json()
    assert j["missing"] == ["excerpts"]
    assert j["metrics"]["model_calls"] == 0     # 모델을 부르지 않았다


def test_모드_지정이_전달된다():
    c, _ = client(["자유롭게 답합니다."])
    j = c.post("/api/chat", json={"question": "연차는?", "mode": "free"}).json()
    assert j["route"] == "free"


def test_잘못된_입력은_거절된다():
    c, _ = client(["- 아무 말 [vacation]"])
    assert c.post("/api/chat", json={"question": "  "}).status_code == 400
    assert c.post("/api/chat", json={"question": "x", "mode": "hmm"}).status_code == 400


# --------------------------------------------------------------------------
# 업로드
# --------------------------------------------------------------------------

def test_업로드한_문서가_즉시_인용된다():
    c, _ = client(["- 경조사 휴가는 5일 부여된다 [family]"])
    r = upload(c, "family.md", "경조사 휴가는 5일 부여된다.")
    assert r.json()["added"] == ["family"]

    j = c.post("/api/chat", json={"question": "경조사 휴가는 며칠인가요?"}).json()
    assert j["sources"] == ["family"]
    assert not j["blocked"]


def test_같은_이름은_교체된다():
    """둘 다 남기면 어느 쪽이 인용된 건지 알 수 없다."""
    c, agent = client(["- 아무 말 [vacation]"])
    upload(c, "note.txt", "첫 번째 내용")
    upload(c, "note.txt", "두 번째 내용")
    ids = [d["id"] for d in c.get("/api/documents").json()["documents"]]
    assert ids.count("note") == 1
    assert agent.corpus["note"].text == "두 번째 내용"


def test_받지_않는_파일은_이유와_함께_거절된다():
    c, _ = client(["- 아무 말 [vacation]"])
    j = upload(c, "evil.pdf", "내용").json()
    assert j["added"] == []
    assert j["skipped"][0]["why"]

    j = upload(c, "empty.md", "   ").json()
    assert j["added"] == [] and j["skipped"]


def test_문서_철회는_즉시_반영된다():
    c, _ = client(["- 아무 말 [vacation]"])
    assert c.delete("/api/documents/vacation").status_code == 200
    assert c.get("/api/documents").json()["documents"] == []
    assert c.delete("/api/documents/vacation").status_code == 404


# --------------------------------------------------------------------------
# 기억
# --------------------------------------------------------------------------

def test_기억_삭제는_문서를_건드리지_않는다():
    c, agent = client(["- 연차는 입사 1년 차에 11일 부여된다 [vacation]"])
    c.post("/api/chat", json={"question": "연차는 며칠인가요?"})
    assert c.get("/api/state").json()["history"]

    assert c.delete("/api/memory").json()["removed"] >= 1
    state = c.get("/api/state").json()
    assert state["history"] == []
    assert [d["id"] for d in state["documents"]] == ["vacation"]


def test_세션_지표가_누적된다():
    c, _ = client(["- 연차는 입사 1년 차에 11일 부여된다 [vacation]"])
    c.post("/api/chat", json={"question": "연차는 며칠인가요?"})
    c.post("/api/chat", json={"question": "연차 신청은 언제까지인가요?"})
    s = c.get("/api/state").json()["session"]
    assert s["turns"] == 2
    assert s["control_ratio"] > 0.8
