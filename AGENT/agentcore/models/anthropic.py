"""Anthropic 레퍼런스 어댑터.

프로바이더 중립 인터페이스의 참조 구현. anthropic 패키지가 없으면
임포트 시점이 아니라 **생성 시점**에 실패하므로, 이 어댑터를 안 쓰는
환경에서는 의존성이 필요 없다.

코어는 동결된다 — 여기서 파인튜닝 API 는 의도적으로 노출하지 않는다.
"""

from __future__ import annotations

from typing import Any

from .base import LLM, Completion, Message


class AnthropicLLM(LLM):
    """Messages API 어댑터.

    model_id 를 명시적으로 고정하는 것이 중요하다. 재현성 요건에서
    "동일 입력·동일 핀 버전"의 '핀 버전'이 이것이다.
    """

    name = "anthropic"

    def __init__(
        self,
        model_id: str = "claude-sonnet-5",
        *,
        api_key: str | None = None,
        max_tokens: int = 4096,
        client: Any = None,
        size_hint_b: float | None = None,
    ) -> None:
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.size_hint_b = size_hint_b
        if client is not None:
            self._client = client
        else:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "AnthropicLLM 을 쓰려면 anthropic 패키지가 필요합니다: "
                    "pip install 'agentcore[anthropic]'"
                ) from exc
            self._client = anthropic.Anthropic(api_key=api_key)

    def _complete(self, messages: list[Message], **params: Any) -> Completion:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        turns = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role in ("user", "assistant")
        ]
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": params.pop("max_tokens", self.max_tokens),
            "messages": turns,
        }
        if system:
            kwargs["system"] = system
        kwargs.update(params)

        resp = self._client.messages.create(**kwargs)
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        )
        return Completion(
            text=text,
            input_tokens=getattr(resp.usage, "input_tokens", 0),
            output_tokens=getattr(resp.usage, "output_tokens", 0),
            model=getattr(resp, "model", self.model_id),
            stop_reason=getattr(resp, "stop_reason", "end_turn") or "end_turn",
        )
