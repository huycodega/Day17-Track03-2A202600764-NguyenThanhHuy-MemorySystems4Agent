from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


def _looks_like_question(message: str) -> bool:
    low = message.lower()
    if "?" in message:
        return True
    cues = ("nhắc lại", "tên gì", "nghề gì", "ở đâu", "là ai", "tóm tắt", "là gì")
    return any(c in low for c in cues)


class BaselineAgent:
    """Agent A: short-term (within-thread) memory only.

    No persistent ``User.md``. A brand new ``thread_id`` starts from an empty
    session, so the baseline *cannot* recall facts across threads/sessions.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None
        if not force_offline:
            self.langchain_agent = self._maybe_build_langchain_agent()

    # -- public API -------------------------------------------------------- #

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None:
            try:
                return self._reply_live(thread_id, message)
            except Exception:  # pragma: no cover - fall back to deterministic path
                pass
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self._session(thread_id).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self._session(thread_id).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    # -- internals --------------------------------------------------------- #

    def _session(self, thread_id: str) -> SessionState:
        return self.sessions.setdefault(thread_id, SessionState())

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self._session(thread_id)
        session.messages.append({"role": "user", "content": message})

        # Baseline re-processes its *entire* in-thread history every turn. This
        # is exactly why its prompt cost grows fast on long threads.
        prompt_context = sum(estimate_tokens(m["content"]) for m in session.messages)
        session.prompt_tokens_processed += prompt_context

        if _looks_like_question(message):
            answer = (
                "Mình chỉ nhớ được vài lượt gần đây trong phiên này và không có "
                "hồ sơ lâu dài, nên chưa chắc thông tin từ phiên trước của bạn."
            )
        else:
            answer = "Ừ, mình ghi nhận thông tin đó trong phiên hiện tại."

        session.messages.append({"role": "assistant", "content": answer})
        session.token_usage += estimate_tokens(answer)
        return {
            "answer": answer,
            "agent_tokens": session.token_usage,
            "prompt_tokens": session.prompt_tokens_processed,
        }

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        """Live path using a LangGraph agent with per-thread short-term memory."""

        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
        )
        answer = result["messages"][-1].content
        session = self._session(thread_id)
        session.messages.append({"role": "user", "content": message})
        session.messages.append({"role": "assistant", "content": answer})
        session.prompt_tokens_processed += sum(
            estimate_tokens(m["content"]) for m in session.messages
        )
        session.token_usage += estimate_tokens(answer)
        return {
            "answer": answer,
            "agent_tokens": session.token_usage,
            "prompt_tokens": session.prompt_tokens_processed,
        }

    def _maybe_build_langchain_agent(self):
        """Optionally wire a live agent with only short-term (thread) memory.

        Returns None when dependencies/credentials are missing so the
        deterministic offline path stays the default. The key property: there
        is *no* persistent store here, only an InMemorySaver keyed by thread.
        """

        if not self.config.model.api_key and self.config.model.provider not in {"ollama", "custom"}:
            return None
        try:
            from langgraph.checkpoint.memory import InMemorySaver
            from langgraph.prebuilt import create_react_agent
        except ImportError:
            return None
        try:
            model = build_chat_model(self.config.model)
            return create_react_agent(
                model,
                tools=[],
                checkpointer=InMemorySaver(),
                prompt=(
                    "Bạn là trợ lý chỉ có trí nhớ trong phiên hiện tại. "
                    "Bạn không có hồ sơ người dùng lâu dài."
                ),
            )
        except Exception:
            return None
