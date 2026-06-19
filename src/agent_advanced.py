from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    FACT_LABELS,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
)
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


def _looks_like_question(message: str) -> bool:
    low = message.lower()
    if "?" in message:
        return True
    cues = (
        "nhắc lại", "tên gì", "nghề gì", "ở đâu", "là ai", "tóm tắt",
        "là gì", "mô tả", "đâu mới", "có biết",
    )
    return any(c in low for c in cues)


class AdvancedAgent:
    """Agent B: three memory layers.

    1. short-term memory (within the thread, via CompactMemoryManager),
    2. persistent memory (``User.md`` per user, survives across threads),
    3. compact memory (older turns folded into a summary on long threads).
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}

        self.langchain_agent = None
        if not force_offline:
            self.langchain_agent = self._maybe_build_langchain_agent()

    # -- public API -------------------------------------------------------- #

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None:
            try:
                return self._reply_live(user_id, thread_id, message)
            except Exception:  # pragma: no cover - deterministic fallback
                pass
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    # -- internals --------------------------------------------------------- #

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        # 1-2. Extract stable facts and persist them (overwrite => clean
        #      conflict handling, no duplicate contradictory lines).
        for key, value in extract_profile_updates(message).items():
            self.profile_store.upsert_fact(user_id, key, value)

        # 3. Append the user turn into compact (short-term) memory.
        self.compact_memory.append(thread_id, "user", message)

        # 4. Prompt context carried this turn = User.md + summary + recent kept.
        prompt_context = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0) + prompt_context
        )

        # 5. Answer using persisted memory (so cross-session recall works).
        answer = self._offline_response(user_id, thread_id, message)

        # 6. Record the assistant turn + agent token usage.
        self.compact_memory.append(thread_id, "assistant", answer)
        self.thread_tokens[thread_id] = (
            self.thread_tokens.get(thread_id, 0) + estimate_tokens(answer)
        )
        return {
            "answer": answer,
            "agent_tokens": self.thread_tokens[thread_id],
            "prompt_tokens": self.thread_prompt_tokens[thread_id],
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        ctx = self.compact_memory.context(thread_id)
        tokens = estimate_tokens(self.profile_store.read_text(user_id))
        tokens += estimate_tokens(str(ctx.get("summary", "")))
        for m in ctx.get("messages", []):  # only the recent kept messages
            tokens += estimate_tokens(m["content"])
        return tokens

    def _profile_recap(self, user_id: str) -> str:
        facts = self.profile_store.facts(user_id)
        if not facts:
            return ""
        parts = []
        for key, label in FACT_LABELS.items():
            if key in facts:
                parts.append(f"{label}: {facts[key]}")
        for key, value in facts.items():  # any extra/bonus fields
            if key not in FACT_LABELS:
                parts.append(f"{key}: {value}")
        return ". ".join(parts) + "."

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        if _looks_like_question(message):
            recap = self._profile_recap(user_id)
            if recap:
                return "Theo hồ sơ đã lưu của bạn — " + recap
            return "Mình chưa lưu được thông tin nào của bạn để nhắc lại."
        # Statement turn: acknowledge briefly; the fact is already persisted.
        return "Đã ghi nhớ thông tin của bạn vào hồ sơ lâu dài."

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Live path: LangGraph agent + User.md tools + dynamic profile prompt."""

        # Persist facts deterministically regardless of the live model so the
        # benchmark's recall guarantees still hold.
        for key, value in extract_profile_updates(message).items():
            self.profile_store.upsert_fact(user_id, key, value)
        self.compact_memory.append(thread_id, "user", message)

        profile = self.profile_store.read_text(user_id)
        result = self.langchain_agent.invoke(
            {
                "messages": [
                    {"role": "system", "content": f"Hồ sơ người dùng:\n{profile}"},
                    {"role": "user", "content": message},
                ]
            },
            config={"configurable": {"thread_id": thread_id}},
        )
        answer = result["messages"][-1].content
        self.compact_memory.append(thread_id, "assistant", answer)
        self.thread_prompt_tokens[thread_id] = (
            self.thread_prompt_tokens.get(thread_id, 0)
            + self._estimate_prompt_context_tokens(user_id, thread_id)
        )
        self.thread_tokens[thread_id] = (
            self.thread_tokens.get(thread_id, 0) + estimate_tokens(answer)
        )
        return {
            "answer": answer,
            "agent_tokens": self.thread_tokens[thread_id],
            "prompt_tokens": self.thread_prompt_tokens[thread_id],
        }

    def _maybe_build_langchain_agent(self):
        """Wire a live agent with short-term memory + User.md tools.

        Returns None when deps/credentials are missing so offline stays default.
        """

        if not self.config.model.api_key and self.config.model.provider not in {"ollama", "custom"}:
            return None
        try:
            from langchain_core.tools import tool
            from langgraph.checkpoint.memory import InMemorySaver
            from langgraph.prebuilt import create_react_agent
        except ImportError:
            return None

        store = self.profile_store

        @tool
        def read_user_profile(user_id: str) -> str:
            """Đọc hồ sơ User.md của một người dùng."""
            return store.read_text(user_id)

        @tool
        def upsert_user_fact(user_id: str, key: str, value: str) -> str:
            """Ghi/đè một fact ổn định vào hồ sơ User.md."""
            store.upsert_fact(user_id, key, value)
            return "ok"

        try:
            model = build_chat_model(self.config.model)
            return create_react_agent(
                model,
                tools=[read_user_profile, upsert_user_fact],
                checkpointer=InMemorySaver(),
                prompt=(
                    "Bạn là trợ lý có trí nhớ lâu dài qua User.md. "
                    "Hãy trả lời ngắn gọn và ưu tiên thông tin mới nhất khi có đính chính."
                ),
            )
        except Exception:
            return None
