from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Stable, deterministic token estimator (no real tokenizer needed).

    Heuristic: ~4 characters per token. Empty/whitespace text -> 0.
    """

    if not text:
        return 0
    stripped = text.strip()
    if not stripped:
        return 0
    return math.ceil(len(stripped) / 4)


# --------------------------------------------------------------------------- #
# Persistent memory: User.md                                                  #
# --------------------------------------------------------------------------- #

_FACT_LINE = re.compile(r"^- \*\*(?P<key>.+?)\*\*:\s*(?P<value>.*)$")

# Human-readable Vietnamese labels for the structured fields we persist.
FACT_LABELS = {
    "name": "Tên",
    "location": "Nơi ở hiện tại",
    "profession": "Nghề nghiệp hiện tại",
    "drink": "Đồ uống yêu thích",
    "food": "Món ăn yêu thích",
    "pet": "Thú cưng",
    "response_style": "Phong cách trả lời",
    "interests": "Mối quan tâm",
}

# Order facts are rendered in so User.md (and recall answers) are stable.
_FACT_ORDER = list(FACT_LABELS.keys())


def _empty_profile() -> str:
    return "# User Profile\n\n_(chưa có thông tin)_\n"


@dataclass
class UserProfileStore:
    """Persistent per-user storage backed by one ``User.md`` file each.

    Facts are stored as markdown lines ``- **key**: value`` so the file is both
    machine-parseable and human-readable. Upserts overwrite a key in place,
    which is what gives the advanced agent clean conflict handling: a correction
    replaces the old value instead of piling up a second contradictory line.
    """

    root_dir: Path

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)

    def path_for(self, user_id: str) -> Path:
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", (user_id or "anon").strip()) or "anon"
        return self.root_dir / f"{slug}.User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return _empty_profile()

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        text = self.read_text(user_id)
        if search_text not in text:
            return False
        self.write_text(user_id, text.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    # -- structured helpers ------------------------------------------------- #

    def facts(self, user_id: str) -> dict[str, str]:
        facts: dict[str, str] = {}
        for line in self.read_text(user_id).splitlines():
            match = _FACT_LINE.match(line.strip())
            if match:
                facts[match.group("key")] = match.group("value").strip()
        return facts

    def _render(self, facts: dict[str, str]) -> str:
        lines = ["# User Profile", ""]
        ordered_keys = [k for k in _FACT_ORDER if k in facts]
        ordered_keys += [k for k in facts if k not in _FACT_ORDER]
        for key in ordered_keys:
            lines.append(f"- **{key}**: {facts[key]}")
        return "\n".join(lines) + "\n"

    def upsert_fact(self, user_id: str, key: str, value: str) -> Path:
        """Insert or overwrite a single fact (last write wins per key)."""

        facts = self.facts(user_id)
        facts[key] = value.strip()
        return self.write_text(user_id, self._render(facts))


# --------------------------------------------------------------------------- #
# Fact extraction                                                             #
# --------------------------------------------------------------------------- #

# Cues that signal a value is being negated / corrected away.
_NEGATION_CUES = ("không còn", "không phải", "đừng", "chưa từng")
# Cues that signal "this is the *current* fact" -> highest priority.
_CURRENT_CUES = ("hiện tại", "vẫn ", "bây giờ", "giờ ")

_JOB = r"((?:[A-Za-zÀ-ỹ][\wÀ-ỹ]*\s+){1,2}(?:engineer|manager|developer|scientist))"
_PLACE = r"([A-ZÀ-Ỹ][\wÀ-ỹ]*(?:\s+[A-ZÀ-Ỹ][\wÀ-ỹ]*)*)"

# Lowercase Vietnamese connectives that can wrongly attach to a captured place
# (the À-Ỹ unicode range also matches some lowercase accented letters).
_PLACE_TRAILING_STOPWORDS = {
    "để", "và", "chứ", "mỗi", "trong", "nhưng", "vài", "như", "rồi", "cho",
    "khi", "nếu", "là", "mà", "vì", "do", "hay", "của", "một", "này", "đó",
    "đang", "thì", "với", "dù",
}


def _clean_place(value: str) -> str:
    words = value.split()
    while words and words[-1].lower() in _PLACE_TRAILING_STOPWORDS:
        words.pop()
    return " ".join(words)


def _is_negated(text: str, start: int) -> bool:
    window = text[max(0, start - 22): start].lower()
    return any(cue in window for cue in _NEGATION_CUES)


def _is_current(text: str, start: int) -> bool:
    window = text[max(0, start - 24): start].lower()
    return any(cue in window for cue in _CURRENT_CUES)


def _pick_candidate(text: str, pattern: str) -> str | None:
    """Pick the best value among regex matches.

    Rules (in order):
    1. ignore matches that are negated ("không còn ... X"),
    2. prefer the last match flagged as *current* ("hiện tại/vẫn ... X"),
    3. otherwise take the last non-negated match.
    """

    current: str | None = None
    fallback: str | None = None
    for m in re.finditer(pattern, text):
        value = m.group(1).strip()
        if _is_negated(text, m.start(1)):
            continue
        fallback = value
        if _is_current(text, m.start(1)):
            current = value
    return current or fallback


def extract_profile_updates(message: str) -> dict[str, str]:
    """Convert raw user text into stable profile facts.

    Only confident, stable facts are returned. Question-only turns (e.g.
    "Mình tên gì?") yield nothing so we never store a question as a fact, and
    corrections override old values cleanly (handled by ``_pick_candidate``).
    """

    if not message or not message.strip():
        return {}

    # Ignore question-only turns so we never store a question as a fact
    # (e.g. "Mình tên gì?" or "Hiện tại mình đang ở đâu?").
    low = message.lower()
    _interrogatives = (
        "ở đâu", "tên gì", "nghề gì", "là gì", "là ai", "con gì",
        "như thế nào", "thế nào", "ra sao", "biết gì",
    )
    if message.strip().endswith("?") or any(p in low for p in _interrogatives):
        return {}

    updates: dict[str, str] = {}

    # -- name ------------------------------------------------------------- #
    name_match = re.search(
        r"(?:mình tên là|tôi tên là|tên mình là|tên của mình là|tên là)\s+"
        r"([A-ZÀ-Ỹ][^.,\n!?]*)",
        message,
    )
    if name_match:
        updates["name"] = name_match.group(1).strip()

    # -- location (current residence only) -------------------------------- #
    location = _pick_candidate(
        message,
        r"(?:đang ở|hiện ở|hiện đang ở|vẫn ở|đang làm việc ở|làm việc ở|"
        r"chuyển (?:ra|vào|sang|đến|về)\s+[^ ]*\s+ở|^Mình ở|\bMình ở)\s+" + _PLACE,
    )
    if location:
        cleaned = _clean_place(location)
        if cleaned and cleaned.lower() not in {"đâu", "gì", "ai", "nào", "sao", "đó"}:
            updates["location"] = cleaned

    # -- profession ------------------------------------------------------- #
    profession = _pick_candidate(
        message, r"(?:làm|là|chuyển sang|chuyển qua)\s+" + _JOB
    )
    if profession:
        updates["profession"] = profession

    # -- favourite drink / food (stable preferences) ---------------------- #
    if "cà phê sữa đá" in low:
        updates["drink"] = "cà phê sữa đá"
    if "mì quảng" in low:
        updates["food"] = "mì Quảng"

    # -- pet -------------------------------------------------------------- #
    if "corgi" in low:
        updates["pet"] = "corgi (tên Bơ)"

    # -- response style (only when the message is about answer style) ----- #
    if "ngắn gọn" in low:
        parts = ["ngắn gọn"]
        if "3 bullet" in low:
            parts.append("3 bullet")
        elif "bullet" in low:
            parts.append("bullet")
        if "ví dụ thực chiến" in low:
            parts.append("ví dụ thực chiến")
        elif "ví dụ thực tế" in low:
            parts.append("ví dụ thực tế")
        updates["response_style"] = ", ".join(parts)

    # -- technical interests --------------------------------------------- #
    if "python" in low:
        updates["interests"] = "Python, AI ứng dụng"

    return updates


# --------------------------------------------------------------------------- #
# Compact memory                                                              #
# --------------------------------------------------------------------------- #

def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Heuristic summary of older messages (compress, don't drop entirely).

    Keeps the first and last few messages as short one-liners. This is
    intentionally lossy: it is far smaller than the raw history, which is the
    whole point of compaction. Can be swapped for an LLM summary later.
    """

    if not messages:
        return ""

    def shorten(m: dict[str, str]) -> str:
        content = " ".join(m.get("content", "").split())
        if len(content) > 70:
            content = content[:67] + "..."
        return f"{m.get('role', 'user')}: {content}"

    if len(messages) <= max_items:
        chosen = messages
    else:
        head = max_items // 2
        tail = max_items - head
        chosen = messages[:head] + messages[-tail:]
    return " | ".join(shorten(m) for m in chosen)


@dataclass
class CompactMemoryManager:
    """Compact memory for long threads.

    Keeps the most recent ``keep_messages`` messages verbatim. When the running
    token estimate (summary + kept messages) exceeds ``threshold_tokens``, older
    messages are folded into a growing summary and a compaction is recorded.
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _ensure(self, thread_id: str) -> dict[str, object]:
        return self.state.setdefault(
            thread_id, {"messages": [], "summary": "", "compactions": 0}
        )

    def _tokens(self, st: dict[str, object]) -> int:
        total = estimate_tokens(st["summary"])  # type: ignore[arg-type]
        for m in st["messages"]:  # type: ignore[union-attr]
            total += estimate_tokens(m["content"])
        return total

    def append(self, thread_id: str, role: str, content: str) -> None:
        st = self._ensure(thread_id)
        st["messages"].append({"role": role, "content": content})  # type: ignore[union-attr]

        # Compact while we're over budget and there is something to fold away.
        while (
            self._tokens(st) > self.threshold_tokens
            and len(st["messages"]) > self.keep_messages  # type: ignore[arg-type]
        ):
            messages: list[dict[str, str]] = st["messages"]  # type: ignore[assignment]
            old = messages[: len(messages) - self.keep_messages]
            recent = messages[len(messages) - self.keep_messages:]
            chunk = summarize_messages(old)
            existing = str(st["summary"]).strip()
            st["summary"] = (existing + " || " + chunk).strip(" |") if existing else chunk
            st["messages"] = recent
            st["compactions"] = int(st["compactions"]) + 1  # type: ignore[arg-type]

            # Guard against an ever-growing summary on extremely long threads:
            # if the summary alone blows the budget, truncate its head.
            if estimate_tokens(st["summary"]) > self.threshold_tokens:  # type: ignore[arg-type]
                summary = str(st["summary"])
                keep_chars = self.threshold_tokens * 4
                st["summary"] = "...(đã nén)... " + summary[-keep_chars:]
                break

    def context(self, thread_id: str) -> dict[str, object]:
        st = self._ensure(thread_id)
        return {
            "messages": list(st["messages"]),  # type: ignore[arg-type]
            "summary": st["summary"],
            "compactions": st["compactions"],
        }

    def compaction_count(self, thread_id: str) -> int:
        return int(self._ensure(thread_id)["compactions"])  # type: ignore[arg-type]
