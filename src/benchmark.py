from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config

try:
    from tabulate import tabulate
except ImportError:  # pragma: no cover - graceful fallback
    tabulate = None


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def recall_points(answer: str, expected: list[str]) -> float:
    """Return 0 / 0.5 / 1 based on how many expected facts appear in answer."""

    if not expected:
        return 0.0
    low = (answer or "").lower()
    found = sum(1 for fact in expected if fact.lower() in low)
    if found == 0:
        return 0.0
    if found == len(expected):
        return 1.0
    return 0.5


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight quality score for offline mode (recall + substance)."""

    if not answer or not answer.strip():
        return 0.0
    frac = 0.0
    if expected:
        frac = sum(1 for f in expected if f.lower() in answer.lower()) / len(expected)
    length_ok = 1.0 if len(answer.strip()) >= 20 else 0.5
    return round(min(1.0, 0.5 * frac + 0.5 * length_ok), 3)


def run_agent_benchmark(
    agent_name: str, agent, conversations: list[dict[str, Any]], config
) -> BenchmarkRow:
    agent_tokens = 0
    prompt_tokens = 0
    recalls: list[float] = []
    quals: list[float] = []
    user_ids: list[str] = []
    compactions = 0

    for conv in conversations:
        user_id = conv["user_id"]
        thread_id = conv["id"]
        if user_id not in user_ids:
            user_ids.append(user_id)

        # 1. Feed every turn of the conversation into one thread.
        for turn in conv["turns"]:
            agent.reply(user_id, thread_id, turn)

        # 2-3. Conversation-level token cost (agent output + prompt context).
        agent_tokens += agent.token_usage(thread_id)
        prompt_tokens += agent.prompt_token_usage(thread_id)
        compactions += agent.compaction_count(thread_id)

        # 4. Cross-session recall: ask in a FRESH thread the agent never saw.
        for i, rq in enumerate(conv.get("recall_questions", [])):
            fresh_thread = f"{thread_id}__recall__{i}"
            result = agent.reply(user_id, fresh_thread, rq["question"])
            answer = result["answer"]
            recalls.append(recall_points(answer, rq["expected_contains"]))
            quals.append(heuristic_quality(answer, rq["expected_contains"]))

    # 6. Memory growth (only the advanced agent keeps persistent files).
    memory_growth = 0
    if hasattr(agent, "memory_file_size"):
        memory_growth = sum(agent.memory_file_size(u) for u in user_ids)

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=round(sum(recalls) / len(recalls), 3) if recalls else 0.0,
        response_quality=round(sum(quals) / len(quals), 3) if quals else 0.0,
        memory_growth_bytes=memory_growth,
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    table = [
        [
            r.agent_name,
            r.agent_tokens_only,
            r.prompt_tokens_processed,
            r.recall_score,
            r.response_quality,
            r.memory_growth_bytes,
            r.compactions,
        ]
        for r in rows
    ]
    if tabulate is not None:
        return tabulate(table, headers=headers, tablefmt="github")

    # Minimal markdown fallback if tabulate is unavailable.
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in table:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def _run_suite(title: str, dataset_path: Path, config) -> None:
    conversations = load_conversations(dataset_path)
    rows = [
        run_agent_benchmark(
            "Baseline", BaselineAgent(config, force_offline=True), conversations, config
        ),
        run_agent_benchmark(
            "Advanced", AdvancedAgent(config, force_offline=True), conversations, config
        ),
    ]
    print(f"\n## {title}\n")
    print(format_rows(rows))


def main() -> None:
    config = load_config(Path(__file__).resolve().parent.parent)

    _run_suite(
        "Standard Benchmark (data/conversations.json)",
        config.data_dir / "conversations.json",
        config,
    )
    _run_suite(
        "Long-Context Stress Benchmark (data/advanced_long_context.json)",
        config.data_dir / "advanced_long_context.json",
        config,
    )


if __name__ == "__main__":
    main()
