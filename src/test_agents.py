from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import UserProfileStore


def make_config(tmp_path: Path):
    """Isolated config: state under tmp_path, low compact threshold."""

    base = load_config()
    return replace(
        base,
        state_dir=tmp_path,
        compact_threshold_tokens=60,
        compact_keep_messages=3,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")

    # Default content before anything is written.
    assert "User Profile" in store.read_text("u1")
    assert store.file_size("u1") == 0

    store.write_text("u1", "# User Profile\n\n- **name**: DũngCT\n")
    assert "DũngCT" in store.read_text("u1")
    assert store.file_size("u1") > 0

    # Upsert overwrites a key in place (clean conflict handling).
    store.upsert_fact("u1", "profession", "backend engineer")
    store.upsert_fact("u1", "profession", "MLOps engineer")
    facts = store.facts("u1")
    assert facts["profession"] == "MLOps engineer"
    assert store.read_text("u1").count("profession") == 1

    # edit_text replaces one occurrence and reports the change.
    assert store.edit_text("u1", "DũngCT", "DũngCT Stress") is True
    assert "DũngCT Stress" in store.read_text("u1")
    assert store.edit_text("u1", "không-tồn-tại", "x") is False


def test_compact_trigger(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    agent = AdvancedAgent(config, force_offline=True)

    thread = "long-thread"
    long_turn = (
        "Đây là một lượt rất dài để ép compact memory hoạt động nhiều lần, "
        "mình nói thêm nhiều chi tiết về tin tức và công việc MLOps."
    )
    for _ in range(12):
        agent.reply("user-x", thread, long_turn)

    assert agent.compaction_count(thread) > 0


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)

    fact_turn = "Mình ở Đà Nẵng và hiện đang làm MLOps engineer."
    advanced.reply("dung", "session-A", fact_turn)
    baseline.reply("dung", "session-A", fact_turn)

    question = "Hiện tại mình làm nghề gì?"
    adv_answer = advanced.reply("dung", "session-B", question)["answer"]
    base_answer = baseline.reply("dung", "session-B", question)["answer"]

    # Advanced recalls across sessions via User.md; baseline cannot.
    assert "MLOps engineer" in adv_answer
    assert "MLOps engineer" not in base_answer


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)

    thread = "stress"
    long_turn = (
        "Mình kể một đoạn dài về bốn tin tức gần đây và nhiều preference để "
        "tạo áp lực ngữ cảnh, lặp lại nhiều lần cho thread thật sự dài ra."
    )
    for _ in range(16):
        advanced.reply("dung", thread, long_turn)
        baseline.reply("dung", thread, long_turn)

    adv_prompt = advanced.prompt_token_usage(thread)
    base_prompt = baseline.prompt_token_usage(thread)

    assert advanced.compaction_count(thread) > 0
    assert adv_prompt < base_prompt
