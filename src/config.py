from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider

try:  # python-dotenv is optional at import time.
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv should be installed per README.
    load_dotenv = None


# Reasonable per-provider default model names when LLM_MODEL is not set.
_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "custom": "gpt-4o-mini",
    "gemini": "gemini-1.5-flash",
    "anthropic": "claude-3-5-haiku-latest",
    "ollama": "llama3.1",
    "openrouter": "openai/gpt-4o-mini",
}


@dataclass
class LabConfig:
    """Shared configuration for the lab."""

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def _api_key_for(provider: str) -> str | None:
    """Pick the right API key env var for the selected provider."""

    mapping = {
        "openai": "OPENAI_API_KEY",
        "custom": "CUSTOM_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "ollama": "OLLAMA_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    return os.getenv(mapping.get(provider, "OPENAI_API_KEY"))


def _base_url_for(provider: str) -> str | None:
    """Pick the right base URL env var for the selected provider."""

    if provider == "custom":
        return os.getenv("CUSTOM_BASE_URL")
    if provider == "ollama":
        return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    if provider == "openrouter":
        return os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    return None


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables and return a populated LabConfig.

    1. Resolve the repo root (default: parent of ``src/``).
    2. Load values from ``.env`` if present.
    3. Create ``state/`` if it does not exist.
    4. Read provider + compact settings from env with sensible defaults.
    """

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    if load_dotenv is not None:
        env_path = root / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        else:
            load_dotenv()  # also honour a CWD .env if present

    data_dir = root / "data"
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    provider = normalize_provider(os.getenv("LLM_PROVIDER", "openai"))
    model_name = os.getenv("LLM_MODEL", _DEFAULT_MODELS.get(provider, "gpt-4o-mini"))
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))

    model = ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=_api_key_for(provider),
        base_url=_base_url_for(provider),
    )

    # Judge model defaults to the same provider but can be overridden.
    judge_provider = normalize_provider(os.getenv("JUDGE_PROVIDER", provider))
    judge_model = ProviderConfig(
        provider=judge_provider,
        model_name=os.getenv("JUDGE_MODEL", _DEFAULT_MODELS.get(judge_provider, model_name)),
        temperature=float(os.getenv("JUDGE_TEMPERATURE", "0.0")),
        api_key=_api_key_for(judge_provider),
        base_url=_base_url_for(judge_provider),
    )

    # Compact defaults: small enough that the long-context stress test triggers
    # several compactions, large enough that short standard threads rarely do.
    compact_threshold_tokens = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "600"))
    compact_keep_messages = int(os.getenv("COMPACT_KEEP_MESSAGES", "6"))

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold_tokens,
        compact_keep_messages=compact_keep_messages,
        model=model,
        judge_model=judge_model,
    )
