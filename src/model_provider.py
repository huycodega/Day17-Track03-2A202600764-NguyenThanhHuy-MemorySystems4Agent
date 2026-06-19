from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents.

    Supported providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


# Canonical provider names supported by the lab.
SUPPORTED_PROVIDERS = {
    "openai",
    "custom",
    "gemini",
    "anthropic",
    "ollama",
    "openrouter",
}

# Common aliases / typos -> canonical name.
_PROVIDER_ALIASES = {
    "open-ai": "openai",
    "open_ai": "openai",
    "oai": "openai",
    "gpt": "openai",
    "azure": "openai",
    "azureopenai": "openai",
    "google": "gemini",
    "googleai": "gemini",
    "google-genai": "gemini",
    "google_genai": "gemini",
    "gemini-pro": "gemini",
    "anthorpic": "anthropic",  # frequent typo
    "antropic": "anthropic",
    "claude": "anthropic",
    "ollama-local": "ollama",
    "local": "ollama",
    "open-router": "openrouter",
    "open_router": "openrouter",
    "router": "openrouter",
    "openai-compatible": "custom",
    "compatible": "custom",
}


def normalize_provider(value: str) -> str:
    """Normalize provider strings: lower-case, strip, and map aliases/typos.

    Example: ``anthorpic`` -> ``anthropic``.
    """

    if not value:
        return "openai"
    key = value.strip().lower().replace(" ", "")
    key = _PROVIDER_ALIASES.get(key, key)
    return key


def build_chat_model(config: ProviderConfig):
    """Instantiate the real chat model for the selected provider.

    Imports are kept lazy so the deterministic offline path never needs the
    provider SDKs installed. Raises a clear error for unsupported providers.
    """

    provider = normalize_provider(config.provider)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
        )

    if provider == "custom":
        # OpenAI-compatible endpoint (vLLM, LM Studio, Together, etc.).
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key or "not-needed",
            base_url=config.base_url,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            google_api_key=config.api_key,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=config.model_name,
            temperature=config.temperature,
            base_url=config.base_url or "http://localhost:11434",
        )

    if provider == "openrouter":
        # langchain-openrouter exposes ChatOpenRouter; fall back to the
        # OpenAI-compatible client against the OpenRouter base URL if absent.
        try:
            from langchain_openrouter import ChatOpenRouter

            return ChatOpenRouter(
                model=config.model_name,
                temperature=config.temperature,
                api_key=config.api_key,
            )
        except ImportError:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                api_key=config.api_key,
                base_url=config.base_url or "https://openrouter.ai/api/v1",
            )

    raise ValueError(
        f"Unsupported provider: {config.provider!r} (normalized: {provider!r}). "
        f"Supported providers: {sorted(SUPPORTED_PROVIDERS)}."
    )
