"""LLM factory: the only place that knows about LLM providers.

CrewAI connects to Ollama through its native OpenAI-compatible client
(``provider="ollama"``), which talks to ``<OLLAMA_HOST>/v1`` and needs no API key.
Constructing an LLM makes no network call.
"""

from __future__ import annotations

from typing import Any

from crewai import LLM

from lead_intel.config import Settings


def build_llm(settings: Settings) -> LLM:
    settings.require_llm_config()
    kwargs: dict[str, Any] = {"model": settings.resolved_llm_model}
    if settings.llm_temperature is not None:
        kwargs["temperature"] = settings.llm_temperature
    if settings.llm_timeout_seconds is not None:
        kwargs["timeout"] = settings.llm_timeout_seconds

    if settings.llm_provider == "ollama":
        return LLM(provider="ollama", base_url=settings.ollama_base_url, **kwargs)
    if settings.llm_provider == "openai":
        return LLM(provider="openai", api_key=settings.openai_api_key.get_secret_value(), **kwargs)
    raise AssertionError(f"unhandled provider {settings.llm_provider!r}")  # pragma: no cover
