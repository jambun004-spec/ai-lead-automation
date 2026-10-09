"""Inbound runtime settings, separate from the outbound research ``Settings``.

Inbound reads only the *shared* LLM variables (LLM_PROVIDER, LEAD_INTEL_LLM_MODEL,
...) plus its own ``LEAD_OPS_*`` variables. Tavily and other research settings
are never read or required here.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from lead_intel.config import ConfigurationError, Settings

# Shared LLM variables -> fields of the existing Settings used by lead_intel.llm.build_llm.
LLM_ENV_MAP = {
    "llm_provider": "LLM_PROVIDER",
    "llm_model": "LEAD_INTEL_LLM_MODEL",
    "llm_temperature": "LEAD_INTEL_LLM_TEMPERATURE",
    "llm_timeout_seconds": "LEAD_INTEL_LLM_TIMEOUT_SECONDS",
    "ollama_base_url": "OLLAMA_HOST",
    "openai_api_key": "OPENAI_API_KEY",
}

INBOUND_ENV_MAP = {
    "business_config_path": "LEAD_OPS_BUSINESS_CONFIG_PATH",
    "extraction_max_retries": "LEAD_OPS_EXTRACTION_MAX_RETRIES",
}


class InboundSettings(BaseModel):
    business_config_path: Path | None = None
    extraction_max_retries: int = Field(default=2, ge=0, le=5)
    llm: Settings = Field(default_factory=Settings)


def _read(env_map: dict[str, str]) -> dict[str, str]:
    return {field: os.environ[env] for field, env in env_map.items() if os.environ.get(env, "").strip()}


def load_inbound_settings(*, use_dotenv: bool = True) -> InboundSettings:
    if use_dotenv:
        load_dotenv(override=False)
    try:
        return InboundSettings(**_read(INBOUND_ENV_MAP), llm=Settings(**_read(LLM_ENV_MAP)))
    except ValueError as exc:  # pydantic.ValidationError subclasses ValueError
        raise ConfigurationError(f"Invalid inbound configuration: {exc}") from exc
