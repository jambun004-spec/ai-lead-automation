"""Environment-driven configuration.

Settings are read from process environment (optionally populated from a local
``.env`` file) and validated by Pydantic. Provider credentials are only
*required* for the providers actually selected, and only when the live research
crew is built, so pure modules (models, scoring) and their tests never need keys.
"""

from __future__ import annotations

import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr, field_validator

DEFAULT_ICP_DESCRIPTION = (
    "Small and medium-sized service businesses that handle customer inquiries, "
    "bookings or quotes and would benefit from AI automation."
)

LlmProvider = Literal["ollama", "openai"]
SearchProviderName = Literal["tavily"]

# Provider defaults. Ollama deliberately has no default model: the operator must
# choose one that is pulled locally.
DEFAULT_OPENAI_MODEL = "gpt-5.6-luna"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


class ConfigurationError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def _is_set(secret: SecretStr | None) -> bool:
    return secret is not None and bool(secret.get_secret_value().strip())


class Settings(BaseModel):
    # LLM
    llm_provider: LlmProvider = "ollama"
    llm_model: str | None = Field(default=None, min_length=1)
    # None = omit temperature and use the model default. GPT-5-family reasoning
    # models reject non-default temperatures, so only set this for models that accept it.
    llm_temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    # None = client default. Local CPU inference can need much longer.
    llm_timeout_seconds: float | None = Field(default=None, gt=0, le=7200)
    ollama_base_url: str = Field(default=DEFAULT_OLLAMA_BASE_URL, min_length=1)
    openai_api_key: SecretStr | None = None

    # Search
    search_provider: SearchProviderName = "tavily"
    tavily_api_key: SecretStr | None = None
    tavily_max_results: int = Field(default=5, ge=1, le=20)

    # Workflow
    guardrail_max_retries: int = Field(default=2, ge=0, le=5)
    icp_description: str = Field(default=DEFAULT_ICP_DESCRIPTION, min_length=10)

    # Supabase
    supabase_url: str | None = None
    supabase_anon_key: SecretStr | None = None


    @field_validator("llm_provider", "search_provider", mode="before")
    @classmethod
    def _normalise_provider(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @property
    def resolved_llm_model(self) -> str:
        """The model for the selected provider; call require_llm_config() first."""
        if self.llm_model:
            return self.llm_model
        if self.llm_provider == "openai":
            return DEFAULT_OPENAI_MODEL
        raise ConfigurationError(
            "LEAD_INTEL_LLM_MODEL must be set to a pulled Ollama model (e.g. qwen2.5:7b)."
        )

    def require_llm_config(self) -> None:
        if self.llm_provider == "ollama":
            self.resolved_llm_model  # raises if no model is configured
        elif self.llm_provider == "openai" and not _is_set(self.openai_api_key):
            raise ConfigurationError("OPENAI_API_KEY is not set but LLM_PROVIDER=openai.")

    def require_search_config(self) -> None:
        if self.search_provider == "tavily" and not _is_set(self.tavily_api_key):
            raise ConfigurationError("TAVILY_API_KEY is not set (see .env.example).")

    def require_research_credentials(self) -> None:
        """Fail fast before any LLM/web call if research cannot run."""
        self.require_search_config()
        self.require_llm_config()


_ENV_MAP = {
    "llm_provider": "LLM_PROVIDER",
    "llm_model": "LEAD_INTEL_LLM_MODEL",
    "llm_temperature": "LEAD_INTEL_LLM_TEMPERATURE",
    "llm_timeout_seconds": "LEAD_INTEL_LLM_TIMEOUT_SECONDS",
    "ollama_base_url": "OLLAMA_HOST",
    "openai_api_key": "OPENAI_API_KEY",
    "search_provider": "LEAD_INTEL_SEARCH_PROVIDER",
    "tavily_api_key": "TAVILY_API_KEY",
    "tavily_max_results": "LEAD_INTEL_TAVILY_MAX_RESULTS",
    "guardrail_max_retries": "LEAD_INTEL_GUARDRAIL_MAX_RETRIES",
    "icp_description": "LEAD_INTEL_ICP_DESCRIPTION",
    "supabase_url": "SUPABASE_URL",
    "supabase_anon_key": "SUPABASE_ANON_KEY",
}


def load_settings(*, use_dotenv: bool = True) -> Settings:
    """Build validated Settings from the environment.

    Empty variables are treated as unset so defaults apply.
    """
    if use_dotenv:
        load_dotenv(override=False)
    values = {
        field: os.environ[env]
        for field, env in _ENV_MAP.items()
        if os.environ.get(env, "").strip()
    }
    try:
        return Settings(**values)
    except ValueError as exc:  # pydantic.ValidationError subclasses ValueError
        raise ConfigurationError(f"Invalid configuration: {exc}") from exc
