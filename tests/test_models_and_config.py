from __future__ import annotations

import pytest
from pydantic import ValidationError

from lead_intel.config import ConfigurationError, Settings, load_settings
from lead_intel.models import LeadInput


def test_lead_requires_name():
    with pytest.raises(ValidationError):
        LeadInput(business_name="  ")


def test_lead_rejects_invalid_website():
    with pytest.raises(ValidationError):
        LeadInput(business_name="Acme", website="acme dot com")


def test_lead_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        LeadInput(business_name="Acme", score=99)


def test_settings_defaults(clean_env):
    settings = load_settings(use_dotenv=False)
    assert settings.tavily_api_key is None
    # First production runtime: Ollama, with no default model (operator must choose one).
    assert settings.llm_provider == "ollama"
    assert settings.llm_model is None
    assert settings.search_provider == "tavily"


def test_settings_read_from_env(clean_env):
    clean_env.setenv("TAVILY_API_KEY", "tvly-test")
    clean_env.setenv("LEAD_INTEL_TAVILY_MAX_RESULTS", "7")
    clean_env.setenv("LEAD_INTEL_LLM_MODEL", "openai/gpt-test")
    settings = load_settings(use_dotenv=False)
    assert settings.tavily_api_key.get_secret_value() == "tvly-test"
    assert settings.tavily_max_results == 7
    assert settings.llm_model == "openai/gpt-test"
    assert "tvly-test" not in repr(settings)


def test_invalid_env_value_raises_configuration_error(clean_env):
    clean_env.setenv("LEAD_INTEL_LLM_TEMPERATURE", "hot")
    with pytest.raises(ConfigurationError):
        load_settings(use_dotenv=False)


def test_missing_tavily_key_fails_fast():
    with pytest.raises(ConfigurationError):
        Settings().require_research_credentials()


def test_llm_resolves_to_openai_without_anthropic(clean_env):
    from lead_intel.agents.research_agent import build_llm

    clean_env.setenv("LLM_PROVIDER", "openai")
    clean_env.setenv("OPENAI_API_KEY", "sk-test")
    clean_env.setenv("LEAD_INTEL_LLM_MODEL", "gpt-5.6-luna")
    llm = build_llm(load_settings(use_dotenv=False))
    assert llm.provider == "openai"
    assert llm.model == "gpt-5.6-luna"
    assert llm.temperature is None  # not sent: GPT-5-family models reject non-default values


def test_llm_temperature_passed_only_when_configured(clean_env):
    from lead_intel.agents.research_agent import build_llm

    clean_env.setenv("LEAD_INTEL_LLM_MODEL", "qwen2.5:7b")
    clean_env.setenv("LEAD_INTEL_LLM_TEMPERATURE", "0.2")
    assert build_llm(load_settings(use_dotenv=False)).temperature == 0.2
