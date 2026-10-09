"""LLM and search provider configuration. No network calls are made."""

from __future__ import annotations

import inspect

import pytest

from lead_intel.agents import research_agent
from lead_intel.config import ConfigurationError, load_settings
from lead_intel.llm import build_llm
from lead_intel.search import SearchError, SearchProvider, SearchResult, TavilySearchProvider, build_search_provider
from lead_intel.tools.web_search import WebSearchTool


@pytest.fixture
def ollama_env(clean_env):
    clean_env.setenv("LLM_PROVIDER", "ollama")
    clean_env.setenv("LEAD_INTEL_LLM_MODEL", "qwen2.5:7b")
    clean_env.setenv("TAVILY_API_KEY", "tvly-test")
    return clean_env


def settings():
    return load_settings(use_dotenv=False)


# --- Ollama provider configuration -------------------------------------------


def test_ollama_llm_uses_crewai_openai_compatible_client(ollama_env):
    llm = build_llm(settings())
    assert type(llm).__name__ == "OpenAICompatibleCompletion"
    assert llm.provider == "ollama"
    assert llm.model == "qwen2.5:7b"
    assert llm.base_url == "http://localhost:11434/v1"
    assert llm.temperature is None


def test_ollama_host_is_configurable(ollama_env):
    ollama_env.setenv("OLLAMA_HOST", "127.0.0.1:11500")
    assert build_llm(settings()).base_url == "http://127.0.0.1:11500/v1"


def test_ollama_timeout_and_temperature_are_passed(ollama_env):
    ollama_env.setenv("LEAD_INTEL_LLM_TIMEOUT_SECONDS", "1800")
    ollama_env.setenv("LEAD_INTEL_LLM_TEMPERATURE", "0.1")
    llm = build_llm(settings())
    assert llm.timeout == 1800
    assert llm.temperature == 0.1


def test_provider_name_is_case_insensitive(ollama_env):
    ollama_env.setenv("LLM_PROVIDER", "  Ollama ")
    assert settings().llm_provider == "ollama"


def test_research_agent_is_provider_agnostic(ollama_env):
    source = inspect.getsource(research_agent).lower()
    assert "ollama" not in source and "openai" not in source and "tavily" not in source
    agent = research_agent.build_research_agent(settings(), tools=[])
    assert agent.llm.provider == "ollama"


# --- missing Ollama configuration -------------------------------------------


def test_ollama_without_model_is_a_configuration_error(ollama_env):
    ollama_env.delenv("LEAD_INTEL_LLM_MODEL")
    with pytest.raises(ConfigurationError, match="LEAD_INTEL_LLM_MODEL"):
        build_llm(settings())
    with pytest.raises(ConfigurationError, match="LEAD_INTEL_LLM_MODEL"):
        settings().require_research_credentials()


def test_ollama_is_the_default_provider_and_still_needs_a_model(clean_env):
    clean_env.setenv("TAVILY_API_KEY", "tvly-test")
    with pytest.raises(ConfigurationError, match="Ollama"):
        settings().require_research_credentials()


def test_blank_ollama_host_falls_back_to_default(ollama_env):
    ollama_env.setenv("OLLAMA_HOST", "   ")
    assert build_llm(settings()).base_url == "http://localhost:11434/v1"


# --- OpenAI / Anthropic keys not required for Ollama ------------------------


def test_openai_and_anthropic_keys_not_required_with_ollama(ollama_env):
    from lead_intel.crew import build_crew
    from lead_intel.models import LeadInput

    s = settings()
    assert s.openai_api_key is None
    s.require_research_credentials()  # does not raise
    crew = build_crew(LeadInput(business_name="Acme"), s)
    assert crew.agents[0].llm.provider == "ollama"
    assert [t.name for t in crew.agents[0].tools] == ["web_search"]


def test_openai_remains_available_as_future_provider(clean_env):
    clean_env.setenv("LLM_PROVIDER", "openai")
    clean_env.setenv("OPENAI_API_KEY", "sk-test")
    llm = build_llm(settings())
    assert llm.provider == "openai"
    assert llm.model == "gpt-5.6-luna"  # OpenAI default model


def test_openai_selected_requires_openai_key(clean_env):
    clean_env.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        build_llm(settings())


def test_openai_key_value_never_appears_in_settings_repr(clean_env):
    clean_env.setenv("OPENAI_API_KEY", "sk-very-secret")
    assert "sk-very-secret" not in repr(settings())


# --- Tavily provider configuration and key requirement -----------------------


def test_tavily_is_default_search_provider(ollama_env):
    provider = build_search_provider(settings())
    assert isinstance(provider, TavilySearchProvider)
    assert isinstance(provider, SearchProvider)
    assert provider.name == "tavily"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_tavily_key_required_when_tavily_selected(ollama_env, value):
    if value is None:
        ollama_env.delenv("TAVILY_API_KEY")
    else:
        ollama_env.setenv("TAVILY_API_KEY", value)
    with pytest.raises(ConfigurationError, match="TAVILY_API_KEY"):
        build_search_provider(settings())
    with pytest.raises(ConfigurationError, match="TAVILY_API_KEY"):
        settings().require_research_credentials()


class _FakeTavilyClient:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def test_tavily_provider_maps_results_and_skips_urlless():
    client = _FakeTavilyClient(
        {"results": [{"title": "A", "url": "https://a.example", "content": "x"}, {"title": "no url"}]}
    )
    results = TavilySearchProvider("tvly-test", client=client).search("acme", 3)
    assert results == [SearchResult(title="A", url="https://a.example", content="x")]
    assert client.calls == [{"query": "acme", "max_results": 3, "search_depth": "basic"}]


def test_tavily_provider_wraps_errors():
    provider = TavilySearchProvider("tvly-test", client=_FakeTavilyClient(error=TimeoutError("slow")))
    with pytest.raises(SearchError, match="TimeoutError"):
        provider.search("acme", 3)


class _FakeProvider:
    name = "fake"

    def __init__(self, results=None, error=None):
        self.results, self.error = results or [], error

    def search(self, query, max_results):
        if self.error:
            raise self.error
        return self.results


def test_web_search_tool_works_with_any_provider():
    tool = WebSearchTool(_FakeProvider([SearchResult(title="T", url="https://t.example", content="c")]))
    assert "URL: https://t.example" in tool.run(query="acme")


def test_web_search_tool_reports_search_errors_to_agent():
    tool = WebSearchTool(_FakeProvider(error=SearchError("quota exceeded")))
    assert "Search failed" in tool.run(query="acme")


# --- invalid provider configuration -------------------------------------------


@pytest.mark.parametrize(
    "var, value",
    [
        ("LLM_PROVIDER", "gemini"),
        ("LLM_PROVIDER", "anthropic"),
        ("LEAD_INTEL_SEARCH_PROVIDER", "serpapi"),
        ("LEAD_INTEL_LLM_TIMEOUT_SECONDS", "-5"),
    ],
)
def test_invalid_provider_configuration_rejected(clean_env, var, value):
    clean_env.setenv(var, value)
    with pytest.raises(ConfigurationError):
        settings()
