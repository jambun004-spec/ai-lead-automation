"""Builders for valid LeadResearchReport fixtures (no LLM or network)."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from lead_intel.models import (
    AutomationOpportunitySignals,
    BusinessSignals,
    Confidence,
    Finding,
    IcpFitSignals,
    LeadResearchReport,
    OnlinePresenceSignals,
    QualificationEvidence,
    ResearchEvidence,
    Signal,
)

SignalFactory = Callable[[str], Signal]


def signal(present: bool = True, confidence: Confidence = Confidence.CONFIRMED, url: str | None = "https://example.com/s") -> Signal:
    return Signal(present=present, confidence=confidence, reasoning="test", source_url=url)


def finding(confidence: Confidence = Confidence.CONFIRMED, url: str | None = "https://example.com/f") -> Finding:
    return Finding(claim="claim", evidence="evidence", confidence=confidence, source_url=url)


def make_qualification(factory: SignalFactory) -> QualificationEvidence:
    def build(model):
        return model(**{name: factory(name) for name in model.model_fields})

    return QualificationEvidence(
        icp_fit=build(IcpFitSignals),
        online_presence=build(OnlinePresenceSignals),
        automation_opportunity=build(AutomationOpportunitySignals),
        business_signals=build(BusinessSignals),
        summary="test summary",
    )


def make_research(core: Finding | None = None, extra: list[Finding] | None = None, source_urls: list[str] | None = None) -> ResearchEvidence:
    core = core or finding()
    return ResearchEvidence(
        business_identity=core,
        industry=core,
        location=core,
        website=core,
        services=extra or [],
        source_urls=source_urls or [],
    )


def make_report(factory: SignalFactory, research: ResearchEvidence | None = None) -> LeadResearchReport:
    return LeadResearchReport(research=research or make_research(), qualification=make_qualification(factory))


@pytest.fixture
def perfect_report() -> LeadResearchReport:
    urls = [f"https://source{i}.example.com/" for i in range(5)]
    return make_report(lambda _: signal(), research=make_research(source_urls=urls))


@pytest.fixture
def clean_env(monkeypatch):
    """Isolate tests from the shell and .env: clear every variable Settings reads."""
    for var in [
        "TAVILY_API_KEY",
        "LEAD_INTEL_LLM_MODEL",
        "LEAD_INTEL_LLM_TEMPERATURE",
        "LEAD_INTEL_TAVILY_MAX_RESULTS",
        "LEAD_INTEL_GUARDRAIL_MAX_RETRIES",
        "LEAD_INTEL_ICP_DESCRIPTION",
        "LLM_PROVIDER",
        "LEAD_INTEL_LLM_TIMEOUT_SECONDS",
        "OLLAMA_HOST",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "LEAD_INTEL_SEARCH_PROVIDER",
    ]:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch
