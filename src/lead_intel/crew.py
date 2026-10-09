"""Workflow: research (LLM) -> validate (Pydantic) -> score (Python) -> classify.

``qualify_lead`` never raises for LLM/research failures: it returns a FAILED
result with the reasons, and a FAILED result is never scored.
"""

from __future__ import annotations

from crewai import Crew, Process

from lead_intel.agents.research_agent import build_research_agent
from lead_intel.config import Settings, load_settings
from lead_intel.models.lead import LeadInput
from lead_intel.models.report import LeadQualificationResult, RunStatus
from lead_intel.scoring.engine import score_report
from lead_intel.search import build_search_provider
from lead_intel.tasks.research_task import build_research_task
from lead_intel.tools.web_search import WebSearchTool
from lead_intel.validation import validate_report_text


def build_crew(lead: LeadInput, settings: Settings) -> Crew:
    settings.require_research_credentials()
    search = WebSearchTool(build_search_provider(settings), max_results=settings.tavily_max_results)
    agent = build_research_agent(settings, tools=[search])
    task = build_research_task(lead, agent, settings)
    return Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)


def result_from_raw_output(lead: LeadInput, raw: str | None) -> LeadQualificationResult:
    """Validate raw LLM text and score it; the single gate before scoring."""
    outcome = validate_report_text(raw)
    if not outcome.ok:
        return LeadQualificationResult(lead=lead, status=RunStatus.FAILED, errors=outcome.errors)
    return LeadQualificationResult(
        lead=lead,
        status=RunStatus.SCORED,
        report=outcome.report,
        score=score_report(outcome.report),
    )


def qualify_lead(lead: LeadInput, settings: Settings | None = None) -> LeadQualificationResult:
    settings = settings or load_settings()
    crew = build_crew(lead, settings)  # configuration errors propagate: they are not lead failures
    try:
        output = crew.kickoff()
    except Exception as exc:  # LLM/tool/guardrail-exhaustion failures
        return LeadQualificationResult(
            lead=lead,
            status=RunStatus.FAILED,
            errors=[f"Research crew failed: {type(exc).__name__}: {exc}"],
        )
    # Re-validate even though the guardrail passed: scoring only trusts this module's check.
    return result_from_raw_output(lead, output.raw)
