"""The lead research agent."""

from __future__ import annotations

from crewai import Agent
from crewai.tools import BaseTool

from lead_intel.config import Settings
from lead_intel.llm import build_llm  # re-exported for callers/tests

__all__ = ["build_llm", "build_research_agent"]


def build_research_agent(settings: Settings, tools: list[BaseTool]) -> Agent:
    return Agent(
        role="Lead Research Analyst",
        goal=(
            "Collect verifiable, source-cited evidence about a business and report "
            "it as structured JSON, clearly separating confirmed facts from inferences "
            "and unknowns."
        ),
        backstory=(
            "You are a meticulous B2B research analyst. You never invent facts, "
            "URLs, numbers or reviews. When evidence is missing you say so and mark "
            "the item 'insufficient' instead of guessing."
        ),
        tools=tools,
        llm=build_llm(settings),
        allow_delegation=False,
        max_iter=15,
        verbose=False,
    )
