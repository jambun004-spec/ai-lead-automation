"""The lead research task: prompt, output contract and guardrail.

No ``from __future__ import annotations`` here: CrewAI inspects the guardrail's
return annotation at runtime and rejects string annotations.
"""

import json
from typing import Any

from crewai import Agent, Task
from crewai.tasks.task_output import TaskOutput

from lead_intel.config import Settings
from lead_intel.models.lead import LeadInput
from lead_intel.models.report import LeadResearchReport
from lead_intel.validation import validate_report_text

RESEARCH_INSTRUCTIONS = """\
Research the business below using the web_search tool and produce structured evidence.

LEAD
{lead}

IDEAL CUSTOMER PROFILE (judge "ICP fit" against this)
{icp}

COLLECT EVIDENCE ABOUT
business identity, industry, location, website, services, online presence,
business signals (hiring, growth, news, reputation), automation opportunities,
existing automation (chatbots, online booking, CRMs, auto-responders),
unknowns, and every source URL you used.

EVIDENCE RULES
- Every finding has: claim, evidence, confidence, source_url.
- confidence = "confirmed" only when a cited source directly states it.
- confidence = "inferred" for reasonable conclusions from indirect evidence.
- confidence = "insufficient" when you could not find evidence.
- source_url must be a URL you actually retrieved; use null if none. Never invent URLs.
- List open questions in "unknowns".

QUALIFICATION RULES
- Fill in EVERY qualification signal: present (true/false), confidence, reasoning, source_url.
- If you cannot tell, set present=false and confidence="insufficient".
- Do NOT output any scores, points, totals or HOT/WARM/COOL/LOW labels. Scoring is done elsewhere.

OUTPUT
Return ONLY one JSON object (no prose, no Markdown) that validates against this JSON Schema:
{schema}
"""


def build_research_description(lead: LeadInput, icp_description: str) -> str:
    schema = json.dumps(LeadResearchReport.model_json_schema(), separators=(",", ":"))
    return RESEARCH_INSTRUCTIONS.format(lead=lead.describe(), icp=icp_description, schema=schema)


def report_guardrail(output: TaskOutput) -> tuple[bool, Any]:
    """Reject schema-invalid output so CrewAI asks the agent to fix it."""
    outcome = validate_report_text(output.raw)
    if not outcome.ok:
        return False, "Output failed schema validation. Fix these and return only JSON:\n" + "\n".join(
            outcome.errors
        )
    return True, outcome.report.model_dump_json()


def build_research_task(lead: LeadInput, agent: Agent, settings: Settings) -> Task:
    return Task(
        description=build_research_description(lead, settings.icp_description),
        expected_output="A single JSON object matching the LeadResearchReport schema.",
        agent=agent,
        guardrail=report_guardrail,
        guardrail_max_retries=settings.guardrail_max_retries,
    )
