"""Malformed LLM output must be rejected, never scored."""

from __future__ import annotations

import json

import pytest

from lead_intel.crew import result_from_raw_output
from lead_intel.models import Classification, LeadInput, RunStatus
from lead_intel.tasks.research_task import build_research_description, report_guardrail
from lead_intel.validation import validate_report_text


@pytest.fixture
def lead() -> LeadInput:
    return LeadInput(business_name="Acme Plumbing", website="https://acme.example")


@pytest.fixture
def valid_json(perfect_report) -> str:
    return perfect_report.model_dump_json()


def test_valid_json_accepted(valid_json):
    assert validate_report_text(valid_json).ok


def test_markdown_fenced_json_accepted(valid_json):
    assert validate_report_text(f"Here you go:\n```json\n{valid_json}\n```").ok


@pytest.mark.parametrize("raw", [None, "", "   ", "I could not find anything.", "{not json}"])
def test_empty_or_non_json_rejected(raw):
    outcome = validate_report_text(raw)
    assert not outcome.ok
    assert outcome.errors


def _mutate(valid_json: str, fn) -> str:
    data = json.loads(valid_json)
    fn(data)
    return json.dumps(data)


@pytest.mark.parametrize(
    "mutation, expected_loc",
    [
        (lambda d: d["research"].pop("industry"), "research.industry"),
        (lambda d: d["research"]["industry"].update(confidence="certain"), "research.industry.confidence"),
        (lambda d: d["research"]["website"].update(source_url="not a url"), "research.website.source_url"),
        (lambda d: d["qualification"]["icp_fit"].pop("industry_matches_icp"), "qualification.icp_fit.industry_matches_icp"),
        # The LLM must not smuggle in its own scores.
        (lambda d: d["qualification"].update(total_score=95), "qualification.total_score"),
        (lambda d: d["qualification"]["icp_fit"]["size_matches_icp"].update(points=3), "points"),
        (lambda d: d["qualification"]["icp_fit"]["size_matches_icp"].update(present="maybe"), "present"),
    ],
)
def test_schema_violations_rejected(valid_json, mutation, expected_loc):
    outcome = validate_report_text(_mutate(valid_json, mutation))
    assert not outcome.ok
    assert any(expected_loc in e for e in outcome.errors), outcome.errors


def test_invalid_output_produces_failed_unscored_result(lead):
    result = result_from_raw_output(lead, '{"research": {}}')
    assert result.status is RunStatus.FAILED
    assert result.score is None
    assert result.report is None
    assert result.errors


def test_valid_output_produces_scored_result(lead, valid_json):
    result = result_from_raw_output(lead, valid_json)
    assert result.status is RunStatus.SCORED
    assert result.score.total == 100
    assert result.score.classification is Classification.HOT


class _FakeOutput:
    def __init__(self, raw: str) -> None:
        self.raw = raw


def test_guardrail_rejects_invalid_with_actionable_errors():
    ok, message = report_guardrail(_FakeOutput('{"research": {}}'))
    assert ok is False
    assert "research.business_identity" in message


def test_guardrail_accepts_and_normalises_valid(valid_json):
    ok, payload = report_guardrail(_FakeOutput(f"```json\n{valid_json}\n```"))
    assert ok is True
    assert validate_report_text(payload).ok


def test_guardrail_signature_accepted_by_crewai():
    from crewai import Task

    task = Task(description="x", expected_output="y", guardrail=report_guardrail)
    assert task.guardrail is report_guardrail


def test_prompt_embeds_schema_and_forbids_scores(lead):
    prompt = build_research_description(lead, "Local service SMBs")
    assert "Acme Plumbing" in prompt
    assert "Local service SMBs" in prompt
    assert '"business_identity"' in prompt and '"no_existing_automation_detected"' in prompt
    assert "Do NOT output any scores" in prompt
