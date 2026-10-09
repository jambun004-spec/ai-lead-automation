from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict

from lead_intel.llm_output import validate_llm_json
from lead_intel.models import LeadResearchReport
from lead_intel.validation import validate_report_text


class Point(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: int
    y: int


def test_valid_json_returns_typed_value():
    outcome = validate_llm_json(Point, '{"x": 1, "y": 2}')
    assert outcome.ok
    assert outcome.value == Point(x=1, y=2)


def test_fenced_json_with_prose_accepted():
    assert validate_llm_json(Point, 'Sure:\n```json\n{"x": 1, "y": 2}\n```\nDone.').ok


@pytest.mark.parametrize("raw", [None, "", "  ", "no json here", "{broken"])
def test_empty_or_non_json_rejected(raw):
    outcome = validate_llm_json(Point, raw)
    assert not outcome.ok and outcome.value is None and outcome.errors


@pytest.mark.parametrize(
    "raw, loc",
    [('{"x": 1}', "y"), ('{"x": "one", "y": 2}', "x"), ('{"x": 1, "y": 2, "score": 9}', "score")],
)
def test_schema_violations_reported_with_location(raw, loc):
    outcome = validate_llm_json(Point, raw)
    assert not outcome.ok
    assert any(e.startswith(loc) for e in outcome.errors), outcome.errors


def test_no_repair_or_coercion_of_data():
    # Values are validated, never defaulted: a missing field is an error, not a zero.
    assert validate_llm_json(Point, '{"x": 1, "y": null}').errors


def test_agrees_with_existing_report_validator(perfect_report):
    raw = perfect_report.model_dump_json()
    assert validate_llm_json(LeadResearchReport, raw).value == validate_report_text(raw).report
    assert validate_llm_json(LeadResearchReport, "{}").errors == validate_report_text("{}").errors
