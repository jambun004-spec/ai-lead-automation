"""Qualification of every offline evaluation case, against hand-labelled expectations.

Expectations are written by hand from the example config's rules (not generated from
the engine): appointment intents = appointment_request + availability_enquiry, high
intent = appointment_request, urgency escalates, urgent keywords include "swelling",
"asap" is a time-sensitive phrase (not urgent), complaints and reschedule/cancel
requests escalate, Thu/Fri 14:00-18:00 and Sat 09:00-13:00 are open hours.
"""

from __future__ import annotations

import pytest

from lead_intel.inbound.extraction import process_extraction_output
from lead_intel.inbound.qualification import qualify_enquiry
from tests.inbound_support import build_enquiry, build_gold_extraction, example_config, load_fixture

# id: (stage, intent_level, urgency, escalation reasons, clarified fields)
EXPECTED = {
    "pricing-whitening": ("ready_for_next_step", "low", "none", [], []),
    "proposal-example-pricing-and-saturday": ("needs_information", "medium", "none", [], ["customer_name"]),
    "appointment-complete": ("ready_for_next_step", "high", "none", [], []),
    "appointment-missing-contact": ("needs_information", "high", "none", [], ["contact_method"]),
    "appointment-missing-everything": ("needs_information", "high", "none", [], ["customer_name", "contact_method", "service", "preferred_datetime"]),
    "ambiguous-next-saturday": ("needs_information", "high", "none", [], ["preferred_datetime"]),
    "ambiguous-todays-weekday": ("needs_information", "medium", "none", [], ["preferred_datetime"]),
    "ambiguous-am-pm": ("needs_information", "high", "none", [], ["preferred_datetime"]),
    "vague-next-week": ("needs_information", "medium", "none", [], ["preferred_datetime"]),
    "date-already-passed": ("needs_information", "high", "none", [], ["preferred_datetime"]),
    "time-already-passed-today": ("needs_information", "high", "none", [], ["preferred_datetime"]),
    "numeric-day-month": ("ready_for_next_step", "high", "none", [], []),
    "weekday-and-date-agree": ("ready_for_next_step", "high", "none", [], []),
    "weekday-and-date-disagree": ("needs_information", "high", "none", [], ["preferred_datetime"]),
    "urgent-pain-today": ("escalated", "high", "urgent", ["urgency_signal", "urgent_keyword"], ["customer_name", "service"]),
    "out-of-scope-medication": ("escalated", "low", "none", ["scope_concern"], ["contact_method"]),
    "unsupported-service": ("escalated", "low", "none", ["unsupported_service"], []),
    "human-request-callback": ("escalated", "low", "none", ["human_requested"], []),
    "human-request-receptionist": ("escalated", "low", "none", ["human_requested"], []),
    "reschedule-vague": ("escalated", "low", "none", ["escalated_intent"], []),
    "complaint-billing": ("escalated", "low", "none", ["escalated_intent"], []),
    "asap-no-date": ("needs_information", "high", "time_sensitive", [], ["preferred_datetime"]),
    "prompt-injection-attempt": ("needs_information", "low", "none", [], ["contact_method"]),
    "two-candidate-days": ("needs_information", "high", "none", [], ["preferred_datetime"]),
    "invalid-email": ("needs_information", "high", "none", [], ["contact_method"]),
    "time-without-date": ("needs_information", "medium", "none", [], ["preferred_datetime"]),
}

CASES = load_fixture()["cases"]


def test_every_fixture_case_has_an_expectation():
    assert set(EXPECTED) == {c["id"] for c in CASES}


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_case_qualification(case):
    config = example_config()
    extraction = process_extraction_output(build_enquiry(case), config, build_gold_extraction(case).model_dump_json())
    result = qualify_enquiry(extraction, config)
    stage, level, urgency, escalations, clarified = EXPECTED[case["id"]]

    assert result.stage.value == stage
    assert result.intent_level.value == level
    assert result.urgency.value == urgency
    assert [e.reason.value for e in result.escalations] == escalations
    assert [c.field.value for c in result.clarifications if c.field is not None] == clarified
