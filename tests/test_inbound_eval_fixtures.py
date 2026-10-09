"""Offline evaluation set: gold extractions through the deterministic pipeline."""

from __future__ import annotations

import datetime as dt
from collections import Counter

import pytest

from lead_intel.inbound.extraction import process_extraction_output
from lead_intel.inbound.models import ExtractionStatus
from tests.inbound_support import EXAMPLE_CONFIG_PATH, ROOT, build_enquiry, build_gold_extraction, example_config, load_fixture

CASES = load_fixture()["cases"]
REQUIRED_CATEGORIES = {
    "pricing", "appointment", "missing_contact", "ambiguous_date", "urgent", "out_of_scope", "human_request",
}  # fmt: skip


def test_fixture_size_and_coverage():
    assert len(CASES) >= 20
    assert len({c["id"] for c in CASES}) == len(CASES)
    counts = Counter(cat for c in CASES for cat in c["categories"])
    assert REQUIRED_CATEGORIES <= counts.keys(), REQUIRED_CATEGORIES - counts.keys()
    assert counts["ambiguous_date"] >= 5


def test_fixture_points_at_example_config():
    assert ROOT / load_fixture()["business_config"] == EXAMPLE_CONFIG_PATH


def test_gold_labels_for_flags():
    by_id = {c["id"]: build_gold_extraction(c) for c in CASES}
    assert by_id["urgent-pain-today"].urgency.value
    assert by_id["human-request-receptionist"].human_requested.value
    assert by_id["out-of-scope-medication"].scope_concern.value
    assert not by_id["prompt-injection-attempt"].urgency.value


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_case_through_pipeline(case):
    config = example_config()
    gold = build_gold_extraction(case)  # also asserts every gold value is verbatim in the message
    result = process_extraction_output(build_enquiry(case), config, gold.model_dump_json())
    expected = case["expected"]

    assert result.status is ExtractionStatus.EXTRACTED
    assert result.notes == [] or all("LLM also listed" in n for n in result.notes)
    assert result.matched_service_id == expected["matched_service_id"]
    assert [m.value for m in result.missing_information] == expected["missing_information"]
    assert bool(result.ambiguities) is expected["has_ambiguities"], result.ambiguities

    when = result.preferred_datetime
    assert when.status.value == expected["date_status"], when
    assert str(when.date) == expected.get("date", "None")
    assert (when.time.strftime("%H:%M") if when.time else None) == expected.get("time")
    assert (when.time_window.value if when.time_window else None) == expected.get("time_window")
    if "candidates" in expected:
        assert when.candidates == [dt.date.fromisoformat(d) for d in expected["candidates"]]
