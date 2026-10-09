from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from lead_intel.config import ConfigurationError
from lead_intel.inbound.business_config import BusinessConfig, Weekday, load_business_config
from lead_intel.inbound.models import Intent, MissingInfo
from tests.inbound_support import EXAMPLE_CONFIG_PATH, example_config

INBOUND_SRC = Path(__file__).resolve().parents[1] / "src" / "lead_intel" / "inbound"


@pytest.fixture
def raw_config() -> dict:
    return json.loads(EXAMPLE_CONFIG_PATH.read_text(encoding="utf-8"))


def test_example_config_is_valid():
    config = example_config()
    assert config.business_id == "example-dental-clinic"
    assert config.tz.key == "Asia/Kolkata"
    assert "teeth-whitening" in config.service_ids()
    assert config.opening_hours[Weekday.SATURDAY][0].close.hour == 13
    assert Weekday.SUNDAY not in config.opening_hours


def test_minimal_config_uses_safe_defaults():
    config = BusinessConfig(
        business_id="acme-plumbing", display_name="Acme Plumbing", timezone="Europe/London",
        services=[{"id": "boiler-repair", "name": "Boiler repair"}],
    )  # fmt: skip
    assert config.escalation_policy.escalate_on_human_request is True
    assert Intent.APPOINTMENT_REQUEST in config.qualification_rules.appointment_intents
    assert MissingInfo.CONTACT_METHOD in config.qualification_rules.required_for_all


def _mutated(raw: dict, fn) -> dict:
    data = copy.deepcopy(raw)
    fn(data)
    return data


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda d: d.update(timezone="Mars/Olympus"), "unknown IANA timezone"),
        (lambda d: d.update(timezone="+05:30"), "unknown IANA timezone"),
        (lambda d: d.update(business_id="Bad ID!"), "business_id"),
        (lambda d: d.update(services=[]), "services"),
        (lambda d: d.update(date_order="YMD"), "date_order"),
        (lambda d: d["services"].append(dict(d["services"][0])), "service ids must be unique"),
        (lambda d: d["services"][1]["aliases"].append("Whitening"), "used by more than one service"),
        (lambda d: d["opening_hours"]["monday"].append({"open": "12:00", "close": "15:00"}), "overlapping"),
        (lambda d: d["opening_hours"].update(friday=[{"open": "18:00", "close": "09:00"}]), "open must be before close"),
        (lambda d: d["opening_hours"].update(funday=[]), "opening_hours"),
        (lambda d: d["qualification_rules"]["required_for_all"].append("shoe_size"), "required_for_all"),
        (lambda d: d["escalation_policy"].update(escalate_on_human_request=False), "escalate_on_human_request"),
        (lambda d: d["escalation_policy"].update(escalate_on_scope_concern=False), "escalate_on_scope_concern"),
        (lambda d: d["escalation_policy"].update(escalate_on_extraction_failure=False), "escalate_on_extraction_failure"),
        (lambda d: d["escalation_policy"].update(min_intent_confidence="certain"), "min_intent_confidence"),
        (lambda d: d.update(price_list={"whitening": 100}), "price_list"),
    ],
)
def test_invalid_config_rejected(raw_config, mutation, message):
    with pytest.raises(ValidationError, match=message):
        BusinessConfig.model_validate(_mutated(raw_config, mutation))


def test_opening_hours_are_sorted(raw_config):
    data = _mutated(raw_config, lambda d: d["opening_hours"]["monday"].reverse())
    ranges = BusinessConfig.model_validate(data).opening_hours[Weekday.MONDAY]
    assert ranges[0].open < ranges[1].open


def test_loader_wraps_errors(tmp_path):
    with pytest.raises(ConfigurationError, match="Cannot read"):
        load_business_config(tmp_path / "missing.json")
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Cannot read"):
        load_business_config(bad_json)
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"business_id": "x"}), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Invalid business config"):
        load_business_config(invalid)


@pytest.mark.parametrize("word", ["dental", "dentist", "teeth", "tooth", "clinic", "whitening", "patient"])
def test_inbound_code_is_vertical_neutral(word):
    for path in INBOUND_SRC.glob("*.py"):
        assert word not in path.read_text(encoding="utf-8").casefold(), f"{word!r} hard-coded in {path.name}"
