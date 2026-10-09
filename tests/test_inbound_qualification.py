"""Deterministic inbound qualification (Phase 2). No LLM, network or persistence."""

from __future__ import annotations

import copy
import json
import subprocess
import sys

import pytest
from pydantic import ValidationError

from lead_intel.inbound.business_config import BusinessConfig
from lead_intel.inbound.extraction import process_extraction_output
from lead_intel.inbound.models import EnquiryExtractionResult, ExtractionStatus, MissingInfo
from lead_intel.inbound.qualification import (
    QUALIFICATION_RULES_VERSION,
    ClarificationReason,
    EscalationReason,
    EscalationTrigger,
    InfoSource,
    InfoStatus,
    IntentLevel,
    QualificationResult,
    QualificationStage,
    UrgencyLevel,
    qualify_enquiry,
)
from lead_intel.models import Classification
from lead_intel.scoring.engine import score_report
from tests.inbound_support import EXAMPLE_CONFIG_PATH, build_enquiry, build_gold_extraction, case_by_id

E, N, R = QualificationStage.ESCALATED, QualificationStage.NEEDS_INFORMATION, QualificationStage.READY_FOR_NEXT_STEP

# A complete, in-hours appointment request with every required field confirmed.
COMPLETE = "appointment-complete"


def raw_config() -> dict:
    return json.loads(EXAMPLE_CONFIG_PATH.read_text(encoding="utf-8"))


def config_with(**sections) -> BusinessConfig:
    """Example config with policy sections merged and other keys replaced, validated like a real file."""
    data = raw_config()
    for key, value in sections.items():
        if key in ("qualification_rules", "escalation_policy"):
            data[key] = {**data[key], **value}
        else:
            data[key] = value
    return BusinessConfig.model_validate(data)


def extracted(case_id: str, *, config=None, message=None, channel_contact="keep", **changes) -> EnquiryExtractionResult:
    """Run a fixture case (optionally altered) through the real Phase 1 pipeline."""
    case = copy.deepcopy(case_by_id(case_id))
    if channel_contact != "keep":
        case["channel_contact"] = channel_contact
    data = build_gold_extraction(case).model_dump(mode="json")
    for path, value in changes.items():
        target = data
        *parents, leaf = path.split("__")
        for key in parents:
            target = target[key]
        target[leaf] = value
    enquiry = build_enquiry(case, message=message)
    return process_extraction_output(enquiry, config or config_with(), json.dumps(data))


def qualify(case_id: str, *, config=None, **kwargs) -> QualificationResult:
    config = config or config_with()
    return qualify_enquiry(extracted(case_id, config=config, **kwargs), config)


def reasons(result: QualificationResult) -> list[EscalationReason]:
    return [e.reason for e in result.escalations]


def field(result: QualificationResult, name: MissingInfo):
    return next(f for f in result.fields if f.field is name)


# --- normal enquiries ------------------------------------------------------------------


def test_complete_appointment_is_ready_high_intent():
    result = qualify(COMPLETE)
    assert result.stage is R and result.intent_level is IntentLevel.HIGH
    assert result.escalations == [] and result.clarifications == []
    assert all(f.status is InfoStatus.CONFIRMED for f in result.fields if f.required)
    assert result.rules_version == QUALIFICATION_RULES_VERSION
    assert not result.requires_human


def test_pricing_question_with_channel_contact_is_ready_low_intent():
    result = qualify("pricing-whitening")
    assert (result.stage, result.intent_level) == (R, IntentLevel.LOW)
    assert field(result, MissingInfo.CONTACT_METHOD).source is InfoSource.CHANNEL
    # Name/service/date are not required for a pricing question.
    assert not field(result, MissingInfo.CUSTOMER_NAME).required


# --- missing / unconfirmed information ------------------------------------------------


def test_missing_required_fields_need_information():
    result = qualify("appointment-missing-everything")
    assert result.stage is N
    missing = [c.field for c in result.clarifications if c.reason is ClarificationReason.MISSING]
    assert missing == list(MissingInfo)


def test_inferred_name_is_not_treated_as_confirmed():
    result = qualify(COMPLETE, contact__name__confidence="inferred")
    assert result.stage is N
    assert field(result, MissingInfo.CUSTOMER_NAME).status is InfoStatus.INFERRED
    assert [c.reason for c in result.clarifications] == [ClarificationReason.UNCONFIRMED]


def test_inferred_service_is_not_treated_as_confirmed():
    # Phase 1 counts any matched service id as present; qualification requires confirmed wording.
    result = qualify(COMPLETE, service_requested__confidence="inferred")
    assert field(result, MissingInfo.SERVICE).status is InfoStatus.INFERRED
    assert result.stage is N


def test_inferred_contact_is_not_treated_as_confirmed():
    result = qualify(COMPLETE, contact__email__confidence="inferred")
    assert field(result, MissingInfo.CONTACT_METHOD).status is InfoStatus.INFERRED
    assert result.stage is N


def test_malformed_contact_is_ambiguous_not_missing():
    result = qualify("invalid-email")
    contact = field(result, MissingInfo.CONTACT_METHOD)
    assert contact.status is InfoStatus.AMBIGUOUS and "does not look valid" in contact.detail
    assert result.stage is N




# --- contradictory information ------------------------------------------------------


def test_matched_service_without_service_wording_is_contradictory():
    result = qualify(COMPLETE, service_requested__value=None, service_requested__confidence="insufficient",
                     service_requested__evidence=None)  # fmt: skip
    service = field(result, MissingInfo.SERVICE)
    assert service.status is InfoStatus.AMBIGUOUS and "no service wording" in service.detail
    assert result.stage is N


def test_keyword_overrides_llm_saying_not_urgent():
    message = "I'd like to book a dental check-up tomorrow at 11am, there is some bleeding. my name is Priya Sharma, priya.sharma@example.com"
    result = qualify(COMPLETE, message=message)
    assert result.matched_urgent_keywords == ["bleeding"]
    assert reasons(result) == [EscalationReason.URGENT_KEYWORD]


def test_weekday_contradicting_date_needs_information():
    result = qualify("weekday-and-date-disagree")
    assert result.stage is N
    assert field(result, MissingInfo.PREFERRED_DATETIME).status is InfoStatus.AMBIGUOUS


def test_never_more_lenient_than_extraction_result():
    extraction = extracted(COMPLETE)
    tampered = extraction.model_copy(update={"missing_information": [MissingInfo.CUSTOMER_NAME]})
    result = qualify_enquiry(tampered, config_with())
    assert field(result, MissingInfo.CUSTOMER_NAME).status is InfoStatus.AMBIGUOUS
    assert result.stage is N


# --- escalation (never overridden by intent or completeness) ---------------------------


@pytest.mark.parametrize(
    "changes, reason",
    [
        ({"human_requested__value": True, "human_requested__confidence": "confirmed", "human_requested__evidence": "reach me"}, EscalationReason.HUMAN_REQUESTED),
        ({"human_requested__value": True, "human_requested__confidence": "inferred"}, EscalationReason.HUMAN_REQUESTED),
        ({"scope_concern__value": True, "scope_concern__confidence": "inferred"}, EscalationReason.SCOPE_CONCERN),
        ({"urgency__value": True, "urgency__confidence": "inferred"}, EscalationReason.URGENCY_SIGNAL),
        ({"primary_intent__confidence": "insufficient", "primary_intent__evidence": None}, EscalationReason.LOW_INTENT_CONFIDENCE),
    ],
)
def test_safety_triggers_escalate_complete_high_intent_enquiries(changes, reason):
    result = qualify(COMPLETE, **changes)
    assert result.stage is E and reasons(result) == [reason]
    assert result.requires_human
    assert result.intent_level is not None  # still reported for staff context


def test_escalation_keeps_clarifications_for_staff():
    result = qualify("out-of-scope-medication")
    assert result.stage is E and reasons(result) == [EscalationReason.SCOPE_CONCERN]
    assert [c.field for c in result.clarifications] == [MissingInfo.CONTACT_METHOD]


def test_multiple_escalations_reported_in_fixed_order():
    result = qualify("urgent-pain-today", human_requested__value=True, human_requested__confidence="inferred")
    assert reasons(result) == [
        EscalationReason.HUMAN_REQUESTED, EscalationReason.URGENCY_SIGNAL, EscalationReason.URGENT_KEYWORD,
    ]  # fmt: skip


def test_unsupported_service_escalates():
    result = qualify("unsupported-service")
    assert reasons(result) == [EscalationReason.UNSUPPORTED_SERVICE]


def test_non_bookable_service_escalates_only_for_appointments():
    services = raw_config()["services"]
    services[2]["bookable"] = False  # check-up
    config = config_with(services=services)
    assert reasons(qualify(COMPLETE, config=config)) == [EscalationReason.NON_BOOKABLE_SERVICE]
    pricing = qualify(COMPLETE, config=config, primary_intent__intent="pricing_enquiry")
    assert EscalationReason.NON_BOOKABLE_SERVICE not in reasons(pricing)


def test_failed_extraction_escalates_with_no_qualification():
    enquiry = build_enquiry(case_by_id(COMPLETE))
    failed = EnquiryExtractionResult(enquiry=enquiry, status=ExtractionStatus.FAILED, errors=["no JSON object"])
    result = qualify_enquiry(failed, config_with())
    assert result.stage is E and reasons(result) == [EscalationReason.EXTRACTION_FAILED]
    assert result.intent_level is None and result.urgency is None and result.fields == []
    assert "no JSON object" in result.escalations[0].detail


# --- configurable rules and boundaries ----------------------------------------------------


@pytest.mark.parametrize(
    "minimum, intent_confidence, escalates",
    [
        ("insufficient", "insufficient", False),
        ("inferred", "inferred", False),
        ("inferred", "insufficient", True),
        ("confirmed", "confirmed", False),
        ("confirmed", "inferred", True),
    ],
)
def test_min_intent_confidence_boundary(minimum, intent_confidence, escalates):
    config = config_with(escalation_policy={"min_intent_confidence": minimum})
    changes = {"primary_intent__confidence": intent_confidence}
    if intent_confidence != "confirmed":
        changes["primary_intent__evidence"] = None
    result = qualify(COMPLETE, config=config, **changes)
    assert (EscalationReason.LOW_INTENT_CONFIDENCE in reasons(result)) is escalates


def test_urgency_escalation_can_be_disabled_but_keywords_still_reported():
    config = config_with(escalation_policy={"escalate_on_urgency": False})
    result = qualify("urgent-pain-today", config=config)
    assert result.escalations == []
    assert result.urgency is UrgencyLevel.URGENT  # classified, just not escalated
    assert result.matched_urgent_keywords == ["swelling"]
    assert result.stage is N  # customer name and service still missing


@pytest.mark.parametrize(
    "message_time, inside",
    [("9am", True), ("12:59", True), ("1pm", False), ("13:30", False), ("2pm", True), ("5:59pm", True), ("6pm", False)],
)
def test_opening_hours_boundaries(message_time, inside):
    # 2026-10-08 is a Thursday: 09:00-13:00 and 14:00-18:00 (close is exclusive).
    message = f"Hello, my name is Priya Sharma. I'd like to book a dental check-up tomorrow at {message_time}. priya.sharma@example.com"
    result = qualify(COMPLETE, message=message, preferred_datetime_text__value=f"tomorrow at {message_time}",
                     preferred_datetime_text__evidence=f"tomorrow at {message_time}")  # fmt: skip
    outside = [c for c in result.clarifications if c.reason is ClarificationReason.OUTSIDE_OPENING_HOURS]
    assert (not outside) is inside, result.clarifications
    assert result.stage is (R if inside else N)


def test_closed_day_needs_information():
    message = "Hello, my name is Priya Sharma. Can I book a dental check-up on Sunday? priya.sharma@example.com"
    result = qualify(COMPLETE, message=message, preferred_datetime_text__value="Sunday",
                     preferred_datetime_text__evidence="Sunday")  # fmt: skip
    assert result.stage is N
    assert "closed on sunday" in result.clarifications[0].detail


def test_opening_hours_not_checked_when_not_configured():
    message = "Hello, my name is Priya Sharma. Can I book a dental check-up on Sunday? priya.sharma@example.com"
    result = qualify(COMPLETE, config=config_with(opening_hours={}), message=message,
                     preferred_datetime_text__value="Sunday", preferred_datetime_text__evidence="Sunday")  # fmt: skip
    assert result.stage is R


def test_empty_requirements_ready_without_contact():
    config = config_with(qualification_rules={"required_for_all": [], "required_for_appointment": []})
    result = qualify("prompt-injection-attempt", config=config)
    assert result.stage is R


def test_appointment_intents_drive_required_fields():
    config = config_with(qualification_rules={"appointment_intents": []})
    result = qualify("appointment-missing-everything", config=config)
    assert [c.field for c in result.clarifications] == [MissingInfo.CONTACT_METHOD]


@pytest.mark.parametrize(
    "primary, confidence, secondary, level",
    [
        ("appointment_request", "confirmed", [], IntentLevel.HIGH),
        ("appointment_request", "inferred", [], IntentLevel.MEDIUM),
        ("availability_enquiry", "confirmed", [], IntentLevel.MEDIUM),
        ("pricing_enquiry", "confirmed", ["appointment_request"], IntentLevel.MEDIUM),
        ("pricing_enquiry", "confirmed", [], IntentLevel.LOW),
        ("appointment_request", "insufficient", [], IntentLevel.LOW),
    ],
)
def test_intent_level_rules(primary, confidence, secondary, level):
    changes = {"primary_intent__intent": primary, "primary_intent__confidence": confidence, "secondary_intents": secondary}
    if confidence != "confirmed":
        changes["primary_intent__evidence"] = None
    config = config_with(escalation_policy={"min_intent_confidence": "insufficient"})
    assert qualify(COMPLETE, config=config, **changes).intent_level is level


# --- invalid configuration and inputs --------------------------------------------------------


@pytest.mark.parametrize(
    "sections",
    [
        {"escalation_policy": {"urgent_keywords": ["pain", "  "]}},
        {"escalation_policy": {"out_of_scope_topics": [""]}},
        {"escalation_policy": {"escalate_on_human_request": False}},
        {"qualification_rules": {"high_intent_intents": ["vip"]}},
        {"qualification_rules": {"required_for_all": ["budget"]}},
    ],
)
def test_invalid_configuration_rejected(sections):
    with pytest.raises(ValidationError):
        config_with(**sections)


def test_keywords_are_trimmed():
    assert config_with(escalation_policy={"urgent_keywords": ["  bleeding "]}).escalation_policy.urgent_keywords == ["bleeding"]


def test_business_mismatch_rejected():
    other = config_with(business_id="another-business")
    with pytest.raises(ValueError, match="another-business"):
        qualify_enquiry(extracted(COMPLETE), other)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stage": R, "intent_level": IntentLevel.HIGH,
         "urgency": UrgencyLevel.NONE,
         "escalations": [EscalationTrigger(reason=EscalationReason.HUMAN_REQUESTED, detail="x")]},
        {"stage": E, "intent_level": IntentLevel.HIGH, "urgency": UrgencyLevel.NONE},
        {"stage": N, "intent_level": IntentLevel.LOW, "urgency": UrgencyLevel.NONE},
        {"stage": R, "intent_level": None, "urgency": None},
        {"stage": R, "intent_level": IntentLevel.LOW, "urgency": None},
    ],
)  # fmt: skip
def test_result_model_rejects_inconsistent_states(kwargs):
    with pytest.raises(ValidationError):
        QualificationResult(business_id="b", external_message_id="m", **kwargs)


# --- determinism and isolation --------------------------------------------------------------


def test_repeatable_for_same_input():
    config = config_with()
    result = extracted("urgent-pain-today", config=config)
    first = qualify_enquiry(result, config)
    assert all(qualify_enquiry(result, config) == first for _ in range(10))
    assert qualify_enquiry(result.model_copy(deep=True), config.model_copy(deep=True)) == first


def test_does_not_mutate_inputs():
    config = config_with()
    result = extracted("invalid-email", config=config)
    before = (result.model_dump_json(), config.model_dump_json())
    qualify_enquiry(result, config)
    assert (result.model_dump_json(), config.model_dump_json()) == before


def test_qualification_does_not_load_outbound_scoring_llm_or_network_modules():
    code = (
        "import sys, lead_intel.inbound.qualification;"
        "print(sorted(m for m in sys.modules if m.startswith(('lead_intel.scoring', 'lead_intel.search', 'tavily', 'lead_intel.crew'))))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_no_outbound_classification_labels_in_inbound_results():
    outbound = {c.value for c in Classification}  # HOT/WARM/COOL/LOW
    inbound = {s.value for s in QualificationStage} | {i.value for i in IntentLevel}
    assert not outbound & inbound
    assert "Classification" not in str(QualificationResult.model_json_schema())


def test_outbound_scoring_unaffected(perfect_report):
    score = score_report(perfect_report)
    assert (score.total, score.classification) == (100, Classification.HOT)


# === Review adjustments (regression tests) =====================================================

PRIYA = "Hello, my name is Priya Sharma. {body} priya.sharma@example.com"


# --- 1. intent-based escalation -----------------------------------------------------------------


@pytest.mark.parametrize(
    "intent, body, evidence",
    [
        ("complaint", "I was charged twice for my dental check-up tomorrow at 11am.", "I was charged twice"),
        ("reschedule_or_cancel", "Please cancel my dental check-up tomorrow at 11am.", "Please cancel my dental check-up"),
        ("reschedule_or_cancel", "I need to reschedule my dental check-up tomorrow at 11am.", "I need to reschedule"),
    ],
    ids=["billing-complaint", "cancellation", "reschedule"],
)
def test_configured_intents_escalate(intent, body, evidence):
    kwargs = {"message": PRIYA.format(body=body), "primary_intent__intent": intent, "primary_intent__evidence": evidence}
    result = qualify(COMPLETE, **kwargs)
    assert result.stage is E and reasons(result) == [EscalationReason.ESCALATED_INTENT]
    assert intent in result.escalations[0].detail
    # Same enquiry with intent escalation switched off is otherwise complete: escalation is what blocks it.
    assert qualify(COMPLETE, config=config_with(escalation_policy={"escalate_intents": []}), **kwargs).stage is R


def test_secondary_intent_escalates_and_keeps_intent_level():
    result = qualify(COMPLETE, secondary_intents=["complaint"])
    assert reasons(result) == [EscalationReason.ESCALATED_INTENT]
    assert result.intent_level is IntentLevel.HIGH


def test_escalate_intents_default_is_safe():
    minimal = BusinessConfig(
        business_id="acme-plumbing", display_name="Acme", timezone="UTC", services=[{"id": "repair", "name": "Repair"}]
    )
    assert set(minimal.escalation_policy.escalate_intents) == {"complaint", "reschedule_or_cancel"}


@pytest.mark.parametrize(
    "changes, reason",
    [
        ({"human_requested__value": True, "human_requested__confidence": "inferred"}, EscalationReason.HUMAN_REQUESTED),
        ({"scope_concern__value": True, "scope_concern__confidence": "inferred"}, EscalationReason.SCOPE_CONCERN),
        ({"matched_service_id": None}, EscalationReason.UNSUPPORTED_SERVICE),
    ],
)
def test_empty_escalate_intents_cannot_disable_fixed_rules(changes, reason):
    config = config_with(escalation_policy={"escalate_intents": []})
    assert reasons(qualify(COMPLETE, config=config, **changes)) == [reason]


def test_empty_escalate_intents_cannot_disable_failed_extraction():
    config = config_with(escalation_policy={"escalate_intents": [], "escalate_on_urgency": False})
    failed = EnquiryExtractionResult(enquiry=build_enquiry(case_by_id(COMPLETE)), status=ExtractionStatus.FAILED, errors=["x"])
    assert reasons(qualify_enquiry(failed, config)) == [EscalationReason.EXTRACTION_FAILED]


def test_unknown_escalate_intent_rejected():
    with pytest.raises(ValidationError):
        config_with(escalation_policy={"escalate_intents": ["refund_now"]})


# --- 2. outcome terminology ------------------------------------------------------------------------


def test_outcome_values_do_not_imply_sales_qualification():
    assert {s.value for s in QualificationStage} == {"ready_for_next_step", "needs_information", "escalated"}


# --- 3. urgency classification vs escalation --------------------------------------------------------


def _urgency(evidence, confidence="confirmed", message=None):
    case_message = case_by_id(COMPLETE)["message"]
    return {
        "message": message or f"{case_message} {evidence}",
        "urgency__value": True, "urgency__confidence": confidence, "urgency__evidence": evidence,
    }  # fmt: skip


@pytest.mark.parametrize("evidence", ["asap", "ASAP!", "as soon as possible", "at the earliest"])
def test_scheduling_haste_is_time_sensitive_not_escalated(evidence):
    result = qualify(COMPLETE, **_urgency(evidence))
    assert result.urgency is UrgencyLevel.TIME_SENSITIVE
    assert result.escalations == [] and result.stage is R


@pytest.mark.parametrize("evidence", ["asap please, it really hurts", "my tooth is broken", "urgent"])
def test_urgency_beyond_scheduling_still_escalates(evidence):
    result = qualify(COMPLETE, **_urgency(evidence))
    assert result.urgency is UrgencyLevel.URGENT
    assert reasons(result) == [EscalationReason.URGENCY_SIGNAL]


def test_inferred_urgency_without_evidence_is_urgent():
    result = qualify(COMPLETE, urgency__value=True, urgency__confidence="inferred", urgency__evidence=None)
    assert result.urgency is UrgencyLevel.URGENT and reasons(result) == [EscalationReason.URGENCY_SIGNAL]


def test_configured_keyword_wins_over_scheduling_phrase():
    message = case_by_id(COMPLETE)["message"] + " asap, there is bleeding"
    result = qualify(COMPLETE, **_urgency("asap", message=message))
    assert result.urgency is UrgencyLevel.URGENT
    assert reasons(result) == [EscalationReason.URGENT_KEYWORD]


def test_without_configured_phrases_scheduling_urgency_stays_urgent():
    config = config_with(escalation_policy={"time_sensitive_phrases": []})
    result = qualify(COMPLETE, config=config, **_urgency("asap"))
    assert result.urgency is UrgencyLevel.URGENT and reasons(result) == [EscalationReason.URGENCY_SIGNAL]


def test_time_sensitive_escalation_is_opt_in():
    config = config_with(escalation_policy={"escalate_on_time_sensitive": True})
    result = qualify(COMPLETE, config=config, **_urgency("asap"))
    assert result.urgency is UrgencyLevel.TIME_SENSITIVE and reasons(result) == [EscalationReason.TIME_SENSITIVE]


def test_no_urgency_signal_is_none():
    assert qualify(COMPLETE).urgency is UrgencyLevel.NONE


@pytest.mark.parametrize(
    "policy",
    [{"time_sensitive_phrases": ["asap", " "]}, {"time_sensitive_phrases": ["Bleeding"]}],
    ids=["blank-phrase", "phrase-also-urgent-keyword"],
)
def test_invalid_urgency_configuration_rejected(policy):
    with pytest.raises(ValidationError):
        config_with(escalation_policy=policy)


# --- 4. identity confidence ---------------------------------------------------------------------------


def test_channel_profile_name_is_not_verified_identity():
    channel = {"phone": "+91 98765 43210", "name": "Asha"}
    result = qualify("proposal-example-pricing-and-saturday", channel_contact=channel)
    name = field(result, MissingInfo.CUSTOMER_NAME)
    assert (name.status, name.source) == (InfoStatus.INFERRED, InfoSource.CHANNEL)
    assert result.stage is N
    assert [(c.reason, c.field) for c in result.clarifications] == [(ClarificationReason.UNCONFIRMED, MissingInfo.CUSTOMER_NAME)]


def test_customer_confirmed_name_outranks_channel_profile_name():
    result = qualify(COMPLETE, channel_contact={"name": "PS", "phone": "+91 98765 43210"})
    name = field(result, MissingInfo.CUSTOMER_NAME)
    assert (name.status, name.source) == (InfoStatus.CONFIRMED, InfoSource.MESSAGE)


def test_channel_contact_identifier_is_usable_but_not_identity():
    result = qualify("pricing-whitening")
    contact = field(result, MissingInfo.CONTACT_METHOD)
    assert (contact.status, contact.source) == (InfoStatus.CONFIRMED, InfoSource.CHANNEL)
    assert "identifier" in contact.detail
    assert field(result, MissingInfo.CUSTOMER_NAME).status is InfoStatus.MISSING


# --- 5. service scope -------------------------------------------------------------------------------------


def test_unsupported_service_appointment_never_ready():
    result = qualify("unsupported-service", primary_intent__intent="appointment_request")
    assert result.stage is E and EscalationReason.UNSUPPORTED_SERVICE in reasons(result)
    assert field(result, MissingInfo.SERVICE).status is InfoStatus.AMBIGUOUS


def test_inferred_unsupported_service_still_escalates():
    # Wording not in the message -> downgraded to inferred by Phase 1; still not something we can promise.
    result = qualify(COMPLETE, service_requested__value="laser hair removal",
                     service_requested__evidence="laser hair removal", matched_service_id=None)  # fmt: skip
    assert reasons(result) == [EscalationReason.UNSUPPORTED_SERVICE]


def test_hallucinated_service_id_escalates():
    result = qualify(COMPLETE, matched_service_id="teeth-straightening-deluxe")
    assert reasons(result) == [EscalationReason.UNSUPPORTED_SERVICE] and result.stage is E


def test_result_carries_no_reply_price_or_booking_fields():
    assert set(QualificationResult.model_fields) == {
        "business_id", "external_message_id", "rules_version", "stage", "intent_level", "urgency",
        "escalations", "clarifications", "fields", "matched_urgent_keywords",
    }  # fmt: skip
