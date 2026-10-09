"""Deterministic inbound qualification (Phase 2).

Input: a Phase 1 ``EnquiryExtractionResult`` and the business's ``BusinessConfig``.
Output: a validated ``QualificationResult``. Pure Python: no LLM, clock, network,
persistence, messaging or other actions. Unrelated to the outbound prospect
scoring engine; there is no numeric score.

Semantics (QUALIFICATION_RULES_VERSION):
- Safety triggers fire on any positive signal, confirmed OR inferred. Credit
  toward qualification requires CONFIRMED information. Uncertainty therefore
  moves an enquiry toward a human or a clarifying question, never toward automation.
- Stage precedence: ESCALATED > NEEDS_INFORMATION > READY_FOR_NEXT_STEP.
  ``intent_level`` and ``urgency`` are reported separately; only the escalation
  rules turn them into a stage.
- READY_FOR_NEXT_STEP means the information needed for the next operational step
  is complete and confirmed. It is NOT a sales qualification.
- Urgency is classified (NONE / TIME_SENSITIVE / URGENT) separately from the
  decision to escalate. No medical triage is performed: URGENT comes only from
  the business's configured keywords or an extracted urgency signal that is not
  merely scheduling haste.
- A channel profile name is never treated as a customer-confirmed name; a
  channel-supplied phone/email is a usable contact identifier, not an identity.
"""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lead_intel.inbound.business_config import BusinessConfig, Weekday
from lead_intel.inbound.extraction import is_valid_email, is_valid_phone
from lead_intel.inbound.models import (
    DateTimeStatus,
    EnquiryExtraction,
    EnquiryExtractionResult,
    ExtractedText,
    ExtractionStatus,
    MissingInfo,
    NormalizedDateTime,
)
from lead_intel.models.evidence import Confidence

QUALIFICATION_RULES_VERSION = "1.0.0"

_CONFIDENCE_RANK = {Confidence.INSUFFICIENT: 0, Confidence.INFERRED: 1, Confidence.CONFIRMED: 2}
_WEEKDAYS = list(Weekday)  # index == date.weekday()


class QualificationStage(str, Enum):
    READY_FOR_NEXT_STEP = "ready_for_next_step"  # required info confirmed; nothing blocks the next step
    NEEDS_INFORMATION = "needs_information"  # ask the customer to supply/clarify
    ESCALATED = "escalated"  # a human must handle it


class UrgencyLevel(str, Enum):
    NONE = "none"
    TIME_SENSITIVE = "time_sensitive"  # wants it soon ("asap"); scheduling, not an emergency
    URGENT = "urgent"  # configured urgent keyword, or an urgency signal beyond scheduling haste


class IntentLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class InfoStatus(str, Enum):
    CONFIRMED = "confirmed"
    INFERRED = "inferred"
    AMBIGUOUS = "ambiguous"
    MISSING = "missing"


class InfoSource(str, Enum):
    MESSAGE = "message"
    CHANNEL = "channel"


class EscalationReason(str, Enum):
    EXTRACTION_FAILED = "extraction_failed"
    HUMAN_REQUESTED = "human_requested"
    SCOPE_CONCERN = "scope_concern"
    URGENCY_SIGNAL = "urgency_signal"
    URGENT_KEYWORD = "urgent_keyword"
    LOW_INTENT_CONFIDENCE = "low_intent_confidence"
    UNSUPPORTED_SERVICE = "unsupported_service"
    ESCALATED_INTENT = "escalated_intent"
    TIME_SENSITIVE = "time_sensitive"
    NON_BOOKABLE_SERVICE = "non_bookable_service"


class ClarificationReason(str, Enum):
    MISSING = "missing"
    UNCONFIRMED = "unconfirmed"
    AMBIGUOUS = "ambiguous"
    EXTRACTION_AMBIGUITY = "extraction_ambiguity"
    OUTSIDE_OPENING_HOURS = "outside_opening_hours"


class FieldAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: MissingInfo
    required: bool
    status: InfoStatus
    source: InfoSource | None = None
    detail: str | None = None


class EscalationTrigger(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: EscalationReason
    detail: str


class Clarification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: ClarificationReason
    field: MissingInfo | None = None
    detail: str


class QualificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    business_id: str
    external_message_id: str
    rules_version: str = QUALIFICATION_RULES_VERSION
    stage: QualificationStage
    intent_level: IntentLevel | None
    urgency: UrgencyLevel | None
    escalations: list[EscalationTrigger] = Field(default_factory=list)
    clarifications: list[Clarification] = Field(default_factory=list)
    fields: list[FieldAssessment] = Field(default_factory=list)
    matched_urgent_keywords: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> QualificationResult:
        escalated = self.stage is QualificationStage.ESCALATED
        if escalated != bool(self.escalations):
            raise ValueError("stage is 'escalated' exactly when there are escalation triggers")
        if self.stage is QualificationStage.NEEDS_INFORMATION and not self.clarifications:
            raise ValueError("'needs_information' requires at least one clarification")
        if self.stage is QualificationStage.READY_FOR_NEXT_STEP and (
            self.clarifications or any(f.required and f.status is not InfoStatus.CONFIRMED for f in self.fields)
        ):
            raise ValueError("'ready_for_next_step' requires every required field confirmed and nothing to clarify")
        failed = any(e.reason is EscalationReason.EXTRACTION_FAILED for e in self.escalations)
        if failed and (self.intent_level is not None or self.urgency is not None or self.fields):
            raise ValueError("a failed extraction carries no intent level, urgency or field assessments")
        if not failed and (self.intent_level is None or self.urgency is None):
            raise ValueError("intent_level and urgency are required unless extraction failed")
        return self

    @property
    def requires_human(self) -> bool:
        return self.stage is QualificationStage.ESCALATED


# --- field assessment ------------------------------------------------------------------


def _text_status(item: ExtractedText) -> InfoStatus:
    return {
        Confidence.CONFIRMED: InfoStatus.CONFIRMED,
        Confidence.INFERRED: InfoStatus.INFERRED,
        Confidence.INSUFFICIENT: InfoStatus.MISSING,
    }[item.confidence]


def _assess_name(result: EnquiryExtractionResult, ext: EnquiryExtraction) -> tuple[InfoStatus, InfoSource | None, str | None]:
    status = _text_status(ext.contact.name)
    if status is InfoStatus.CONFIRMED:
        return status, InfoSource.MESSAGE, None
    channel = result.enquiry.channel_contact
    if channel and channel.name:  # a profile name may be a nickname: never verified identity
        return InfoStatus.INFERRED, InfoSource.CHANNEL, "channel profile name, not confirmed by the customer"
    return status, (None if status is InfoStatus.MISSING else InfoSource.MESSAGE), None


def _assess_contact(result: EnquiryExtractionResult, ext: EnquiryExtraction) -> tuple[InfoStatus, InfoSource | None, str | None]:
    channel = result.enquiry.channel_contact
    if channel and (channel.phone or channel.email):
        # Usable reply address supplied by the channel; says nothing about identity.
        return InfoStatus.CONFIRMED, InfoSource.CHANNEL, "contact identifier supplied by the channel"
    statuses: list[tuple[InfoStatus, str | None]] = []
    for label, item, valid in (("phone", ext.contact.phone, is_valid_phone), ("email", ext.contact.email, is_valid_email)):
        status = _text_status(item)
        if status is InfoStatus.CONFIRMED and not valid(item.value):
            statuses.append((InfoStatus.AMBIGUOUS, f"{label} {item.value!r} does not look valid"))
        else:
            statuses.append((status, None))
    # Best evidence wins; a malformed explicit value outranks an inferred one.
    priority = [InfoStatus.CONFIRMED, InfoStatus.AMBIGUOUS, InfoStatus.INFERRED, InfoStatus.MISSING]
    status, detail = min(statuses, key=lambda s: priority.index(s[0]))
    return status, (None if status is InfoStatus.MISSING else InfoSource.MESSAGE), detail


def _assess_service(result: EnquiryExtractionResult, ext: EnquiryExtraction) -> tuple[InfoStatus, InfoSource | None, str | None]:
    text, matched = ext.service_requested, result.matched_service_id
    if matched is None:
        if text.value is None:
            return InfoStatus.MISSING, None, None
        return InfoStatus.AMBIGUOUS, InfoSource.MESSAGE, f"{text.value!r} is not a configured service"
    if text.confidence is Confidence.INSUFFICIENT:
        return InfoStatus.AMBIGUOUS, InfoSource.MESSAGE, f"service {matched!r} matched but no service wording was extracted"
    return _text_status(text), InfoSource.MESSAGE, None


def _assess_datetime(result: EnquiryExtractionResult) -> tuple[InfoStatus, InfoSource | None, str | None]:
    when = result.preferred_datetime
    if when is None or when.status is DateTimeStatus.NOT_PROVIDED:
        return InfoStatus.MISSING, None, None
    if when.status is DateTimeStatus.RESOLVED:  # only confirmed wording is ever resolved
        return InfoStatus.CONFIRMED, InfoSource.MESSAGE, None
    return InfoStatus.AMBIGUOUS, InfoSource.MESSAGE, when.reason


def _assess_fields(result: EnquiryExtractionResult, ext: EnquiryExtraction, required: set[MissingInfo]) -> list[FieldAssessment]:
    assessors = {
        MissingInfo.CUSTOMER_NAME: lambda: _assess_name(result, ext),
        MissingInfo.CONTACT_METHOD: lambda: _assess_contact(result, ext),
        MissingInfo.SERVICE: lambda: _assess_service(result, ext),
        MissingInfo.PREFERRED_DATETIME: lambda: _assess_datetime(result),
    }
    fields = []
    for field in MissingInfo:
        status, source, detail = assessors[field]()
        if status is InfoStatus.CONFIRMED and field in result.missing_information:
            # Never be more lenient than the extraction stage's own rules.
            status, detail = InfoStatus.AMBIGUOUS, "extraction result lists this as missing"
        fields.append(FieldAssessment(field=field, required=field in required, status=status, source=source, detail=detail))
    return fields


# --- rules ------------------------------------------------------------------------------


def _matched_keywords(message: str, keywords: list[str]) -> list[str]:
    text = " ".join(message.casefold().split())
    matched = []
    for keyword in keywords:
        phrase = re.escape(" ".join(keyword.casefold().split()))
        if re.search(rf"(?<!\w){phrase}(?!\w)", text) and keyword not in matched:
            matched.append(keyword)
    return matched


def _opening_hours_problem(config: BusinessConfig, when: NormalizedDateTime) -> str | None:
    if not config.opening_hours or when.date is None:
        return None  # no hours configured: nothing to check against
    day = _WEEKDAYS[when.date.weekday()]
    ranges = config.opening_hours.get(day, [])
    if not ranges:
        return f"the business is closed on {day.value}"
    if when.time is not None and not any(r.open <= when.time < r.close for r in ranges):
        return f"{when.time:%H:%M} on {day.value} is outside opening hours"
    return None


def _normalise_phrase(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())


def _is_scheduling_only(evidence: str | None, phrases: list[str]) -> bool:
    """True when the urgency evidence is exactly one configured scheduling phrase."""
    return bool(evidence) and _normalise_phrase(evidence) in {_normalise_phrase(p) for p in phrases}


def _classify_urgency(ext: EnquiryExtraction, config: BusinessConfig, keywords: list[str]) -> UrgencyLevel:
    if keywords:
        return UrgencyLevel.URGENT
    if ext.urgency.value:
        if _is_scheduling_only(ext.urgency.evidence, config.escalation_policy.time_sensitive_phrases):
            return UrgencyLevel.TIME_SENSITIVE
        return UrgencyLevel.URGENT  # includes inferred signals without evidence: cannot prove it is mere haste
    return UrgencyLevel.NONE


def _intent_level(ext: EnquiryExtraction, config: BusinessConfig) -> IntentLevel:
    rules, primary = config.qualification_rules, ext.primary_intent
    if primary.confidence is Confidence.INSUFFICIENT:
        return IntentLevel.LOW
    high, appointment = set(rules.high_intent_intents), set(rules.appointment_intents)
    if primary.intent in high and primary.confidence is Confidence.CONFIRMED:
        return IntentLevel.HIGH
    # Secondary intents carry no confidence of their own, so they can raise the level to MEDIUM at most.
    if primary.intent in high | appointment or set(ext.secondary_intents) & (high | appointment):
        return IntentLevel.MEDIUM
    return IntentLevel.LOW


def _escalations(
    result: EnquiryExtractionResult, ext: EnquiryExtraction, config: BusinessConfig, appointment_related: bool,
    keywords: list[str], urgency: UrgencyLevel,
) -> list[EscalationTrigger]:  # fmt: skip
    policy, triggers = config.escalation_policy, []

    def add(reason: EscalationReason, detail: str) -> None:
        triggers.append(EscalationTrigger(reason=reason, detail=detail))

    def quoted(flag) -> str:
        return f"{flag.confidence.value}" + (f": {flag.evidence!r}" if flag.evidence else "")

    # Fixed safety rules (cannot be disabled by configuration).
    if ext.human_requested.value:
        add(EscalationReason.HUMAN_REQUESTED, quoted(ext.human_requested))
    if ext.scope_concern.value:
        add(EscalationReason.SCOPE_CONCERN, quoted(ext.scope_concern))
    if ext.service_requested.value is not None and result.matched_service_id is None:
        add(EscalationReason.UNSUPPORTED_SERVICE, f"{ext.service_requested.value!r} is not a configured service")

    # Configurable rules.
    intents = [ext.primary_intent.intent, *ext.secondary_intents]
    escalated_intents = [i.value for i in policy.escalate_intents if i in intents]
    if escalated_intents:
        add(EscalationReason.ESCALATED_INTENT, ", ".join(escalated_intents))
    if urgency is UrgencyLevel.URGENT and policy.escalate_on_urgency:
        if ext.urgency.value and not _is_scheduling_only(ext.urgency.evidence, policy.time_sensitive_phrases):
            add(EscalationReason.URGENCY_SIGNAL, quoted(ext.urgency))
        if keywords:
            add(EscalationReason.URGENT_KEYWORD, ", ".join(keywords))
    if urgency is UrgencyLevel.TIME_SENSITIVE and policy.escalate_on_time_sensitive:
        add(EscalationReason.TIME_SENSITIVE, quoted(ext.urgency))
    if _CONFIDENCE_RANK[ext.primary_intent.confidence] < _CONFIDENCE_RANK[policy.min_intent_confidence]:
        add(
            EscalationReason.LOW_INTENT_CONFIDENCE,
            f"intent confidence {ext.primary_intent.confidence.value!r} is below {policy.min_intent_confidence.value!r}",
        )
    service = next((s for s in config.services if s.id == result.matched_service_id), None)
    if appointment_related and service is not None and not service.bookable:
        add(EscalationReason.NON_BOOKABLE_SERVICE, f"{service.id!r} is not bookable")
    return triggers


def qualify_enquiry(result: EnquiryExtractionResult, config: BusinessConfig) -> QualificationResult:
    """Qualify one extracted enquiry. Deterministic and side-effect free."""
    enquiry = result.enquiry
    if enquiry.business_id != config.business_id:
        raise ValueError(f"enquiry is for {enquiry.business_id!r}, config is for {config.business_id!r}")
    base = {"business_id": enquiry.business_id, "external_message_id": enquiry.external_message_id}

    if result.status is ExtractionStatus.FAILED or result.extraction is None:
        detail = "; ".join(result.errors) or "extraction failed"
        return QualificationResult(
            **base, stage=QualificationStage.ESCALATED, intent_level=None, urgency=None,
            escalations=[EscalationTrigger(reason=EscalationReason.EXTRACTION_FAILED, detail=detail)],
        )  # fmt: skip

    ext, rules = result.extraction, config.qualification_rules
    intents = {ext.primary_intent.intent, *ext.secondary_intents}
    appointment_related = bool(intents & set(rules.appointment_intents))
    required = set(rules.required_for_all) | (set(rules.required_for_appointment) if appointment_related else set())

    fields = _assess_fields(result, ext, required)
    keywords = _matched_keywords(enquiry.message_text, config.escalation_policy.urgent_keywords)
    urgency = _classify_urgency(ext, config, keywords)
    escalations = _escalations(result, ext, config, appointment_related, keywords, urgency)

    clarifications = []
    reasons = {
        InfoStatus.MISSING: ClarificationReason.MISSING,
        InfoStatus.INFERRED: ClarificationReason.UNCONFIRMED,
        InfoStatus.AMBIGUOUS: ClarificationReason.AMBIGUOUS,
    }
    for f in fields:
        if f.required and f.status is not InfoStatus.CONFIRMED:
            detail = f.detail or f"{f.field.value} is {f.status.value}"
            clarifications.append(Clarification(reason=reasons[f.status], field=f.field, detail=detail))
    clarifications += [Clarification(reason=ClarificationReason.EXTRACTION_AMBIGUITY, detail=a) for a in result.ambiguities]
    if appointment_related and result.preferred_datetime is not None:
        if problem := _opening_hours_problem(config, result.preferred_datetime):
            clarifications.append(
                Clarification(reason=ClarificationReason.OUTSIDE_OPENING_HOURS, field=MissingInfo.PREFERRED_DATETIME, detail=problem)
            )

    if escalations:
        stage = QualificationStage.ESCALATED
    elif clarifications:
        stage = QualificationStage.NEEDS_INFORMATION
    else:
        stage = QualificationStage.READY_FOR_NEXT_STEP

    return QualificationResult(
        **base,
        stage=stage,
        intent_level=_intent_level(ext, config),
        urgency=urgency,
        escalations=escalations,
        clarifications=clarifications,
        fields=fields,
        matched_urgent_keywords=keywords,
    )
