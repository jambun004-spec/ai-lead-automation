"""Per-business configuration, validated before use.

All vertical-specific content (service names, out-of-scope topics, urgency
keywords) belongs in a config file, not in code.
"""

from __future__ import annotations

import json
from datetime import time
from enum import Enum
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from lead_intel.config import ConfigurationError
from lead_intel.inbound.models import BUSINESS_ID_PATTERN, Intent, MissingInfo
from lead_intel.models.evidence import Confidence


class Weekday(str, Enum):
    MONDAY = "monday"
    TUESDAY = "tuesday"
    WEDNESDAY = "wednesday"
    THURSDAY = "thursday"
    FRIDAY = "friday"
    SATURDAY = "saturday"
    SUNDAY = "sunday"


class TimeRange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    open: time
    close: time

    @model_validator(mode="after")
    def _ordered(self) -> TimeRange:
        if self.open >= self.close:
            raise ValueError("open must be before close")
        return self


class ServiceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    name: str = Field(min_length=1, max_length=120)
    aliases: list[str] = Field(default_factory=list)
    bookable: bool = True
    duration_minutes: int | None = Field(default=None, ge=5, le=480)


class QualificationRules(BaseModel):
    """Operational rules (data only in Phase 1; evaluated by the Phase 2 engine)."""

    model_config = ConfigDict(extra="forbid")

    appointment_intents: list[Intent] = Field(
        default_factory=lambda: [Intent.APPOINTMENT_REQUEST, Intent.AVAILABILITY_ENQUIRY]
    )
    required_for_appointment: list[MissingInfo] = Field(
        default_factory=lambda: [
            MissingInfo.CUSTOMER_NAME,
            MissingInfo.CONTACT_METHOD,
            MissingInfo.SERVICE,
            MissingInfo.PREFERRED_DATETIME,
        ]
    )
    required_for_all: list[MissingInfo] = Field(default_factory=lambda: [MissingInfo.CONTACT_METHOD])
    high_intent_intents: list[Intent] = Field(default_factory=lambda: [Intent.APPOINTMENT_REQUEST])


class EscalationPolicy(BaseModel):
    """Escalation configuration. The safety rules from the product proposal (1.7)
    are fixed to True and cannot be disabled by configuration."""

    model_config = ConfigDict(extra="forbid")

    escalate_on_human_request: Literal[True] = True
    escalate_on_scope_concern: Literal[True] = True
    escalate_on_extraction_failure: Literal[True] = True
    escalate_on_urgency: bool = True
    min_intent_confidence: Confidence = Confidence.INFERRED
    urgent_keywords: list[str] = Field(default_factory=list)
    out_of_scope_topics: list[str] = Field(default_factory=list)


class BusinessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    business_id: str = Field(pattern=BUSINESS_ID_PATTERN)
    display_name: str = Field(min_length=1, max_length=200)
    timezone: str
    # How to read numeric dates like 05/06. None = treat as ambiguous unless unambiguous.
    date_order: Literal["DMY", "MDY"] | None = None
    services: list[ServiceConfig] = Field(min_length=1)
    opening_hours: dict[Weekday, list[TimeRange]] = Field(default_factory=dict)
    qualification_rules: QualificationRules = Field(default_factory=QualificationRules)
    escalation_policy: EscalationPolicy = Field(default_factory=EscalationPolicy)

    @field_validator("timezone")
    @classmethod
    def _valid_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown IANA timezone {value!r}") from exc
        return value

    @field_validator("opening_hours")
    @classmethod
    def _no_overlaps(cls, hours: dict[Weekday, list[TimeRange]]) -> dict[Weekday, list[TimeRange]]:
        for day, ranges in hours.items():
            ordered = sorted(ranges, key=lambda r: r.open)
            for earlier, later in zip(ordered, ordered[1:]):
                if later.open < earlier.close:
                    raise ValueError(f"overlapping opening hours on {day.value}")
            hours[day] = ordered
        return hours

    @model_validator(mode="after")
    def _unique_services(self) -> BusinessConfig:
        ids = [s.id for s in self.services]
        if len(ids) != len(set(ids)):
            raise ValueError("service ids must be unique")
        labels: dict[str, str] = {}
        for service in self.services:
            for label in [service.name, *service.aliases]:
                key = label.casefold()
                if labels.setdefault(key, service.id) != service.id:
                    raise ValueError(f"service label {label!r} is used by more than one service")
        return self

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    def service_ids(self) -> set[str]:
        return {s.id for s in self.services}


def load_business_config(path: str | Path) -> BusinessConfig:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Cannot read business config {path}: {exc}") from exc
    try:
        return BusinessConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigurationError(f"Invalid business config {path}: {exc}") from exc
