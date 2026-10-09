"""Qualification evidence: the LLM's structured assessment, with no numbers.

The LLM fills in a fixed checklist of named signals per criterion. It never
assigns points; ``lead_intel.scoring`` converts signals to scores.

Criterion 5 (Evidence / Confidence) has no LLM signals at all - it is computed
purely from the research findings.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from lead_intel.models.evidence import Confidence


class Signal(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    present: bool
    confidence: Confidence
    reasoning: str = Field(min_length=1)
    source_url: HttpUrl | None = None


class IcpFitSignals(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_operating_business: Signal
    industry_matches_icp: Signal
    location_in_target_market: Signal
    size_matches_icp: Signal


class OnlinePresenceSignals(BaseModel):
    model_config = ConfigDict(extra="forbid")

    has_working_website: Signal
    website_describes_services: Signal
    active_social_profiles: Signal
    listed_on_review_or_directory_sites: Signal
    recent_online_activity: Signal


class AutomationOpportunitySignals(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manual_inquiry_or_booking_process: Signal
    high_volume_customer_interactions: Signal
    repetitive_workflows_evident: Signal
    no_existing_automation_detected: Signal


class BusinessSignals(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hiring_activity: Signal
    growth_or_expansion: Signal
    recent_news_or_updates: Signal
    established_reputation: Signal
    multiple_services_or_locations: Signal


class QualificationEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    icp_fit: IcpFitSignals
    online_presence: OnlinePresenceSignals
    automation_opportunity: AutomationOpportunitySignals
    business_signals: BusinessSignals
    summary: str = Field(min_length=1, description="Short qualitative reasoning; not used for scoring.")
