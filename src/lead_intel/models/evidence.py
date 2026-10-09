"""Research evidence schema: what the agent found, and how sure it is."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class Confidence(str, Enum):
    CONFIRMED = "confirmed"  # directly stated by a cited source
    INFERRED = "inferred"  # reasonable conclusion from indirect evidence
    INSUFFICIENT = "insufficient"  # not enough evidence to say


class Finding(BaseModel):
    """One factual finding. Every important claim is expressed as a Finding."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    claim: str = Field(min_length=1)
    evidence: str = Field(min_length=1, description="Quote or description of what supports the claim.")
    confidence: Confidence
    source_url: HttpUrl | None = None


class ResearchEvidence(BaseModel):
    """Structured evidence gathered about a lead, grouped by topic."""

    model_config = ConfigDict(extra="forbid")

    business_identity: Finding
    industry: Finding
    location: Finding
    website: Finding
    services: list[Finding] = Field(default_factory=list)
    online_presence: list[Finding] = Field(default_factory=list)
    business_signals: list[Finding] = Field(default_factory=list)
    automation_opportunities: list[Finding] = Field(default_factory=list)
    existing_automation: list[Finding] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list, description="Questions research could not answer.")
    source_urls: list[HttpUrl] = Field(default_factory=list)

    def all_findings(self) -> list[Finding]:
        return [
            self.business_identity,
            self.industry,
            self.location,
            self.website,
            *self.services,
            *self.online_presence,
            *self.business_signals,
            *self.automation_opportunities,
            *self.existing_automation,
        ]
