"""Top-level contracts: the LLM output and the final pipeline result."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from lead_intel.models.evidence import ResearchEvidence
from lead_intel.models.lead import LeadInput
from lead_intel.models.qualification import QualificationEvidence


class LeadResearchReport(BaseModel):
    """Exactly what the research agent must return (validated before use)."""

    model_config = ConfigDict(extra="forbid")

    research: ResearchEvidence
    qualification: QualificationEvidence


class Classification(str, Enum):
    HOT = "HOT"
    WARM = "WARM"
    COOL = "COOL"
    LOW = "LOW"


class CriterionScore(BaseModel):
    name: str
    points: int = Field(ge=0, le=20)
    max_points: int = 20
    breakdown: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class ScoreBreakdown(BaseModel):
    icp_fit: CriterionScore
    online_presence: CriterionScore
    automation_opportunity: CriterionScore
    business_signals: CriterionScore
    evidence_confidence: CriterionScore
    total: int = Field(ge=0, le=100)
    classification: Classification

    def criteria(self) -> list[CriterionScore]:
        return [
            self.icp_fit,
            self.online_presence,
            self.automation_opportunity,
            self.business_signals,
            self.evidence_confidence,
        ]


class RunStatus(str, Enum):
    SCORED = "scored"
    FAILED = "failed"


class LeadQualificationResult(BaseModel):
    """Final output of the pipeline. A FAILED result is never scored."""

    lead: LeadInput
    status: RunStatus
    report: LeadResearchReport | None = None
    score: ScoreBreakdown | None = None
    errors: list[str] = Field(default_factory=list)
