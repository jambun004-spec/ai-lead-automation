"""Deterministic qualification scoring.

Pure functions only: the same validated ``LeadResearchReport`` always produces
the same ``ScoreBreakdown``. No LLM, network, randomness or clock is involved.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from pydantic import BaseModel

from lead_intel.models.evidence import Confidence, Finding, ResearchEvidence
from lead_intel.models.qualification import Signal
from lead_intel.models.report import CriterionScore, LeadResearchReport, ScoreBreakdown
from lead_intel.scoring import weights as w
from lead_intel.scoring.classification import classify


def _round_points(value: float) -> int:
    """Round half up (never banker's rounding) and clamp to the criterion range."""
    rounded = int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    return max(0, min(w.CRITERION_MAX, rounded))


def effective_confidence(confidence: Confidence, source_url: object | None) -> Confidence:
    """A 'confirmed' claim without a source URL is scored as 'inferred'."""
    if confidence is Confidence.CONFIRMED and source_url is None:
        return Confidence.INFERRED
    return confidence


def signal_points(signal: Signal, weight: int) -> float:
    if not signal.present:
        return 0.0
    conf = effective_confidence(signal.confidence, signal.source_url)
    return weight * w.CONFIDENCE_MULTIPLIER[conf]


def score_signals(name: str, signals: BaseModel, weights: dict[str, int]) -> CriterionScore:
    breakdown: dict[str, float] = {}
    notes: list[str] = []
    for field, weight in weights.items():
        signal: Signal = getattr(signals, field)
        breakdown[field] = signal_points(signal, weight)
        if signal.present and signal.confidence is Confidence.CONFIRMED and signal.source_url is None:
            notes.append(f"{field}: 'confirmed' without source URL, scored as inferred")
    return CriterionScore(
        name=name,
        points=_round_points(sum(breakdown.values())),
        breakdown=breakdown,
        notes=notes,
    )


def _is_confirmed(finding: Finding) -> bool:
    return effective_confidence(finding.confidence, finding.source_url) is Confidence.CONFIRMED


def score_evidence_confidence(research: ResearchEvidence) -> CriterionScore:
    findings = research.all_findings()  # always >= 4 (the core identity findings)
    confirmed_ratio = sum(_is_confirmed(f) for f in findings) / len(findings)

    urls = {str(u) for u in research.source_urls}
    urls.update(str(f.source_url) for f in findings if f.source_url is not None)

    breakdown = {
        "confirmed_ratio": confirmed_ratio * w.EVIDENCE_CONFIRMED_RATIO_MAX,
        "source_diversity": float(
            min(len(urls) * w.EVIDENCE_SOURCE_POINTS_PER_URL, w.EVIDENCE_SOURCE_MAX)
        ),
    }
    for field, weight in w.EVIDENCE_CORE_IDENTITY_WEIGHTS.items():
        finding: Finding = getattr(research, field)
        breakdown[f"core_{field}_confirmed"] = float(weight) if _is_confirmed(finding) else 0.0

    return CriterionScore(
        name="Evidence / Confidence",
        points=_round_points(sum(breakdown.values())),
        breakdown=breakdown,
        notes=[f"{len(findings)} findings, {len(urls)} distinct sources, {len(research.unknowns)} unknowns"],
    )


def score_report(report: LeadResearchReport) -> ScoreBreakdown:
    q = report.qualification
    icp = score_signals("Business / ICP Fit", q.icp_fit, w.ICP_FIT_WEIGHTS)
    online = score_signals("Online Presence", q.online_presence, w.ONLINE_PRESENCE_WEIGHTS)
    automation = score_signals(
        "Automation Opportunity", q.automation_opportunity, w.AUTOMATION_OPPORTUNITY_WEIGHTS
    )
    business = score_signals("Business Signals", q.business_signals, w.BUSINESS_SIGNALS_WEIGHTS)
    evidence = score_evidence_confidence(report.research)

    total = icp.points + online.points + automation.points + business.points + evidence.points
    return ScoreBreakdown(
        icp_fit=icp,
        online_presence=online,
        automation_opportunity=automation,
        business_signals=business,
        evidence_confidence=evidence,
        total=total,
        classification=classify(total),
    )
