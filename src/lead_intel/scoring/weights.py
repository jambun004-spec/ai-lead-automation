"""Every point value used by the scoring engine, in one auditable place.

Each criterion is worth exactly 20 points; the weights inside a criterion must
sum to 20 (enforced by tests). Changing a weight changes scores deterministically
for every lead - treat edits here as a versioned scoring-policy change.
"""

from __future__ import annotations

from lead_intel.models.evidence import Confidence

SCORING_VERSION = "1.0.0"
CRITERION_MAX = 20

# How much of a signal's weight is earned at each confidence level.
CONFIDENCE_MULTIPLIER: dict[Confidence, float] = {
    Confidence.CONFIRMED: 1.0,
    Confidence.INFERRED: 0.5,
    Confidence.INSUFFICIENT: 0.0,
}

# Criteria 1-4: signal name -> weight. Names must match the signal model fields.
ICP_FIT_WEIGHTS: dict[str, int] = {
    "is_operating_business": 5,
    "industry_matches_icp": 8,
    "location_in_target_market": 4,
    "size_matches_icp": 3,
}

ONLINE_PRESENCE_WEIGHTS: dict[str, int] = {
    "has_working_website": 6,
    "website_describes_services": 4,
    "active_social_profiles": 4,
    "listed_on_review_or_directory_sites": 4,
    "recent_online_activity": 2,
}

AUTOMATION_OPPORTUNITY_WEIGHTS: dict[str, int] = {
    "manual_inquiry_or_booking_process": 6,
    "high_volume_customer_interactions": 5,
    "repetitive_workflows_evident": 5,
    "no_existing_automation_detected": 4,
}

BUSINESS_SIGNALS_WEIGHTS: dict[str, int] = {
    "hiring_activity": 5,
    "growth_or_expansion": 5,
    "recent_news_or_updates": 3,
    "established_reputation": 4,
    "multiple_services_or_locations": 3,
}

# Criterion 5 (Evidence / Confidence) - computed from research findings only.
EVIDENCE_CONFIRMED_RATIO_MAX = 10  # share of findings that are confirmed
EVIDENCE_SOURCE_POINTS_PER_URL = 1  # distinct source URLs...
EVIDENCE_SOURCE_MAX = 5  # ...capped
EVIDENCE_CORE_IDENTITY_WEIGHTS: dict[str, int] = {  # core facts confirmed
    "business_identity": 2,
    "website": 2,
    "location": 1,
}
