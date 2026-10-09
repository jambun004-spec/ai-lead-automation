from __future__ import annotations

import pytest

from lead_intel.models import (
    AutomationOpportunitySignals,
    BusinessSignals,
    Classification,
    Confidence,
    IcpFitSignals,
    OnlinePresenceSignals,
)
from lead_intel.scoring import weights as w
from lead_intel.scoring.engine import score_evidence_confidence, score_report

from tests.conftest import finding, make_report, make_research, signal

C, I, X = Confidence.CONFIRMED, Confidence.INFERRED, Confidence.INSUFFICIENT


# --- scoring policy integrity -------------------------------------------------


@pytest.mark.parametrize(
    "weights, model",
    [
        (w.ICP_FIT_WEIGHTS, IcpFitSignals),
        (w.ONLINE_PRESENCE_WEIGHTS, OnlinePresenceSignals),
        (w.AUTOMATION_OPPORTUNITY_WEIGHTS, AutomationOpportunitySignals),
        (w.BUSINESS_SIGNALS_WEIGHTS, BusinessSignals),
    ],
)
def test_signal_weights_sum_to_20_and_cover_every_signal(weights, model):
    assert sum(weights.values()) == w.CRITERION_MAX
    assert set(weights) == set(model.model_fields)


def test_evidence_criterion_max_is_20():
    total = (
        w.EVIDENCE_CONFIRMED_RATIO_MAX
        + w.EVIDENCE_SOURCE_MAX
        + sum(w.EVIDENCE_CORE_IDENTITY_WEIGHTS.values())
    )
    assert total == w.CRITERION_MAX


# --- end-to-end scoring -------------------------------------------------------


def test_perfect_report_scores_100_hot(perfect_report):
    score = score_report(perfect_report)
    assert [c.points for c in score.criteria()] == [20, 20, 20, 20, 20]
    assert score.total == 100
    assert score.classification is Classification.HOT


def test_nothing_known_scores_0_low():
    report = make_report(
        lambda _: signal(present=False, confidence=X, url=None),
        research=make_research(core=finding(X, url=None)),
    )
    score = score_report(report)
    assert score.total == 0
    assert score.classification is Classification.LOW


def test_all_inferred_signals_earn_half_points():
    score = score_report(make_report(lambda _: signal(confidence=I)))
    for criterion in score.criteria()[:4]:
        assert criterion.points == 10


def test_absent_signal_earns_nothing_even_if_confirmed():
    score = score_report(make_report(lambda _: signal(present=False, confidence=C)))
    assert [c.points for c in score.criteria()[:4]] == [0, 0, 0, 0]


def test_insufficient_signal_earns_nothing_even_if_present():
    report = make_report(lambda _: signal(present=True, confidence=X))
    assert score_report(report).icp_fit.points == 0


def test_confirmed_without_source_url_is_scored_as_inferred_and_noted():
    icp = score_report(make_report(lambda _: signal(confidence=C, url=None))).icp_fit
    assert icp.points == 10
    assert len(icp.notes) == len(w.ICP_FIT_WEIGHTS)


def test_individual_signal_weights_are_applied():
    def factory(name: str):
        return signal() if name == "industry_matches_icp" else signal(present=False)

    icp = score_report(make_report(factory)).icp_fit
    assert icp.points == w.ICP_FIT_WEIGHTS["industry_matches_icp"]
    assert icp.breakdown["industry_matches_icp"] == 8.0


@pytest.mark.parametrize("signal_name, expected", [("size_matches_icp", 2), ("is_operating_business", 3)])
def test_half_points_round_half_up(signal_name, expected):
    # Inferred weight-3 signal -> 1.5 -> 2; weight-5 -> 2.5 -> 3 (Python's round() would give 2 for both).
    def factory(name: str):
        return signal(confidence=I) if name == signal_name else signal(present=False)

    assert score_report(make_report(factory)).icp_fit.points == expected


def test_total_is_sum_of_criteria():
    report = make_report(lambda n: signal(confidence=I) if "website" in n else signal(present=False))
    score = score_report(report)
    assert score.total == sum(c.points for c in score.criteria())


def test_scoring_is_deterministic(perfect_report):
    first = score_report(perfect_report)
    for _ in range(5):
        assert score_report(perfect_report) == first


# --- criterion 5: evidence / confidence --------------------------------------


def test_evidence_counts_distinct_urls_once_and_caps_source_points():
    urls = ["https://a.example/", "https://a.example/"] + [f"https://s{i}.example/" for i in range(10)]
    score = score_evidence_confidence(make_research(source_urls=urls))
    assert score.breakdown["source_diversity"] == w.EVIDENCE_SOURCE_MAX


def test_evidence_duplicate_urls_are_not_double_counted():
    # core findings all cite https://example.com/f; plus the same URL in source_urls
    score = score_evidence_confidence(make_research(source_urls=["https://example.com/f"]))
    assert score.breakdown["source_diversity"] == 1.0


def test_evidence_confirmed_ratio():
    # 4 confirmed core findings + 4 insufficient extras -> ratio 0.5 -> 5 points
    score = score_evidence_confidence(make_research(extra=[finding(X, url=None)] * 4))
    assert score.breakdown["confirmed_ratio"] == 5.0


def test_evidence_core_identity_requires_confirmed_with_url():
    score = score_evidence_confidence(make_research(core=finding(C, url=None)))
    assert score.breakdown["confirmed_ratio"] == 0.0
    assert score.breakdown["core_business_identity_confirmed"] == 0.0
    assert score.breakdown["source_diversity"] == 0.0
    assert score.points == 0


def test_evidence_score_ignores_llm_qualification_signals():
    research = make_research()
    strong = score_report(make_report(lambda _: signal(), research=research))
    weak = score_report(make_report(lambda _: signal(present=False, confidence=X), research=research))
    assert strong.evidence_confidence == weak.evidence_confidence


# --- classification through the full engine ----------------------------------


@pytest.mark.parametrize(
    "inferred_criteria, expected_total, expected_class",
    [
        (0, 100, Classification.HOT),
        (1, 90, Classification.HOT),
        (2, 80, Classification.HOT),  # boundary
        (3, 70, Classification.WARM),
        (4, 60, Classification.WARM),  # boundary
    ],
)
def test_engine_classification(inferred_criteria, expected_total, expected_class):
    groups = [w.ICP_FIT_WEIGHTS, w.ONLINE_PRESENCE_WEIGHTS, w.AUTOMATION_OPPORTUNITY_WEIGHTS, w.BUSINESS_SIGNALS_WEIGHTS]
    inferred_names = {n for g in groups[:inferred_criteria] for n in g}
    urls = [f"https://source{i}.example.com/" for i in range(5)]
    report = make_report(
        lambda n: signal(confidence=I) if n in inferred_names else signal(),
        research=make_research(source_urls=urls),
    )
    score = score_report(report)
    assert score.total == expected_total
    assert score.classification is expected_class
