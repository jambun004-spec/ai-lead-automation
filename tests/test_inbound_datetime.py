from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pytest

from lead_intel.inbound.datetime_normalization import normalize_preferred_datetime
from lead_intel.inbound.models import DateTimeStatus, TimeWindow

KOLKATA = ZoneInfo("Asia/Kolkata")
REF = dt.datetime(2026, 10, 7, 10, 0, tzinfo=KOLKATA)  # Wednesday 10:00
R, A, U, N = DateTimeStatus.RESOLVED, DateTimeStatus.AMBIGUOUS, DateTimeStatus.UNRECOGNIZED, DateTimeStatus.NOT_PROVIDED
D = dt.date


def norm(text, ref=REF, tz=KOLKATA, order="DMY"):
    return normalize_preferred_datetime(text, reference=ref, tz=tz, date_order=order)


@pytest.mark.parametrize(
    "text, status, date, time",
    [
        ("this Saturday", R, D(2026, 10, 10), None),
        ("Saturday", R, D(2026, 10, 10), None),
        ("coming sat", R, D(2026, 10, 10), None),
        ("today", R, D(2026, 10, 7), None),
        ("tomorrow at 11am", R, D(2026, 10, 8), dt.time(11)),
        ("day after tomorrow", R, D(2026, 10, 9), None),
        ("Friday at 4pm", R, D(2026, 10, 9), dt.time(16)),
        ("14:30 tomorrow", R, D(2026, 10, 8), dt.time(14, 30)),
        ("tomorrow at noon", R, D(2026, 10, 8), dt.time(12)),
        ("12am tomorrow", R, D(2026, 10, 8), dt.time(0)),
        ("15/10 at 2pm", R, D(2026, 10, 15), dt.time(14)),
        ("17th October", R, D(2026, 10, 17), None),
        ("October 17", R, D(2026, 10, 17), None),
        ("on the 17th of October at 11:30am", R, D(2026, 10, 17), dt.time(11, 30)),
        ("at 10 October", R, D(2026, 10, 10), None),
        ("2026-10-20 9:00", R, D(2026, 10, 20), dt.time(9)),
        ("October 17th 2027 at noon", R, D(2027, 10, 17), dt.time(12)),
        ("Saturday 10 October", R, D(2026, 10, 10), None),
        ("Monday around 3 in the afternoon", R, D(2026, 10, 12), dt.time(15)),
        ("sometime on Saturday", R, D(2026, 10, 10), None),
    ],
)
def test_resolved(text, status, date, time):
    result = norm(text)
    assert (result.status, result.date, result.time) == (status, date, time)
    assert result.timezone == "Asia/Kolkata"


@pytest.mark.parametrize(
    "text, candidates",
    [
        ("next Saturday", [D(2026, 10, 10), D(2026, 10, 17)]),
        ("Wednesday", [D(2026, 10, 7), D(2026, 10, 14)]),  # said on a Wednesday
        ("this Wednesday", [D(2026, 10, 7), D(2026, 10, 14)]),
        ("Monday around 3", [D(2026, 10, 12)]),
        ("Monday at 3 o'clock", [D(2026, 10, 12)]),
        ("3 October", [D(2027, 10, 3)]),  # already passed this year
        ("3 October 2026", []),  # explicitly in the past
        ("today at 9am", [D(2026, 10, 7)]),  # time already passed
        ("Friday 10 October", [D(2026, 10, 9), D(2026, 10, 10)]),  # weekday contradicts date
        ("Monday or Tuesday", [D(2026, 10, 12), D(2026, 10, 13)]),
        ("tomorrow at 10am or 2pm", [D(2026, 10, 8)]),
        ("tomorrow morning or evening", [D(2026, 10, 8)]),
        ("the 20th", [D(2026, 10, 20)]),
        ("next week", []),
        ("this weekend", []),
        ("in a few days", []),
        ("asap", []),
        ("whenever suits", []),
        ("5pm", []),  # time without a date
        ("31/02", []),  # impossible date
    ],
)
def test_ambiguous_never_guesses(text, candidates):
    result = norm(text)
    assert result.status is A, result
    assert result.date is None and result.start is None
    assert result.candidates == candidates
    assert result.reason


@pytest.mark.parametrize("text", ["blah", "at 25", "13pm tomorrow", "after lunch"])
def test_unrecognized(text):
    result = norm(text)
    assert result.status is U and result.date is None and result.reason


@pytest.mark.parametrize("text", [None, "", "   "])
def test_not_provided(text):
    assert norm(text).status is N


def test_time_window_kept():
    result = norm("Saturday 10 October, morning")
    assert result.time_window is TimeWindow.MORNING and result.time is None
    assert norm("tonight").time_window is TimeWindow.EVENING


def test_start_is_aware_in_business_timezone():
    result = norm("tomorrow at 11am")
    assert result.start == dt.datetime(2026, 10, 8, 11, 0, tzinfo=KOLKATA)
    assert result.start.utcoffset() == dt.timedelta(hours=5, minutes=30)


# --- numeric dates and configured date order ---------------------------------


def test_numeric_dates_follow_configured_order():
    assert norm("05/11", order="DMY").date == D(2026, 11, 5)
    assert norm("11/05", order="MDY").date == D(2026, 11, 5)


def test_numeric_date_ambiguous_without_configured_order():
    result = norm("11/12", order=None)
    assert result.status is A
    assert result.candidates == [D(2026, 11, 12), D(2026, 12, 11)]


def test_numeric_date_unambiguous_without_order_when_one_part_exceeds_12():
    assert norm("25/12", order=None).date == D(2026, 12, 25)


# --- timezone handling ----------------------------------------------------------


def test_relative_dates_use_business_timezone_not_utc():
    # Friday 20:00 UTC is already Saturday 01:30 in Kolkata.
    ref = dt.datetime(2026, 10, 9, 20, 0, tzinfo=dt.timezone.utc)
    assert norm("today", ref=ref).date == D(2026, 10, 10)
    assert norm("tomorrow", ref=ref).date == D(2026, 10, 11)
    new_york = ZoneInfo("America/New_York")
    assert norm("today", ref=ref, tz=new_york).date == D(2026, 10, 9)


def test_same_instant_different_business_weekday():
    ref = dt.datetime(2026, 10, 9, 20, 0, tzinfo=dt.timezone.utc)  # Fri in NY, Sat in Kolkata
    assert norm("Saturday", ref=ref).status is A  # today's weekday in Kolkata
    assert norm("Saturday", ref=ref, tz=ZoneInfo("America/New_York")).date == D(2026, 10, 10)


def test_dst_gap_is_ambiguous_not_shifted():
    london = ZoneInfo("Europe/London")
    ref = dt.datetime(2026, 3, 27, 9, 0, tzinfo=london)
    result = norm("29/03 at 1:30am", ref=ref, tz=london)
    assert result.status is A and "does not exist" in result.reason


def test_dst_overlap_is_ambiguous():
    london = ZoneInfo("Europe/London")
    ref = dt.datetime(2026, 10, 20, 9, 0, tzinfo=london)
    result = norm("25/10 at 1:30am", ref=ref, tz=london)
    assert result.status is A and "occurs twice" in result.reason


def test_naive_reference_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        normalize_preferred_datetime("today", reference=dt.datetime(2026, 10, 7), tz=KOLKATA)


def test_deterministic():
    assert all(norm("Friday at 4pm") == norm("Friday at 4pm") for _ in range(5))
