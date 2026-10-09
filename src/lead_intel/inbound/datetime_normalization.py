"""Deterministic, conservative date/time normalization in the business timezone.

The LLM only copies the customer's wording ("this Saturday at 10am"); this module
resolves it. It never guesses: wording with more than one reasonable reading is
AMBIGUOUS (with candidate dates where possible), and wording it does not
understand is UNRECOGNIZED. Pure functions: no clock, network or randomness -
the reference time is always the enquiry's ``received_at``.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from typing import Literal
from zoneinfo import ZoneInfo

from lead_intel.inbound.models import DateTimeStatus, NormalizedDateTime, TimeWindow

DateOrder = Literal["DMY", "MDY"] | None

_WEEKDAYS = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tues": 1, "tue": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thurs": 3, "thur": 3, "thu": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}  # fmt: skip
_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sept": 9, "sep": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}  # fmt: skip
_WINDOWS = {"morning": TimeWindow.MORNING, "afternoon": TimeWindow.AFTERNOON, "evening": TimeWindow.EVENING}


def _alternation(words: dict[str, int]) -> str:
    return "|".join(sorted(words, key=len, reverse=True))


_WD = _alternation(_WEEKDAYS)
_MON = _alternation(_MONTHS)

# Wording that names a range or no specific day -> always ambiguous.
_RANGE_TERMS = re.compile(
    r"\b(next week|this week|following week|week after|weekend|weekday|weekdays|next month|this month|"
    r"end of the month|in a few days|in a couple of days|in a week)\b"
)
# Wording with no specific day; harmless next to a concrete date.
_VAGUE_TERMS = re.compile(r"\b(soon|asap|as soon as possible|any ?time|whenever|later|sometime|flexible|now)\b")

_AMPM = re.compile(r"\b(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)(?=\W|$)")
_H24 = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
_NOON = re.compile(r"\b(noon|midday)\b")
# "at 3" is a time; "at 10 October" / "on the 3rd" are not.
_BARE_HOUR = re.compile(
    rf"\b(?:at|around|about|by|after|before)\s+(\d{{1,2}})(?:\s*o'?clock)?\b"
    rf"(?![:/.\-]?\d)(?!\s*(?:st|nd|rd|th|of|{_MON})\b)"
)
_OCLOCK = re.compile(r"\b(\d{1,2})\s*o'?clock\b")
_WINDOW = re.compile(r"\b(morning|afternoon|evening|tonight)\b")

_RELATIVE = re.compile(r"\b(day after tomorrow|tomorrow|today|tonight)\b")
_ISO = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_NUMERIC = re.compile(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{4}|\d{2}))?\b")
_DOTTED = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4}|\d{2})\b")
_DAY_MONTH = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?(?:\s+of)?\s+({_MON})\b\.?(?:\s+(\d{{4}}))?")
_MONTH_DAY = re.compile(rf"\b({_MON})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:\s+(\d{{4}}))?")
_DAY_ONLY = re.compile(r"\bthe\s+(\d{1,2})(?:st|nd|rd|th)\b")
_WEEKDAY = re.compile(rf"\b(?:(this|next|coming)\s+)?({_WD})\b")


@dataclass
class _DateMatch:
    text: str
    dates: list[dt.date] = field(default_factory=list)  # exactly one when unambiguous
    reason: str | None = None  # set when ambiguous/invalid
    weekday_only: int | None = None  # weekday index for plain weekday mentions

    @property
    def resolved(self) -> dt.date | None:
        return self.dates[0] if self.reason is None and len(self.dates) == 1 else None


def _consume(pattern: re.Pattern[str], text: str) -> tuple[list[re.Match[str]], str]:
    """Return all matches and the text with matched spans blanked out."""
    matches = list(pattern.finditer(text))
    for m in matches:
        text = text[: m.start()] + " " * (m.end() - m.start()) + text[m.end() :]
    return matches, text


def _safe_date(year: int, month: int, day: int) -> dt.date | None:
    try:
        return dt.date(year, month, day)
    except ValueError:
        return None


def _with_year(text: str, month: int, day: int, year: int | None, today: dt.date) -> _DateMatch:
    if year is not None:
        if year < 100:
            year += 2000
        d = _safe_date(year, month, day)
        if d is None:
            return _DateMatch(text, reason=f"'{text}' is not a valid date")
        if d < today:
            return _DateMatch(text, reason=f"'{text}' is in the past")
        return _DateMatch(text, [d])
    d = _safe_date(today.year, month, day)
    if d is not None and d >= today:
        return _DateMatch(text, [d])
    following = _safe_date(today.year + 1, month, day)
    if following is None:
        return _DateMatch(text, reason=f"'{text}' is not a valid date")
    return _DateMatch(text, [following], reason=f"'{text}' has already passed this year")


def _parse_dates(s: str, today: dt.date, date_order: DateOrder) -> tuple[list[_DateMatch], str]:
    found: list[_DateMatch] = []

    matches, s = _consume(_RELATIVE, s)
    offsets = {"today": 0, "tonight": 0, "tomorrow": 1, "day after tomorrow": 2}
    found += [_DateMatch(m.group(0), [today + dt.timedelta(days=offsets[m.group(1)])]) for m in matches]

    matches, s = _consume(_ISO, s)
    for m in matches:
        found.append(_with_year(m.group(0), int(m.group(2)), int(m.group(3)), int(m.group(1)), today))

    for pattern in (_DOTTED, _NUMERIC):
        matches, s = _consume(pattern, s)
        for m in matches:
            a, b = int(m.group(1)), int(m.group(2))
            year = int(m.group(3)) if m.group(3) else None
            readings = {"DMY": (b, a), "MDY": (a, b)}  # -> (month, day)
            orders = [date_order] if date_order else ["DMY", "MDY"]
            # 2000 is a leap year, so this only rejects impossible month/day pairs.
            parsed = [_with_year(m.group(0), *readings[o], year, today) for o in orders if _safe_date(2000, *readings[o])]
            if not parsed:
                found.append(_DateMatch(m.group(0), reason=f"'{m.group(0)}' is not a valid date"))
            elif all((p.dates, p.reason) == (parsed[0].dates, parsed[0].reason) for p in parsed):
                found.append(parsed[0])
            else:
                options = sorted({d for p in parsed for d in p.dates})
                found.append(_DateMatch(m.group(0), options, reason=f"'{m.group(0)}' could be day/month or month/day"))

    for pattern, day_group, month_group in ((_DAY_MONTH, 1, 2), (_MONTH_DAY, 2, 1)):
        matches, s = _consume(pattern, s)
        for m in matches:
            year = int(m.group(3)) if m.group(3) else None
            found.append(_with_year(m.group(0), _MONTHS[m.group(month_group)], int(m.group(day_group)), year, today))

    matches, s = _consume(_DAY_ONLY, s)
    for m in matches:
        day = int(m.group(1))
        this_month = _safe_date(today.year, today.month, day)
        if this_month is not None and this_month >= today:
            candidate = this_month
        else:
            nxt = today.replace(day=1) + dt.timedelta(days=32)
            candidate = _safe_date(nxt.year, nxt.month, day)
        found.append(_DateMatch(m.group(0), [candidate] if candidate else [], reason=f"'{m.group(0)}' does not state the month"))

    matches, s = _consume(_WEEKDAY, s)
    for m in matches:
        qualifier, wd = m.group(1), _WEEKDAYS[m.group(2)]
        ahead = (wd - today.weekday()) % 7
        if qualifier == "next":
            first = today + dt.timedelta(days=ahead or 7)
            found.append(_DateMatch(m.group(0), [first, first + dt.timedelta(days=7)],
                                    reason=f"'{m.group(0)}' can mean the coming one or the one after"))
        elif ahead == 0:
            found.append(_DateMatch(m.group(0), [today, today + dt.timedelta(days=7)],
                                    reason=f"'{m.group(0)}' is today's weekday: today or next week"))
        else:
            found.append(_DateMatch(m.group(0), [today + dt.timedelta(days=ahead)], weekday_only=wd))
    return found, s


@dataclass
class _TimeParse:
    times: list[dt.time] = field(default_factory=list)
    ambiguous: list[str] = field(default_factory=list)
    invalid: list[str] = field(default_factory=list)
    windows: list[TimeWindow] = field(default_factory=list)


def _parse_times(s: str) -> tuple[_TimeParse, str]:
    out = _TimeParse()

    window_matches = list(_WINDOW.finditer(s))  # read, not consumed: "tonight" is also a date
    out.windows = [TimeWindow.EVENING if m.group(1) == "tonight" else _WINDOWS[m.group(1)] for m in window_matches]
    _, s = _consume(re.compile(r"\b(morning|afternoon|evening)\b"), s)

    matches, s = _consume(_AMPM, s)
    for m in matches:
        hour, minute = int(m.group(1)), int(m.group(2) or 0)
        if not 1 <= hour <= 12 or minute > 59:
            out.invalid.append(m.group(0))
            continue
        hour = hour % 12 + (12 if m.group(3).startswith("p") else 0)
        out.times.append(dt.time(hour, minute))

    matches, s = _consume(_H24, s)
    out.times += [dt.time(int(m.group(1)), int(m.group(2))) for m in matches]

    matches, s = _consume(_NOON, s)
    out.times += [dt.time(12, 0) for _ in matches]

    for pattern in (_BARE_HOUR, _OCLOCK):
        matches, s = _consume(pattern, s)
        for m in matches:
            hour = int(m.group(1))
            if 13 <= hour <= 23:
                out.times.append(dt.time(hour, 0))
            elif 1 <= hour <= 11 and out.windows in ([TimeWindow.AFTERNOON], [TimeWindow.EVENING]):
                out.times.append(dt.time(hour + 12, 0))
            elif 5 <= hour <= 11 and out.windows == [TimeWindow.MORNING]:
                out.times.append(dt.time(hour, 0))
            elif 1 <= hour <= 12:
                out.ambiguous.append(m.group(0).strip())
            else:
                out.invalid.append(m.group(0).strip())
    return out, s


def _localize(d: dt.date, t: dt.time, tz: ZoneInfo) -> tuple[dt.datetime | None, str | None]:
    """Attach the business timezone, refusing DST gaps and overlaps."""
    naive = dt.datetime.combine(d, t)
    first = naive.replace(tzinfo=tz, fold=0)
    if first.astimezone(dt.timezone.utc).astimezone(tz).replace(tzinfo=None) != naive:
        return None, f"{naive:%Y-%m-%d %H:%M} does not exist in {tz.key} (clock change)"
    if first.utcoffset() != naive.replace(tzinfo=tz, fold=1).utcoffset():
        return None, f"{naive:%Y-%m-%d %H:%M} occurs twice in {tz.key} (clock change)"
    return first, None


def normalize_preferred_datetime(
    raw_text: str | None,
    *,
    reference: dt.datetime,
    tz: ZoneInfo,
    date_order: DateOrder = None,
) -> NormalizedDateTime:
    """Resolve customer date/time wording relative to ``reference`` (an aware datetime)."""
    if reference.tzinfo is None:
        raise ValueError("reference must be timezone-aware")

    def result(status: DateTimeStatus, **kwargs) -> NormalizedDateTime:
        return NormalizedDateTime(status=status, raw_text=raw_text, timezone=tz.key, **kwargs)

    if raw_text is None or not raw_text.strip():
        return result(DateTimeStatus.NOT_PROVIDED)

    now = reference.astimezone(tz)
    today = now.date()
    s = " ".join(raw_text.casefold().replace(",", " ").split())

    times, s = _parse_times(s)
    dates, s = _parse_dates(s, today, date_order)

    def ambiguous(reason: str, candidates: list[dt.date] | None = None) -> NormalizedDateTime:
        return result(DateTimeStatus.AMBIGUOUS, reason=reason, candidates=sorted(set(candidates or [])))

    all_candidates = [d for m in dates for d in m.dates]
    if range_term := _RANGE_TERMS.search(s):
        return ambiguous(f"'{range_term.group(0)}' is a range, not a specific day", all_candidates)
    if times.invalid:
        return result(DateTimeStatus.UNRECOGNIZED, reason=f"invalid time {times.invalid[0]!r}")
    if not dates:
        if times.times or times.ambiguous or times.windows:
            return ambiguous("a time was given without a date")
        if vague := _VAGUE_TERMS.search(s):
            return ambiguous(f"'{vague.group(0)}' does not name a specific day")
        return result(DateTimeStatus.UNRECOGNIZED, reason="no recognizable date")

    # A weekday next to an explicit date ("Saturday 10 October") must agree with it.
    explicit = [m for m in dates if m.weekday_only is None]
    weekdays = [m for m in dates if m.weekday_only is not None]
    if len(explicit) == 1 and weekdays:
        match = explicit[0]
        if match.resolved is None:
            return ambiguous(match.reason or "unclear date", match.dates)
        if any(w.weekday_only != match.resolved.weekday() for w in weekdays):
            return ambiguous(f"weekday does not match '{match.text}'", match.dates + all_candidates)
        dates = [match]
    if len(dates) > 1:
        return ambiguous("more than one date mentioned", all_candidates)

    match = dates[0]
    if match.resolved is None:
        return ambiguous(match.reason or "unclear date", match.dates)
    day = match.resolved

    if len(set(times.times)) + len(times.ambiguous) > 1:
        return ambiguous("more than one time mentioned", [day])
    if times.ambiguous:
        return ambiguous(f"'{times.ambiguous[0]}' could be morning or afternoon", [day])
    window = times.windows[0] if len(set(times.windows)) == 1 else None
    if len(set(times.windows)) > 1:
        return ambiguous("more than one part of the day mentioned", [day])

    if not times.times:
        if day < today:  # pragma: no cover - relative/explicit parsing never yields past dates
            return ambiguous("date is in the past", [day])
        return result(DateTimeStatus.RESOLVED, date=day, time_window=window)

    start, problem = _localize(day, times.times[0], tz)
    if problem:
        return ambiguous(problem, [day])
    if start < now:
        return ambiguous(f"{start:%Y-%m-%d %H:%M} has already passed", [day])
    return result(DateTimeStatus.RESOLVED, date=day, time=times.times[0], time_window=window, start=start)
