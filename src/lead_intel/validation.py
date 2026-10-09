"""Safe parsing of raw LLM text into a validated ``LeadResearchReport``.

The only tolerance applied is stripping a Markdown code fence or surrounding
prose around a single JSON object. The data itself is never repaired or
defaulted: anything that fails the schema is reported as an error.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ValidationError

from lead_intel.models.report import LeadResearchReport

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


class ValidationOutcome(BaseModel):
    report: LeadResearchReport | None = None
    errors: list[str] = []

    @property
    def ok(self) -> bool:
        return self.report is not None and not self.errors


def extract_json_object(raw: str) -> str | None:
    """Return the JSON object text in ``raw``, or None if there isn't one."""
    text = raw.strip()
    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    return text[start : end + 1]


def _format_errors(exc: ValidationError, limit: int = 20) -> list[str]:
    errors = []
    for err in exc.errors()[:limit]:
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        errors.append(f"{loc}: {err['msg']}")
    if exc.error_count() > limit:
        errors.append(f"... and {exc.error_count() - limit} more errors")
    return errors


def validate_report_text(raw: str | None) -> ValidationOutcome:
    if not raw or not raw.strip():
        return ValidationOutcome(errors=["LLM returned empty output"])
    payload = extract_json_object(raw)
    if payload is None:
        return ValidationOutcome(errors=["LLM output contains no JSON object"])
    try:
        report = LeadResearchReport.model_validate_json(payload)
    except ValidationError as exc:
        return ValidationOutcome(errors=_format_errors(exc))
    return ValidationOutcome(report=report)
