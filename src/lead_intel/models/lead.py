"""Lead input: the minimal facts we are asked to research."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class LeadInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    business_name: str = Field(min_length=1, max_length=200)
    website: HttpUrl | None = None
    location: str | None = Field(default=None, max_length=200)
    industry_hint: str | None = Field(default=None, max_length=200)
    notes: str | None = Field(default=None, max_length=2000)

    def describe(self) -> str:
        """Human-readable summary used in the research prompt."""
        lines = [f"Business name: {self.business_name}"]
        if self.website:
            lines.append(f"Website: {self.website}")
        if self.location:
            lines.append(f"Location: {self.location}")
        if self.industry_hint:
            lines.append(f"Industry hint: {self.industry_hint}")
        if self.notes:
            lines.append(f"Notes: {self.notes}")
        return "\n".join(lines)
