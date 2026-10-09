"""Generic validation of raw LLM text against any Pydantic model.

Same policy as ``lead_intel.validation`` (which stays dedicated to the outbound
research report): a Markdown fence or surrounding prose around a single JSON
object is tolerated, but the data itself is never repaired or defaulted.
"""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError

from lead_intel.validation import _format_errors, extract_json_object

ModelT = TypeVar("ModelT", bound=BaseModel)


class LlmJsonOutcome(BaseModel, Generic[ModelT]):
    value: ModelT | None = None
    errors: list[str] = []

    @property
    def ok(self) -> bool:
        return self.value is not None and not self.errors


def validate_llm_json(model_cls: type[ModelT], raw_text: str | None) -> LlmJsonOutcome[ModelT]:
    if not raw_text or not raw_text.strip():
        return LlmJsonOutcome[model_cls](errors=["LLM returned empty output"])
    payload = extract_json_object(raw_text)
    if payload is None:
        return LlmJsonOutcome[model_cls](errors=["LLM output contains no JSON object"])
    try:
        value = model_cls.model_validate_json(payload)
    except ValidationError as exc:
        return LlmJsonOutcome[model_cls](errors=_format_errors(exc))
    return LlmJsonOutcome[model_cls](value=value)
