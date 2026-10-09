"""Total score -> HOT / WARM / COOL / LOW."""

from __future__ import annotations

from lead_intel.models.report import Classification

# (minimum total inclusive, class), checked from highest to lowest.
THRESHOLDS: tuple[tuple[int, Classification], ...] = (
    (80, Classification.HOT),
    (60, Classification.WARM),
    (40, Classification.COOL),
    (0, Classification.LOW),
)


def classify(total: int) -> Classification:
    if isinstance(total, bool) or not isinstance(total, int):
        raise TypeError(f"total must be an int, got {type(total).__name__}")
    if not 0 <= total <= 100:
        raise ValueError(f"total must be within 0-100, got {total}")
    for minimum, label in THRESHOLDS:
        if total >= minimum:
            return label
    raise AssertionError("unreachable")  # pragma: no cover
