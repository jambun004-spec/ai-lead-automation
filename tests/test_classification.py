from __future__ import annotations

import pytest

from lead_intel.models import Classification
from lead_intel.scoring.classification import classify


@pytest.mark.parametrize(
    "total, expected",
    [
        (100, Classification.HOT),
        (80, Classification.HOT),
        (79, Classification.WARM),
        (60, Classification.WARM),
        (59, Classification.COOL),
        (40, Classification.COOL),
        (39, Classification.LOW),
        (0, Classification.LOW),
    ],
)
def test_boundaries(total, expected):
    assert classify(total) is expected


@pytest.mark.parametrize("total", [-1, 101])
def test_out_of_range_rejected(total):
    with pytest.raises(ValueError):
        classify(total)


@pytest.mark.parametrize("total", [79.5, "80", True, None])
def test_non_int_rejected(total):
    with pytest.raises(TypeError):
        classify(total)
