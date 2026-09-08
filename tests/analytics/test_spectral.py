from __future__ import annotations

import numpy as np
import pytest

from satquery.analytics.spectral import MissingRequiredBandError, compute_index


def test_indices_match_hand_calculation_and_preserve_nodata() -> None:
    red = np.array([[1.0, 2.0], [0.0, 4.0]])
    nir = np.array([[3.0, 2.0], [0.0, 0.0]])
    valid = np.array([[True, True], [False, True]])

    ndvi = compute_index("ndvi", {"RED": red, "NIR": nir}, valid)

    np.testing.assert_allclose(ndvi.compressed(), [0.5, 0.0, -1.0])
    assert ndvi.mask[1, 0]


def test_ndvi_rejects_unknown_or_missing_nir() -> None:
    with pytest.raises(MissingRequiredBandError):
        compute_index(
            "ndvi",
            {"RED": np.ones((2, 2))},
            np.ones((2, 2), dtype=bool),
        )
