"""Frangi vesselness evaluated in z-slabs must match the full-volume filter."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tools


def _reference(volume, gauss_sigma, sigma):
    smoothed = ndimage.gaussian_filter(volume, sigma=gauss_sigma)
    smoothed = tools.normalize(smoothed)
    hessian, zero_mask = tools.hessian_of_normalized(smoothed, sigma)
    return tools.frangi_from_hessian(hessian, zero_mask)


def test_z_slabs_match_the_full_hessian_and_stay_inside_the_slab(monkeypatch):
    rng = np.random.default_rng(0)
    volume = rng.random((40, 18, 16))
    # A spike in one slab: normalizing each slab on its own extrema would move
    # this value and the rest of the cube apart.
    volume[3, 2, 2] = 50.0
    volume[-1, 10, 8] = 0.0
    reference = _reference(volume, gauss_sigma=2, sigma=6)

    seen = []
    real = tools.hessian_of_normalized

    def record(slab, sigma=6):
        seen.append(slab.shape[0])
        return real(slab, sigma)

    monkeypatch.setattr(tools, "hessian_of_normalized", record)
    got = tools.detect_vesselness(volume, gauss_sigma=2, sigma=6, slab=8)

    assert np.array_equal(got, reference)
    assert seen
    assert max(seen) <= 8 + 4
    assert max(seen) < volume.shape[0]


def test_short_volume_uses_one_slab_and_float32_matches():
    rng = np.random.default_rng(1)
    short = rng.random((5, 8, 8))
    assert np.array_equal(
        tools.detect_vesselness(short, gauss_sigma=1, sigma=3, slab=64),
        _reference(short, gauss_sigma=1, sigma=3),
    )
    narrow = rng.random((20, 12, 12), dtype=np.float32)
    assert np.array_equal(
        tools.detect_vesselness(narrow, gauss_sigma=2, sigma=6, slab=7),
        _reference(narrow, gauss_sigma=2, sigma=6),
    )
