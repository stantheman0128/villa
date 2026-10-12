"""Positive full_3d dilation must run on CPU when cuCIM is absent.

apply_label_dilation is what training calls for every batch. On a CPU
tensor the cuCIM helper raises, so a non-zero dilation distance cannot
be used. The CPU path uses the edt package that is already a dependency.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
import scipy.ndimage as ndi
import torch

import vesuvius.ink_detection.training.dilation as dilation_module
from vesuvius.ink_detection.training.dilation import apply_label_dilation


def _center_batch(shape=(1, 1, 5, 5, 5), dtype=torch.float32):
    labels = torch.zeros(*shape, dtype=dtype)
    labels[(0, 0) + tuple(size // 2 for size in shape[2:])] = 1
    return {
        "inklabels": labels,
        "supervision_mask": torch.zeros_like(labels),
    }


def test_cpu_positive_dilation_fills_orthogonal_neighbors():
    output = apply_label_dilation(_center_batch(), 1.0, 0.0)
    ink = output["inklabels"][0, 0]
    assert int(ink.sum()) == 7
    assert ink[2, 2, 2] == 1
    assert ink[2, 2, 3] == 1
    assert ink[2, 3, 3] == 0
    assert int(output["supervision_mask"].sum()) == 7


def test_cpu_fallback_warns_once(caplog):
    dilation_module._fallback_announced = False
    batch = _center_batch()
    with caplog.at_level(logging.WARNING):
        apply_label_dilation(batch, 1.0, 0.0)
        apply_label_dilation(
            {
                "inklabels": batch["inklabels"].clone(),
                "supervision_mask": batch["supervision_mask"].clone(),
            },
            1.0,
            0.0,
        )
    messages = [record.message for record in caplog.records if record.levelno >= logging.WARNING]
    assert len(messages) == 1
    assert "edt" in messages[0]


def test_validity_mask_blocks_dilation():
    labels = torch.zeros(1, 1, 5, 5, 5)
    labels[0, 0, 2, 2, 2] = 1
    valid = torch.ones(1, 1, 5, 5, 5)
    valid[0, 0, 2, 2, 3] = 0
    output = dilation_module.dilate_label_batch_with_edt(labels, valid, 1.0)
    assert output[0, 0, 2, 2, 2] == 1
    assert output[0, 0, 2, 2, 3] == 0
    assert output[0, 0, 2, 2, 1] == 1
    assert int(output.sum()) == 6


def test_edt_matches_scipy_away_from_the_threshold():
    rng = np.random.default_rng(0)
    labels_np = (rng.random((2, 1, 8, 8, 8)) < 0.05).astype(np.float32)
    valid_np = (rng.random((2, 1, 8, 8, 8)) < 0.85).astype(np.float32)
    output = dilation_module.dilate_label_batch_with_edt(
        torch.from_numpy(labels_np.copy()),
        torch.from_numpy(valid_np.copy()),
        1.0,
    ).numpy()
    for batch in range(labels_np.shape[0]):
        source = (labels_np[batch, 0] == 1) & (valid_np[batch, 0] > 0)
        distance = ndi.distance_transform_edt(~source)
        close = (labels_np[batch, 0] == 0) & (valid_np[batch, 0] > 0) & (distance < 0.99)
        far = (labels_np[batch, 0] == 0) & (distance > 1.01)
        assert np.all(output[batch, 0][close] == 1)
        assert np.all(output[batch, 0][far] == 0)
        assert np.all(output[batch, 0][labels_np[batch, 0] == 1] == 1)


def test_bfloat16_round_trip_keeps_dtype():
    output = apply_label_dilation(_center_batch(dtype=torch.bfloat16), 1.0, 0.0)
    assert output["inklabels"].dtype == torch.bfloat16
    assert int(output["inklabels"].sum()) == 7


def test_empty_labels_stay_empty():
    labels = torch.zeros(1, 1, 4, 4, 4)
    output = apply_label_dilation(
        {"inklabels": labels, "supervision_mask": torch.zeros_like(labels)},
        2.0,
        0.0,
    )
    assert int(output["inklabels"].sum()) == 0
