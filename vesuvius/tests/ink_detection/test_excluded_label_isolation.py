"""Held-out ink must not select training patches or become a native target."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vesuvius.ink_detection.config import InkDataConfig
from vesuvius.ink_detection.data.geometry import project_labels_and_supervision
from vesuvius.ink_detection.data.normalization import isolate_excluded_labels
from vesuvius.ink_detection.data.patch_finding_default import (
    find_segment_patches as find_default_patches,
)
from vesuvius.ink_detection.data.patch_finding_subtiling import (
    find_segment_patches as find_subtiling_patches,
)
from vesuvius.ink_detection.types import Segment


def _config(tmp_path: Path, **overrides) -> InkDataConfig:
    authored = {
        "mode": "flat",
        "patch_size": [1, 4, 4],
        "patch_overlap": 1.0,
        "patch_min_labeled_coverage": 0.5,
        "image_normalization": "none",
        "out_dir": str(tmp_path),
        "dataloader_workers": 1,
        "datasets": [{"segments_path": str(tmp_path), "volume_scale": 0}],
    }
    authored.update(overrides)
    return InkDataConfig.from_mapping(authored)


def _segment(config: InkDataConfig, tmp_path: Path) -> Segment:
    return Segment(
        data_config=config,
        source=config.datasets[0],
        dataset_idx=0,
        segment_relpath="segment-a",
        segment_dir=tmp_path / "segment-a",
        segment_name="segment-a",
        image_volume="image",
        inklabels=Path("labels"),
        supervision_mask=Path("supervision"),
        validation_mask=Path("validation"),
    )


def _volumes(label_value: int):
    image = np.zeros((1, 4, 4), dtype=np.uint8)
    image[0] = 1
    labels = np.zeros_like(image)
    labels[0, 0, 0] = label_value
    labels[0, 3, 3] = label_value
    supervision = np.zeros_like(image)
    supervision[0] = 1
    validation = np.zeros_like(image)
    validation[0, 0, 0] = 1
    validation[0, 3, 3] = 1
    return {
        "image": image,
        "labels": labels,
        "supervision": supervision,
        "validation": validation,
    }


def test_default_discovery_ignores_held_out_ink(tmp_path):
    config = _config(tmp_path)
    segment = _segment(config, tmp_path)
    volumes = _volumes(1)
    training, held_out = find_default_patches(
        segment, lambda path, resolution: volumes[str(path)]
    )
    assert training == []
    assert len(held_out) == 1


def test_subtiling_discovery_ignores_held_out_ink(tmp_path):
    config = _config(
        tmp_path,
        patch_finding_type="subtiling",
        patch_finding_filter_empty_tile=True,
    )
    segment = _segment(config, tmp_path)
    volumes = _volumes(3)
    training, held_out = find_subtiling_patches(
        segment, lambda path, resolution: volumes[str(path)]
    )
    assert training == []
    assert len(held_out) == 1


def test_training_ink_outside_the_holdout_still_selects_a_patch(tmp_path):
    config = _config(tmp_path, patch_min_labeled_coverage=0.05)
    segment = _segment(config, tmp_path)
    volumes = _volumes(1)
    volumes["labels"][0, 1, 1] = 1
    training, held_out = find_default_patches(
        segment, lambda path, resolution: volumes[str(path)]
    )
    assert len(training) == 1
    assert len(held_out) == 1


def test_excluded_ink_is_not_projected_as_supervision():
    labels = np.array([[1]], dtype=np.uint8)
    cleared_supervision = np.array([[0]], dtype=np.uint8)
    validation = np.array([[1]], dtype=np.uint8)
    leaked, leaked_supervision = project_labels_and_supervision(
        positions_zyx=np.zeros((1, 1, 3), dtype=np.float32),
        valid_mask=np.ones((1, 1), dtype=bool),
        inklabels_flat=labels,
        supervision_flat=cleared_supervision,
        crop_bbox_zyx=(0, 0, 0, 1, 1, 1),
        normals_zyx=np.zeros((1, 1, 3), dtype=np.float32),
        label_half_thickness=0.0,
        background_half_thickness=0.0,
    )
    assert leaked_supervision[0, 0, 0] == 1
    assert leaked[0, 0, 0] == 1

    isolated = isolate_excluded_labels(
        labels,
        supervision=cleared_supervision,
        validation=validation,
        restrict_to_supervision=False,
    )
    kept, kept_supervision = project_labels_and_supervision(
        positions_zyx=np.zeros((1, 1, 3), dtype=np.float32),
        valid_mask=np.ones((1, 1), dtype=bool),
        inklabels_flat=isolated,
        supervision_flat=cleared_supervision,
        crop_bbox_zyx=(0, 0, 0, 1, 1, 1),
        normals_zyx=np.zeros((1, 1, 3), dtype=np.float32),
        label_half_thickness=0.0,
        background_half_thickness=0.0,
    )
    assert int(kept_supervision[0, 0, 0]) == 0
    assert int(kept[0, 0, 0]) == 0

    restricted = isolate_excluded_labels(
        labels,
        supervision=cleared_supervision,
        validation=None,
        restrict_to_supervision=True,
    )
    assert int(restricted[0, 0]) == 0
