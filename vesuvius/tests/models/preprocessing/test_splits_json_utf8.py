"""The packer splits file is UTF-8. A legacy text open must not drop it.

``discover_samples`` opens ``splits_json`` with no encoding and no
handler. On cp1252 a sample name containing U+201D raises
``UnicodeDecodeError`` before any TIF is read. A UTF-8 BOM is invalid
JSON and raises ``JSONDecodeError``. An ASCII splits file still assigns
the split, and a malformed file still raises.
"""
from __future__ import annotations

import builtins
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "vesuvius"
    / "models"
    / "preprocessing"
    / "pack_tifs_to_sparse_zarr.py"
)
_spec = importlib.util.spec_from_file_location("pack_tifs_splits_utf8", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
discover_samples = _mod.discover_samples

_STEM = "scroll \u201d"


def _open_text_as_cp1252(real_open):
    def wrapped(file, mode="r", *args, **kwargs):
        if isinstance(mode, str) and "b" not in mode and "encoding" not in kwargs:
            kwargs = dict(kwargs)
            kwargs["encoding"] = "cp1252"
        return real_open(file, mode, *args, **kwargs)

    return wrapped


def _write_pair(source: Path, stem: str) -> None:
    images = source / "images"
    labels = source / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    image = np.zeros((4, 4), dtype=np.uint8)
    label = np.zeros((4, 4), dtype=np.uint8)
    label[0, 0] = 1
    tifffile.imwrite(images / f"{stem}.tif", image)
    tifffile.imwrite(labels / f"{stem}.tif", label)


def _write_splits(path: Path, stem: str, bom: bool = False) -> None:
    payload = json.dumps({"train": [stem], "val": []}, ensure_ascii=False).encode("utf-8")
    if bom:
        payload = b"\xef\xbb\xbf" + payload
    path.write_bytes(payload)


def test_legacy_locale_keeps_unicode_split(tmp_path, monkeypatch):
    _write_pair(tmp_path, _STEM)
    splits = tmp_path / "splits.json"
    _write_splits(splits, _STEM)
    monkeypatch.setattr(builtins, "open", _open_text_as_cp1252(builtins.open))

    samples = discover_samples(tmp_path, splits, num_workers=1)

    assert len(samples) == 1
    assert samples[0].split == "train"
    assert samples[0].filename == f"{_STEM}.tif"


def test_utf8_bom_keeps_unicode_split(tmp_path):
    _write_pair(tmp_path, _STEM)
    splits = tmp_path / "splits.json"
    _write_splits(splits, _STEM, bom=True)

    samples = discover_samples(tmp_path, splits, num_workers=1)

    assert len(samples) == 1
    assert samples[0].split == "train"


def test_ascii_split_still_assigned(tmp_path):
    _write_pair(tmp_path, "scroll")
    splits = tmp_path / "splits.json"
    _write_splits(splits, "scroll")

    samples = discover_samples(tmp_path, splits, num_workers=1)

    assert len(samples) == 1
    assert samples[0].split == "train"


def test_malformed_splits_still_raises(tmp_path):
    _write_pair(tmp_path, "scroll")
    splits = tmp_path / "splits.json"
    splits.write_text("{", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        discover_samples(tmp_path, splits, num_workers=1)
