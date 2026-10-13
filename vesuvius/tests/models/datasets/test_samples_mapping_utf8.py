"""_load_from_mapping reads samples_mapping.json with the process locale.

Under cp1252, U+201D in a target name raises UnicodeDecodeError before the
name is compared with the configured targets, and a UTF-8 BOM raises
JSONDecodeError. The target name should still be compared. A missing file
is still reported. Malformed JSON is still reported. An ASCII target name
that is configured does not warn. An unknown ASCII target name still warns.
Related to #1524.
"""
import builtins
import importlib.util
import json
import logging
import sys
import types
from pathlib import Path

import pytest


def _package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    module.__package__ = name
    sys.modules[name] = module
    return module


for _name in (
    "vesuvius",
    "vesuvius.utils",
    "vesuvius.utils.utils",
    "vesuvius.models",
    "vesuvius.models.datasets",
    "vesuvius.models.datasets.intensity_properties",
    "vesuvius.models.training",
    "vesuvius.models.training.normalization",
    "vesuvius.models.augmentation",
    "vesuvius.models.augmentation.pipelines",
    "vesuvius.models.augmentation.transforms",
    "vesuvius.models.augmentation.transforms.utils",
    "vesuvius.models.augmentation.transforms.utils.perf",
):
    _package(_name)
sys.modules["vesuvius.utils.utils"].pad_or_crop_3d = object
sys.modules["vesuvius.utils.utils"].pad_or_crop_2d = object
sys.modules["vesuvius.models.datasets.intensity_properties"].initialize_intensity_properties = object
sys.modules["vesuvius.models.training.normalization"].get_normalization = object
sys.modules["vesuvius.models.augmentation.pipelines"].create_training_transforms = object
sys.modules["vesuvius.models.augmentation.transforms.utils.perf"].collect_augmentation_names = object

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "src/vesuvius/models/datasets/zarr_dataset.py"
spec = importlib.util.spec_from_file_location(
    "vesuvius.models.datasets.zarr_dataset_utf8_under_test",
    SCRIPT,
)
mod = importlib.util.module_from_spec(spec)
mod.__package__ = "vesuvius.models.datasets"
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def cp1252_text(monkeypatch):
    real = builtins.open

    def fake(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
        if "b" not in str(mode) and encoding in (None, "locale"):
            encoding = "cp1252"
        return real(file, mode, buffering, encoding, errors, newline, closefd, opener)

    monkeypatch.setattr(builtins, "open", fake)


def write_json(path, document, *, bom=False, raw=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = raw if raw is not None else json.dumps(document, ensure_ascii=False).encode("utf-8")
    if bom:
        payload = b"\xef\xbb\xbf" + payload
    path.write_bytes(payload)


def dataset():
    reader = mod.ZarrDataset.__new__(mod.ZarrDataset)
    reader.target_names = ["ink"]
    reader.skip_patch_validation = True
    reader.enumerated = False

    def enumerate_all():
        reader.enumerated = True

    reader._enumerate_all_patches = enumerate_all
    return reader


def mapping(name):
    return {"target_name": name, "layout": {}, "samples": []}


def test_unicode_target_name_is_compared(tmp_path, monkeypatch, caplog):
    cp1252_text(monkeypatch)
    path = tmp_path / "samples_mapping.json"
    write_json(path, mapping("scroll \u201d"))
    reader = dataset()
    with caplog.at_level(logging.WARNING):
        reader._load_from_mapping(path)
    assert "scroll \u201d" in caplog.text
    assert reader.enumerated is True


def test_bom_target_name_is_compared(tmp_path, caplog):
    path = tmp_path / "samples_mapping.json"
    write_json(path, mapping("scroll \u201d"), bom=True)
    reader = dataset()
    with caplog.at_level(logging.WARNING):
        reader._load_from_mapping(path)
    assert "scroll \u201d" in caplog.text
    assert reader.enumerated is True


def test_ascii_configured_target_does_not_warn(tmp_path, monkeypatch, caplog):
    cp1252_text(monkeypatch)
    path = tmp_path / "samples_mapping.json"
    write_json(path, mapping("ink"))
    reader = dataset()
    with caplog.at_level(logging.WARNING):
        reader._load_from_mapping(path)
    assert "not in config targets" not in caplog.text
    assert reader.enumerated is True


def test_unknown_ascii_target_warns(tmp_path, monkeypatch, caplog):
    cp1252_text(monkeypatch)
    path = tmp_path / "samples_mapping.json"
    write_json(path, mapping("dust"))
    reader = dataset()
    with caplog.at_level(logging.WARNING):
        reader._load_from_mapping(path)
    assert "dust" in caplog.text
    assert reader.enumerated is True


def test_missing_file_raises(tmp_path):
    reader = dataset()
    with pytest.raises(FileNotFoundError):
        reader._load_from_mapping(tmp_path / "missing.json")
    assert reader.enumerated is False


def test_malformed_json_raises(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    path = tmp_path / "samples_mapping.json"
    write_json(path, None, raw=b"{")
    reader = dataset()
    with pytest.raises(json.JSONDecodeError):
        reader._load_from_mapping(path)
    assert reader.enumerated is False
