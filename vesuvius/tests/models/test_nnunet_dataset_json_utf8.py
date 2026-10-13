"""load_json reads dataset.json and plans.json with the process locale.

Under cp1252, U+201D in a label name raises UnicodeDecodeError before the
labels are returned, and a UTF-8 BOM raises JSONDecodeError. The label
name should still be returned. A missing file is still reported. Malformed
JSON is still reported. An ASCII label name is still kept. A plans
configuration name is still kept. The Python source reader in the same
file already passes encoding="utf-8" and is unchanged.
Related to #1524.
"""
import builtins
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


def _package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
    return module


for _name in (
    "nnunetv2",
    "nnunetv2.utilities",
    "nnunetv2.utilities.label_handling",
    "nnunetv2.utilities.label_handling.label_handling",
    "nnunetv2.utilities.find_class_by_name",
    "nnunetv2.utilities.get_network_from_plans",
    "nnunetv2.utilities.plans_handling",
    "nnunetv2.utilities.plans_handling.plans_handler",
):
    _package(_name)
sys.modules["nnunetv2.utilities.label_handling.label_handling"].determine_num_input_channels = object
sys.modules["nnunetv2.utilities.find_class_by_name"].recursive_find_python_class = object
sys.modules["nnunetv2.utilities.get_network_from_plans"].get_network_from_plans = object
sys.modules["nnunetv2.utilities.plans_handling.plans_handler"].PlansManager = object

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "src/vesuvius/utils/models/load_nnunet_model.py"
spec = importlib.util.spec_from_file_location("nnunet_dataset_json_utf8_under_test", SCRIPT)
mod = importlib.util.module_from_spec(spec)
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


def dataset(name="ink"):
    return {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, name: 1},
        "file_ending": ".nii.gz",
    }


def test_unicode_label_is_returned(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    path = tmp_path / "dataset.json"
    write_json(path, dataset("scroll \u201d"))
    loaded = mod.load_json(path)
    assert loaded["labels"]["scroll \u201d"] == 1
    assert loaded["labels"]["background"] == 0
    assert loaded["channel_names"]["0"] == "CT"


def test_utf8_bom_keeps_the_label(tmp_path):
    path = tmp_path / "dataset.json"
    write_json(path, dataset("ink"), bom=True)
    loaded = mod.load_json(path)
    assert loaded["labels"]["ink"] == 1
    assert loaded["labels"]["background"] == 0
    assert loaded["channel_names"]["0"] == "CT"


def test_ascii_label_is_kept(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    path = tmp_path / "dataset.json"
    write_json(path, dataset("ink"))
    loaded = mod.load_json(path)
    assert loaded["labels"]["ink"] == 1
    assert loaded["channel_names"]["0"] == "CT"
    assert loaded["file_ending"] == ".nii.gz"


def test_plans_configuration_name_is_kept(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    path = tmp_path / "plans.json"
    write_json(path, {"dataset_name": "Dataset001", "configurations": {"3d_fullres": {"batch_size": 2}}})
    loaded = mod.load_json(path)
    assert loaded["dataset_name"] == "Dataset001"
    assert loaded["configurations"]["3d_fullres"]["batch_size"] == 2


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        mod.load_json(tmp_path / "dataset.json")


def test_malformed_json_still_raises(tmp_path):
    path = tmp_path / "dataset.json"
    write_json(path, None, raw=b"{")
    with pytest.raises(json.JSONDecodeError):
        mod.load_json(path)
