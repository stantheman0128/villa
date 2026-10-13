"""ResizeTask._get_resolutions reads OME-Zarr .zattrs with the process locale.

Under cp1252, U+201D in a multiscale dataset path raises UnicodeDecodeError
before that path is returned, and a UTF-8 BOM raises JSONDecodeError. The
path should still be returned. A missing store is still reported. Malformed
JSON is still reported. An ASCII dataset path is still returned. A store
without .zattrs still falls back to numeric level directories.
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
    module.__package__ = name
    sys.modules[name] = module
    return module


for _name in (
    "vesuvius",
    "vesuvius.image_proc",
    "vesuvius.image_proc.run",
    "vesuvius.image_proc.run.zarr_tasks",
    "vesuvius.image_proc.run.zarr_tasks.tasks",
    "vesuvius.image_proc.run.zarr_tasks.base",
    "vesuvius.image_proc.run.zarr_tasks.registry",
    "cv2",
):
    _package(_name)


class _TaskConfig:
    pass


class _ZarrTask:
    pass


def _register_task(_name):
    def decorator(cls):
        return cls

    return decorator


sys.modules["vesuvius.image_proc.run.zarr_tasks.base"].TaskConfig = _TaskConfig
sys.modules["vesuvius.image_proc.run.zarr_tasks.base"].ZarrTask = _ZarrTask
sys.modules["vesuvius.image_proc.run.zarr_tasks.base"].make_task_config = object
sys.modules["vesuvius.image_proc.run.zarr_tasks.registry"].register_task = _register_task

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "src/vesuvius/image_proc/run/zarr_tasks/tasks/resize.py"
spec = importlib.util.spec_from_file_location(
    "vesuvius.image_proc.run.zarr_tasks.tasks.resize_zattrs_utf8_under_test",
    SCRIPT,
)
mod = importlib.util.module_from_spec(spec)
mod.__package__ = "vesuvius.image_proc.run.zarr_tasks.tasks"
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def cp1252_text(monkeypatch):
    real = builtins.open

    def fake(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
        if "b" not in str(mode) and encoding in (None, "locale"):
            encoding = "cp1252"
        return real(file, mode, buffering, encoding, errors, newline, closefd, opener)

    monkeypatch.setattr(builtins, "open", fake)


def stub_zarr_open(monkeypatch):
    def fake_open(path, mode="r"):
        return object()

    monkeypatch.setattr(mod.zarr, "open", fake_open)


def write_json(path, document, *, bom=False, raw=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = raw if raw is not None else json.dumps(document, ensure_ascii=False).encode("utf-8")
    if bom:
        payload = b"\xef\xbb\xbf" + payload
    path.write_bytes(payload)


def attrs(path):
    return {"multiscales": [{"datasets": [{"path": path}]}]}


def reader():
    return mod.ResizeTask.__new__(mod.ResizeTask)


def test_unicode_dataset_path_is_returned(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    stub_zarr_open(monkeypatch)
    store = tmp_path / "volume.zarr"
    write_json(store / ".zattrs", attrs("scroll \u201d"))
    assert reader()._get_resolutions(str(store)) == ["scroll \u201d"]


def test_bom_dataset_path_is_returned(tmp_path, monkeypatch):
    stub_zarr_open(monkeypatch)
    store = tmp_path / "volume.zarr"
    write_json(store / ".zattrs", attrs("scroll \u201d"), bom=True)
    assert reader()._get_resolutions(str(store)) == ["scroll \u201d"]


def test_ascii_dataset_path_is_returned(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    stub_zarr_open(monkeypatch)
    store = tmp_path / "volume.zarr"
    write_json(store / ".zattrs", attrs("0"))
    assert reader()._get_resolutions(str(store)) == ["0"]


def test_missing_zattrs_uses_numeric_levels(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    stub_zarr_open(monkeypatch)
    store = tmp_path / "volume.zarr"
    (store / "0").mkdir(parents=True)
    assert reader()._get_resolutions(str(store)) == ["0"]


def test_malformed_zattrs_raises(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    stub_zarr_open(monkeypatch)
    store = tmp_path / "volume.zarr"
    write_json(store / ".zattrs", None, raw=b"{")
    with pytest.raises(json.JSONDecodeError):
        reader()._get_resolutions(str(store))


def test_missing_store_raises(tmp_path, monkeypatch):
    stub_zarr_open(monkeypatch)
    with pytest.raises(FileNotFoundError):
        reader()._get_resolutions(str(tmp_path / "missing.zarr"))
