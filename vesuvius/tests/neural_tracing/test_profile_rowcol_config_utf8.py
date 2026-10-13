"""_load_config reads a row/col dataset JSON with the process locale.

Under cp1252, U+201D in training_mode raises UnicodeDecodeError before that
mode is checked, and a UTF-8 BOM raises JSONDecodeError. The mode should
still be checked. A missing file is still reported. Malformed JSON is still
reported. An ASCII mode is still accepted. An unknown ASCII mode is still
rejected.
Related to #1524.
"""
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[2] / "src"
PKG = SRC / "vesuvius"


def _namespace(name, path):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    sys.modules[name] = module
    return module


def _stub(name):
    module = types.ModuleType(name)
    module.__path__ = []
    module.__package__ = name
    sys.modules[name] = module
    return module


_namespace("vesuvius", PKG)
_namespace("vesuvius.neural_tracing", PKG / "neural_tracing")
_namespace("vesuvius.neural_tracing.datasets", PKG / "neural_tracing" / "datasets")
_namespace("vesuvius.neural_tracing.scripts", PKG / "neural_tracing" / "scripts")
_stub("vesuvius.neural_tracing.datasets.dataset_rowcol_cond").EdtSegDataset = object
targets = _stub("vesuvius.neural_tracing.datasets.targets")
targets.CopyNeighborTargets = object
targets.RowColTargets = object

SCRIPT = PKG / "neural_tracing/scripts/profile_rowcol_cond_dataset.py"
spec = importlib.util.spec_from_file_location(
    "vesuvius.neural_tracing.scripts.profile_rowcol_config_utf8_under_test",
    SCRIPT,
)
mod = importlib.util.module_from_spec(spec)
mod.__package__ = "vesuvius.neural_tracing.scripts"
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def cp1252_text(monkeypatch):
    real = Path.open

    def fake(self, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
        if "b" not in str(mode) and encoding in (None, "locale"):
            encoding = "cp1252"
        return real(self, mode, buffering, encoding, errors, newline)

    monkeypatch.setattr(Path, "open", fake)


def write_config(path, mode, *, bom=False, extra=None):
    document = {"training_mode": mode}
    if extra:
        document.update(extra)
    payload = json.dumps(document, ensure_ascii=False).encode("utf-8")
    if bom:
        payload = b"\xef\xbb\xbf" + payload
    path.write_bytes(payload)


def checked(path):
    with pytest.raises(ValueError) as caught:
        mod._load_config(path, disable_force_recompute_patches=False)
    assert not isinstance(caught.value, UnicodeDecodeError)
    assert "scroll \u201d" in str(caught.value)


def test_unicode_training_mode_is_checked(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    path = tmp_path / "config.json"
    write_config(path, "scroll \u201d")
    checked(path)


def test_bom_training_mode_is_checked(tmp_path):
    path = tmp_path / "config.json"
    write_config(path, "scroll \u201d", bom=True)
    checked(path)


def test_ascii_mode_is_kept(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    path = tmp_path / "config.json"
    write_config(path, "copy_neighbors", extra={"force_recompute_patches": True})
    config = mod._load_config(path, disable_force_recompute_patches=True)
    assert config["training_mode"] == "copy_neighbors"
    assert config["force_recompute_patches"] is False


def test_unknown_ascii_mode_is_rejected(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    path = tmp_path / "config.json"
    write_config(path, "dust")
    with pytest.raises(ValueError, match="dust") as caught:
        mod._load_config(path, disable_force_recompute_patches=False)
    assert not isinstance(caught.value, UnicodeDecodeError)


def test_malformed_json_raises(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    path = tmp_path / "config.json"
    path.write_bytes(b"{")
    with pytest.raises(json.JSONDecodeError):
        mod._load_config(path, disable_force_recompute_patches=False)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        mod._load_config(tmp_path / "missing.json", disable_force_recompute_patches=False)
