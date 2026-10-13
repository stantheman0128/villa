"""The winding trainer reads a base_config JSON with the process locale.

Under cp1252, U+201D in the base file's validation_mode raises
UnicodeDecodeError before that mode is checked, and a UTF-8 BOM raises
JSONDecodeError. The mode should still be checked. A missing base file is
still reported. Malformed base JSON is still reported. An unknown ASCII
mode in the base file is still rejected. train_batch with val_datasets in
the base file is still rejected. An overlay validation_mode still replaces
the base value. The primary config open and the config.json write are
unchanged. Related to #1524.
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
    "vesuvius.models",
    "vesuvius.models.training",
    "vesuvius.neural_tracing",
    "vesuvius.neural_tracing.nets",
    "vesuvius.neural_tracing.winding_models",
    "accelerate",
    "wandb",
):
    _package(_name)

sys.modules["vesuvius.models.training.lr_schedulers"] = _package(
    "vesuvius.models.training.lr_schedulers"
)
sys.modules["vesuvius.models.training.lr_schedulers"].get_scheduler = object
sys.modules["vesuvius.models.training.optimizers"] = _package(
    "vesuvius.models.training.optimizers"
)
sys.modules["vesuvius.models.training.optimizers"].create_optimizer = object
sys.modules["vesuvius.neural_tracing.nets.models"] = _package(
    "vesuvius.neural_tracing.nets.models"
)
sys.modules["vesuvius.neural_tracing.nets.models"].strip_state = object

_winding = sys.modules["vesuvius.neural_tracing.winding_models"]
_targets = _package("vesuvius.neural_tracing.winding_models.winding_targets")
_winding.winding_targets = _targets
_model = _package("vesuvius.neural_tracing.winding_models.winding_model")
_model.WindingModel = object
_dataset = _package("vesuvius.neural_tracing.winding_models.winding_model_dataset")
_dataset.WindingModelDataset = object
_viz = _package("vesuvius.neural_tracing.winding_models.winding_visualization")
_viz.make_winding_visualization = object

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "src/vesuvius/neural_tracing/winding_models/train_winding_model.py"
spec = importlib.util.spec_from_file_location(
    "vesuvius.neural_tracing.winding_models.train_winding_base_config_utf8_under_test",
    SCRIPT,
)
mod = importlib.util.module_from_spec(spec)
mod.__package__ = "vesuvius.neural_tracing.winding_models"
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def cp1252_text(monkeypatch):
    real_builtin = builtins.open

    def fake_builtin(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
        if "b" not in str(mode) and encoding in (None, "locale"):
            encoding = "cp1252"
        return real_builtin(file, mode, buffering, encoding, errors, newline, closefd, opener)

    monkeypatch.setattr(builtins, "open", fake_builtin)

    real_path = Path.open

    def fake_path(self, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
        if "b" not in str(mode) and encoding in (None, "locale"):
            encoding = "cp1252"
        return real_path(self, mode, buffering, encoding, errors, newline)

    monkeypatch.setattr(Path, "open", fake_path)


def write_json(path, document, *, bom=False):
    payload = json.dumps(document, ensure_ascii=False).encode("utf-8")
    if bom:
        payload = b"\xef\xbb\xbf" + payload
    path.write_bytes(payload)


def write_overlay(path, out, extra=None):
    document = {"base_config": "base.json", "out_dir": str(out)}
    if extra:
        document.update(extra)
    write_json(path, document)


def write_base(directory, mode, *, bom=False, extra=None):
    document = {"validation_mode": mode}
    if extra:
        document.update(extra)
    write_json(directory / "base.json", document, bom=bom)


def invoke(path):
    mod.train.callback(str(path))


def saved_mode(out):
    return json.loads((out / "config.json").read_text(encoding="utf-8"))["validation_mode"]


def test_unicode_base_validation_mode_is_checked(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    out = tmp_path / "run"
    overlay = tmp_path / "config.json"
    write_overlay(overlay, out)
    write_base(tmp_path, "scroll \u201d")
    with pytest.raises(ValueError) as caught:
        invoke(overlay)
    assert not isinstance(caught.value, UnicodeDecodeError)
    assert "validation_mode must be either" in str(caught.value)
    assert saved_mode(out) == "scroll \u201d"


def test_bom_base_validation_mode_is_checked(tmp_path):
    out = tmp_path / "run"
    overlay = tmp_path / "config.json"
    write_overlay(overlay, out)
    write_base(tmp_path, "scroll \u201d", bom=True)
    with pytest.raises(ValueError) as caught:
        invoke(overlay)
    assert "validation_mode must be either" in str(caught.value)
    assert saved_mode(out) == "scroll \u201d"


def test_unknown_ascii_base_mode_is_rejected(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    out = tmp_path / "run"
    overlay = tmp_path / "config.json"
    write_overlay(overlay, out)
    write_base(tmp_path, "dust")
    with pytest.raises(ValueError, match="validation_mode must be either") as caught:
        invoke(overlay)
    assert not isinstance(caught.value, UnicodeDecodeError)
    assert saved_mode(out) == "dust"


def test_train_batch_in_base_with_val_datasets_is_rejected(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    out = tmp_path / "run"
    overlay = tmp_path / "config.json"
    write_overlay(overlay, out)
    write_base(tmp_path, "train_batch", extra={"val_datasets": []})
    with pytest.raises(ValueError, match="train_batch") as caught:
        invoke(overlay)
    assert not isinstance(caught.value, UnicodeDecodeError)
    assert saved_mode(out) == "train_batch"


def test_overlay_mode_replaces_base_mode(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    out = tmp_path / "run"
    overlay = tmp_path / "config.json"
    write_overlay(overlay, out, extra={"validation_mode": "dust"})
    write_base(tmp_path, "separate")
    with pytest.raises(ValueError, match="validation_mode must be either") as caught:
        invoke(overlay)
    assert not isinstance(caught.value, UnicodeDecodeError)
    assert saved_mode(out) == "dust"


def test_malformed_base_json_raises(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    overlay = tmp_path / "config.json"
    write_overlay(overlay, tmp_path / "run")
    (tmp_path / "base.json").write_bytes(b"{")
    with pytest.raises(json.JSONDecodeError):
        invoke(overlay)


def test_missing_base_file_raises(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    overlay = tmp_path / "config.json"
    write_overlay(overlay, tmp_path / "run")
    with pytest.raises(FileNotFoundError):
        invoke(overlay)
