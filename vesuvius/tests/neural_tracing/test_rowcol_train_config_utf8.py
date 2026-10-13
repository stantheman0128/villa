"""The row/col trainer reads its JSON config with the process locale.

Under cp1252, U+201D in training_mode raises UnicodeDecodeError before that
mode is checked, and a UTF-8 BOM raises JSONDecodeError. The mode should
still be checked. A missing file is still reported. Malformed JSON is still
reported. An unknown ASCII mode is still rejected. A later numeric check
still runs after an accepted ASCII mode. This is not the profile
_load_config reader. Related to #1524.
"""
import builtins
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
_namespace("vesuvius.neural_tracing.trainers", PKG / "neural_tracing" / "trainers")
_namespace("vesuvius.neural_tracing.loss", PKG / "neural_tracing" / "loss")
_namespace("vesuvius.neural_tracing.nets", PKG / "neural_tracing" / "nets")
_namespace("vesuvius.models", PKG / "models")
_namespace("vesuvius.models.training", PKG / "models" / "training")

_stub("vesuvius.neural_tracing.datasets.dataset_rowcol_cond").EdtSegDataset = object
targets = _stub("vesuvius.neural_tracing.datasets.targets")
targets.CopyNeighborTargets = object
targets.RowColTargets = object
_stub("accelerate")
_stub("wandb")
displacement = _stub("vesuvius.neural_tracing.loss.displacement_losses")
displacement.dense_displacement_loss = object
displacement.displaced_source_edt_loss = object
displacement.smoothness_loss = object
displacement.triplet_min_displacement_loss = object
_stub("vesuvius.neural_tracing.loss.trace_losses").compute_trace_losses = object
_stub("vesuvius.models.training.optimizers").create_optimizer = object
_stub("vesuvius.models.training.lr_schedulers").get_scheduler = object
nets = _stub("vesuvius.neural_tracing.nets.models")
nets.checkpoint_wandb_run_id = object
nets.load_checkpoint = object
nets.load_checkpoint_payload = object
nets.make_model = object
nets.strip_state = object
_stub("vesuvius.neural_tracing.trainers.loss_config").TraceLossConfig = object
viz = _stub("vesuvius.neural_tracing.trainers.rowcol_cond_visualization")
viz.make_dense_visualization = object
viz.make_trace_visualization = object

SCRIPT = PKG / "neural_tracing/trainers/train_rowcol_cond.py"
spec = importlib.util.spec_from_file_location(
    "vesuvius.neural_tracing.trainers.train_rowcol_config_utf8_under_test",
    SCRIPT,
)
mod = importlib.util.module_from_spec(spec)
mod.__package__ = "vesuvius.neural_tracing.trainers"
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def cp1252_text(monkeypatch):
    real = builtins.open

    def fake(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
        if "b" not in str(mode) and encoding in (None, "locale"):
            encoding = "cp1252"
        return real(file, mode, buffering, encoding, errors, newline, closefd, opener)

    monkeypatch.setattr(builtins, "open", fake)


def write_config(path, mode, *, bom=False, extra=None):
    document = {"training_mode": mode}
    if extra:
        document.update(extra)
    payload = json.dumps(document, ensure_ascii=False).encode("utf-8")
    if bom:
        payload = b"\xef\xbb\xbf" + payload
    path.write_bytes(payload)


def invoke(path):
    mod.train.callback(str(path))


def test_unicode_training_mode_is_checked(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    path = tmp_path / "config.json"
    write_config(path, "scroll \u201d")
    with pytest.raises(ValueError) as caught:
        invoke(path)
    assert not isinstance(caught.value, UnicodeDecodeError)
    assert "training_mode must be" in str(caught.value)
    assert "scroll \u201d" in str(caught.value)


def test_bom_training_mode_is_checked(tmp_path):
    path = tmp_path / "config.json"
    write_config(path, "scroll \u201d", bom=True)
    with pytest.raises(ValueError) as caught:
        invoke(path)
    assert "training_mode must be" in str(caught.value)
    assert "scroll \u201d" in str(caught.value)


def test_unknown_ascii_mode_is_rejected(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    path = tmp_path / "config.json"
    write_config(path, "dust")
    with pytest.raises(ValueError, match="dust") as caught:
        invoke(path)
    assert not isinstance(caught.value, UnicodeDecodeError)
    assert "training_mode must be" in str(caught.value)


def test_accepted_ascii_mode_still_checks_numbers(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    path = tmp_path / "config.json"
    write_config(path, "copy_neighbors", extra={"trace_target_dilation_radius": "nope"})
    with pytest.raises(ValueError, match="could not convert string to float") as caught:
        invoke(path)
    assert not isinstance(caught.value, UnicodeDecodeError)


def test_malformed_json_raises(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    path = tmp_path / "config.json"
    path.write_bytes(b"{")
    with pytest.raises(json.JSONDecodeError):
        invoke(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        invoke(tmp_path / "missing.json")
