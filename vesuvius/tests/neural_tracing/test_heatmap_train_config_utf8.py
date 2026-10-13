"""The heatmap trainer reads its JSON config with the process locale.

Under cp1252, U+201D in out_dir raises UnicodeDecodeError before that
directory is created, and a UTF-8 BOM raises JSONDecodeError. The
directory should still be created. A missing file is still reported.
Malformed JSON is still reported. A missing out_dir is still reported.
An ASCII out_dir is still created. The seed check still runs after that.
Related to #1524.
"""
import builtins
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PKG = SRC / "vesuvius"


def _namespace(name, path):
    module = types.ModuleType(name)
    module.__path__ = [str(path)] if path is not None else []
    module.__package__ = name
    sys.modules[name] = module
    return module


def _stub(name, **attrs):
    module = types.ModuleType(name)
    module.__package__ = name.rpartition(".")[0]
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


_namespace("vesuvius", PKG)
_namespace("vesuvius.neural_tracing", PKG / "neural_tracing")
_namespace("vesuvius.neural_tracing.heatmap_single_point", PKG / "neural_tracing" / "heatmap_single_point")
_namespace("vesuvius.neural_tracing.loss", PKG / "neural_tracing" / "loss")
_namespace("vesuvius.neural_tracing.nets", PKG / "neural_tracing" / "nets")
_namespace("vesuvius.models", PKG / "models")
_namespace("vesuvius.models.training", PKG / "models" / "training")
_namespace("vesuvius.models.training.loss", PKG / "models" / "training" / "loss")
_stub("wandb")
_stub("accelerate").Accelerator = object
_stub("einops").rearrange = object
_stub(
    "vesuvius.neural_tracing.heatmap_single_point.dataset",
    HeatmapDatasetV2=object,
    load_datasets=object,
    make_heatmaps=object,
)
_stub(
    "vesuvius.neural_tracing.heatmap_single_point.cropping",
    safe_crop_with_padding=object,
    transform_to_first_crop_space=object,
)
_stub(
    "vesuvius.neural_tracing.heatmap_single_point.visualization",
    make_canvas=object,
    print_training_config=object,
)
_stub(
    "vesuvius.neural_tracing.loss.deep_supervision",
    _resize_for_ds=object,
    _compute_ds_weights=object,
)
_stub(
    "vesuvius.neural_tracing.nets.models",
    make_model=object,
    strip_state=object,
    resolve_checkpoint_path=object,
)
_stub(
    "vesuvius.models.training.loss.nnunet_losses",
    DeepSupervisionWrapper=object,
    MemoryEfficientSoftDiceLoss=object,
)
_stub("vesuvius.models.training.loss.losses", CosineSimilarityLoss=object)
_stub("vesuvius.models.training.optimizers", create_optimizer=object)
_stub("vesuvius.models.training.lr_schedulers", get_scheduler=object)

SCRIPT = PKG / "neural_tracing/heatmap_single_point/train.py"
spec = importlib.util.spec_from_file_location(
    "vesuvius.neural_tracing.heatmap_train_config_utf8_under_test",
    SCRIPT,
)
mod = importlib.util.module_from_spec(spec)
mod.__package__ = "vesuvius.neural_tracing.heatmap_single_point"
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def cp1252_text(monkeypatch):
    real = builtins.open

    def fake(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
        if "b" not in str(mode) and encoding in (None, "locale"):
            encoding = "cp1252"
        return real(file, mode, buffering, encoding, errors, newline, closefd, opener)

    monkeypatch.setattr(builtins, "open", fake)


def write_config(directory, document, *, bom=False):
    payload = json.dumps(document, ensure_ascii=False).encode("utf-8")
    if bom:
        payload = b"\xef\xbb\xbf" + payload
    path = directory / "config.json"
    path.write_bytes(payload)
    return path


def run_train(path):
    mod.train.callback(str(path))


def test_unicode_out_dir_is_created(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    monkeypatch.chdir(tmp_path)
    path = write_config(tmp_path, {"out_dir": "scroll \u201d"})
    with pytest.raises(KeyError, match="seed") as caught:
        run_train(path)
    assert not isinstance(caught.value, UnicodeDecodeError)
    assert (tmp_path / "scroll \u201d").is_dir()


def test_bom_out_dir_is_created(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = write_config(tmp_path, {"out_dir": "scroll \u201d"}, bom=True)
    with pytest.raises(KeyError, match="seed") as caught:
        run_train(path)
    assert (tmp_path / "scroll \u201d").is_dir()
    assert "seed" in str(caught.value)


def test_ascii_out_dir_still_reaches_seed(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    monkeypatch.chdir(tmp_path)
    path = write_config(tmp_path, {"out_dir": "runs"})
    with pytest.raises(KeyError, match="seed") as caught:
        run_train(path)
    assert not isinstance(caught.value, UnicodeDecodeError)
    assert (tmp_path / "runs").is_dir()


def test_missing_out_dir_is_reported(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    path = write_config(tmp_path, {"seed": 1})
    with pytest.raises(KeyError, match="out_dir") as caught:
        run_train(path)
    assert not isinstance(caught.value, UnicodeDecodeError)


def test_malformed_config_is_reported(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    path = tmp_path / "config.json"
    path.write_bytes(b"{")
    with pytest.raises(json.JSONDecodeError):
        run_train(path)


def test_missing_config_is_reported(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    with pytest.raises(FileNotFoundError):
        run_train(tmp_path / "missing.json")
