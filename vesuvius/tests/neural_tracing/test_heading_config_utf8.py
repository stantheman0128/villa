"""The heading-model config is UTF-8. A legacy text open must not drop it.

``read_config`` reads the JSON with ``Path.read_text()`` and no
encoding. On cp1252 a dataset path containing U+201D raises
``UnicodeDecodeError`` before defaults are applied. A UTF-8 BOM is
invalid JSON. An ASCII config still resolves its dataset path, and a
malformed file still raises.
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import types
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "vesuvius"
    / "neural_tracing"
    / "tracers"
    / "fibers"
    / "heading_model"
    / "train.py"
)
_DATASET = "scroll \u201d.json"


def _package(name: str) -> None:
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module


def _stub(name: str, **attrs: object) -> None:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module


def _reader():
    for name in (
        "vesuvius",
        "vesuvius.neural_tracing",
        "vesuvius.neural_tracing.tracers",
        "vesuvius.neural_tracing.tracers.fibers",
        "vesuvius.neural_tracing.tracers.fibers.heading_model",
        "vesuvius.neural_tracing.tracers.shared",
        "vesuvius.neural_tracing.tracers.shared.crop_sampling",
    ):
        _package(name)
    _stub(
        "vesuvius.neural_tracing.tracers.fibers.heading_model.data",
        load_sources=None,
        mixed_heading_states=None,
        sampling_from_dict=None,
        sampling_to_dict=None,
        validation_states=None,
    )
    _stub(
        "vesuvius.neural_tracing.tracers.fibers.heading_model.evaluate",
        evaluate_states=None,
        format_report=None,
        normal_summary_metric=None,
        frame_summary_metric=None,
        summary_metric=None,
    )
    _stub(
        "vesuvius.neural_tracing.tracers.fibers.heading_model.model",
        HeadingConfig=object,
        HeadingNet=object,
        save_heading_model=None,
    )
    _stub(
        "vesuvius.neural_tracing.tracers.fibers.heading_model.normals",
        normal_loss=None,
        normal_target_policy=None,
    )
    _stub(
        "vesuvius.neural_tracing.tracers.fibers.heading_model.frames",
        roll_supervision=None,
    )
    _stub(
        "vesuvius.neural_tracing.tracers.shared.crop_sampling.prefetch",
        attach_remote_prefetch=None,
    )
    _stub(
        "vesuvius.neural_tracing.tracers.shared.system",
        raise_open_file_limit=None,
    )
    spec = importlib.util.spec_from_file_location("heading_config_utf8", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.read_config


read_config = _reader()


def _open_text_as_cp1252(real_open):
    def wrapped(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, *args, **kwargs):
        if isinstance(mode, str) and "b" not in mode and encoding in (None, "locale"):
            encoding = "cp1252"
        return real_open(file, mode, buffering, encoding, errors, newline, *args, **kwargs)

    return wrapped


def _write(path: Path, dataset: str, bom: bool = False) -> None:
    raw = json.dumps({"dataset_config": dataset}, ensure_ascii=False).encode("utf-8")
    if bom:
        raw = b"\xef\xbb\xbf" + raw
    path.write_bytes(raw)


def test_legacy_locale_keeps_unicode_dataset(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    _write(path, _DATASET)
    monkeypatch.setattr(io, "open", _open_text_as_cp1252(io.open))

    config = read_config(path)

    assert config["dataset_config"].endswith(_DATASET)


def test_utf8_bom_keeps_unicode_dataset(tmp_path):
    path = tmp_path / "config.json"
    _write(path, _DATASET, bom=True)

    config = read_config(path)

    assert config["dataset_config"].endswith(_DATASET)


def test_ascii_dataset_still_resolved(tmp_path):
    path = tmp_path / "config.json"
    _write(path, "scroll.json")

    config = read_config(path)

    assert config["dataset_config"].endswith("scroll.json")
    assert config["name"] == "heading_model"


def test_malformed_config_still_raises(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        read_config(path)
