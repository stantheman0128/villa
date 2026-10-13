"""Prefetch reads .zarray with the process locale.

Under cp1252, U+201D in the dimension separator raises UnicodeDecodeError
before the separator is used, and a UTF-8 BOM raises JSONDecodeError. The
separator should still be returned. A missing file is still reported.
Malformed JSON is still reported. A file without a separator still
defaults to a dot.
Related to #1524.
"""
import importlib.util
import json
import sys
import types
from pathlib import Path

import pathlib
import pytest


def _package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
    return module


for _name in (
    "vesuvius",
    "vesuvius.neural_tracing",
    "vesuvius.neural_tracing.tracers",
    "vesuvius.neural_tracing.tracers.shared",
    "vesuvius.neural_tracing.tracers.shared.crop_sampling",
    "vesuvius.neural_tracing.tracers.shared.crop_sampling.volume",
    "vesuvius.neural_tracing.tracers.shared.io",
):
    _package(_name)
_volume = sys.modules["vesuvius.neural_tracing.tracers.shared.crop_sampling.volume"]
_volume.RemoteChunkedArray = object()
_volume.chunk_lock_path = lambda *args, **kwargs: None
_volume.chunk_slices = lambda *args, **kwargs: None
_volume.open_remote_array = lambda *args, **kwargs: None
_volume.padded_chunk = lambda *args, **kwargs: None
sys.modules["vesuvius.neural_tracing.tracers.shared.io"].atomic_write = lambda *args, **kwargs: None

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "src/vesuvius/neural_tracing/tracers/shared/crop_sampling/prefetch.py"
spec = importlib.util.spec_from_file_location("prefetch_zarray_utf8_under_test", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def cp1252_text(monkeypatch):
    real = pathlib.Path.open

    def fake(self, mode="r", buffering=-1, encoding=None, errors=None, newline=None, **kwargs):
        if "b" not in str(mode) and encoding in (None, "locale"):
            encoding = "cp1252"
        return real(self, mode, buffering, encoding, errors, newline, **kwargs)

    monkeypatch.setattr(pathlib.Path, "open", fake)


def write_json(path, document, *, bom=False, raw=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = raw if raw is not None else json.dumps(document, ensure_ascii=False).encode("utf-8")
    if bom:
        payload = b"\xef\xbb\xbf" + payload
    path.write_bytes(payload)


def document(separator="."):
    return {
        "shape": [8, 4, 2],
        "chunks": [8, 4, 2],
        "dtype": "|u1",
        "dimension_separator": separator,
    }


def load(path):
    return mod._chunk_separator(path)


def test_unicode_separator_is_returned(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    write_json(tmp_path / ".zarray", document("scroll \u201d"))
    assert load(tmp_path) == "scroll \u201d"


def test_utf8_bom_keeps_the_separator(tmp_path):
    write_json(tmp_path / ".zarray", document("/"), bom=True)
    assert load(tmp_path) == "/"


def test_ascii_separator_is_kept(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    write_json(tmp_path / ".zarray", document("."))
    assert load(tmp_path) == "."


def test_missing_separator_defaults_to_dot(tmp_path):
    payload = document(".")
    del payload["dimension_separator"]
    write_json(tmp_path / ".zarray", payload)
    assert load(tmp_path) == "."


def test_missing_file_still_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load(tmp_path)


def test_malformed_json_still_raises(tmp_path):
    write_json(tmp_path / ".zarray", None, raw=b"{")
    with pytest.raises(json.JSONDecodeError):
        load(tmp_path)
