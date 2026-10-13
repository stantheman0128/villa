"""ChunkedArray loads .zarray with the process locale.

Under cp1252, U+201D in the dimension separator raises UnicodeDecodeError
before the separator is used, and a UTF-8 BOM raises JSONDecodeError. The
separator and shape should still be kept. A missing file is still reported.
Malformed JSON is still reported.
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
    "vesuvius.neural_tracing.tracers.shared.io",
    "vesuvius.neural_tracing.tracers.shared.retired",
):
    _package(_name)
sys.modules["vesuvius.neural_tracing.tracers.shared.io"].atomic_write = lambda *args, **kwargs: None
sys.modules["vesuvius.neural_tracing.tracers.shared.retired"].ANY = object()
sys.modules["vesuvius.neural_tracing.tracers.shared.retired"].retire = lambda *args, **kwargs: None

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "src/vesuvius/neural_tracing/tracers/shared/crop_sampling/volume.py"
spec = importlib.util.spec_from_file_location("chunked_zarray_utf8_under_test", SCRIPT)
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
        "compressor": None,
        "fill_value": 0,
        "dimension_separator": separator,
    }


def load(path):
    return mod.ChunkedArray(path, cache_bytes=64)


def test_unicode_separator_keeps_the_shape(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    write_json(tmp_path / ".zarray", document("scroll \u201d"))
    array = load(tmp_path)
    assert array.sep == "scroll \u201d"
    assert array.shape == (8, 4, 2)
    assert array.codec is None


def test_utf8_bom_keeps_the_shape(tmp_path):
    write_json(tmp_path / ".zarray", document("/"), bom=True)
    array = load(tmp_path)
    assert array.sep == "/"
    assert array.shape == (8, 4, 2)


def test_ascii_separator_is_kept(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    write_json(tmp_path / ".zarray", document("."))
    array = load(tmp_path)
    assert array.sep == "."
    assert array.chunks == (8, 4, 2)
    assert array.dtype.str == "|u1"


def test_missing_file_still_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load(tmp_path)


def test_malformed_json_still_raises(tmp_path):
    write_json(tmp_path / ".zarray", None, raw=b"{")
    with pytest.raises(json.JSONDecodeError):
        load(tmp_path)


def test_bad_dtype_still_raises(tmp_path):
    payload = document(".")
    payload["dtype"] = "not-a-dtype"
    write_json(tmp_path / ".zarray", payload)
    with pytest.raises(TypeError):
        load(tmp_path)
