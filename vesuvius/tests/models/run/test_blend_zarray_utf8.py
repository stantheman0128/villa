"""SpatialPatchGrid reads .zarray with the process locale.

Under cp1252, U+201D in zarr_format raises UnicodeDecodeError before that
value is checked, and a UTF-8 BOM raises JSONDecodeError. The format
should still be checked. A missing file is still reported. Malformed JSON
is still reported. An ASCII format is still rejected. A later chunks check
still runs. The S3 open, the cache reads, and the cache write are
unchanged. Related to #1524.
"""
import builtins
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[3] / "src"
PKG = SRC / "vesuvius"


def _namespace(name, path):
    module = types.ModuleType(name)
    module.__path__ = [str(path)] if path is not None else []
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
_namespace("vesuvius.models", PKG / "models")
_namespace("vesuvius.models.run", PKG / "models" / "run")
_namespace("vesuvius.data", PKG / "data")
_namespace("vesuvius.utils", PKG / "utils")
_namespace("vesuvius.utils.io", PKG / "utils" / "io")
_stub("fsspec")
_stub("numcodecs")
_stub("scipy")
_stub("scipy.ndimage").gaussian_filter = object
_namespace("tqdm", None)
_stub("tqdm.auto").tqdm = object
_stub("vesuvius.data.utils").open_zarr = object
_stub("vesuvius.utils.io.zarr_utils").wait_for_zarr_creation = object
_stub("vesuvius.utils.k8s").get_tqdm_kwargs = object
_stub("vesuvius.utils.cli").HyphenUnderscoreParser = object

SCRIPT = PKG / "models/run/blending.py"
spec = importlib.util.spec_from_file_location(
    "vesuvius.models.run.blend_zarray_utf8_under_test",
    SCRIPT,
)
mod = importlib.util.module_from_spec(spec)
mod.__package__ = "vesuvius.models.run"
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def cp1252_text(monkeypatch):
    real = builtins.open

    def fake(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
        if "b" not in str(mode) and encoding in (None, "locale"):
            encoding = "cp1252"
        return real(file, mode, buffering, encoding, errors, newline, closefd, opener)

    monkeypatch.setattr(builtins, "open", fake)


def write_zarray(directory, document, *, bom=False):
    payload = json.dumps(document, ensure_ascii=False).encode("utf-8")
    if bom:
        payload = b"\xef\xbb\xbf" + payload
    (directory / ".zarray").write_bytes(payload)


def validate(directory):
    mod.SpatialPatchGrid((8, 8, 8))._validate_zarr_format(str(directory))


def test_unicode_zarr_format_is_checked(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    write_zarray(tmp_path, {"zarr_format": "scroll \u201d"})
    with pytest.raises(ValueError) as caught:
        validate(tmp_path)
    assert not isinstance(caught.value, UnicodeDecodeError)
    assert "Expected zarr_format=2" in str(caught.value)
    assert "scroll \u201d" in str(caught.value)


def test_bom_zarr_format_is_checked(tmp_path):
    write_zarray(tmp_path, {"zarr_format": "scroll \u201d"}, bom=True)
    with pytest.raises(ValueError) as caught:
        validate(tmp_path)
    assert "Expected zarr_format=2" in str(caught.value)
    assert "scroll \u201d" in str(caught.value)


def test_ascii_format_is_rejected(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    write_zarray(tmp_path, {"zarr_format": 3, "chunks": [1, 8, 8, 8, 8]})
    with pytest.raises(ValueError, match="got 3") as caught:
        validate(tmp_path)
    assert not isinstance(caught.value, UnicodeDecodeError)


def test_ascii_chunk_layout_is_still_checked(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    write_zarray(tmp_path, {"zarr_format": 2, "chunks": [2, 8, 8, 8, 8]})
    with pytest.raises(ValueError, match="chunks\\[0\\]=1") as caught:
        validate(tmp_path)
    assert not isinstance(caught.value, UnicodeDecodeError)


def test_malformed_json_raises(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    (tmp_path / ".zarray").write_bytes(b"{")
    with pytest.raises(json.JSONDecodeError):
        validate(tmp_path)


def test_missing_file_raises(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    with pytest.raises(FileNotFoundError):
        validate(tmp_path)
