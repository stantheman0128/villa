"""SpatialPatchGrid reloads the aggregate bbox cache with the process locale.

Under cp1252, U+201D in a cached box raises UnicodeDecodeError before that
box is returned, and a UTF-8 BOM is treated as a missing cache. The box
should still be returned. A missing file still returns None. Malformed
JSON still returns None. An ASCII cache is still returned. A non-integer
key still returns None. The .zarray open and the non-empty cache open are
unchanged. Related to #1524.
"""
import builtins
import importlib.util
import json
import sys
import types
import warnings
from pathlib import Path


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
    "vesuvius.models.run.blend_bbox_cache_utf8_under_test",
    SCRIPT,
)
mod = importlib.util.module_from_spec(spec)
mod.__package__ = "vesuvius.models.run"
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


BOX = {
    "name": "scroll \u201d",
    "z_min": 1,
    "z_max": 2,
    "y_min": 3,
    "y_max": 4,
    "x_min": 5,
    "x_max": 6,
}
ASCII_BOX = {
    "z_min": 1,
    "z_max": 2,
    "y_min": 3,
    "y_max": 4,
    "x_min": 5,
    "x_max": 6,
}


def cp1252_text(monkeypatch):
    real = builtins.open

    def fake(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
        if "b" not in str(mode) and encoding in (None, "locale"):
            encoding = "cp1252"
        return real(file, mode, buffering, encoding, errors, newline, closefd, opener)

    monkeypatch.setattr(builtins, "open", fake)


def write_cache(directory, document, *, bom=False):
    payload = json.dumps(document, ensure_ascii=False).encode("utf-8")
    if bom:
        payload = b"\xef\xbb\xbf" + payload
    path = directory / ".bbox_cache.json"
    path.write_bytes(payload)
    return path, payload


def load(directory):
    return mod.SpatialPatchGrid((8, 8, 8))._load_aggregate_bbox_cache(str(directory))


def test_unicode_box_is_returned(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    _path, payload = write_cache(tmp_path, {"7": BOX})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = load(tmp_path)
    assert result == {7: BOX}, [str(item.message) for item in caught]
    assert payload[payload.index(b"\x9d")] == 0x9D
    assert not any("UnicodeDecodeError" in str(item.message) for item in caught)


def test_bom_box_is_returned(tmp_path):
    write_cache(tmp_path, {"7": BOX}, bom=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = load(tmp_path)
    assert result == {7: BOX}, [str(item.message) for item in caught]
    assert result[7]["name"] == "scroll \u201d"


def test_ascii_box_is_returned(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    write_cache(tmp_path, {"7": ASCII_BOX})
    assert load(tmp_path) == {7: ASCII_BOX}


def test_missing_cache_returns_none(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    assert load(tmp_path) is None


def test_malformed_cache_returns_none(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    (tmp_path / ".bbox_cache.json").write_bytes(b"{")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = load(tmp_path)
    assert result is None
    assert caught
    assert not any("UnicodeDecodeError" in str(item.message) for item in caught)


def test_non_integer_key_returns_none(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    write_cache(tmp_path, {"note": ASCII_BOX})
    assert load(tmp_path) is None
