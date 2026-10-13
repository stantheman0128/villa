"""Tifxyz.get_patches_3d reads patches_cache.json with the process locale.

Under cp1252, U+201D in the cache raises UnicodeDecodeError before the
cached patches are returned, and a UTF-8 BOM is treated as a corrupt
cache and recomputed. The cached patches should still be returned. A
missing cache is still recomputed. Malformed JSON is still recomputed.
An ASCII cache is still returned. An in-memory cache still wins.
Related to #1524.
"""
import builtins
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
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
_namespace("vesuvius.tifxyz", PKG / "tifxyz")
_stub("cv2")
_stub("vesuvius.tifxyz_canvas", full_resolution_extent=lambda stored, scale: int(stored))

TILER_CALLS = []


def _tiler(*args, **kwargs):
    TILER_CALLS.append((args, kwargs))
    raise RuntimeError("tiler")


_stub("vesuvius.tifxyz.hierarchical_tiling", multipass_hierarchical_tiling=_tiler)

SCRIPT = PKG / "tifxyz" / "types.py"
spec = importlib.util.spec_from_file_location(
    "vesuvius.tifxyz.patches_cache_utf8_under_test",
    SCRIPT,
)
mod = importlib.util.module_from_spec(spec)
mod.__package__ = "vesuvius.tifxyz"
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


CACHE_KEY = "8,8,8,1.0,1.0"
PATCHES = [((0, 2, 0, 2), (1.0, 2.0, 3.0, 4.0, 5.0, 6.0))]


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
    path = directory / "patches_cache.json"
    path.write_bytes(payload)
    return path, payload


def surface(directory):
    grid = np.zeros((4, 4), dtype=np.float32)
    return mod.Tifxyz(
        _x=grid,
        _y=grid.copy(),
        _z=grid.copy(),
        path=directory,
        _scale=(1.0, 1.0),
    )


def test_unicode_cache_returns_patches(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    TILER_CALLS.clear()
    _path, payload = write_cache(
        tmp_path,
        {CACHE_KEY: [[[0, 2, 0, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]], "note": "scroll \u201d"},
    )
    found = surface(tmp_path).get_patches_3d((8, 8, 8))
    assert found == PATCHES
    assert payload[payload.index(b"\x9d")] == 0x9D
    assert TILER_CALLS == []


def test_bom_cache_returns_patches(tmp_path):
    TILER_CALLS.clear()
    write_cache(
        tmp_path,
        {CACHE_KEY: [[[0, 2, 0, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]], "note": "scroll \u201d"},
        bom=True,
    )
    found = surface(tmp_path).get_patches_3d((8, 8, 8))
    assert found == PATCHES
    assert TILER_CALLS == []


def test_ascii_cache_returns_patches(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    TILER_CALLS.clear()
    write_cache(tmp_path, {CACHE_KEY: [[[0, 2, 0, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]]})
    found = surface(tmp_path).get_patches_3d((8, 8, 8))
    assert found == PATCHES
    assert TILER_CALLS == []


def test_malformed_cache_recomputes(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    TILER_CALLS.clear()
    (tmp_path / "patches_cache.json").write_bytes(b"{")
    with pytest.raises(RuntimeError, match="tiler"):
        surface(tmp_path).get_patches_3d((8, 8, 8))
    assert len(TILER_CALLS) == 1


def test_missing_cache_recomputes(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    TILER_CALLS.clear()
    with pytest.raises(RuntimeError, match="tiler"):
        surface(tmp_path).get_patches_3d((8, 8, 8))
    assert len(TILER_CALLS) == 1


def test_memory_cache_skips_disk(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    TILER_CALLS.clear()
    write_cache(
        tmp_path,
        {CACHE_KEY: [[[0, 2, 0, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]], "note": "scroll \u201d"},
    )
    stored = surface(tmp_path)
    sentinel = [((9, 9, 9, 9), (0.0,))]
    object.__setattr__(stored, "_patches_cache", sentinel)
    assert stored.get_patches_3d((8, 8, 8)) is sentinel
    assert TILER_CALLS == []
