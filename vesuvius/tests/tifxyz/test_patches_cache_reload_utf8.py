"""Tifxyz.get_patches_3d reloads patches_cache.json with the process locale.

Under cp1252, U+201D in the cache raises UnicodeDecodeError before a
recomputed entry is merged, and a UTF-8 BOM is treated as a corrupt
cache so the other entries are dropped. The other entries should still
be kept. A missing cache is still created. Malformed JSON is still
replaced. An ASCII cache still keeps its other entry. The first cache
read is unchanged.
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
NEW_PATCHES = [((1, 3, 1, 3), (9.0, 8.0, 7.0, 6.0, 5.0, 4.0))]
NEW_BOXES = [[[1, 3, 1, 3], [9.0, 8.0, 7.0, 6.0, 5.0, 4.0]]]
OLD_KEY = "1,1,1,1.0,1.0"
NEW_KEY = "8,8,8,1.0,1.0"
OLD_BOXES = [[[0, 2, 0, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]]


def _tiler(*args, **kwargs):
    TILER_CALLS.append(kwargs.get("target_size"))
    return list(NEW_PATCHES)


_stub("vesuvius.tifxyz.hierarchical_tiling", multipass_hierarchical_tiling=_tiler)

SCRIPT = PKG / "tifxyz" / "types.py"
spec = importlib.util.spec_from_file_location(
    "vesuvius.tifxyz.patches_cache_reload_utf8_under_test",
    SCRIPT,
)
mod = importlib.util.module_from_spec(spec)
mod.__package__ = "vesuvius.tifxyz"
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


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


def load_cache(path):
    return json.loads(path.read_bytes().decode("utf-8"))


def surface(directory):
    grid = np.zeros((4, 4), dtype=np.float32)
    return mod.Tifxyz(
        _x=grid,
        _y=grid.copy(),
        _z=grid.copy(),
        path=directory,
        _scale=(1.0, 1.0),
    )


def test_unicode_reload_keeps_other_entry(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    TILER_CALLS.clear()
    path, payload = write_cache(
        tmp_path,
        {OLD_KEY: OLD_BOXES, "note": "scroll \u201d"},
    )
    before = path.read_bytes()
    found = surface(tmp_path).get_patches_3d((8, 8, 8), force_recompute=True)
    assert found == NEW_PATCHES
    saved = load_cache(path)
    assert saved[OLD_KEY] == OLD_BOXES
    assert saved[NEW_KEY] == NEW_BOXES
    assert saved["note"] == "scroll \u201d"
    assert payload[payload.index(b"\x9d")] == 0x9D
    assert before != path.read_bytes()
    assert TILER_CALLS == [(8, 8, 8)]


def test_bom_reload_keeps_other_entry(tmp_path):
    TILER_CALLS.clear()
    write_cache(
        tmp_path,
        {OLD_KEY: OLD_BOXES, "note": "scroll \u201d"},
        bom=True,
    )
    found = surface(tmp_path).get_patches_3d((8, 8, 8), force_recompute=True)
    assert found == NEW_PATCHES
    saved = load_cache(tmp_path / "patches_cache.json")
    assert saved[OLD_KEY] == OLD_BOXES
    assert saved[NEW_KEY] == NEW_BOXES
    assert saved["note"] == "scroll \u201d"
    assert TILER_CALLS == [(8, 8, 8)]


def test_ascii_reload_keeps_other_entry(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    TILER_CALLS.clear()
    write_cache(tmp_path, {OLD_KEY: OLD_BOXES})
    found = surface(tmp_path).get_patches_3d((8, 8, 8), force_recompute=True)
    assert found == NEW_PATCHES
    saved = load_cache(tmp_path / "patches_cache.json")
    assert saved[OLD_KEY] == OLD_BOXES
    assert saved[NEW_KEY] == NEW_BOXES
    assert TILER_CALLS == [(8, 8, 8)]


def test_malformed_reload_replaces_cache(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    TILER_CALLS.clear()
    path = tmp_path / "patches_cache.json"
    path.write_bytes(b"{")
    found = surface(tmp_path).get_patches_3d((8, 8, 8), force_recompute=True)
    assert found == NEW_PATCHES
    assert load_cache(path) == {NEW_KEY: NEW_BOXES}
    assert TILER_CALLS == [(8, 8, 8)]


def test_missing_cache_writes_new_entry(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    TILER_CALLS.clear()
    found = surface(tmp_path).get_patches_3d((8, 8, 8), force_recompute=True)
    assert found == NEW_PATCHES
    assert load_cache(tmp_path / "patches_cache.json") == {NEW_KEY: NEW_BOXES}
    assert TILER_CALLS == [(8, 8, 8)]


def test_cache_hit_still_uses_locale(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    TILER_CALLS.clear()
    path, _payload = write_cache(
        tmp_path,
        {NEW_KEY: OLD_BOXES, "note": "scroll \u201d"},
    )
    before = path.read_bytes()
    with pytest.raises(UnicodeDecodeError) as caught:
        surface(tmp_path).get_patches_3d((8, 8, 8))
    assert caught.value.object[caught.value.start] == 0x9D
    assert path.read_bytes() == before
    assert TILER_CALLS == []
