"""A winding patch pack reads names.json with the process locale.

Under cp1252, U+201D in a patch directory name raises
UnicodeDecodeError before the pack is built, and a UTF-8 BOM raises
JSONDecodeError. The document should still be kept. An empty name list
still builds a pack, and a missing file still raises.
Related to #1524.
"""
import importlib.util
import io
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

vc = types.ModuleType("vc")
grid = types.ModuleType("vc.grid_raycast")
grid.GridRaycaster = type("GridRaycaster", (), {})
vc.grid_raycast = grid
sys.modules.setdefault("vc", vc)
sys.modules.setdefault("vc.grid_raycast", grid)
skimage = types.ModuleType("skimage")
restoration = types.ModuleType("skimage.restoration")
restoration.unwrap_phase = lambda *_args, **_kwargs: None
skimage.restoration = restoration
sys.modules.setdefault("skimage", skimage)
sys.modules.setdefault("skimage.restoration", restoration)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
SCRIPT = ROOT / "src/vesuvius/neural_tracing/winding_models/winding_model_dataset.py"
spec = importlib.util.spec_from_file_location("patch_pack_names_utf8_under_test", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def cp1252_text(monkeypatch):
    real = io.open

    def fake(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, *args, **kwargs):
        if "b" not in str(mode) and encoding in (None, "locale"):
            encoding = "cp1252"
        return real(file, mode, buffering, encoding, errors, newline, *args, **kwargs)

    monkeypatch.setattr(io, "open", fake)


def write_pack(directory, names, bom=False, raw=None):
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "xyz.npy", np.zeros((4, 3), np.float32))
    np.save(directory / "quads.npy", np.array([True]))
    np.save(directory / "block_min.npy", np.zeros((1, 3), np.float32))
    np.save(directory / "block_max.npy", np.ones((1, 3), np.float32))
    np.savez(
        directory / "index.npz",
        shapes=np.array([[2, 2]], np.int32),
        vertex_offsets=np.array([0, 4], np.int64),
        quad_offsets=np.array([0, 1], np.int64),
        block_shapes=np.array([[1, 1]], np.int32),
        block_offsets=np.array([0, 1], np.int64),
        quad_counts=np.array([1], np.int64),
        bbox_min=np.zeros((1, 3), np.float64),
        bbox_max=np.ones((1, 3), np.float64),
    )
    if raw is None and names is None:
        return
    payload = raw if raw is not None else json.dumps(names, ensure_ascii=False).encode("utf-8")
    if bom:
        payload = b"\xef\xbb\xbf" + payload
    (directory / "names.json").write_bytes(payload)


def read_names(directory):
    return mod.WindingModelDataset._read_patch_pack(directory).names


def test_unicode_patch_name_is_kept(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    write_pack(tmp_path, ["scroll \u201d"])
    assert read_names(tmp_path) == ["scroll \u201d"]


def test_utf8_bom_is_kept(tmp_path):
    write_pack(tmp_path, ["plain"], bom=True)
    assert read_names(tmp_path) == ["plain"]


def test_ascii_name_is_kept(tmp_path):
    write_pack(tmp_path, ["plain"])
    assert read_names(tmp_path) == ["plain"]


def test_empty_name_list_still_builds(tmp_path):
    write_pack(tmp_path, [])
    assert read_names(tmp_path) == []


def test_malformed_names_still_raise(tmp_path):
    write_pack(tmp_path, None, raw=b"{")
    with pytest.raises(json.JSONDecodeError):
        read_names(tmp_path)


def test_missing_names_still_raise(tmp_path):
    write_pack(tmp_path, None)
    with pytest.raises(FileNotFoundError):
        read_names(tmp_path)
