"""rasterize_patches reads the patch-grid stamp with the process locale.

Under cp1252, U+201D in the cached patches directory raises
UnicodeDecodeError before the stamp is compared, and a UTF-8 BOM raises
JSONDecodeError. A matching stamp should reload the saved grid.
Related to #1524.
"""
import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "src/vesuvius/neural_tracing/winding_models/merge_winding_field.py"
spec = importlib.util.spec_from_file_location("patch_grid_stamp_utf8_under_test", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

BAND = mod.BandGeometry((128, 64, 64), 0, 128)
SHAPE = BAND.band_shape(4)


class Cfg:
    patch_erode = 0
    patch_subdiv = 1
    patch_dilate = 0


def stamp_for(patches_dir, names):
    return {
        "patches_dir": str(patches_dir),
        "erode": Cfg.patch_erode,
        "subdiv": Cfg.patch_subdiv,
        "dilate": Cfg.patch_dilate,
        "band": [BAND.z_lo, BAND.z_hi],
        "patches": len(names),
    }


def write_json(path, payload, bom=False):
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if bom:
        raw = b"\xef\xbb\xbf" + raw
    path.write_bytes(raw)


def cp1252_text(monkeypatch):
    real = io.open

    def fake(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, *args, **kwargs):
        if "b" not in str(mode) and encoding in (None, "locale"):
            encoding = "cp1252"
        return real(file, mode, buffering, encoding, errors, newline, *args, **kwargs)

    monkeypatch.setattr(io, "open", fake)


def saved_grid(scratch, patches_dir, names, fill=7, bom=False, payload=None):
    scratch.mkdir(parents=True, exist_ok=True)
    grid = np.memmap(scratch / "patch_grid_ds4.dat", dtype=np.int32, mode="w+", shape=SHAPE)
    grid[:] = fill
    grid.flush()
    del grid
    np.savez(scratch / "patch_vertices.npz", xyz=np.zeros((0, 3), np.float32))
    write_json(scratch / "patch_grid_params.json", payload if payload is not None else stamp_for(patches_dir, names), bom=bom)


def test_unicode_stamp_reloads_the_grid(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    path = "scroll \u201d/patches"
    scratch = tmp_path / "scratch"
    saved_grid(scratch, path, ["p0"])
    grid = mod.rasterize_patches(["p0"], path, BAND, Cfg(), "cpu", scratch)
    assert int(grid.cpu().numpy()[0, 0, 0]) == 7


def test_utf8_bom_stamp_reloads_the_grid(tmp_path):
    scratch = tmp_path / "scratch"
    saved_grid(scratch, "plain/patches", ["p0"], bom=True)
    grid = mod.rasterize_patches(["p0"], "plain/patches", BAND, Cfg(), "cpu", scratch)
    assert int(grid.cpu().numpy()[0, 0, 0]) == 7


def test_malformed_stamp_still_raises(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    scratch = tmp_path / "scratch"
    saved_grid(scratch, "plain/patches", ["p0"], payload={"patches_dir": "plain/patches"})
    (scratch / "patch_grid_params.json").write_text("{", encoding="ascii")
    with pytest.raises(json.JSONDecodeError):
        mod.rasterize_patches(["p0"], "plain/patches", BAND, Cfg(), "cpu", scratch)


def test_different_stamp_still_rebuilds(tmp_path):
    scratch = tmp_path / "scratch"
    saved_grid(scratch, "plain/patches", ["p0"], fill=7, payload=stamp_for("plain/patches", ["p0", "extra"]))
    grid = mod.rasterize_patches([], "plain/patches", BAND, Cfg(), "cpu", scratch)
    assert int(grid.cpu().numpy().reshape(-1)[0]) == -1


def test_ascii_stamp_reloads_the_grid(tmp_path):
    scratch = tmp_path / "scratch"
    saved_grid(scratch, "plain/patches", ["p0"], fill=3)
    grid = mod.rasterize_patches(["p0"], "plain/patches", BAND, Cfg(), "cpu", scratch)
    assert int(grid.cpu().numpy()[0, 0, 0]) == 3
