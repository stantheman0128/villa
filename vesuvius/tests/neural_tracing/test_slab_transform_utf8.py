"""A winding slab loads its registration transform with the process locale.

Under cp1252, U+201D in the transform name raises UnicodeDecodeError
before the matrix is used, and a UTF-8 BOM raises JSONDecodeError. The
matrix should still be returned. A missing file is still reported. A
matrix that is not 3x4 is still reported. Skipping registration still
returns the scale-only transform.
Related to #1524.
"""
import importlib.util
import json
import sys
import types
from pathlib import Path

import pathlib
import numpy as np
import pytest

vc = types.ModuleType("vc")
vc.Volume = type("Volume", (), {})
sys.modules["vc"] = vc

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "src/vesuvius/neural_tracing/winding_models/volume_slab_extractor.py"
spec = importlib.util.spec_from_file_location("slab_transform_utf8_under_test", SCRIPT)
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


def document(name="scroll", shift=4.0):
    return {
        "name": name,
        "transformation_matrix": [
            [2, 0, 0, shift],
            [0, 2, 0, 0],
            [0, 0, 2, 0],
        ],
    }


def load(path, **kwargs):
    return mod.VolumeSlabExtractor.load_segment_to_volume_transform(path, 0, **kwargs)


def test_unicode_name_keeps_the_matrix(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    write_json(tmp_path / "transform.json", document("scroll \u201d"))
    transform = load(tmp_path)
    assert transform.shape == (4, 4)
    assert transform[0, 0] == pytest.approx(0.5)
    assert transform[0, 3] == pytest.approx(-2.0)


def test_utf8_bom_keeps_the_matrix(tmp_path):
    write_json(tmp_path / "transform.json", document("scroll"), bom=True)
    transform = load(tmp_path)
    assert transform[0, 3] == pytest.approx(-2.0)


def test_ascii_matrix_is_kept(tmp_path):
    write_json(tmp_path / "transform.json", document("scroll", shift=8.0))
    transform = load(tmp_path)
    assert transform[0, 0] == pytest.approx(0.5)
    assert transform[0, 3] == pytest.approx(-4.0)


def test_missing_file_still_raises(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        load(tmp_path)


def test_bad_shape_still_raises(tmp_path):
    write_json(tmp_path / "transform.json", {"name": "scroll", "transformation_matrix": [[1, 0], [0, 1]]})
    with pytest.raises(ValueError, match="3x4"):
        load(tmp_path)


def test_malformed_json_still_raises(tmp_path):
    write_json(tmp_path / "transform.json", None, raw=b"{")
    with pytest.raises(json.JSONDecodeError):
        load(tmp_path)


def test_skipped_registration_ignores_the_file(tmp_path):
    write_json(tmp_path / "transform.json", None, raw=b"{")
    transform = load(tmp_path, use_registration=False)
    assert np.allclose(transform, np.eye(4))


def test_nonpositive_downscale_still_raises(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        load(tmp_path, segment_downscale=0)
