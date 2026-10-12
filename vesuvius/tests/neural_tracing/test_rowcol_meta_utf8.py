"""Row-column inference reads tifxyz metadata with the process locale.

Under cp1252, U+201D in the uuid raises UnicodeDecodeError before the
scale or the voxel size is used, and a UTF-8 BOM raises JSONDecodeError.
The document should still be kept.
Related to #1524.
"""
import builtins
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

sys.modules.setdefault("vc", types.ModuleType("vc"))

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
SCRIPT = ROOT / "src/vesuvius/neural_tracing/inference/infer_rowcol_triplet_wraps.py"
spec = importlib.util.spec_from_file_location("rowcol_meta_utf8_under_test", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def write_meta(directory, payload, bom=False):
    directory.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if bom:
        raw = b"\xef\xbb\xbf" + raw
    (directory / "meta.json").write_bytes(raw)


def cp1252_text(monkeypatch):
    real = builtins.open

    def fake(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, *args, **kwargs):
        if "b" not in str(mode) and encoding in (None, "locale"):
            encoding = "cp1252"
        return real(file, mode, buffering, encoding, errors, newline, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake)


def document(uuid):
    return {"uuid": uuid, "scale": [0.05, 0.1], "voxelsize": 7.91, "format": "tifxyz"}


def test_unicode_uuid_keeps_the_scale(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    write_meta(tmp_path, document("scroll \u201d"))
    assert mod._read_tifxyz_scale_from_meta(tmp_path) == (0.05, 0.1)


def test_unicode_uuid_keeps_the_voxel_size(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    write_meta(tmp_path, document("scroll \u201d"))
    assert mod._read_volume_voxel_size_um(tmp_path) == 7.91


def test_utf8_bom_keeps_the_scale(tmp_path):
    write_meta(tmp_path, document("plain"), bom=True)
    assert mod._read_tifxyz_scale_from_meta(tmp_path) == (0.05, 0.1)


def test_utf8_bom_keeps_the_voxel_size(tmp_path):
    write_meta(tmp_path, document("plain"), bom=True)
    assert mod._read_volume_voxel_size_um(tmp_path) == 7.91


def test_missing_tifxyz_meta_still_raises(tmp_path):
    with pytest.raises(RuntimeError, match="missing meta.json"):
        mod._read_tifxyz_scale_from_meta(tmp_path)


def test_missing_volume_meta_still_returns_none(tmp_path):
    assert mod._read_volume_voxel_size_um(tmp_path) is None


def test_invalid_scale_still_raises(tmp_path):
    write_meta(tmp_path, {"uuid": "plain", "scale": [0, 1]})
    with pytest.raises(RuntimeError, match="Invalid"):
        mod._read_tifxyz_scale_from_meta(tmp_path)


def test_absent_voxel_size_still_returns_none(tmp_path):
    write_meta(tmp_path, {"uuid": "plain", "scale": [1, 1]})
    assert mod._read_volume_voxel_size_um(tmp_path) is None


def test_none_tifxyz_path_still_returns_none():
    assert mod._read_tifxyz_scale_from_meta(None) is None


def test_malformed_meta_still_raises(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    (tmp_path / "meta.json").write_bytes(b"{")
    with pytest.raises(json.JSONDecodeError):
        mod._read_tifxyz_scale_from_meta(tmp_path)


def test_ascii_metadata_is_kept(tmp_path):
    write_meta(tmp_path, document("plain"))
    assert mod._read_tifxyz_scale_from_meta(tmp_path) == (0.05, 0.1)
    assert mod._read_volume_voxel_size_um(tmp_path) == 7.91
