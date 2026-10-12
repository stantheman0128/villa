"""load_patch_metas reads its cache and each patch meta.json with the process locale.

Under cp1252, U+201D in the cached patches directory raises UnicodeDecodeError
before the cache is used, and a UTF-8 BOM raises JSONDecodeError. The same
byte in a patch meta.json is swallowed as ValueError, so the patch is dropped
before its bbox is checked.
Related to #1524.
"""
import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "src/vesuvius/neural_tracing/winding_models/merge_winding_field.py"
spec = importlib.util.spec_from_file_location("merge_patch_metas_utf8_under_test", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

BAND = mod.BandGeometry((128, 64, 64), 0, 128)


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


def cache(scratch, patches_dir, names, bom=False):
    scratch.mkdir(parents=True, exist_ok=True)
    write_json(
        scratch / "patch_metas.json",
        {"patches_dir": patches_dir, "names": names},
        bom=bom,
    )


def patch_dir(root, name, bbox, label="plain"):
    entry = root / name
    entry.mkdir(parents=True)
    write_json(entry / "meta.json", {"name": label, "bbox": bbox})
    return entry


def test_unicode_cache_path_is_read_under_cp1252(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    path = "scroll \u201d/patches"
    scratch = tmp_path / "scratch"
    cache(scratch, path, ["p0"])
    assert mod.load_patch_metas(path, BAND, scratch) == ["p0"]


def test_utf8_bom_cache_is_read(tmp_path):
    scratch = tmp_path / "scratch"
    cache(scratch, "plain/patches", ["p0"], bom=True)
    assert mod.load_patch_metas("plain/patches", BAND, scratch) == ["p0"]


def test_unicode_patch_meta_is_kept(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    patches = tmp_path / "patches"
    patch_dir(patches, "p0", [[0, 0, 10], [1, 1, 20]], label="scroll \u201d")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    assert mod.load_patch_metas(patches, BAND, scratch) == ["p0"]


def test_malformed_cache_still_raises(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "patch_metas.json").write_text("{", encoding="ascii")
    with pytest.raises(json.JSONDecodeError):
        mod.load_patch_metas("plain/patches", BAND, scratch)


def test_patch_outside_the_band_is_dropped(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    patches = tmp_path / "patches"
    patch_dir(patches, "far", [[0, 0, 1000], [1, 1, 1100]], label="scroll \u201d")
    patch_dir(patches, "bad", [[0, 0, 10], [1, 1, 20]])
    (patches / "bad" / "meta.json").write_text("{", encoding="ascii")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    assert mod.load_patch_metas(patches, BAND, scratch) == []


def test_ascii_cache_returns_names(tmp_path):
    scratch = tmp_path / "scratch"
    cache(scratch, "plain/patches", ["p0", "p1"])
    assert mod.load_patch_metas("plain/patches", BAND, scratch) == ["p0", "p1"]
