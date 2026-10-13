"""A spiral supervision part reads its manifest with the process locale.

Under cp1252, U+201D in a shard name raises UnicodeDecodeError before
the cached part is reused, and a UTF-8 BOM raises JSONDecodeError. The
document should still be kept when its signature and arrays match. A
malformed file still raises. A different signature is still not reused.
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
sys.path.insert(0, str(ROOT / "src"))
SCRIPT = ROOT / "src/vesuvius/neural_tracing/winding_models/export_spiral_supervision.py"
spec = importlib.util.spec_from_file_location("spiral_part_manifest_utf8_under_test", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

ARRAY_NAMES = (
    "ray_origin_zyx",
    "ray_step_zyx",
    "crossing_offsets",
    "crossing_t",
    "crossing_level",
    "seed_winding",
)
SIGNATURE = {"marker": "current"}


def cp1252_text(monkeypatch):
    real = io.open

    def fake(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, *args, **kwargs):
        if "b" not in str(mode) and encoding in (None, "locale"):
            encoding = "cp1252"
        return real(file, mode, buffering, encoding, errors, newline, *args, **kwargs)

    monkeypatch.setattr(io, "open", fake)


def stub_source(monkeypatch, signature=SIGNATURE):
    monkeypatch.setattr(mod, "_get_export_group", lambda _path: None)
    monkeypatch.setattr(
        mod, "_source_signature", lambda *_args, **_kwargs: signature)


def write_part(partial, name, *, signature=SIGNATURE, bom=False, raw=None, valid=True):
    destination = partial / "parts" / name / "00000000_00000001"
    destination.mkdir(parents=True)
    arrays = {}
    if valid:
        for key in ARRAY_NAMES:
            arrays[key] = mod._save_array(destination, key, np.zeros(1, np.float32))
    if raw is None:
        document = {"name": name, "source_signature": signature, "arrays": arrays}
        payload = json.dumps(document, ensure_ascii=False).encode("utf-8")
    else:
        payload = raw
    if bom:
        payload = b"\xef\xbb\xbf" + payload
    (destination / "manifest.json").write_bytes(payload)


def export(partial, name):
    task = ("source.zarr", str(partial), {"name": name, "lo": 0, "hi": 1}, 0, 1, 8, 2)
    return mod._export_part(task)


def test_unicode_shard_name_is_reused(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    stub_source(monkeypatch)
    write_part(tmp_path, "scroll \u201d")
    assert export(tmp_path, "scroll \u201d")["name"] == "scroll \u201d"


def test_utf8_bom_is_reused(tmp_path, monkeypatch):
    stub_source(monkeypatch)
    write_part(tmp_path, "plain", bom=True)
    assert export(tmp_path, "plain")["name"] == "plain"


def test_ascii_name_is_reused(tmp_path, monkeypatch):
    stub_source(monkeypatch)
    write_part(tmp_path, "plain")
    assert export(tmp_path, "plain")["name"] == "plain"


def test_malformed_manifest_still_raises(tmp_path, monkeypatch):
    stub_source(monkeypatch)
    write_part(tmp_path, "plain", raw=b"{", valid=False)
    with pytest.raises(json.JSONDecodeError):
        export(tmp_path, "plain")


def test_different_signature_is_not_reused(tmp_path, monkeypatch):
    stub_source(monkeypatch, signature={"marker": "current"})
    write_part(tmp_path, "plain", signature={"marker": "stale"})
    with pytest.raises(TypeError):
        export(tmp_path, "plain")
