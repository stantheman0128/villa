"""validate_spiral_supervision reads manifest.json with the process locale.

Under cp1252, U+201D in source_cache raises UnicodeDecodeError before the
artifact type is checked, and a UTF-8 BOM raises JSONDecodeError.
Related to #1524.
"""
import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "src/vesuvius/neural_tracing/winding_models/export_spiral_supervision.py"
spec = importlib.util.spec_from_file_location("spiral_manifest_utf8_under_test", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def document(source_cache="plain/cache", artifact_type=None, version=None):
    manifest = {
        "artifact_type": mod.FORMAT_NAME if artifact_type is None else artifact_type,
        "format_version": mod.FORMAT_VERSION if version is None else version,
        "source_cache": source_cache,
        "num_rays": 0,
        "num_crossings": 0,
        "shards": [],
    }
    manifest["fingerprint"] = mod._canonical_digest({
        key: value for key, value in manifest.items() if key != "fingerprint"
    })
    return manifest


def cp1252_text(monkeypatch):
    real = io.open

    def fake(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, *args, **kwargs):
        if "b" not in str(mode) and encoding in (None, "locale"):
            encoding = "cp1252"
        return real(file, mode, buffering, encoding, errors, newline, *args, **kwargs)

    monkeypatch.setattr(io, "open", fake)


def store(tmp_path, payload, bom=False):
    root = tmp_path / "store"
    root.mkdir()
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if bom:
        raw = b"\xef\xbb\xbf" + raw
    (root / "manifest.json").write_bytes(raw)
    return root


def test_unicode_source_cache_is_read_under_cp1252(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    source = "scroll \u201d/cache"
    summary = mod.validate_spiral_supervision(store(tmp_path, document(source_cache=source)))
    assert summary["num_rays"] == 0
    assert summary["num_crossings"] == 0
    assert summary["fingerprint"]


def test_utf8_bom_manifest_is_read(tmp_path):
    summary = mod.validate_spiral_supervision(store(tmp_path, document(), bom=True))
    assert summary["num_rays"] == 0
    assert summary["num_crossings"] == 0


def test_malformed_manifest_still_raises(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    root = tmp_path / "store"
    root.mkdir()
    (root / "manifest.json").write_text("{", encoding="ascii")
    with pytest.raises(json.JSONDecodeError):
        mod.validate_spiral_supervision(root)


def test_wrong_artifact_type_still_raises(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    payload = document(artifact_type="other")
    with pytest.raises(ValueError, match="not a compact winding supervision store"):
        mod.validate_spiral_supervision(store(tmp_path, payload))


def test_ascii_manifest_is_validated(tmp_path):
    summary = mod.validate_spiral_supervision(store(tmp_path, document()))
    assert summary["num_rays"] == 0
    assert summary["num_crossings"] == 0
    assert summary["path"].endswith("store")
