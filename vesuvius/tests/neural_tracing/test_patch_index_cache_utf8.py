"""The sheet patch index reads its cache with the process locale.

Under cp1252, U+201D in a cached patch name raises UnicodeDecodeError
before the index is returned, and a UTF-8 BOM raises JSONDecodeError.
The cached name should still be kept. A malformed cache still raises.
An empty cache still returns no patches. The meta.json scan is unchanged.
Related to #1524.
"""
import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
SCRIPT = ROOT / "src/vesuvius/neural_tracing/tracers/sheets/data/patches.py"
spec = importlib.util.spec_from_file_location("patch_index_cache_utf8_under_test", SCRIPT)
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


def cache_path(patches, cache_dir):
    names = sorted(p.name for p in patches.iterdir() if (p / "x.tif").is_file())
    key = hashlib.sha256(
        json.dumps([mod.INDEX_VERSION, str(patches.resolve()), names]).encode()
    ).hexdigest()[:16]
    return cache_dir / f"patch_index_{key}.json"


def write_cache(patches, cache_dir, rows, *, bom=False, raw=None):
    patches.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(patches, cache_dir)
    payload = raw if raw is not None else json.dumps(rows, ensure_ascii=False).encode("utf-8")
    if bom:
        payload = b"\xef\xbb\xbf" + payload
    path.write_bytes(payload)


def row(name, root):
    return {
        "name": name,
        "rows": 2,
        "cols": 3,
        "cell": 1.0,
        "valid": 4,
        "bbox": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "root": root,
        "stated_cell": 0.0,
    }


def test_unicode_cached_name_is_kept(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    patches, cache_dir = tmp_path / "patches", tmp_path / "cache"
    write_cache(patches, cache_dir, [row("scroll \u201d", str(patches))])
    index = mod.build_index(patches, cache_dir)
    assert index[0].name == "scroll \u201d"
    assert index[0].bbox == (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)


def test_utf8_bom_is_kept(tmp_path):
    patches, cache_dir = tmp_path / "patches", tmp_path / "cache"
    write_cache(patches, cache_dir, [row("plain", str(patches))], bom=True)
    assert mod.build_index(patches, cache_dir)[0].name == "plain"


def test_ascii_cached_name_is_kept(tmp_path):
    patches, cache_dir = tmp_path / "patches", tmp_path / "cache"
    write_cache(patches, cache_dir, [row("plain", str(patches))])
    assert mod.build_index(patches, cache_dir)[0].name == "plain"


def test_empty_cache_still_returns_no_patches(tmp_path):
    patches, cache_dir = tmp_path / "patches", tmp_path / "cache"
    write_cache(patches, cache_dir, [])
    assert mod.build_index(patches, cache_dir) == []


def test_malformed_cache_still_raises(tmp_path):
    patches, cache_dir = tmp_path / "patches", tmp_path / "cache"
    write_cache(patches, cache_dir, None, raw=b"{")
    with pytest.raises(json.JSONDecodeError):
        mod.build_index(patches, cache_dir)
