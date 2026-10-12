"""The OME-Zarr patch cache is UTF-8. A legacy text open must not drop it.

``load_patch_cache`` reads the cache with ``Path.open`` and no encoding.
On cp1252 a volume name containing U+201D raises ``UnicodeDecodeError``
before the cache is used. A UTF-8 BOM is caught as invalid JSON and the
cache is treated as missing. A malformed file still returns ``None``.
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "vesuvius"
    / "models"
    / "preprocessing"
    / "patches"
    / "cache.py"
)
_spec = importlib.util.spec_from_file_location("patch_cache_utf8", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
_cache = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _cache
_spec.loader.exec_module(_cache)
PatchCacheParams = _cache.PatchCacheParams
cache_filename = _cache.cache_filename
load_patch_cache = _cache.load_patch_cache


def _cp1252_for_locale(real_open):
    def wrapped(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
        if isinstance(mode, str) and "b" not in mode and encoding in (None, "locale"):
            encoding = "cp1252"
        return real_open(file, mode, buffering, encoding, errors, newline, closefd, opener)

    return wrapped


def _params(volume_name: str) -> PatchCacheParams:
    return PatchCacheParams(
        data_path="/data/scroll",
        volume_ids=(volume_name,),
        patch_size=(8, 8, 8),
        min_labeled_ratio=0.1,
        bbox_threshold=0.2,
        valid_patch_find_resolution=0,
    )


def _write(cache_dir: Path, params: PatchCacheParams, bom: bool = False, raw: bytes | None = None) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    if raw is None:
        payload = {
            "schema_version": 3,
            "metadata": {
                "created_at": "2026-10-12T00:00:00",
                "cache_params": params.to_dict(),
                "counts": {"fg": 1, "bg": 0, "unlabeled_fg": 0},
            },
            "fg_patches": [{"vol_idx": 0, "vol_name": params.volume_ids[0], "pos": [1, 2, 3]}],
            "bg_patches": [],
            "unlabeled_fg_patches": [],
        }
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    path = cache_dir / cache_filename(params)
    path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + raw)


def test_legacy_locale_keeps_the_cache(tmp_path, monkeypatch):
    params = _params("scroll \u201d")
    _write(tmp_path, params)
    monkeypatch.setattr(io, "open", _cp1252_for_locale(io.open))
    loaded = load_patch_cache(tmp_path, params)
    assert loaded is not None
    assert loaded.fg_patches[0].volume_name == "scroll \u201d"
    assert loaded.fg_patches[0].position == (1, 2, 3)


def test_utf8_bom_keeps_the_cache(tmp_path):
    params = _params("scroll")
    _write(tmp_path, params, bom=True)
    loaded = load_patch_cache(tmp_path, params)
    assert loaded is not None
    assert loaded.fg_patches[0].position == (1, 2, 3)


def test_malformed_cache_still_returns_none(tmp_path):
    params = _params("scroll")
    _write(tmp_path, params, raw=b"{")
    assert load_patch_cache(tmp_path, params) is None
