"""The training patch cache is UTF-8. A legacy text open must not drop it.

``load_cached_patches`` and ``get_cache_info`` open the cache with no
encoding. On cp1252 a volume name containing U+201D raises
``UnicodeDecodeError``, which neither function catches. A UTF-8 BOM is
invalid JSON, so the cache is treated as missing. A malformed file still
returns ``None``.
"""
from __future__ import annotations

import builtins
import importlib.util
import json
import sys
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "vesuvius"
    / "models"
    / "datasets"
    / "save_valid_patches.py"
)
_spec = importlib.util.spec_from_file_location("save_valid_patches_utf8", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
generate_cache_filename = _mod.generate_cache_filename
get_cache_info = _mod.get_cache_info
load_cached_patches = _mod.load_cached_patches

_PATHS = ["/data/scroll \u201d"]
_LABELS = ["/data/labels"]
_PATCH = (8, 8, 8)


def _open_text_as_cp1252(real_open):
    def wrapped(file, mode="r", *args, **kwargs):
        if isinstance(mode, str) and "b" not in mode and "encoding" not in kwargs:
            kwargs = dict(kwargs)
            kwargs["encoding"] = "cp1252"
        return real_open(file, mode, *args, **kwargs)

    return wrapped


def _write(cache_dir: Path, bom: bool = False, raw: bytes | None = None) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = generate_cache_filename(_PATHS, _LABELS, _PATCH, 0.1, 0.97)
    if raw is None:
        payload = {
            "metadata": {
                "timestamp": "2026-10-12T00:00:00",
                "train_data_paths": _PATHS,
                "label_paths": _LABELS,
                "patch_size": list(_PATCH),
                "min_labeled_ratio": 0.1,
                "bbox_threshold": 0.97,
                "valid_patch_find_resolution": 1,
                "valid_patch_value": None,
                "bg_sampling_enabled": False,
                "bg_to_fg_ratio": 0.5,
                "unlabeled_fg_enabled": False,
                "unlabeled_fg_threshold": 0.05,
                "unlabeled_fg_bbox_threshold": 0.15,
            },
            "valid_patches": [
                {
                    "volume_path": _PATHS[0],
                    "volume_index": 0,
                    "volume_name": "scroll \u201d",
                    "start_position": [1, 2, 3],
                    "patch_size": list(_PATCH),
                }
            ],
            "bg_only_patches": [],
            "unlabeled_fg_patches": [],
        }
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    (cache_dir / name).write_bytes((b"\xef\xbb\xbf" if bom else b"") + raw)


def _load(cache_dir: Path):
    return load_cached_patches(_PATHS, _LABELS, _PATCH, 0.1, cache_path=str(cache_dir))


def test_legacy_locale_keeps_the_patches(tmp_path, monkeypatch):
    _write(tmp_path)
    monkeypatch.setattr(builtins, "open", _open_text_as_cp1252(builtins.open))
    loaded = _load(tmp_path)
    assert loaded is not None
    assert loaded["fg_patches"][0]["volume_name"] == "scroll \u201d"
    assert loaded["fg_patches"][0]["position"] == [1, 2, 3]


def test_utf8_bom_keeps_the_patches(tmp_path):
    _write(tmp_path, bom=True)
    loaded = _load(tmp_path)
    assert loaded is not None
    assert loaded["fg_patches"][0]["position"] == [1, 2, 3]


def test_cache_info_legacy_locale(tmp_path, monkeypatch):
    _write(tmp_path)
    monkeypatch.setattr(builtins, "open", _open_text_as_cp1252(builtins.open))
    info = get_cache_info(_PATHS, _LABELS, _PATCH, 0.1, cache_path=str(tmp_path))
    assert info is not None
    assert info["train_data_paths"] == _PATHS


def test_cache_info_utf8_bom(tmp_path):
    _write(tmp_path, bom=True)
    info = get_cache_info(_PATHS, _LABELS, _PATCH, 0.1, cache_path=str(tmp_path))
    assert info is not None
    assert info["min_labeled_ratio"] == 0.1


def test_malformed_cache_still_returns_none(tmp_path):
    _write(tmp_path, raw=b"{")
    assert _load(tmp_path) is None
    assert get_cache_info(_PATHS, _LABELS, _PATCH, 0.1, cache_path=str(tmp_path)) is None
