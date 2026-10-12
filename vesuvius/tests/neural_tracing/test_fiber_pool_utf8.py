"""Fiber pool names are UTF-8. A legacy text open must not drop them.

``FiberPool`` reads ``keys.json`` and ``aliases.json`` with
``Path.read_text()`` and no encoding. On cp1252 a fiber name containing
U+201D raises ``UnicodeDecodeError`` before the index exists. A UTF-8
BOM is invalid JSON. An ASCII key list still indexes, and a malformed
keys file still raises.
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "vesuvius"
    / "neural_tracing"
    / "tracers"
    / "sheets"
    / "data"
    / "fibers.py"
)
_spec = importlib.util.spec_from_file_location("fiber_pool_utf8", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
FiberPool = _mod.FiberPool

_NAME = "scroll \u201d"


def _open_text_as_cp1252(real_open):
    def wrapped(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, *args, **kwargs):
        if isinstance(mode, str) and "b" not in mode and encoding in (None, "locale"):
            encoding = "cp1252"
        return real_open(file, mode, buffering, encoding, errors, newline, *args, **kwargs)

    return wrapped


def _write_arrays(pool: Path) -> None:
    pool.mkdir(parents=True, exist_ok=True)
    arrays = {
        "points": np.zeros((0, 3), np.float32),
        "tangent": np.zeros((0, 3), np.float32),
        "arc": np.zeros((0,), np.float32),
        "instance": np.zeros((0,), np.int64),
        "family": np.zeros((0,), np.int64),
        "cell_keys": np.zeros((0,), np.int64),
        "cell_starts": np.zeros((1,), np.int64),
    }
    for name, array in arrays.items():
        np.save(pool / f"{name}.npy", array)


def _write_json(path: Path, payload, bom: bool = False) -> None:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if bom:
        raw = b"\xef\xbb\xbf" + raw
    path.write_bytes(raw)


def test_legacy_locale_reads_unicode_key(tmp_path, monkeypatch):
    _write_arrays(tmp_path)
    _write_json(tmp_path / "keys.json", [_NAME])
    monkeypatch.setattr(io, "open", _open_text_as_cp1252(io.open))

    pool = FiberPool(tmp_path)

    assert pool.index[_NAME] == 0


def test_legacy_locale_reads_unicode_key_and_alias(tmp_path, monkeypatch):
    _write_arrays(tmp_path)
    _write_json(tmp_path / "keys.json", ["scroll"])
    _write_json(tmp_path / "aliases.json", {_NAME: "scroll"})
    monkeypatch.setattr(io, "open", _open_text_as_cp1252(io.open))

    pool = FiberPool(tmp_path)

    assert pool.index["scroll"] == 0
    assert pool.index[_NAME] == 0


def test_utf8_bom_reads_unicode_key(tmp_path):
    _write_arrays(tmp_path)
    _write_json(tmp_path / "keys.json", [_NAME], bom=True)

    pool = FiberPool(tmp_path)

    assert pool.index[_NAME] == 0


def test_utf8_bom_reads_unicode_alias(tmp_path):
    _write_arrays(tmp_path)
    _write_json(tmp_path / "keys.json", ["scroll"])
    _write_json(tmp_path / "aliases.json", {_NAME: "scroll"}, bom=True)

    pool = FiberPool(tmp_path)

    assert pool.index[_NAME] == 0


def test_ascii_key_still_indexed(tmp_path):
    _write_arrays(tmp_path)
    _write_json(tmp_path / "keys.json", ["scroll"])

    pool = FiberPool(tmp_path)

    assert pool.index["scroll"] == 0


def test_malformed_keys_still_raises(tmp_path):
    _write_arrays(tmp_path)
    (tmp_path / "keys.json").write_text("{", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        FiberPool(tmp_path)
