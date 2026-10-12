"""Sheet seed JSON is UTF-8. A legacy text open must not drop it.

``seed_from_json`` and ``add_area`` both call ``Path.read_text()`` with no
encoding. On cp1252 a name containing U+201D raises ``UnicodeDecodeError``
before the points or the area are used. A UTF-8 BOM is invalid JSON. An
ASCII file still loads, and a malformed file still raises.
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[2] / "src"
_SCRIPT = (
    _SRC
    / "vesuvius"
    / "neural_tracing"
    / "tracers"
    / "sheets"
    / "tracing"
    / "inference.py"
)
_NAME = "scroll \u201d"


def _load():
    sys.path.insert(0, str(_SRC))
    spec = importlib.util.spec_from_file_location("sheet_seed_json_utf8_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sheet = _load()


def _open_text_as_cp1252(real_open):
    def wrapped(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, *args, **kwargs):
        if isinstance(mode, str) and "b" not in mode and encoding in (None, "locale"):
            encoding = "cp1252"
        return real_open(file, mode, buffering, encoding, errors, newline, *args, **kwargs)

    return wrapped


def _bytes(document: dict, *, bom: bool = False) -> bytes:
    raw = json.dumps(document, ensure_ascii=False).encode("utf-8")
    return b"\xef\xbb\xbf" + raw if bom else raw


def _seed(name: str) -> dict:
    return {
        "name": name,
        "points": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        "direction": [0.0, 0.0, 1.0],
    }


def _write(path: Path, document: dict, *, bom: bool = False) -> None:
    path.write_bytes(_bytes(document, bom=bom))


def test_legacy_locale_keeps_unicode_seed_name(tmp_path, monkeypatch):
    path = tmp_path / "seed.json"
    _write(path, _seed(_NAME))
    monkeypatch.setattr(io, "open", _open_text_as_cp1252(io.open))

    points, direction = sheet.seed_from_json(path)

    assert points.shape == (3, 3)
    assert points[2, 0] == 2.0
    assert direction.shape == (3, 3)


def test_utf8_bom_seed_still_loads(tmp_path, monkeypatch):
    path = tmp_path / "seed.json"
    _write(path, _seed("plain"), bom=True)
    monkeypatch.setattr(io, "open", _open_text_as_cp1252(io.open))

    points, _direction = sheet.seed_from_json(path)

    assert points.shape == (3, 3)


def test_ascii_seed_loads(tmp_path):
    path = tmp_path / "seed.json"
    _write(path, _seed("plain"))

    points, direction = sheet.seed_from_json(path)

    assert points[0].tolist() == [0.0, 0.0, 0.0]
    assert direction[0].tolist() == [0.0, 0.0, 1.0]


def test_ascii_seed_loads_under_legacy_locale(tmp_path, monkeypatch):
    path = tmp_path / "seed.json"
    _write(path, _seed("plain"))
    monkeypatch.setattr(io, "open", _open_text_as_cp1252(io.open))

    points, _direction = sheet.seed_from_json(path)

    assert len(points) == 3


def test_malformed_seed_still_raises(tmp_path, monkeypatch):
    path = tmp_path / "seed.json"
    path.write_bytes(b"{")
    monkeypatch.setattr(io, "open", _open_text_as_cp1252(io.open))

    with pytest.raises(json.JSONDecodeError):
        sheet.seed_from_json(path)


def test_legacy_locale_keeps_unicode_meta_name(tmp_path, monkeypatch):
    path = tmp_path / "meta.json"
    _write(path, {"name": _NAME, "scale": [1, 1]})
    monkeypatch.setattr(io, "open", _open_text_as_cp1252(io.open))
    xyz = np.full((2, 2, 3), -1.0)

    meta = sheet.add_area(path, xyz, None)

    assert meta["name"] == _NAME
    assert meta["area_vx2"] == 0.0


def test_utf8_bom_meta_still_loads(tmp_path, monkeypatch):
    path = tmp_path / "meta.json"
    _write(path, {"name": "plain"}, bom=True)
    monkeypatch.setattr(io, "open", _open_text_as_cp1252(io.open))

    meta = sheet.add_area(path, np.full((2, 2, 3), -1.0), None)

    assert meta["name"] == "plain"
    assert "area_vx2" in meta


def test_ascii_meta_loads(tmp_path):
    path = tmp_path / "meta.json"
    _write(path, {"name": "plain"})

    meta = sheet.add_area(path, np.full((2, 2, 3), -1.0), 2.4)

    assert meta["name"] == "plain"
    assert meta["voxel_size_um"] == 2.4


def test_malformed_meta_still_raises(tmp_path, monkeypatch):
    path = tmp_path / "meta.json"
    path.write_bytes(b"{")
    monkeypatch.setattr(io, "open", _open_text_as_cp1252(io.open))

    with pytest.raises(json.JSONDecodeError):
        sheet.add_area(path, np.full((2, 2, 3), -1.0), None)
