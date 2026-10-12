"""Fiber snap JSON is UTF-8. A legacy text open must not drop it.

``snap_json`` reads each VC3D fiber with ``Path.read_text()`` and no
encoding. On cp1252 a fiber name containing U+201D raises
``UnicodeDecodeError`` before any point is snapped. A UTF-8 BOM is
invalid JSON. An ASCII fiber still snaps, and a malformed file still
raises.
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
_SCRIPT = _SRC / "vesuvius" / "neural_tracing" / "tracers" / "fibers" / "data" / "fiber_snapping.py"
_NAME = "scroll \u201d"


def _load():
    sys.path.insert(0, str(_SRC))
    spec = importlib.util.spec_from_file_location("fiber_snap_json_utf8_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fiber_snapping = _load()


def _open_text_as_cp1252(real_open):
    def wrapped(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, *args, **kwargs):
        if isinstance(mode, str) and "b" not in mode and encoding in (None, "locale"):
            encoding = "cp1252"
        return real_open(file, mode, buffering, encoding, errors, newline, *args, **kwargs)

    return wrapped


class _Volume:
    def __init__(self, values):
        self.values, self.shape, self.dtype = values, values.shape, values.dtype

    def read(self, start, size):
        return self.values[tuple(slice(a, a + n) for a, n in zip(start, size))]


def _wavy(offset=(0.0, 0.0, 0.0)):
    x = np.arange(4.0, 92.0, 1.0)
    return np.c_[x, 24 + 3 * np.sin(x / 9.0), 24 + 2 * np.cos(x / 13.0)] + np.asarray(offset)


def _tube(centre, shape=(48, 48, 96), sigma=0.8):
    from scipy.spatial import cKDTree

    z, y, x = np.indices(shape)
    distance = cKDTree(fiber_snapping.resample(centre, 0.1)).query(
        np.c_[x.ravel(), y.ravel(), z.ravel()])[0].reshape(shape)
    return (255 * np.exp(-0.5 * (distance / sigma) ** 2)).astype(np.uint8)


def _document(name: str, line: np.ndarray) -> dict:
    return {
        "type": "vc3d_fiber",
        "version": 3,
        "name": name,
        "line_points": line.tolist(),
        "control_points": [{"position": line[0].tolist(), "label": name}],
    }


def _write(path: Path, document: dict, *, bom: bool = False) -> None:
    raw = json.dumps(document, ensure_ascii=False).encode("utf-8")
    if bom:
        raw = b"\xef\xbb\xbf" + raw
    path.write_bytes(raw)


def _snap(src: Path, out: Path):
    presence = _Volume(_tube(_wavy()))
    return fiber_snapping.snap_json(
        [src], out, presence, 8.0, 0.0,
        fiber_snapping.SnapConfig(tile=32, threads=2), "cpu", 1.0, {"method": "test"})


def test_legacy_locale_keeps_unicode_fiber_name(tmp_path, monkeypatch):
    src = tmp_path / "in" / "f.json"
    src.parent.mkdir()
    line = _wavy((0.0, 1.4, -0.9)) * 8.0
    _write(src, _document(_NAME, line))
    monkeypatch.setattr(io, "open", _open_text_as_cp1252(io.open))

    result = _snap(src, tmp_path / "out")

    saved = json.loads((tmp_path / "out" / "f.json").read_text())
    assert result["fibers"] == 1
    assert saved["name"] == _NAME
    assert saved["control_points"][0]["label"] == _NAME
    assert json.loads((tmp_path / "out" / "snapping.json").read_text())["files"] == ["f.json"]


def test_utf8_bom_fiber_still_snaps(tmp_path, monkeypatch):
    src = tmp_path / "in" / "f.json"
    src.parent.mkdir()
    line = _wavy((0.0, 1.4, -0.9)) * 8.0
    _write(src, _document("plain", line), bom=True)
    monkeypatch.setattr(io, "open", _open_text_as_cp1252(io.open))

    _snap(src, tmp_path / "out")

    assert json.loads((tmp_path / "out" / "f.json").read_text())["name"] == "plain"


def test_ascii_fiber_snaps(tmp_path):
    src = tmp_path / "in" / "f.json"
    src.parent.mkdir()
    line = _wavy((0.0, 1.4, -0.9)) * 8.0
    _write(src, _document("plain", line))

    result = _snap(src, tmp_path / "out")

    assert result["fibers"] == 1
    assert json.loads((tmp_path / "out" / "f.json").read_text())["name"] == "plain"


def test_ascii_fiber_snaps_under_legacy_locale(tmp_path, monkeypatch):
    src = tmp_path / "in" / "f.json"
    src.parent.mkdir()
    line = _wavy((0.0, 1.4, -0.9)) * 8.0
    _write(src, _document("plain", line))
    monkeypatch.setattr(io, "open", _open_text_as_cp1252(io.open))

    _snap(src, tmp_path / "out")

    assert json.loads((tmp_path / "out" / "f.json").read_text())["name"] == "plain"


def test_malformed_fiber_still_raises(tmp_path, monkeypatch):
    src = tmp_path / "in" / "f.json"
    src.parent.mkdir()
    src.write_bytes(b"{")
    monkeypatch.setattr(io, "open", _open_text_as_cp1252(io.open))

    with pytest.raises(json.JSONDecodeError):
        _snap(src, tmp_path / "out")
