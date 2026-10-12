"""The snap_fibers JSON filter is UTF-8. A legacy text open must not drop it.

``snap_fibers.py`` reads each fiber with ``Path.read_text()`` and no
encoding before it checks ``type``. On cp1252 a fiber name containing
U+201D raises ``UnicodeDecodeError`` before that check. A UTF-8 BOM is
invalid JSON. An ASCII fiber of another type is still dropped, and a
malformed file still raises.
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
_SCRIPT = _SRC / "vesuvius" / "neural_tracing" / "tracers" / "fibers" / "scripts" / "snap_fibers.py"
_NAME = "scroll \u201d"


def _load():
    sys.path.insert(0, str(_SRC))
    spec = importlib.util.spec_from_file_location("snap_fibers_filter_utf8_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


snap_fibers = _load()


def _open_text_as_cp1252(real_open):
    def wrapped(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, *args, **kwargs):
        if isinstance(mode, str) and "b" not in mode and encoding in (None, "locale"):
            encoding = "cp1252"
        return real_open(file, mode, buffering, encoding, errors, newline, *args, **kwargs)

    return wrapped


class _Presence:
    dtype = np.dtype("uint8")


def _write(path: Path, document: dict, *, bom: bool = False) -> None:
    raw = json.dumps(document, ensure_ascii=False).encode("utf-8")
    if bom:
        raw = b"\xef\xbb\xbf" + raw
    path.write_bytes(raw)


def _filter(src: Path, out: Path):
    snap_fibers.ChunkedArray = lambda path, budget: _Presence()
    calls = []

    def snap_json(paths, destination, presence, scale, offset, cfg, device, output_step, provenance):
        calls.append([Path(p).name for p in paths])
        return {"fibers": len(paths)}

    snap_fibers.snap_json = snap_json
    snap_fibers.main([str(src), str(out), "--presence", "presence", "--native-per-voxel", "8"])
    return calls


def test_legacy_locale_keeps_unicode_fiber_name(tmp_path, monkeypatch):
    src = tmp_path / "in"
    src.mkdir()
    _write(src / "f.json", {"type": "vc3d_fiber", "name": _NAME})
    _write(src / "note.json", {"type": "note", "name": "plain"})
    monkeypatch.setattr(io, "open", _open_text_as_cp1252(io.open))

    calls = _filter(src, tmp_path / "out")

    assert calls == [["f.json"]]


def test_utf8_bom_fiber_still_passes_the_filter(tmp_path, monkeypatch):
    src = tmp_path / "in"
    src.mkdir()
    _write(src / "f.json", {"type": "vc3d_fiber", "name": "plain"}, bom=True)
    monkeypatch.setattr(io, "open", _open_text_as_cp1252(io.open))

    calls = _filter(src, tmp_path / "out")

    assert calls == [["f.json"]]


def test_ascii_other_type_is_dropped(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    _write(src / "f.json", {"type": "vc3d_fiber", "name": "plain"})
    _write(src / "note.json", {"type": "note", "name": "plain"})

    calls = _filter(src, tmp_path / "out")

    assert calls == [["f.json"]]


def test_ascii_filter_under_legacy_locale(tmp_path, monkeypatch):
    src = tmp_path / "in"
    src.mkdir()
    _write(src / "f.json", {"type": "vc3d_fiber", "name": "plain"})
    _write(src / "note.json", {"type": "note", "name": "plain"})
    monkeypatch.setattr(io, "open", _open_text_as_cp1252(io.open))

    calls = _filter(src, tmp_path / "out")

    assert calls == [["f.json"]]


def test_malformed_fiber_still_raises(tmp_path, monkeypatch):
    src = tmp_path / "in"
    src.mkdir()
    (src / "f.json").write_bytes(b"{")
    monkeypatch.setattr(io, "open", _open_text_as_cp1252(io.open))

    with pytest.raises(json.JSONDecodeError):
        _filter(src, tmp_path / "out")
