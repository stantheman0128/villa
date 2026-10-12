"""Resample meta.json is UTF-8. A legacy text open must not drop it.

``resample_isotropic`` reads each patch ``meta.json`` with
``Path.read_text()`` and no encoding. On cp1252 a uuid containing U+201D
raises ``UnicodeDecodeError`` before the grid spacing is checked. A
UTF-8 BOM is invalid JSON. An ASCII meta still copies a square patch,
and a malformed file still raises.
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "vesuvius"
    / "neural_tracing"
    / "tracers"
    / "sheets"
    / "scripts"
    / "resample_isotropic.py"
)
_UUID = "scroll \u201d"


def _load():
    spec = importlib.util.spec_from_file_location("resample_meta_utf8_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


resample = _load()


def _open_text_as_cp1252(real_open):
    def wrapped(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, *args, **kwargs):
        if isinstance(mode, str) and "b" not in mode and encoding in (None, "locale"):
            encoding = "cp1252"
        return real_open(file, mode, buffering, encoding, errors, newline, *args, **kwargs)

    return wrapped


def _meta(uuid: str, *, bom: bool = False) -> bytes:
    raw = json.dumps({"uuid": uuid, "scale": [1, 1]}, ensure_ascii=False).encode("utf-8")
    return b"\xef\xbb\xbf" + raw if bom else raw


def _write_square_patch(folder: Path, meta: bytes) -> None:
    folder.mkdir(parents=True)
    rows = np.arange(3, dtype=np.float32)[:, None]
    cols = np.arange(3, dtype=np.float32)[None, :]
    tifffile.imwrite(folder / "x.tif", np.broadcast_to(cols, (3, 3)).copy())
    tifffile.imwrite(folder / "y.tif", np.broadcast_to(rows, (3, 3)).copy())
    tifffile.imwrite(folder / "z.tif", np.zeros((3, 3), np.float32))
    (folder / "meta.json").write_bytes(meta)


def _copy(src: Path, out: Path, capsys):
    resample.main([str(src), str(out)])
    return capsys.readouterr().out


def test_legacy_locale_keeps_unicode_uuid(tmp_path, monkeypatch, capsys):
    src, out = tmp_path / "src", tmp_path / "out"
    meta = _meta(_UUID)
    _write_square_patch(src / "p0", meta)
    out.mkdir()
    monkeypatch.setattr(io, "open", _open_text_as_cp1252(io.open))

    text = _copy(src, out, capsys)

    assert "copied" in text
    assert (out / "p0" / "meta.json").read_bytes() == meta


def test_utf8_bom_meta_still_copies(tmp_path, monkeypatch, capsys):
    src, out = tmp_path / "src", tmp_path / "out"
    _write_square_patch(src / "p0", _meta("plain", bom=True))
    out.mkdir()
    monkeypatch.setattr(io, "open", _open_text_as_cp1252(io.open))

    text = _copy(src, out, capsys)

    assert "copied" in text
    assert (out / "p0" / "x.tif").is_file()


def test_ascii_meta_copies_square_patch(tmp_path, capsys):
    src, out = tmp_path / "src", tmp_path / "out"
    _write_square_patch(src / "p0", _meta("plain"))
    out.mkdir()

    text = _copy(src, out, capsys)

    assert "square" in text
    assert (out / "p0" / "x.tif").is_file()


def test_ascii_meta_copies_under_legacy_locale(tmp_path, monkeypatch, capsys):
    src, out = tmp_path / "src", tmp_path / "out"
    _write_square_patch(src / "p0", _meta("plain"))
    out.mkdir()
    monkeypatch.setattr(io, "open", _open_text_as_cp1252(io.open))

    text = _copy(src, out, capsys)

    assert "copied" in text


def test_malformed_meta_still_raises(tmp_path, monkeypatch):
    src, out = tmp_path / "src", tmp_path / "out"
    _write_square_patch(src / "p0", b"{")
    out.mkdir()
    monkeypatch.setattr(io, "open", _open_text_as_cp1252(io.open))

    with pytest.raises(json.JSONDecodeError):
        resample.main([str(src), str(out)])
