"""tifxyz meta.json is UTF-8, including on a legacy locale.

``surface_preflight`` already opens ``meta.json`` with ``encoding="utf-8"``.
``TifxyzReader.read_metadata`` still uses the process code page. On cp1252 a
byte that code page does not define aborts the load. A UTF-8 BOM, which
Windows editors write, makes ``json.load`` fail on a UTF-8 locale too.
"""
from __future__ import annotations

import builtins
import json

from vesuvius.tifxyz.reader import TifxyzReader


def _open_text_as_cp1252(real_open):
    def wrapped(file, mode="r", *args, **kwargs):
        if isinstance(mode, str) and "b" not in mode and "encoding" not in kwargs:
            kwargs = dict(kwargs)
            kwargs["encoding"] = "cp1252"
        return real_open(file, mode, *args, **kwargs)

    return wrapped


def _write(path, payload: dict, bom: bool = False) -> None:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + raw)


def test_legacy_locale_reads_a_non_cp1252_name(tmp_path, monkeypatch):
    segment = tmp_path / "segment"
    segment.mkdir()
    # U+201D is UTF-8 e2 80 9d. Byte 0x9d is undefined in cp1252.
    _write(
        segment / "meta.json",
        {"uuid": "seg", "scale": [1, 2], "name": "scroll \u201d label"},
    )
    monkeypatch.setattr(builtins, "open", _open_text_as_cp1252(builtins.open))

    meta = TifxyzReader(segment).read_metadata()

    assert meta["extra"]["name"] == "scroll \u201d label"
    assert meta["scale"] == (2.0, 1.0)


def test_utf8_bom_does_not_hide_meta(tmp_path):
    segment = tmp_path / "segment"
    segment.mkdir()
    _write(segment / "meta.json", {"uuid": "seg", "scale": [1, 2]}, bom=True)

    meta = TifxyzReader(segment).read_metadata()

    assert meta["uuid"] == "seg"
    assert meta["scale"] == (2.0, 1.0)
