"""chunks.json is UTF-8. A legacy text open must not drop the manifest.

``load_chunks_json`` catches every ``Exception`` and returns ``None``. On
cp1252 a byte the code page does not define becomes that warning, so
inference proceeds as if no chunk list existed. A UTF-8 BOM is the same
silent miss: ``json.load`` fails and the manifest is ignored.
"""
from __future__ import annotations

import builtins
import json

from vesuvius.data.chunks_filter import load_chunks_json


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


def _manifest() -> dict:
    return {
        "chunks_by_level": {"0": [[0, 1, 2]]},
        "name": "scroll \u201d label",
    }


def test_legacy_locale_keeps_the_manifest(tmp_path, monkeypatch):
    zarr_dir = tmp_path / "volume.zarr"
    zarr_dir.mkdir()
    # U+201D is UTF-8 e2 80 9d. Byte 0x9d is undefined in cp1252.
    _write(zarr_dir / "chunks.json", _manifest())
    monkeypatch.setattr(builtins, "open", _open_text_as_cp1252(builtins.open))

    loaded = load_chunks_json(str(zarr_dir))

    assert loaded is not None
    assert loaded["chunks_by_level"] == {"0": [[0, 1, 2]]}
    assert loaded["name"] == "scroll \u201d label"


def test_utf8_bom_does_not_drop_the_manifest(tmp_path):
    zarr_dir = tmp_path / "volume.zarr"
    zarr_dir.mkdir()
    _write(zarr_dir / "chunks.json", _manifest(), bom=True)

    loaded = load_chunks_json(str(zarr_dir))

    assert loaded is not None
    assert loaded["chunks_by_level"] == {"0": [[0, 1, 2]]}
