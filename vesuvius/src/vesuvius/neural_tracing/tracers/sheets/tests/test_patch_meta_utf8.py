"""Sheet patch meta.json is UTF-8. A legacy text read must not drop the patch.

``_scan`` uses ``Path.read_text`` with no encoding, then treats ``ValueError``
as "not a patch". On cp1252 the U+201D byte is undefined, so the patch
disappears from the index. A UTF-8 BOM is the same miss: ``json.loads``
fails and the patch is skipped.
"""
from __future__ import annotations

import io
import json

import numpy as np
import tifffile

from vesuvius.neural_tracing.tracers.sheets.data.patches import build_index


def _cp1252_for_meta(real_open):
    def wrapped(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
        if (
            isinstance(mode, str)
            and "b" not in mode
            and str(file).endswith("meta.json")
            and encoding in (None, "locale")
        ):
            encoding = "cp1252"
        return real_open(
            file,
            mode,
            buffering,
            encoding,
            errors,
            newline,
            closefd,
            opener,
        )

    return wrapped


def _write_patch(patch, note="sheet", bom=False) -> None:
    patch.mkdir(parents=True)
    grid = np.ones((4, 4), np.float32)
    for name in "xyz":
        tifffile.imwrite(patch / f"{name}.tif", grid)
    raw = json.dumps({"scale": [0.5, 0.5], "note": note}, ensure_ascii=False).encode("utf-8")
    (patch / "meta.json").write_bytes((b"\xef\xbb\xbf" if bom else b"") + raw)


def test_legacy_locale_keeps_the_patch(tmp_path, monkeypatch):
    _write_patch(tmp_path / "patches" / "p0", note="sheet \u201d")
    monkeypatch.setattr(io, "open", _cp1252_for_meta(io.open))
    index = build_index(tmp_path / "patches", tmp_path / "cache", workers=1)
    assert [info.name for info in index] == ["p0"]


def test_utf8_bom_keeps_the_patch(tmp_path):
    _write_patch(tmp_path / "patches" / "p0", bom=True)
    index = build_index(tmp_path / "patches", tmp_path / "cache", workers=1)
    assert [info.name for info in index] == ["p0"]
