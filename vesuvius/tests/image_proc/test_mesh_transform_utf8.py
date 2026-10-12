"""Mesh transform JSON is UTF-8, including on a legacy locale.

``data.affine`` already decodes ``transform.json`` as UTF-8.
``image_proc.mesh.affine.load_transform_from_json`` is the loader
``apply_affine`` and ``voxelize_objs`` call, and it still uses the
process code page. On cp1252 a byte the code page does not define
aborts the load before the matrix is read. A UTF-8 BOM, which
Windows editors write, fails the same way on a UTF-8 locale because
``json.load`` then sees U+FEFF.
"""
from __future__ import annotations

import builtins
import json

import numpy as np

from vesuvius.image_proc.mesh.affine import load_transform_from_json

_MATRIX = [[1, 0, 0, 2], [0, 1, 0, 3], [0, 0, 1, 4]]


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


def test_legacy_locale_loads_a_matrix_beside_a_non_cp1252_label(tmp_path, monkeypatch):
    path = tmp_path / "transform.json"
    # U+201D is UTF-8 e2 80 9d. Byte 0x9d is undefined in cp1252.
    _write(path, {"name": "scroll \u201d label", "transformation_matrix": _MATRIX})
    monkeypatch.setattr(builtins, "open", _open_text_as_cp1252(builtins.open))

    matrix = load_transform_from_json(path)

    np.testing.assert_allclose(matrix, [[1, 0, 0, 2], [0, 1, 0, 3], [0, 0, 1, 4], [0, 0, 0, 1]])


def test_utf8_bom_does_not_hide_the_matrix(tmp_path):
    path = tmp_path / "transform.json"
    _write(path, {"transformation_matrix": _MATRIX}, bom=True)

    matrix = load_transform_from_json(path)

    np.testing.assert_allclose(matrix[0], [1, 0, 0, 2])
