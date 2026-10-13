"""decode_store.read_meta loads .zarray with the process locale.

Under cp1252, U+201D in the compressor id raises UnicodeDecodeError
before the id is used, and a UTF-8 BOM raises JSONDecodeError. The
compressor id should still be returned. A missing file is still
reported. Malformed JSON is still reported.
Related to #1524.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pathlib
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "src/vesuvius/neural_tracing/tracers/shared/crop_sampling/decode_store.py"
spec = importlib.util.spec_from_file_location("decode_zarray_utf8_under_test", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def cp1252_text(monkeypatch):
    real = pathlib.Path.open

    def fake(self, mode="r", buffering=-1, encoding=None, errors=None, newline=None, **kwargs):
        if "b" not in str(mode) and encoding in (None, "locale"):
            encoding = "cp1252"
        return real(self, mode, buffering, encoding, errors, newline, **kwargs)

    monkeypatch.setattr(pathlib.Path, "open", fake)


def write_json(path, document, *, bom=False, raw=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = raw if raw is not None else json.dumps(document, ensure_ascii=False).encode("utf-8")
    if bom:
        payload = b"\xef\xbb\xbf" + payload
    path.write_bytes(payload)


def document(name="blosc"):
    return {
        "shape": [8, 4, 2],
        "chunks": [8, 4, 2],
        "dtype": "|u1",
        "compressor": {"id": name},
        "dimension_separator": ".",
    }


def load(path):
    return mod.read_meta(path)


def test_unicode_compressor_id_is_kept(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    write_json(tmp_path / ".zarray", document("scroll \u201d"))
    meta = load(tmp_path)
    assert meta["compressor"]["id"] == "scroll \u201d"
    assert meta["shape"] == [8, 4, 2]


def test_utf8_bom_keeps_the_compressor_id(tmp_path):
    write_json(tmp_path / ".zarray", document("blosc"), bom=True)
    meta = load(tmp_path)
    assert meta["compressor"]["id"] == "blosc"
    assert meta["shape"] == [8, 4, 2]


def test_ascii_compressor_id_is_kept(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    write_json(tmp_path / ".zarray", document("blosc"))
    meta = load(tmp_path)
    assert meta["compressor"]["id"] == "blosc"
    assert meta["chunks"] == [8, 4, 2]


def test_missing_file_still_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load(tmp_path)


def test_malformed_json_still_raises(tmp_path):
    write_json(tmp_path / ".zarray", None, raw=b"{")
    with pytest.raises(json.JSONDecodeError):
        load(tmp_path)
