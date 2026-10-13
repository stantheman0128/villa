"""A winding centerline reads local Zarr metadata with the process locale.

Under cp1252, U+201D in a Zarr v3 chunk-key name or a Zarr v2 compressor
id raises UnicodeDecodeError before the array shape is kept, and a UTF-8
BOM raises JSONDecodeError. Those strings should still be kept. A group
is still rejected. A missing array still raises. A float array still
raises.
Related to #1524.
"""
import importlib.util
import json
import sys
from pathlib import Path

import builtins
import click
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
SCRIPT = ROOT / "src/vesuvius/neural_tracing/winding_models/winding_centerlines.py"
spec = importlib.util.spec_from_file_location("centerline_zarr_utf8_under_test", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def cp1252_text(monkeypatch):
    real = builtins.open

    def fake(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, *args, **kwargs):
        if "b" not in str(mode) and encoding in (None, "locale"):
            encoding = "cp1252"
        return real(file, mode, buffering, encoding, errors, newline, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake)


def write_json(path, document, *, bom=False, raw=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = raw if raw is not None else json.dumps(document, ensure_ascii=False).encode("utf-8")
    if bom:
        payload = b"\xef\xbb\xbf" + payload
    path.write_bytes(payload)


def v3_document(name="default"):
    return {
        "node_type": "array",
        "shape": [2, 3, 4],
        "chunk_grid": {"configuration": {"chunk_shape": [1, 1, 1]}},
        "data_type": "uint8",
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "chunk_key_encoding": {"name": name, "configuration": {"separator": "/"}},
    }


def v2_document(codec_id="zlib"):
    return {
        "shape": [2, 3, 4],
        "chunks": [1, 1, 1],
        "dtype": "<u1",
        "compressor": {"id": codec_id},
        "dimension_separator": ".",
    }


def test_unicode_v3_chunk_key_is_kept(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    write_json(tmp_path / "zarr.json", v3_document("scroll \u201d"))
    array = mod.LocalZarrArray(tmp_path)
    assert array.chunk_key_name == "scroll \u201d"
    assert array.shape == (2, 3, 4)


def test_unicode_v2_compressor_is_kept(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    write_json(tmp_path / ".zarray", v2_document("scroll \u201d"))
    array = mod.LocalZarrArray(tmp_path)
    assert array.compressor["id"] == "scroll \u201d"
    assert array.shape == (2, 3, 4)


def test_utf8_bom_v3_is_kept(tmp_path):
    write_json(tmp_path / "zarr.json", v3_document("default"), bom=True)
    assert mod.LocalZarrArray(tmp_path).chunk_key_name == "default"


def test_utf8_bom_v2_is_kept(tmp_path):
    write_json(tmp_path / ".zarray", v2_document("zlib"), bom=True)
    assert mod.LocalZarrArray(tmp_path).compressor["id"] == "zlib"


def test_ascii_v3_shape_is_kept(tmp_path):
    write_json(tmp_path / "zarr.json", v3_document("default"))
    array = mod.LocalZarrArray(tmp_path)
    assert array.version == 3
    assert array.shape == (2, 3, 4)


def test_group_is_still_rejected(tmp_path):
    document = v3_document("default")
    document["node_type"] = "group"
    write_json(tmp_path / "zarr.json", document)
    with pytest.raises(click.UsageError, match="not an array"):
        mod.LocalZarrArray(tmp_path)


def test_float_v2_is_still_rejected(tmp_path):
    document = v2_document("zlib")
    document["dtype"] = "<f4"
    write_json(tmp_path / ".zarray", document)
    with pytest.raises(click.UsageError, match="expected an integer array"):
        mod.LocalZarrArray(tmp_path)


def test_malformed_v3_still_raises(tmp_path):
    write_json(tmp_path / "zarr.json", None, raw=b"{")
    with pytest.raises(json.JSONDecodeError):
        mod.LocalZarrArray(tmp_path)


def test_missing_array_still_raises(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    with pytest.raises(click.UsageError, match="not a local Zarr"):
        mod.LocalZarrArray(tmp_path)
