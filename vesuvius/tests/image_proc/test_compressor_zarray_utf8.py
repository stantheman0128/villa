"""update_compressor_metadata reads .zarray with the process locale.

Under cp1252, U+201D in the dimension separator raises UnicodeDecodeError
before the compressor is replaced, and a UTF-8 BOM raises JSONDecodeError.
The separator should still be written back. A missing file is still
reported. Malformed JSON is still reported. An ASCII separator is still
kept. The rewrite open is unchanged.
Related to #1524.
"""
import builtins
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


def _package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
    return module


for _name in ("vesuvius", "vesuvius.data", "vesuvius.data.utils"):
    _package(_name)
sys.modules["vesuvius.data.utils"]._ZARR_V3 = False
sys.modules["vesuvius.data.utils"].open_zarr = lambda *args, **kwargs: None

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "src/vesuvius/image_proc/run/zarr_tasks/utils.py"
spec = importlib.util.spec_from_file_location("compressor_zarray_utf8_under_test", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


class _Compressor:
    def get_config(self):
        return {"id": "blosc", "cname": "zstd", "clevel": 1, "shuffle": 1}


def cp1252_text(monkeypatch):
    real = builtins.open

    def fake(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
        if "b" not in str(mode) and encoding in (None, "locale"):
            encoding = "cp1252"
        return real(file, mode, buffering, encoding, errors, newline, closefd, opener)

    monkeypatch.setattr(builtins, "open", fake)


def write_json(path, document, *, bom=False, raw=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = raw if raw is not None else json.dumps(document, ensure_ascii=False).encode("utf-8")
    if bom:
        payload = b"\xef\xbb\xbf" + payload
    path.write_bytes(payload)


def document(separator="."):
    return {
        "zarr_format": 2,
        "shape": [8, 4, 2],
        "chunks": [4, 4, 2],
        "dtype": "|u1",
        "compressor": {"id": "zlib", "level": 1},
        "dimension_separator": separator,
        "order": "C",
        "fill_value": 0,
    }


def update(path):
    mod.update_compressor_metadata(path, _Compressor())
    return json.loads((Path(path) / ".zarray").read_bytes().decode("utf-8"))


def test_unicode_separator_is_kept(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    write_json(tmp_path / ".zarray", document("scroll \u201d"))
    written = update(tmp_path)
    assert written["dimension_separator"] == "scroll \u201d"
    assert written["compressor"]["id"] == "blosc"
    assert written["order"] == "C"


def test_utf8_bom_keeps_the_separator(tmp_path):
    write_json(tmp_path / ".zarray", document("/"), bom=True)
    written = update(tmp_path)
    assert written["dimension_separator"] == "/"
    assert written["compressor"]["id"] == "blosc"


def test_ascii_separator_is_kept(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    write_json(tmp_path / ".zarray", document("."))
    written = update(tmp_path)
    assert written["dimension_separator"] == "."
    assert written["compressor"]["id"] == "blosc"
    assert written["order"] == "C"
    assert written["fill_value"] == 0


def test_missing_compressor_key_is_added(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    payload = document(".")
    del payload["compressor"]
    write_json(tmp_path / ".zarray", payload)
    written = update(tmp_path)
    assert written["compressor"]["id"] == "blosc"
    assert written["dimension_separator"] == "."


def test_missing_file_still_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        update(tmp_path)


def test_malformed_json_still_raises(tmp_path):
    target = tmp_path / ".zarray"
    write_json(target, None, raw=b"{")
    with pytest.raises(json.JSONDecodeError):
        update(tmp_path)
    assert target.read_bytes() == b"{"
