"""prepare_scroll.py reads open-data metadata.json with the process locale.

Under cp1252, U+201D in a volume path raises UnicodeDecodeError before the
path is used, and a UTF-8 BOM raises JSONDecodeError.
Related to #1524.
"""
import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

SCRIPT = SRC / "vesuvius/neural_tracing/tracers/sheets/scripts/prepare_scroll.py"
spec = importlib.util.spec_from_file_location("scroll_metadata_utf8_under_test", SCRIPT)
prep = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = prep
spec.loader.exec_module(prep)


class Stop(Exception):
    def __init__(self, src):
        self.src = src


def install_stubs():
    def mirror(src, dst, attrs):
        raise Stop(src)

    prep.mirror_array = mirror


def document(*, path="scans/volume", name="plain"):
    return {
        "name": name,
        "samples": {
            "PHerc1667": {
                "volumes": {
                    "vol": {"data": [{"origins": [{"path": path}]}]},
                },
                "segments": {},
            }
        },
    }


def write_json(path, payload):
    path.write_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"))


def cp1252_text(monkeypatch):
    real = io.open

    def fake(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, *args, **kwargs):
        if "b" not in str(mode) and encoding in (None, "locale"):
            encoding = "cp1252"
        return real(file, mode, buffering, encoding, errors, newline, *args, **kwargs)

    monkeypatch.setattr(io, "open", fake)


def run(tmp_path, metadata):
    install_stubs()
    prep.main([
        "--metadata", str(metadata),
        "--scroll", "PHerc1667",
        "--volume-id", "vol",
        "--coord-scale", "0.25",
        "--levels", "2",
        "--out", str(tmp_path / "out"),
    ])


def test_unicode_volume_path_is_read_under_cp1252(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    path = tmp_path / "metadata.json"
    write_json(path, document(name="scroll \u201d", path="scroll \u201d/volume"))
    with pytest.raises(Stop) as caught:
        run(tmp_path, path)
    assert caught.value.src.endswith("scroll \u201d/volume/2")


def test_utf8_bom_metadata_is_read(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps(document(path="scans/volume")).encode("utf-8"))
    with pytest.raises(Stop) as caught:
        run(tmp_path, path)
    assert caught.value.src.endswith("scans/volume/2")


def test_malformed_metadata_still_raises(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    path = tmp_path / "metadata.json"
    path.write_text("{", encoding="ascii")
    with pytest.raises(json.JSONDecodeError):
        run(tmp_path, path)


def test_unknown_scroll_still_raises(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    path = tmp_path / "metadata.json"
    write_json(path, {"name": "plain", "samples": {}})
    with pytest.raises(KeyError):
        run(tmp_path, path)


def test_ascii_volume_path_is_used(tmp_path):
    path = tmp_path / "metadata.json"
    write_json(path, document(path="scans/volume"))
    with pytest.raises(Stop) as caught:
        run(tmp_path, path)
    assert caught.value.src == f"{prep.ROOT}/scans/volume/2"
