"""Volume-run JSON is UTF-8. A legacy text open must not drop it.

``export_afv`` reads ``run.json`` with ``Path.read_text()`` and no encoding.
On cp1252 a checkpoint path containing U+201D raises ``UnicodeDecodeError``
before any fiber is written. A UTF-8 BOM is invalid JSON. An ASCII run
still exports, and a malformed file still raises.
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src"
_SCRIPT = _SRC / "vesuvius" / "neural_tracing" / "tracers" / "fibers" / "tracing" / "volume_afv.py"
_CHECKPOINT = "scroll \u201d.pt"


def _load():
    sys.path.insert(0, str(_SRC))
    spec = importlib.util.spec_from_file_location("volume_run_json_utf8_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


volume_afv = _load()


def _open_text_as_cp1252(real_open):
    def wrapped(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, *args, **kwargs):
        if isinstance(mode, str) and "b" not in mode and encoding in (None, "locale"):
            encoding = "cp1252"
        return real_open(file, mode, buffering, encoding, errors, newline, *args, **kwargs)

    return wrapped


def _record(checkpoint: str) -> dict:
    return {
        "checkpoint": checkpoint,
        "box": {"lo": [0, 0, 0], "hi": [1, 1, 1]},
        "grid_scale": 1.0,
        "operating_policy": {"confidence": 0.5},
    }


def _write_run(out: Path, document: dict, *, bom: bool = False) -> None:
    out.mkdir()
    raw = json.dumps(document, ensure_ascii=False).encode("utf-8")
    if bom:
        raw = b"\xef\xbb\xbf" + raw
    (out / "run.json").write_bytes(raw)


def _export(out: Path, captured: list):
    def write_afv(template, tmp, fibers, metadata, keep_metadata):
        captured.append(metadata)
        Path(tmp).write_bytes(b"afv")
        return {"fibers": 0}

    volume_afv.write_afv = write_afv
    return volume_afv.export_afv(out, template="template.afv")


def test_legacy_locale_keeps_unicode_checkpoint(tmp_path, monkeypatch):
    out = tmp_path / "run"
    _write_run(out, _record(_CHECKPOINT))
    monkeypatch.setattr(io, "open", _open_text_as_cp1252(io.open))
    captured = []

    result = _export(out, captured)

    assert captured[0]["provenance"]["checkpoint"] == _CHECKPOINT
    assert result["fibers"] == 0
    assert (out / "fibers.afv").is_file()


def test_utf8_bom_run_still_exports(tmp_path, monkeypatch):
    out = tmp_path / "run"
    _write_run(out, _record("plain.pt"), bom=True)
    monkeypatch.setattr(io, "open", _open_text_as_cp1252(io.open))
    captured = []

    _export(out, captured)

    assert captured[0]["provenance"]["checkpoint"] == "plain.pt"


def test_ascii_run_exports(tmp_path):
    out = tmp_path / "run"
    _write_run(out, _record("plain.pt"))
    captured = []

    result = _export(out, captured)

    assert captured[0]["provenance"]["grid_scale"] == 1.0
    assert result["path"].endswith("fibers.afv")


def test_ascii_run_exports_under_legacy_locale(tmp_path, monkeypatch):
    out = tmp_path / "run"
    _write_run(out, _record("plain.pt"))
    monkeypatch.setattr(io, "open", _open_text_as_cp1252(io.open))
    captured = []

    _export(out, captured)

    assert captured[0]["provenance"]["checkpoint"] == "plain.pt"


def test_malformed_run_still_raises(tmp_path, monkeypatch):
    out = tmp_path / "run"
    out.mkdir()
    (out / "run.json").write_bytes(b"{")
    monkeypatch.setattr(io, "open", _open_text_as_cp1252(io.open))

    with pytest.raises(json.JSONDecodeError):
        volume_afv.export_afv(out, template="template.afv")
