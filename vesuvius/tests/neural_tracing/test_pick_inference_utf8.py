"""Inference-report JSON is UTF-8. A legacy text open must not drop it.

``pick_inference`` reads each ``report.json`` with ``Path.read_text()`` and
no encoding. On cp1252 a seed name containing U+201D raises
``UnicodeDecodeError`` before any configuration is ranked. A UTF-8 BOM is
invalid JSON. An ASCII report still ranks, and a malformed file still
raises.
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "vesuvius"
    / "neural_tracing"
    / "tracers"
    / "sheets"
    / "scripts"
    / "pick_inference.py"
)
_SEED = "scroll \u201d"


def _load():
    spec = importlib.util.spec_from_file_location("pick_inference_utf8_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pick = _load()


def _open_text_as_cp1252(real_open):
    def wrapped(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, *args, **kwargs):
        if isinstance(mode, str) and "b" not in mode and encoding in (None, "locale"):
            encoding = "cp1252"
        return real_open(file, mode, buffering, encoding, errors, newline, *args, **kwargs)

    return wrapped


def _report(seed_name: str) -> dict:
    return {
        "front": 60,
        "seeds": [{"name": seed_name}],
        "models": {
            "sheet": {
                "summary": {
                    "points_covered": 10,
                    "off_kept_points": 1,
                    "coverage_mean": 0.5,
                    "precision_mean": 0.9,
                    "traces_real_switch": 2,
                    "traces_lost": 0,
                }
            }
        },
    }


def _write_report(folder: Path, document: dict, *, bom: bool = False) -> None:
    raw = json.dumps(document, ensure_ascii=False).encode("utf-8")
    if bom:
        raw = b"\xef\xbb\xbf" + raw
    folder.mkdir()
    (folder / "report.json").write_bytes(raw)


def _rank(folder: Path, capsys):
    pick.main(["--model", "sheet", str(folder)])
    return capsys.readouterr().out


def test_legacy_locale_keeps_unicode_seed_name(tmp_path, monkeypatch, capsys):
    folder = tmp_path / "sweep_t0.30"
    _write_report(folder, _report(_SEED))
    monkeypatch.setattr(io, "open", _open_text_as_cp1252(io.open))

    out = _rank(folder, capsys)

    assert "0.500" in out
    assert "10" in out


def test_utf8_bom_report_still_ranks(tmp_path, monkeypatch, capsys):
    folder = tmp_path / "sweep_t0.30"
    _write_report(folder, _report("plain"), bom=True)
    monkeypatch.setattr(io, "open", _open_text_as_cp1252(io.open))

    out = _rank(folder, capsys)

    assert "0.500" in out


def test_ascii_report_ranks(tmp_path, capsys):
    folder = tmp_path / "sweep_t0.30"
    _write_report(folder, _report("plain"))

    out = _rank(folder, capsys)

    assert "0.500" in out
    assert "10" in out


def test_ascii_report_ranks_under_legacy_locale(tmp_path, monkeypatch, capsys):
    folder = tmp_path / "sweep_t0.30"
    _write_report(folder, _report("plain"))
    monkeypatch.setattr(io, "open", _open_text_as_cp1252(io.open))

    out = _rank(folder, capsys)

    assert "0.500" in out


def test_malformed_report_still_raises(tmp_path, monkeypatch):
    folder = tmp_path / "sweep_t0.30"
    folder.mkdir()
    (folder / "report.json").write_bytes(b"{")
    monkeypatch.setattr(io, "open", _open_text_as_cp1252(io.open))

    with pytest.raises(json.JSONDecodeError):
        pick.main(["--model", "sheet", str(folder)])
