"""Scatter resume reads cursor.json with the process locale.

Under cp1252, U+201D in a note stored beside the checkpoint time raises
UnicodeDecodeError before the next block is used, and a UTF-8 BOM raises
JSONDecodeError. The document should still be kept. A newer cursor still
wins, and a finished scatter still skips the cursors.
Related to #1524.
"""
import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
SCRIPT = ROOT / "src/vesuvius/neural_tracing/winding_models/merge_winding_field.py"
spec = importlib.util.spec_from_file_location("scatter_cursor_utf8_under_test", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


class Stop(Exception):
    """Raised once the cursor has been accepted and rays would be built."""


def _load(self, directory, _device):
    self.loaded_from = directory


class _Rays:
    def __init__(self, _archive, _device):
        raise Stop("past the cursor")


mod.ScatterAccumulator.load = _load
mod.RaysGpu = _Rays

BAND = mod.BandGeometry((32, 32, 32), 0, 32)


def call(scratch):
    cfg = type("Cfg", (), {"winding_center": 0.0})()
    return mod.run_scatter(None, None, [], BAND, cfg, "cpu", scratch, None)


def cp1252_text(monkeypatch):
    real = io.open

    def fake(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, *args, **kwargs):
        if "b" not in str(mode) and encoding in (None, "locale"):
            encoding = "cp1252"
        return real(file, mode, buffering, encoding, errors, newline, *args, **kwargs)

    monkeypatch.setattr(io, "open", fake)


def write_cursor(directory, payload, bom=False):
    directory.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if bom:
        raw = b"\xef\xbb\xbf" + raw
    (directory / "cursor.json").write_bytes(raw)


def test_unicode_note_still_resumes(tmp_path, monkeypatch, capsys):
    cp1252_text(monkeypatch)
    write_cursor(tmp_path / "scatter_ckpt_a", {"time": 2.0, "next_position": 6, "label": "scroll \u201d"})
    with pytest.raises(Stop):
        call(tmp_path)
    assert "block 6" in capsys.readouterr().out


def test_utf8_bom_still_resumes(tmp_path, capsys):
    write_cursor(tmp_path / "scatter_ckpt_a", {"time": 2.0, "next_position": 4}, bom=True)
    with pytest.raises(Stop):
        call(tmp_path)
    assert "block 4" in capsys.readouterr().out


def test_ascii_cursor_still_resumes(tmp_path, capsys):
    write_cursor(tmp_path / "scatter_ckpt_a", {"time": 1.0, "next_position": 4})
    with pytest.raises(Stop):
        call(tmp_path)
    assert "block 4" in capsys.readouterr().out


def test_newer_cursor_still_wins(tmp_path, capsys):
    write_cursor(tmp_path / "scatter_ckpt_a", {"time": 1.0, "next_position": 2})
    write_cursor(tmp_path / "scatter_ckpt_b", {"time": 5.0, "next_position": 9})
    with pytest.raises(Stop):
        call(tmp_path)
    out = capsys.readouterr().out
    assert "scatter_ckpt_b" in out
    assert "block 9" in out


def test_malformed_cursor_still_raises(tmp_path):
    directory = tmp_path / "scatter_ckpt_a"
    directory.mkdir()
    (directory / "cursor.json").write_text("{", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        call(tmp_path)


def test_finished_scatter_still_skips_the_cursor(tmp_path):
    done = tmp_path / "scatter"
    done.mkdir()
    (done / "complete.json").write_text("{}", encoding="utf-8")
    write_cursor(tmp_path / "scatter_ckpt_a", {"time": 1.0, "next_position": 3, "label": "scroll \u201d"})
    accumulator = call(tmp_path)
    assert accumulator.loaded_from == done
