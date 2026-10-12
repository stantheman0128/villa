"""benchmark.rescore reads report.json with the process locale.

Under cp1252, U+201D in a model name raises UnicodeDecodeError before the
switch counts are checked, and a UTF-8 BOM raises JSONDecodeError.
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

SCRIPT = SRC / "vesuvius/neural_tracing/tracers/sheets/evaluation/benchmark.py"
spec = importlib.util.spec_from_file_location("bench_rescore_utf8_under_test", SCRIPT)
bench = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bench
spec.loader.exec_module(bench)


def document(*, model="plain", name="plain", held_out=False):
    report = {
        "name": name,
        "steps": 4,
        "seeds": [],
        "models": {
            model: {
                "first_switch": [],
                "summary": {
                    "traces_lost": 0,
                    "coverage_mean": 1.0,
                    "precision_mean": 1.0,
                },
            }
        },
    }
    if held_out:
        report["held_out_scoring"] = True
    return report


def write_json(path, payload):
    path.write_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"))


def cp1252_text(monkeypatch):
    real = io.open

    def fake(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, *args, **kwargs):
        if "b" not in str(mode) and encoding in (None, "locale"):
            encoding = "cp1252"
        return real(file, mode, buffering, encoding, errors, newline, *args, **kwargs)

    monkeypatch.setattr(io, "open", fake)


def test_unicode_model_name_is_read_under_cp1252(tmp_path, monkeypatch, capsys):
    cp1252_text(monkeypatch)
    folder = tmp_path / "bench"
    folder.mkdir()
    write_json(folder / "report.json", document(name="scroll \u201d", model="scroll \u201d"))
    bench.rescore(folder)
    assert "scroll \u201d" in capsys.readouterr().out
    saved = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    assert "scroll \u201d" in saved["models"]


def test_utf8_bom_report_is_read(tmp_path, capsys):
    folder = tmp_path / "bench"
    folder.mkdir()
    (folder / "report.json").write_bytes(b"\xef\xbb\xbf" + json.dumps(document(model="bom")).encode("utf-8"))
    bench.rescore(folder)
    assert "bom" in capsys.readouterr().out
    saved = (folder / "report.json").read_bytes()
    assert not saved.startswith(b"\xef\xbb\xbf")


def test_malformed_report_still_raises(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    folder = tmp_path / "bench"
    folder.mkdir()
    (folder / "report.json").write_text("{", encoding="ascii")
    with pytest.raises(json.JSONDecodeError):
        bench.rescore(folder)


def test_held_out_report_still_raises(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    folder = tmp_path / "bench"
    folder.mkdir()
    write_json(folder / "report.json", document(held_out=True))
    with pytest.raises(ValueError, match="held-out-scoring"):
        bench.rescore(folder)


def test_ascii_model_name_is_rescored(tmp_path, capsys):
    folder = tmp_path / "bench"
    folder.mkdir()
    write_json(folder / "report.json", document(model="plain"))
    bench.rescore(folder)
    assert "plain" in capsys.readouterr().out
    saved = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    assert saved["models"]["plain"]["first_real_switch"] == []
