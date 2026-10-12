"""trace_from_fibers.py reads VC3D fiber JSON with the process locale.

Under cp1252, U+201D in a fiber name raises UnicodeDecodeError before the
family tag is checked, and a UTF-8 BOM raises JSONDecodeError. Related to #1524.
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

SCRIPT = SRC / "vesuvius/neural_tracing/tracers/sheets/scripts/trace_from_fibers.py"
spec = importlib.util.spec_from_file_location("trace_from_fibers_utf8_under_test", SCRIPT)
tracer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tracer
spec.loader.exec_module(tracer)


class _Data:
    volume = "volume"
    volume_level = 0
    point_spacing = 1.0
    step = 1.0
    patch_volumes = []
    patches_dir = []
    hard_windows = None
    fiber_runs = None
    fiber_pool = None


class _Trace:
    confidence_threshold = 0.4


class _Cfg:
    data = _Data()
    trace = _Trace()


class _Volume:
    shape = (8, 8, 8)


def install_stubs():
    tracer.resolve_checkpoint = lambda path: path
    tracer.load_checkpoint = lambda path, device: (object(), _Cfg(), None)
    tracer.checkpoint_source = lambda requested, resolved: "test"
    tracer.ChunkedArray = lambda path, cache: _Volume()
    tracer.C.path_mapper = lambda texts: (lambda cfg: cfg)


def write_fiber(path, *, name, tag):
    path.write_bytes(json.dumps({
        "name": name,
        "hv_classification": {"automatic_tag": tag},
        "line_points": [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
    }, ensure_ascii=False).encode("utf-8"))


def cp1252_text(monkeypatch):
    real = io.open

    def fake(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, *args, **kwargs):
        if "b" not in str(mode) and encoding in (None, "locale"):
            encoding = "cp1252"
        return real(file, mode, buffering, encoding, errors, newline, *args, **kwargs)

    monkeypatch.setattr(io, "open", fake)


def run(tmp_path, capsys):
    install_stubs()
    tracer.main([
        "--fibers", str(tmp_path / "fibers"),
        "--out", str(tmp_path / "out"),
        "--count", "0",
        "--tag", "H",
    ])
    return capsys.readouterr().out


def test_unicode_fiber_name_is_kept_under_cp1252(tmp_path, monkeypatch, capsys):
    cp1252_text(monkeypatch)
    src = tmp_path / "fibers"
    src.mkdir()
    write_fiber(src / "scroll.json", name="scroll \u201d", tag="H")
    write_fiber(src / "other.json", name="plain", tag="Z")
    assert run(tmp_path, capsys).startswith("1 H fibers")


def test_utf8_bom_fiber_json_is_read(tmp_path, capsys):
    src = tmp_path / "fibers"
    src.mkdir()
    body = json.dumps({
        "name": "bom",
        "hv_classification": {"automatic_tag": "H"},
        "line_points": [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
    }).encode("utf-8")
    (src / "bom.json").write_bytes(b"\xef\xbb\xbf" + body)
    assert run(tmp_path, capsys).startswith("1 H fibers")


def test_tag_outside_family_is_skipped(tmp_path, monkeypatch, capsys):
    cp1252_text(monkeypatch)
    src = tmp_path / "fibers"
    src.mkdir()
    write_fiber(src / "skip.json", name="plain", tag="Z")
    assert run(tmp_path, capsys).startswith("0 H fibers")


def test_malformed_fiber_json_still_raises(tmp_path, monkeypatch, capsys):
    cp1252_text(monkeypatch)
    src = tmp_path / "fibers"
    src.mkdir()
    (src / "bad.json").write_text("{", encoding="ascii")
    with pytest.raises(json.JSONDecodeError):
        run(tmp_path, capsys)


def test_ascii_fiber_is_kept(tmp_path, capsys):
    src = tmp_path / "fibers"
    src.mkdir()
    write_fiber(src / "ok.json", name="plain", tag="H")
    assert run(tmp_path, capsys).startswith("1 H fibers")
