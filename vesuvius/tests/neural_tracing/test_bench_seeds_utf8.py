"""benchmark.seeds_from reads an earlier report with the process locale.

Under cp1252, U+201D in a seed root raises UnicodeDecodeError before the
patch is looked up, and a UTF-8 BOM raises JSONDecodeError.
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
spec = importlib.util.spec_from_file_location("bench_seeds_utf8_under_test", SCRIPT)
bench = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bench
spec.loader.exec_module(bench)


class Info:
    def __init__(self, root, name):
        self.root = root
        self.name = name


class Sampler:
    def __init__(self, infos):
        self.index = infos
        self.spec = "spec"
        self.calls = []

    def _episode(self, info, axis, sign_u, sign_v, anchor):
        self.calls.append((info.root, info.name, axis, sign_u, sign_v, tuple(anchor)))
        return {"root": info.root, "patch": info.name}


def install(infos):
    sampler = Sampler(infos)

    def stub(*args, **kwargs):
        return sampler, {"point_spacing": 20.0}

    bench._seed_sampler = stub
    return sampler


def document(seeds, name="plain"):
    return {"name": name, "seeds": seeds}


def seed(root, patch, axis=2):
    return {
        "root": root,
        "patch": patch,
        "axis": axis,
        "sign_u": 1,
        "sign_v": -1,
        "anchor": [1, 2, 3],
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


def call(report):
    return bench.seeds_from(report, object(), steps=4, front=40.0, min_valid=0.5)


def test_unicode_seed_root_is_read_under_cp1252(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    root = "scroll \u201d"
    sampler = install([Info(root, "p0")])
    report = tmp_path / "report.json"
    write_json(report, document([seed(root, "p0")], name=root))
    seeds, spec, data, kept = call(report)
    assert spec == "spec"
    assert data == {"point_spacing": 20.0}
    assert kept == [0]
    assert seeds == [{"root": root, "patch": "p0"}]
    assert sampler.calls == [(root, "p0", 2, 1, -1, (1, 2, 3))]


def test_utf8_bom_report_is_read(tmp_path):
    install([Info("plain", "p0")])
    report = tmp_path / "report.json"
    report.write_bytes(b"\xef\xbb\xbf" + json.dumps(document([seed("plain", "p0")])).encode("utf-8"))
    seeds, spec, data, kept = call(report)
    assert kept == [0]
    assert seeds == [{"root": "plain", "patch": "p0"}]


def test_malformed_report_still_raises(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    install([Info("plain", "p0")])
    report = tmp_path / "report.json"
    report.write_text("{", encoding="ascii")
    with pytest.raises(json.JSONDecodeError):
        call(report)


def test_unknown_patch_is_dropped(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    sampler = install([Info("plain", "p0")])
    report = tmp_path / "report.json"
    write_json(report, document([seed("plain", "missing")]))
    seeds, spec, data, kept = call(report)
    assert seeds == []
    assert kept == []
    assert sampler.calls == []


def test_ascii_seed_is_kept(tmp_path):
    install([Info("plain", "p0"), Info("plain", "p1")])
    report = tmp_path / "report.json"
    write_json(report, document([seed("plain", "p1"), seed("other", "p0")]))
    seeds, spec, data, kept = call(report)
    assert kept == [0]
    assert seeds == [{"root": "plain", "patch": "p1"}]
