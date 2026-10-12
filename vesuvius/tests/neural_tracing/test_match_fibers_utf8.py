"""match_fibers_to_patches.py reads umbilicus and fiber JSON with the process locale.

Under cp1252, U+201D in a name raises UnicodeDecodeError before the control
points or the family tag are used, and a UTF-8 BOM raises JSONDecodeError.
Related to #1524.
"""
import importlib.util
import io
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from vesuvius.neural_tracing.tracers.shared.annotations.afv import AFV_SCHEMA

SCRIPT = SRC / "vesuvius/neural_tracing/tracers/sheets/scripts/match_fibers_to_patches.py"
spec = importlib.util.spec_from_file_location("match_fibers_utf8_under_test", SCRIPT)
matcher = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = matcher
spec.loader.exec_module(matcher)


class Info:
    cell = 8.0
    root = "root"
    name = "p0"
    path = Path(".")


class _Pool:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def imap_unordered(self, fn, jobs, chunksize=1):
        return []


class _Ctx:
    def Pool(self, workers):
        return _Pool()


def install_stubs():
    matcher.build_index = lambda dirs, cache: [Info()]
    matcher.patch_nodes = lambda info: (
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 2.0]], np.float32),
        np.tile(np.array([[0.0, 0.0, 1.0]], np.float32), (3, 1)),
        np.array([[0, 0], [0, 1], [1, 0]], np.int32),
    )
    matcher.multiprocessing.get_context = lambda kind: _Ctx()


def make_afv(path):
    xyz = np.c_[np.linspace(0.0, 40.0, 8), np.full(8, 400.0), np.zeros(8)]
    c = sqlite3.connect(path)
    c.execute("PRAGMA application_id = %d" % 0x56434643)
    c.execute("PRAGMA user_version = 1")
    for sql in AFV_SCHEMA:
        c.execute(sql)
    c.execute(
        "INSERT INTO fibers VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (1, "away", "V", len(xyz), 0.0, *[0.0] * 6, json.dumps({"source": "test"})),
    )
    c.execute("INSERT INTO blocks VALUES (?,?,?,?)", (1, 1, 0, np.ascontiguousarray(xyz, "<f8").tobytes()))
    c.execute(
        "INSERT INTO block_bounds VALUES (?,?,?,?,?,?,?)",
        (1, *[v for k in range(3) for v in (xyz[:, k].min(), xyz[:, k].max())]),
    )
    meta = dict(frame=dict(vc_open_data_coordinate_space="test/1"), root={}, fiber_count=1, point_count=len(xyz), uuid="u")
    c.executemany("INSERT INTO metadata VALUES (?, ?)", [(k, json.dumps(v)) for k, v in meta.items()])
    c.execute("INSERT INTO metadata VALUES ('complete', 'true')")
    c.commit()
    c.close()


def umbilicus(*, name="axis"):
    return {
        "name": name,
        "control_points": [
            {"x": 0.0, "y": 0.0, "z": 0.0},
            {"x": 0.0, "y": 0.0, "z": 10.0},
        ],
    }


def fiber(*, name="plain", tag="H"):
    return {
        "name": name,
        "hv_classification": {"automatic_tag": tag},
        "line_points": [[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
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


def run(tmp_path, capsys, *, fiber_dir=None):
    install_stubs()
    argv = [
        "--afv", str(tmp_path / "one.afv"),
        "--patches-dir", str(tmp_path / "patches"),
        "--umbilicus", str(tmp_path / "umbilicus.json"),
        "--out", str(tmp_path / "matches.jsonl"),
        "--family", "H",
        "--grid-scale", "4",
    ]
    if fiber_dir is not None:
        argv.extend(["--fiber-jsons", str(fiber_dir)])
    matcher.main(argv)
    return capsys.readouterr().out


def test_unicode_umbilicus_name_is_read_under_cp1252(tmp_path, monkeypatch, capsys):
    cp1252_text(monkeypatch)
    make_afv(tmp_path / "one.afv")
    write_json(tmp_path / "umbilicus.json", umbilicus(name="scroll \u201d"))
    out = run(tmp_path, capsys)
    assert "1 patches, 3 nodes" in out
    assert "0 H fibers of 1" in out


def test_utf8_bom_umbilicus_is_read(tmp_path, capsys):
    make_afv(tmp_path / "one.afv")
    (tmp_path / "umbilicus.json").write_bytes(b"\xef\xbb\xbf" + json.dumps(umbilicus(name="bom")).encode("utf-8"))
    assert "1 patches, 3 nodes" in run(tmp_path, capsys)


def test_unicode_fiber_name_is_kept_under_cp1252(tmp_path, monkeypatch, capsys):
    cp1252_text(monkeypatch)
    make_afv(tmp_path / "one.afv")
    write_json(tmp_path / "umbilicus.json", umbilicus())
    src = tmp_path / "json"
    src.mkdir()
    write_json(src / "scroll.json", fiber(name="scroll \u201d", tag="H"))
    write_json(src / "other.json", fiber(name="plain", tag="Z"))
    assert "1 H fibers from" in run(tmp_path, capsys, fiber_dir=src)


def test_utf8_bom_fiber_json_is_read(tmp_path, capsys):
    make_afv(tmp_path / "one.afv")
    write_json(tmp_path / "umbilicus.json", umbilicus())
    src = tmp_path / "json"
    src.mkdir()
    (src / "bom.json").write_bytes(b"\xef\xbb\xbf" + json.dumps(fiber(name="bom", tag="H")).encode("utf-8"))
    assert "1 H fibers from" in run(tmp_path, capsys, fiber_dir=src)


def test_tag_outside_family_is_skipped(tmp_path, monkeypatch, capsys):
    cp1252_text(monkeypatch)
    make_afv(tmp_path / "one.afv")
    write_json(tmp_path / "umbilicus.json", umbilicus())
    src = tmp_path / "json"
    src.mkdir()
    write_json(src / "skip.json", fiber(name="plain", tag="Z"))
    assert "0 H fibers from" in run(tmp_path, capsys, fiber_dir=src)


def test_malformed_fiber_json_still_raises(tmp_path, monkeypatch, capsys):
    cp1252_text(monkeypatch)
    make_afv(tmp_path / "one.afv")
    write_json(tmp_path / "umbilicus.json", umbilicus())
    src = tmp_path / "json"
    src.mkdir()
    (src / "bad.json").write_text("{", encoding="ascii")
    with pytest.raises(json.JSONDecodeError):
        run(tmp_path, capsys, fiber_dir=src)


def test_malformed_umbilicus_still_raises(tmp_path, monkeypatch, capsys):
    cp1252_text(monkeypatch)
    make_afv(tmp_path / "one.afv")
    (tmp_path / "umbilicus.json").write_text("{", encoding="ascii")
    with pytest.raises(json.JSONDecodeError):
        run(tmp_path, capsys)


def test_ascii_fiber_is_kept(tmp_path, capsys):
    make_afv(tmp_path / "one.afv")
    write_json(tmp_path / "umbilicus.json", umbilicus())
    src = tmp_path / "json"
    src.mkdir()
    write_json(src / "ok.json", fiber(name="plain", tag="H"))
    assert "1 H fibers from" in run(tmp_path, capsys, fiber_dir=src)
