"""build_fiber_pool.py reads VC3D source fiber JSON with the process locale.

Under cp1252, U+201D in a fiber name raises UnicodeDecodeError before the
family tag is checked, and a UTF-8 BOM raises JSONDecodeError. Related to #1524.
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

SCRIPT = SRC / "vesuvius/neural_tracing/tracers/sheets/scripts/build_fiber_pool.py"
spec = importlib.util.spec_from_file_location("build_fiber_pool_utf8_under_test", SCRIPT)
pool = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pool
spec.loader.exec_module(pool)


def make_afv(path, fibers):
    c = sqlite3.connect(path)
    c.execute("PRAGMA application_id = %d" % 0x56434643)
    c.execute("PRAGMA user_version = 1")
    for sql in AFV_SCHEMA:
        c.execute(sql)
    for fid, (name, family, xyz) in enumerate(fibers, 1):
        c.execute(
            "INSERT INTO fibers VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (fid, name, family, len(xyz), 0.0, *[0.0] * 6, json.dumps({"source": "test"})),
        )
        c.execute(
            "INSERT INTO blocks VALUES (?,?,?,?)",
            (fid, fid, 0, np.ascontiguousarray(xyz, "<f8").tobytes()),
        )
        c.execute(
            "INSERT INTO block_bounds VALUES (?,?,?,?,?,?,?)",
            (fid, *[v for k in range(3) for v in (xyz[:, k].min(), xyz[:, k].max())]),
        )
    meta = dict(
        frame=dict(vc_open_data_coordinate_space="test/1"),
        root={},
        fiber_count=len(fibers),
        point_count=sum(len(f[2]) for f in fibers),
        uuid="u",
    )
    c.executemany("INSERT INTO metadata VALUES (?, ?)", [(k, json.dumps(v)) for k, v in meta.items()])
    c.execute("INSERT INTO metadata VALUES ('complete', 'true')")
    c.commit()
    c.close()


def line(y, n=21):
    x = np.linspace(0.0, 80.0, n)
    return np.c_[x, np.full(n, y), np.zeros(n)]


def write_fiber(path, *, name, tag, points):
    path.write_bytes(json.dumps({
        "name": name,
        "hv_classification": {"automatic_tag": tag},
        "line_points": points.tolist(),
    }, ensure_ascii=False).encode("utf-8"))


def cp1252_text(monkeypatch):
    real = io.open

    def fake(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, *args, **kwargs):
        if "b" not in str(mode) and encoding in (None, "locale"):
            encoding = "cp1252"
        return real(file, mode, buffering, encoding, errors, newline, *args, **kwargs)

    monkeypatch.setattr(io, "open", fake)


def run(tmp_path, monkeypatch, *, locale=False):
    if locale:
        cp1252_text(monkeypatch)
    out = tmp_path / "pool"
    src = tmp_path / "json"
    pool.main([
        "--afv", str(tmp_path / "one.afv"),
        "--fiber-jsons", str(src),
        "--out", str(out),
        "--grid-scale", "4",
        "--sample", "5",
    ])
    return json.loads((out / "keys.json").read_text(encoding="utf-8"))


def test_unicode_fiber_name_is_kept_under_cp1252(tmp_path, monkeypatch):
    make_afv(tmp_path / "one.afv", [("away", "V", line(400.0))])
    src = tmp_path / "json"
    src.mkdir()
    write_fiber(src / "scroll.json", name="scroll \u201d", tag="H", points=line(0.0))
    write_fiber(src / "other.json", name="plain", tag="Z", points=line(0.0))
    keys = run(tmp_path, monkeypatch, locale=True)
    assert keys == ["json:scroll", "afv:1"]
    assert json.loads((tmp_path / "pool" / "aliases.json").read_text(encoding="utf-8")) == {}


def test_utf8_bom_fiber_json_is_read(tmp_path, monkeypatch):
    make_afv(tmp_path / "one.afv", [("away", "V", line(400.0))])
    src = tmp_path / "json"
    src.mkdir()
    body = json.dumps({
        "name": "bom",
        "hv_classification": {"automatic_tag": "H"},
        "line_points": line(0.0).tolist(),
    }).encode("utf-8")
    (src / "bom.json").write_bytes(b"\xef\xbb\xbf" + body)
    keys = run(tmp_path, monkeypatch)
    assert keys == ["json:bom", "afv:1"]


def test_tag_outside_family_is_skipped(tmp_path, monkeypatch):
    make_afv(tmp_path / "one.afv", [("away", "V", line(400.0))])
    src = tmp_path / "json"
    src.mkdir()
    write_fiber(src / "skip.json", name="plain", tag="Z", points=line(0.0))
    keys = run(tmp_path, monkeypatch, locale=True)
    assert keys == ["afv:1"]


def test_malformed_fiber_json_still_raises(tmp_path, monkeypatch):
    make_afv(tmp_path / "one.afv", [("away", "V", line(400.0))])
    src = tmp_path / "json"
    src.mkdir()
    (src / "bad.json").write_text("{", encoding="ascii")
    with pytest.raises(json.JSONDecodeError):
        run(tmp_path, monkeypatch, locale=True)


def test_ascii_fiber_is_kept(tmp_path, monkeypatch):
    make_afv(tmp_path / "one.afv", [("away", "V", line(400.0))])
    src = tmp_path / "json"
    src.mkdir()
    write_fiber(src / "ok.json", name="plain", tag="H", points=line(0.0))
    keys = run(tmp_path, monkeypatch)
    assert keys == ["json:ok", "afv:1"]
