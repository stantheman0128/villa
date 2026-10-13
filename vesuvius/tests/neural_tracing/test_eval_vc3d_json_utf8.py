"""VC3D evaluation reads its JSON inputs with the process locale.

Under cp1252, U+201D in a volume name or a trace socket raises
UnicodeDecodeError before that string is used, and a UTF-8 BOM raises
JSONDecodeError. Those strings should still be kept. A missing volume
is still reported. A missing socket is still reported. A missing grow
binary is still reported.
Related to #1524.
"""
import importlib.util
import json
import sys
from pathlib import Path

import builtins
import click
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "src/vesuvius/neural_tracing/evaluation/evaluate_vc3d.py"
spec = importlib.util.spec_from_file_location("eval_vc3d_json_utf8_under_test", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def cp1252_text(monkeypatch):
    real = builtins.open

    def fake(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, *args, **kwargs):
        if "b" not in str(mode) and encoding in (None, "locale"):
            encoding = "cp1252"
        return real(file, mode, buffering, encoding, errors, newline, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake)


def write_json(path, document, *, bom=False, raw=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = raw if raw is not None else json.dumps(document, ensure_ascii=False).encode("utf-8")
    if bom:
        payload = b"\xef\xbb\xbf" + payload
    path.write_bytes(payload)


def points(name="scroll", **extra):
    document = {"name": name}
    document.update(extra)
    return [document]


def complete_points(name="scroll"):
    return points(
        name,
        volume_zarr="/vol",
        volume_scale=1,
        paths_dir="/paths",
        point_xyzs=[[1, 2, 3]],
    )


def run(tmp_path, points_document, params_document, *, grow=True, raw_points=None, raw_params=None, bom_points=False, bom_params=False):
    points_path = tmp_path / "points.json"
    params_path = tmp_path / "params.json"
    write_json(points_path, points_document, bom=bom_points, raw=raw_points)
    write_json(params_path, params_document, bom=bom_params, raw=raw_params)
    checkpoint = tmp_path / "ckpt"
    checkpoint.write_bytes(b"x")
    grow_path = tmp_path / "grow" if grow else tmp_path / "missing-grow"
    if grow:
        grow_path.write_bytes(b"x")
    return mod.main.callback(
        points_path=str(points_path),
        prefix="eval",
        checkpoint_path=str(checkpoint),
        params_path=str(params_path),
        grow_seg_bin=str(grow_path),
    )


def test_unicode_volume_name_is_kept(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    with pytest.raises(click.UsageError, match="scroll \u201d"):
        run(tmp_path, points("scroll \u201d"), {"neural_socket": "/tmp/sock"})


def test_unicode_socket_is_kept(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    seen = {}

    def start(**kwargs):
        seen.update(kwargs)
        raise RuntimeError("stopped")

    monkeypatch.setattr(mod, "_start_trace_service", start)
    with pytest.raises(RuntimeError, match="stopped"):
        run(tmp_path, complete_points(), {"neural_socket": "scroll \u201d.sock"})
    assert seen["socket_path"] == "scroll \u201d.sock"
    assert seen["volume_zarr"] == "/vol"


def test_utf8_bom_points_is_kept(tmp_path):
    with pytest.raises(click.UsageError, match="volume_zarr"):
        run(tmp_path, points("scroll"), {"neural_socket": "/tmp/sock"}, bom_points=True)


def test_utf8_bom_params_is_kept(tmp_path):
    with pytest.raises(click.UsageError, match="volume_zarr"):
        run(tmp_path, points("scroll"), {"neural_socket": "/tmp/sock"}, bom_params=True)


def test_ascii_name_still_reports_missing_volume(tmp_path):
    with pytest.raises(click.UsageError, match="scroll"):
        run(tmp_path, points("scroll"), {"neural_socket": "/tmp/sock"})


def test_malformed_points_still_raises(tmp_path):
    with pytest.raises(json.JSONDecodeError):
        run(tmp_path, None, {"neural_socket": "/tmp/sock"}, raw_points=b"{")


def test_missing_socket_still_raises(tmp_path):
    with pytest.raises(click.UsageError, match="neural_socket"):
        run(tmp_path, points("scroll"), {})


def test_missing_grow_bin_still_raises(tmp_path):
    with pytest.raises(click.UsageError, match="not found"):
        run(tmp_path, points("scroll"), {"neural_socket": "/tmp/sock"}, grow=False)


def test_ascii_socket_still_starts(tmp_path, monkeypatch):
    seen = {}

    def start(**kwargs):
        seen.update(kwargs)
        raise RuntimeError("stopped")

    monkeypatch.setattr(mod, "_start_trace_service", start)
    with pytest.raises(RuntimeError, match="stopped"):
        run(tmp_path, complete_points(), {"neural_socket": "/tmp/sock"})
    assert seen["socket_path"] == "/tmp/sock"
    assert seen["volume_scale"] == 1
