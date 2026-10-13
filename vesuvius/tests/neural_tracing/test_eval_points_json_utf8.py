"""Neural-tracing evaluation reads its points JSON with the process locale.

Under cp1252, U+201D in a volume name or a volume path raises
UnicodeDecodeError before that string is used, and a UTF-8 BOM raises
JSONDecodeError. Those strings should still be kept. A missing volume
is still reported. A missing scale is still reported. A missing paths
directory is still reported.
Related to #1524.
"""
import importlib.util
import json
import sys
import types
from pathlib import Path

import builtins
import click
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "src/vesuvius/neural_tracing/evaluation/evaluate.py"

trace_mod = types.ModuleType("vesuvius.neural_tracing.trace")
trace_mod.trace = types.SimpleNamespace(main=lambda *args, **kwargs: None)
neural = types.ModuleType("vesuvius.neural_tracing")
neural.trace = trace_mod
vesuvius = types.ModuleType("vesuvius")
vesuvius.neural_tracing = neural
sys.modules["vesuvius"] = vesuvius
sys.modules["vesuvius.neural_tracing"] = neural
sys.modules["vesuvius.neural_tracing.trace"] = trace_mod

spec = importlib.util.spec_from_file_location("eval_points_json_utf8_under_test", SCRIPT)
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


def run(tmp_path, document, *, raw=None, bom=False, checkpoints=("ckpt",)):
    points_path = tmp_path / "points.json"
    write_json(points_path, document, bom=bom, raw=raw)
    return mod.main.callback(
        points_path=str(points_path),
        prefix="eval",
        checkpoint_paths=checkpoints,
        steps_per_crop=1,
        max_size=60,
    )


def test_unicode_volume_name_is_kept(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    with pytest.raises(click.UsageError, match="scroll \u201d"):
        run(tmp_path, [{"name": "scroll \u201d"}])


def test_unicode_volume_path_is_traced(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    seen = {}

    def stop(args, standalone_mode=False):
        seen["args"] = list(args)
        raise RuntimeError("stopped")

    monkeypatch.setattr(trace_mod.trace, "main", stop)
    document = [{
        "name": "scroll",
        "volume_zarr": "scroll \u201d.zarr",
        "volume_scale": 2,
        "paths_dir": "/paths",
        "point_xyzs": [[4, 5, 6]],
    }]
    with pytest.raises(RuntimeError, match="stopped"):
        run(tmp_path, document)
    assert "scroll \u201d.zarr" in seen["args"]
    assert "--volume_scale" in seen["args"]
    assert seen["args"][seen["args"].index("--volume_scale") + 1] == "2"
    assert "--uuid" in seen["args"]
    assert "nt-eval_4-5-6_eval_ckpt" in seen["args"]


def test_utf8_bom_points_is_kept(tmp_path):
    with pytest.raises(click.UsageError, match="volume_zarr"):
        run(tmp_path, [{"name": "scroll"}], bom=True)


def test_ascii_name_still_reports_missing_volume(tmp_path):
    with pytest.raises(click.UsageError, match="scroll"):
        run(tmp_path, [{"name": "scroll"}])


def test_malformed_points_still_raises(tmp_path):
    with pytest.raises(json.JSONDecodeError):
        run(tmp_path, None, raw=b"{")


def test_missing_scale_still_raises(tmp_path):
    with pytest.raises(click.UsageError, match="volume_scale"):
        run(tmp_path, [{"name": "scroll", "volume_zarr": "/vol"}])


def test_missing_paths_dir_still_raises(tmp_path):
    with pytest.raises(click.UsageError, match="paths_dir"):
        run(tmp_path, [{"name": "scroll", "volume_zarr": "/vol", "volume_scale": 1}])


def test_ascii_volume_still_traces(tmp_path, monkeypatch):
    seen = {}

    def stop(args, standalone_mode=False):
        seen["args"] = list(args)
        raise RuntimeError("stopped")

    monkeypatch.setattr(trace_mod.trace, "main", stop)
    document = [{
        "name": "scroll",
        "volume_zarr": "/vol",
        "volume_scale": 1,
        "paths_dir": "/paths",
        "point_xyzs": [[1, 2, 3]],
    }]
    with pytest.raises(RuntimeError, match="stopped"):
        run(tmp_path, document, checkpoints=("models/ckpt.pt",))
    assert "/vol" in seen["args"]
    assert "--max_size" in seen["args"]
    assert seen["args"][seen["args"].index("--max_size") + 1] == "60"
    assert "nt-eval_1-2-3_eval_ckpt.pt" in seen["args"]
