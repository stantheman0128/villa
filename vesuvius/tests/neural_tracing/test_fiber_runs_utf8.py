"""EpisodeSampler reads fiber-run JSONL with the process locale.

Under cp1252, U+201D in a patch root raises UnicodeDecodeError before the
run is matched, and a UTF-8 BOM raises JSONDecodeError. A long enough run
whose patch is in the index should be kept.
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
SCRIPT = ROOT / "src/vesuvius/neural_tracing/tracers/sheets/data/episodes.py"
spec = importlib.util.spec_from_file_location("fiber_runs_utf8_under_test", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


class Data:
    history = 1
    step = 10.0
    gt_margin = 0.0
    front_points = 3
    point_spacing = 10.0
    gt_resolution = 10.0
    patch_weight_power = 0.0
    dir_weights = None
    volume_level = 0
    hard_fraction = 0
    fiber_follow = False
    fiber_fraction = 1
    split_block = 64
    val_fraction = 0.0
    split_seed = 0


def patch(root):
    return mod.PatchInfo(
        name="p0", rows=40, cols=40, cell=2.0, valid=10,
        bbox=(0, 0, 0, 10, 10, 10), root=root,
    )


def run_line(root, *, end=40.0, patch_name="p0"):
    return {
        "root": root,
        "patch": patch_name,
        "fiber": "f1",
        "start": 0.0,
        "end": end,
        "xyz": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        "rc": [[0.0, 0.0], [1.0, 0.0]],
    }


def write_jsonl(path, payload, bom=False):
    raw = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
    if bom:
        raw = b"\xef\xbb\xbf" + raw
    path.write_bytes(raw)


def cp1252_text(monkeypatch):
    real = io.open

    def fake(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, *args, **kwargs):
        if "b" not in str(mode) and encoding in (None, "locale"):
            encoding = "cp1252"
        return real(file, mode, buffering, encoding, errors, newline, *args, **kwargs)

    monkeypatch.setattr(io, "open", fake)


def sampler(path, root):
    data = Data()
    data.fiber_runs = str(path)
    return mod.EpisodeSampler(data, [patch(root)], "train", 2, 64, 0.0)


def test_unicode_root_keeps_the_fiber_run(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    root = "scroll \u201d/patches"
    path = tmp_path / "runs.jsonl"
    write_jsonl(path, run_line(root))
    found = sampler(path, root)
    assert [(run["fiber"], run["root"]) for run in found.fiber_runs] == [("f1", root)]


def test_utf8_bom_keeps_the_fiber_run(tmp_path):
    root = "plain/patches"
    path = tmp_path / "runs.jsonl"
    write_jsonl(path, run_line(root), bom=True)
    found = sampler(path, root)
    assert [run["fiber"] for run in found.fiber_runs] == ["f1"]


def test_malformed_line_still_raises(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    path = tmp_path / "runs.jsonl"
    path.write_text("{\n", encoding="ascii")
    with pytest.raises(json.JSONDecodeError):
        sampler(path, "plain/patches")


def test_unknown_patch_is_still_dropped(tmp_path):
    path = tmp_path / "runs.jsonl"
    write_jsonl(path, run_line("plain/patches", patch_name="missing"))
    found = sampler(path, "plain/patches")
    assert found.fiber_runs == []


def test_short_run_is_still_dropped(tmp_path):
    path = tmp_path / "runs.jsonl"
    write_jsonl(path, run_line("plain/patches", end=1.0))
    found = sampler(path, "plain/patches")
    assert found.fiber_runs == []


def test_ascii_run_is_kept(tmp_path):
    root = "plain/patches"
    path = tmp_path / "runs.jsonl"
    write_jsonl(path, run_line(root))
    found = sampler(path, root)
    assert [run["fiber"] for run in found.fiber_runs] == ["f1"]
