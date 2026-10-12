"""EpisodeSampler reads hard-window JSON with the process locale.

Under cp1252, U+201D in the recorded checkpoint path raises
UnicodeDecodeError before the failure records are used, and a UTF-8 BOM
raises JSONDecodeError. The records should still be kept.
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
spec = importlib.util.spec_from_file_location("hard_windows_utf8_under_test", SCRIPT)
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
    hard_fraction = 1
    fiber_follow = False
    fiber_fraction = 0
    split_block = 64
    val_fraction = 0.0
    split_seed = 0


def patch():
    return mod.PatchInfo(
        name="p0", rows=40, cols=40, cell=2.0, valid=10,
        bbox=(0, 0, 0, 10, 10, 10), root="plain/patches",
    )


def document(checkpoint):
    return {
        "checkpoint": checkpoint,
        "records": [{"dir": 0, "patch": "p0", "kind": "stop"}],
    }


def write_json(path, payload, bom=False):
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
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


def sampler(path, *, split="train", hard_fraction=1):
    data = Data()
    data.hard_windows = str(path)
    data.hard_fraction = hard_fraction
    return mod.EpisodeSampler(data, [patch()], split, 2, 64, 0.0)


def test_unicode_checkpoint_keeps_the_records(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    path = tmp_path / "hard.json"
    write_json(path, document("scroll \u201d/last.pt"))
    found = sampler(path)
    assert [(row["patch"], row["kind"]) for row in found.hard] == [("p0", "stop")]


def test_utf8_bom_keeps_the_records(tmp_path):
    path = tmp_path / "hard.json"
    write_json(path, document("plain/last.pt"), bom=True)
    found = sampler(path)
    assert [row["patch"] for row in found.hard] == ["p0"]


def test_malformed_file_still_raises(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    path = tmp_path / "hard.json"
    path.write_text("{", encoding="ascii")
    with pytest.raises(json.JSONDecodeError):
        sampler(path)


def test_validation_split_still_skips_the_file(tmp_path):
    path = tmp_path / "hard.json"
    write_json(path, document("scroll \u201d/last.pt"))
    data = Data()
    data.hard_windows = str(path)
    data.val_fraction = 1.0
    found = mod.EpisodeSampler(data, [patch()], "val", 2, 64, 0.0)
    assert found.hard == []


def test_zero_fraction_still_skips_the_file(tmp_path):
    path = tmp_path / "hard.json"
    write_json(path, document("scroll \u201d/last.pt"))
    found = sampler(path, hard_fraction=0)
    assert found.hard == []


def test_ascii_records_are_kept(tmp_path):
    path = tmp_path / "hard.json"
    write_json(path, document("plain/last.pt"))
    found = sampler(path)
    assert [row["patch"] for row in found.hard] == ["p0"]
