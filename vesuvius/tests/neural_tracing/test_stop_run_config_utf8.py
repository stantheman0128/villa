"""stop_run.trainer_matches reads a run config with the process locale.

Under cp1252, U+201D in a volume path is swallowed as ValueError, so the
trainer is treated as a different run before its name is compared. A
UTF-8 BOM is swallowed the same way. A matching name should still match.
Related to #1524.
"""
import importlib.util
import io
import json
import sys
import types
from pathlib import Path

import pytest

sys.modules.setdefault("psutil", types.ModuleType("psutil"))

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "src/vesuvius/neural_tracing/tracers/fibers/scripts/stop_run.py"
spec = importlib.util.spec_from_file_location("stop_run_config_utf8_under_test", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def write_json(path, payload, bom=False):
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if bom:
        raw = b"\xef\xbb\xbf" + raw
    path.write_bytes(raw)


def trainer_args(config):
    return [
        "python", "-m", "vesuvius.neural_tracing.tracers.fibers.train.train",
        "--config", config,
    ]


def cp1252_text(monkeypatch):
    real = io.open

    def fake(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, *args, **kwargs):
        if "b" not in str(mode) and encoding in (None, "locale"):
            encoding = "cp1252"
        return real(file, mode, buffering, encoding, errors, newline, *args, **kwargs)

    monkeypatch.setattr(io, "open", fake)


def test_unicode_volume_still_matches_the_run_name(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    write_json(tmp_path / "run.json", {"name": "scroll", "volume": "scroll \u201d/ct"})
    assert mod.trainer_matches(trainer_args("run.json"), "scroll", tmp_path) is True


def test_utf8_bom_config_still_matches_the_run_name(tmp_path):
    write_json(tmp_path / "run.json", {"name": "scroll", "volume": "plain/ct"}, bom=True)
    assert mod.trainer_matches(trainer_args("run.json"), "scroll", tmp_path) is True


def test_malformed_config_is_still_ignored(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    (tmp_path / "run.json").write_text("{", encoding="ascii")
    assert mod.trainer_matches(trainer_args("run.json"), "scroll", tmp_path) is False


def test_different_name_still_does_not_match(tmp_path):
    write_json(tmp_path / "run.json", {"name": "other", "volume": "plain/ct"})
    assert mod.trainer_matches(trainer_args("run.json"), "scroll", tmp_path) is False


def test_ascii_config_matches_the_run_name(tmp_path):
    write_json(tmp_path / "run.json", {"name": "scroll", "volume": "plain/ct"})
    assert mod.trainer_matches(trainer_args("run.json"), "scroll", tmp_path) is True


def test_other_process_is_still_ignored(tmp_path):
    write_json(tmp_path / "run.json", {"name": "scroll", "volume": "scroll \u201d/ct"})
    args = ["python", "-m", "other.module", "--config", "run.json"]
    assert mod.trainer_matches(args, "scroll", tmp_path) is False
