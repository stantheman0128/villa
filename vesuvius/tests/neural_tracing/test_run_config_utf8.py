"""run_config.load reads the training run JSON with the process locale.

Under cp1252, U+201D in the run name raises UnicodeDecodeError before
defaults are filled, and a UTF-8 BOM raises JSONDecodeError. Related to #1524.
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

SCRIPT = SRC / "vesuvius/neural_tracing/tracers/fibers/train/run_config.py"
spec = importlib.util.spec_from_file_location("run_config_utf8_under_test", SCRIPT)
run_config = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = run_config
spec.loader.exec_module(run_config)


def document(*, name="plain"):
    return {
        "name": name,
        "dataset": {
            "version": 1,
            "cache_dir": "cache",
            "sources": [{
                "name": "paris4",
                "kind": "paris4",
                "weight": 1.0,
                "fibers": "fibers",
                "ct": "s3://example/ct.zarr",
                "manifest": "seeds.json",
                "val_z": [1.0, 2.0],
                "validation": {"strategy": "fiber_hash", "fraction": 0.1, "seed": 1},
            }],
        },
    }


def write_config(path, payload):
    path.write_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"))


def cp1252_text(monkeypatch):
    real = io.open

    def fake(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, *args, **kwargs):
        if "b" not in str(mode) and encoding in (None, "locale"):
            encoding = "cp1252"
        return real(file, mode, buffering, encoding, errors, newline, *args, **kwargs)

    monkeypatch.setattr(io, "open", fake)


def test_unicode_run_name_is_kept_under_cp1252(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    path = tmp_path / "run.json"
    write_config(path, document(name="scroll \u201d"))
    loaded = run_config.load(path)
    assert loaded["name"] == "scroll \u201d"
    assert loaded["dataset"]["sources"][0]["kind"] == "paris4"


def test_utf8_bom_run_config_is_read(tmp_path):
    path = tmp_path / "run.json"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps(document(name="bom")).encode("utf-8"))
    loaded = run_config.load(path)
    assert loaded["name"] == "bom"


def test_missing_name_still_raises(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    path = tmp_path / "run.json"
    payload = document()
    del payload["name"]
    write_config(path, payload)
    with pytest.raises(ValueError, match="needs a name"):
        run_config.load(path)


def test_malformed_run_config_still_raises(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    path = tmp_path / "run.json"
    path.write_text("{", encoding="ascii")
    with pytest.raises(json.JSONDecodeError):
        run_config.load(path)


def test_ascii_run_name_is_kept(tmp_path):
    path = tmp_path / "run.json"
    write_config(path, document(name="plain"))
    assert run_config.load(path)["name"] == "plain"
