"""fibers/tracing/inference.py reads a calibration policy with the process locale.

Under cp1252, U+201D in the selection raises UnicodeDecodeError before the
operating policy is applied, and a UTF-8 BOM raises JSONDecodeError.
Related to #1524.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

SCRIPT = SRC / "vesuvius/neural_tracing/tracers/fibers/tracing/inference.py"
spec = importlib.util.spec_from_file_location("fiber_trace_policy_utf8_under_test", SCRIPT)
tracer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tracer
spec.loader.exec_module(tracer)


class _Cfg:
    model_type = "follower"


class _Model:
    cfg = _Cfg()


class _Spec:
    grid_scale = 1.0


class Stop(Exception):
    def __init__(self, confidence, n_commit, gate):
        self.confidence = confidence
        self.n_commit = n_commit
        self.gate = gate


def install_stubs():
    tracer.resolve_checkpoint = lambda checkpoint: "ckpt.pt"
    tracer.configure_volume = lambda *args, **kwargs: None

    def loader(path, device):
        return _Model(), None, 1, _Spec(), {}

    tracer.load_checkpoint = loader

    def policy(ck, cfg, *, confidence=None, n_commit=None, gate=None):
        raise Stop(confidence, n_commit, gate)

    tracer.checkpoint_policy = policy


def selection(*, checkpoint="plain/model.pt", confidence=0.3):
    return {
        "checkpoint": checkpoint,
        "operating_policy": {"confidence": confidence, "n_commit": 4, "gate": "prefix"},
    }


def write_json(path, payload):
    path.write_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"))


def cp1252_text(monkeypatch):
    real = open

    def fake(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
        if "b" not in str(mode) and encoding in (None, "locale"):
            encoding = "cp1252"
        return real(file, mode, buffering, encoding, errors, newline, closefd, opener)

    monkeypatch.setattr("builtins.open", fake)


def run(tmp_path, policy, extra=()):
    install_stubs()
    tracer.main(
        ["--family", "H", "--out", str(tmp_path / "out"), "--policy", str(policy), "--device", "cpu", *extra],
        checkpoint_loader=tracer.load_checkpoint,
    )


def test_unicode_checkpoint_path_is_read_under_cp1252(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    path = tmp_path / "selection.json"
    write_json(path, selection(checkpoint="scroll \u201d/model.pt"))
    with pytest.raises(Stop) as caught:
        run(tmp_path, path)
    assert caught.value.confidence == 0.3
    assert caught.value.n_commit == 4
    assert caught.value.gate == "prefix"


def test_utf8_bom_policy_is_read(tmp_path):
    path = tmp_path / "selection.json"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps(selection()).encode("utf-8"))
    with pytest.raises(Stop) as caught:
        run(tmp_path, path)
    assert caught.value.confidence == 0.3
    assert caught.value.n_commit == 4
    assert caught.value.gate == "prefix"


def test_malformed_policy_still_raises(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    path = tmp_path / "selection.json"
    path.write_text("{", encoding="ascii")
    with pytest.raises(json.JSONDecodeError):
        run(tmp_path, path)


def test_ascii_policy_is_applied(tmp_path):
    path = tmp_path / "selection.json"
    write_json(path, selection(confidence=0.42))
    with pytest.raises(Stop) as caught:
        run(tmp_path, path)
    assert caught.value.confidence == 0.42
    assert caught.value.n_commit == 4
    assert caught.value.gate == "prefix"


def test_cli_confidence_still_overrides_the_file(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    path = tmp_path / "selection.json"
    write_json(path, selection(confidence=0.3))
    with pytest.raises(Stop) as caught:
        run(tmp_path, path, extra=("--confidence", "0.5"))
    assert caught.value.confidence == 0.5
    assert caught.value.n_commit == 4
    assert caught.value.gate == "prefix"
