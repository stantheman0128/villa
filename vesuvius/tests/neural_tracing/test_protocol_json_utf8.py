"""Protocol JSON is UTF-8. A legacy text open must not drop it.

``compare`` reads both result files with ``Path.read_text()`` and no
encoding. ``run`` reads the calibration policy the same way. On cp1252 a
checkpoint path containing U+201D raises ``UnicodeDecodeError`` before
the paired report or the selection check. A UTF-8 BOM is invalid JSON.
An ASCII report still compares, a non-calibration policy still raises,
and a malformed file still raises.
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_SRC = Path(__file__).resolve().parents[2] / "src"
_SCRIPT = _SRC / "vesuvius" / "neural_tracing" / "tracers" / "fibers" / "evaluation" / "protocol.py"
_CHECKPOINT = "scroll \u201d.pt"


def _load():
    sys.path.insert(0, str(_SRC))
    spec = importlib.util.spec_from_file_location("protocol_json_utf8_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


protocol = _load()


def _open_text_as_cp1252(real_open):
    def wrapped(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, *args, **kwargs):
        if isinstance(mode, str) and "b" not in mode and encoding in (None, "locale"):
            encoding = "cp1252"
        return real_open(file, mode, buffering, encoding, errors, newline, *args, **kwargs)

    return wrapped


def _report(checkpoint: str) -> dict:
    return {
        "checkpoint": checkpoint,
        "operating_policy": {"confidence": 0.5},
        "manifest_sha256": "abc",
        "rows": [{
            "source": "scroll",
            "split": "final",
            "fiber": "f0",
            "t0": 0,
            "sign": 1,
            "correct": 3.0,
        }],
    }


def _write(path: Path, document: dict, *, bom: bool = False) -> None:
    raw = json.dumps(document, ensure_ascii=False).encode("utf-8")
    if bom:
        raw = b"\xef\xbb\xbf" + raw
    path.write_bytes(raw)


def _compare(base: Path, new: Path):
    return protocol.compare(SimpleNamespace(base=str(base), new=str(new), out=None))


def _stub_checkpoint():
    def load_checkpoint(path, device):
        return object(), None, None, None, {}

    protocol.load_checkpoint = load_checkpoint


def _run_policy(path: Path):
    _stub_checkpoint()
    args = SimpleNamespace(splits=["final"], policy=str(path), confidence=None, checkpoint="ckpt.pt")
    protocol.run(args, [])


def test_legacy_locale_keeps_unicode_checkpoint(tmp_path, monkeypatch):
    base = tmp_path / "base.json"
    new = tmp_path / "new.json"
    _write(base, _report(_CHECKPOINT))
    _write(new, _report("plain.pt"))
    monkeypatch.setattr(io, "open", _open_text_as_cp1252(io.open))

    report = _compare(base, new)

    assert report["base"]["checkpoint"] == _CHECKPOINT
    assert report["paired"]["all"]["correct"]["base"] == 3.0


def test_utf8_bom_reports_still_compare(tmp_path, monkeypatch):
    base = tmp_path / "base.json"
    new = tmp_path / "new.json"
    _write(base, _report("plain.pt"), bom=True)
    _write(new, _report("plain.pt"), bom=True)
    monkeypatch.setattr(io, "open", _open_text_as_cp1252(io.open))

    report = _compare(base, new)

    assert report["base"]["checkpoint"] == "plain.pt"
    assert report["new"]["checkpoint"] == "plain.pt"


def test_ascii_reports_compare(tmp_path):
    base = tmp_path / "base.json"
    new = tmp_path / "new.json"
    _write(base, _report("plain.pt"))
    _write(new, _report("plain.pt"))

    report = _compare(base, new)

    assert report["paired"]["all"]["traces"] == 1


def test_ascii_reports_compare_under_legacy_locale(tmp_path, monkeypatch):
    base = tmp_path / "base.json"
    new = tmp_path / "new.json"
    _write(base, _report("plain.pt"))
    _write(new, _report("plain.pt"))
    monkeypatch.setattr(io, "open", _open_text_as_cp1252(io.open))

    report = _compare(base, new)

    assert report["base"]["checkpoint"] == "plain.pt"


def test_malformed_report_still_raises(tmp_path, monkeypatch):
    base = tmp_path / "base.json"
    new = tmp_path / "new.json"
    base.write_bytes(b"{")
    _write(new, _report("plain.pt"))
    monkeypatch.setattr(io, "open", _open_text_as_cp1252(io.open))

    with pytest.raises(json.JSONDecodeError):
        _compare(base, new)


def test_legacy_locale_reads_policy_before_selection_check(tmp_path, monkeypatch):
    path = tmp_path / "policy.json"
    _write(path, {"split": "final", "checkpoint": _CHECKPOINT})
    monkeypatch.setattr(io, "open", _open_text_as_cp1252(io.open))

    with pytest.raises(ValueError, match="not a calibration selection"):
        _run_policy(path)


def test_utf8_bom_policy_reaches_selection_check(tmp_path, monkeypatch):
    path = tmp_path / "policy.json"
    _write(path, {"split": "final", "checkpoint": "plain.pt"}, bom=True)
    monkeypatch.setattr(io, "open", _open_text_as_cp1252(io.open))

    with pytest.raises(ValueError, match="not a calibration selection"):
        _run_policy(path)


def test_ascii_policy_still_rejects_non_calibration(tmp_path):
    path = tmp_path / "policy.json"
    _write(path, {"split": "final", "checkpoint": "plain.pt"})

    with pytest.raises(ValueError, match="not a calibration selection"):
        _run_policy(path)


def test_ascii_policy_rejects_non_calibration_under_legacy_locale(tmp_path, monkeypatch):
    path = tmp_path / "policy.json"
    _write(path, {"split": "final", "checkpoint": "plain.pt"})
    monkeypatch.setattr(io, "open", _open_text_as_cp1252(io.open))

    with pytest.raises(ValueError, match="not a calibration selection"):
        _run_policy(path)


def test_malformed_policy_still_raises(tmp_path, monkeypatch):
    path = tmp_path / "policy.json"
    path.write_bytes(b"{")
    monkeypatch.setattr(io, "open", _open_text_as_cp1252(io.open))

    with pytest.raises(json.JSONDecodeError):
        _run_policy(path)
