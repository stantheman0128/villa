"""Frozen seed manifests are UTF-8. A legacy text open must not drop them.

``read_manifest`` reads the JSON with ``Path.read_text()`` and no
encoding. On cp1252 a fiber name containing U+201D raises
``UnicodeDecodeError`` before the hash is checked. A UTF-8 BOM is
invalid JSON. An ASCII manifest still returns, and a malformed file
still raises.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2] / "src" / "vesuvius" / "neural_tracing" / "tracers"
_NAME = "scroll \u201d"


def _package(name: str) -> None:
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module


def _stub(name: str, **attrs: object) -> None:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _seeds():
    for name in (
        "vesuvius",
        "vesuvius.neural_tracing",
        "vesuvius.neural_tracing.tracers",
        "vesuvius.neural_tracing.tracers.shared",
        "vesuvius.neural_tracing.tracers.shared.annotations",
        "vesuvius.neural_tracing.tracers.fibers",
        "vesuvius.neural_tracing.tracers.fibers.frames",
    ):
        _package(name)
    _stub("vesuvius.neural_tracing.tracers.shared.annotations.traced", TracedFiber=object)
    _stub(
        "vesuvius.neural_tracing.tracers.fibers.frames.ct",
        oriented_seed_heading=lambda *args, **kwargs: None,
        SEED_HEADING_POLICY="stub",
    )
    _stub(
        "vesuvius.neural_tracing.tracers.shared.ct_structure",
        SeedHeadingError=type("SeedHeadingError", (Exception,), {}),
    )
    _stub(
        "vesuvius.neural_tracing.tracers.shared.geometry",
        interp_at=lambda *args, **kwargs: None,
        tangent_at=lambda *args, **kwargs: None,
    )
    _load(
        "vesuvius.neural_tracing.tracers.shared.io",
        _ROOT / "shared" / "io.py",
    )
    return _load(
        "seed_manifest_utf8",
        _ROOT / "fibers" / "data" / "seeds.py",
    )


read_manifest = _seeds().read_manifest


def _open_text_as_cp1252(real_open):
    def wrapped(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, *args, **kwargs):
        if isinstance(mode, str) and "b" not in mode and encoding in (None, "locale"):
            encoding = "cp1252"
        return real_open(file, mode, buffering, encoding, errors, newline, *args, **kwargs)

    return wrapped


def _manifest(name: str) -> dict:
    payload = {
        "monitor": [],
        "calibration": [{"fiber": name}],
        "final": [],
        "monitor_fibers": [],
        "calibration_fibers": [name],
        "final_fibers": [],
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return {**payload, "sha256": digest}


def _write(path: Path, name: str, bom: bool = False) -> None:
    raw = json.dumps(_manifest(name), ensure_ascii=False).encode("utf-8")
    if bom:
        raw = b"\xef\xbb\xbf" + raw
    path.write_bytes(raw)


def test_legacy_locale_keeps_unicode_fiber(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    _write(path, _NAME)
    monkeypatch.setattr(io, "open", _open_text_as_cp1252(io.open))

    saved = read_manifest(path)

    assert saved["calibration_fibers"] == [_NAME]
    assert saved["calibration"][0]["fiber"] == _NAME


def test_utf8_bom_keeps_unicode_fiber(tmp_path):
    path = tmp_path / "manifest.json"
    _write(path, _NAME, bom=True)

    saved = read_manifest(path)

    assert saved["calibration_fibers"] == [_NAME]


def test_ascii_manifest_still_returns(tmp_path):
    path = tmp_path / "manifest.json"
    _write(path, "scroll")

    saved = read_manifest(path)

    assert saved["calibration_fibers"] == ["scroll"]


def test_malformed_manifest_still_raises(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        read_manifest(path)
