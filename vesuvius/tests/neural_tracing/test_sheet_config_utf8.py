"""The sheet tracer config is UTF-8. A legacy text open must not drop it.

``load`` reads the JSON with ``Path.read_text()`` and no encoding. On
cp1252 a volume path containing U+201D raises ``UnicodeDecodeError``
before the config exists. A UTF-8 BOM is invalid JSON. An ASCII file
still applies its override, and a malformed file still raises.
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import types
from pathlib import Path

import pytest

_ROOT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "vesuvius"
    / "neural_tracing"
    / "tracers"
)
_VOLUME = "scroll \u201d"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _config_module():
    for name in (
        "vesuvius",
        "vesuvius.neural_tracing",
        "vesuvius.neural_tracing.tracers",
        "vesuvius.neural_tracing.tracers.shared",
        "vesuvius.neural_tracing.tracers.sheets",
    ):
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = []
            sys.modules[name] = package
    _load(
        "vesuvius.neural_tracing.tracers.shared.retired",
        _ROOT / "shared" / "retired.py",
    )
    _load(
        "vesuvius.neural_tracing.tracers.shared.paths",
        _ROOT / "shared" / "paths.py",
    )
    return _load(
        "vesuvius.neural_tracing.tracers.sheets.config_utf8",
        _ROOT / "sheets" / "config.py",
    )


_config = _config_module()
load = _config.load


def _open_text_as_cp1252(real_open):
    def wrapped(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, *args, **kwargs):
        if isinstance(mode, str) and "b" not in mode and encoding in (None, "locale"):
            encoding = "cp1252"
        return real_open(file, mode, buffering, encoding, errors, newline, *args, **kwargs)

    return wrapped


def _write(path: Path, volume: str, bom: bool = False) -> None:
    payload = json.dumps({"data": {"volume": volume}}, ensure_ascii=False).encode("utf-8")
    if bom:
        payload = b"\xef\xbb\xbf" + payload
    path.write_bytes(payload)


def test_legacy_locale_keeps_unicode_volume(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    _write(config_path, _VOLUME)
    monkeypatch.setattr(io, "open", _open_text_as_cp1252(io.open))

    config = load(config_path)

    assert config.data.volume == _VOLUME


def test_utf8_bom_keeps_unicode_volume(tmp_path):
    config_path = tmp_path / "config.json"
    _write(config_path, _VOLUME, bom=True)

    config = load(config_path)

    assert config.data.volume == _VOLUME


def test_ascii_volume_still_applied(tmp_path):
    config_path = tmp_path / "config.json"
    _write(config_path, "/data/scroll")

    config = load(config_path)

    assert config.data.volume == "/data/scroll"


def test_malformed_config_still_raises(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text("{", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        load(config_path)
