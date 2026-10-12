"""CT normalization JSON is UTF-8. A legacy text open must not drop it.

``prepare_normalization`` reads ``ct_normalization.json`` with
``Path.read_text()`` and no encoding, twice. On cp1252 a volume key
containing U+201D raises ``UnicodeDecodeError`` before the policy is
checked. A UTF-8 BOM is invalid JSON. An ASCII document still loads, a
missing file still gets the default policy, and a malformed file still
raises.
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import types
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "vesuvius"
    / "neural_tracing"
    / "tracers"
    / "shared"
    / "crop_sampling"
    / "normalization.py"
)
_VOLUME = "scroll \u201d"


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


def _reader():
    for name in (
        "vesuvius",
        "vesuvius.neural_tracing",
        "vesuvius.neural_tracing.tracers",
        "vesuvius.neural_tracing.tracers.shared",
        "vesuvius.neural_tracing.tracers.shared.crop_sampling",
    ):
        _package(name)

    def open_ct_array(*_args, **_kwargs):
        raise AssertionError("empty specs must not open a CT array")

    def atomic_write(path, data):
        Path(path).write_bytes(data)

    _stub(
        "vesuvius.neural_tracing.tracers.shared.crop_sampling.volume",
        open_ct_array=open_ct_array,
    )
    _stub("vesuvius.neural_tracing.tracers.shared.io", atomic_write=atomic_write)
    spec = importlib.util.spec_from_file_location("ct_normalization_utf8", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


normalization = _reader()


def _open_text_as_cp1252(real_open):
    def wrapped(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, *args, **kwargs):
        if isinstance(mode, str) and "b" not in mode and encoding in (None, "locale"):
            encoding = "cp1252"
        return real_open(file, mode, buffering, encoding, errors, newline, *args, **kwargs)

    return wrapped


def _write(directory: Path, volume: str, *, bom: bool = False) -> None:
    raw = json.dumps(
        {
            "method": normalization.ZSCORE_METHOD,
            "volumes": {volume: {"method": normalization.ZSCORE_METHOD, "volume": volume}},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    if bom:
        raw = b"\xef\xbb\xbf" + raw
    (directory / "ct_normalization.json").write_bytes(raw)


def test_legacy_locale_keeps_unicode_volume(tmp_path, monkeypatch):
    _write(tmp_path, _VOLUME)
    monkeypatch.setattr(io, "open", _open_text_as_cp1252(io.open))

    document = normalization.prepare_normalization(tmp_path, [])

    assert _VOLUME in document["volumes"]
    assert document["method"] == normalization.ZSCORE_METHOD


def test_utf8_bom_keeps_unicode_volume(tmp_path):
    _write(tmp_path, _VOLUME, bom=True)

    document = normalization.prepare_normalization(tmp_path, [])

    assert _VOLUME in document["volumes"]


def test_ascii_volume_still_loads(tmp_path):
    _write(tmp_path, "scroll")

    document = normalization.prepare_normalization(tmp_path, [])

    assert document["volumes"]["scroll"]["volume"] == "scroll"
    assert document["method"] == normalization.ZSCORE_METHOD


def test_missing_file_uses_default_policy(tmp_path):
    document = normalization.prepare_normalization(tmp_path, [])

    assert document == {"method": normalization.ZSCORE_METHOD, "volumes": {}}
    assert (tmp_path / "ct_normalization.json").is_file()


def test_malformed_document_still_raises(tmp_path):
    (tmp_path / "ct_normalization.json").write_bytes(b"{")

    with pytest.raises(json.JSONDecodeError):
        normalization.prepare_normalization(tmp_path, [])
