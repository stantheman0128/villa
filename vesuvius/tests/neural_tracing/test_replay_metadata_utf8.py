"""Replay metadata.json is UTF-8. A legacy text open must not drop it.

``OnPolicyStates._open`` and the npz refresh both read ``metadata.json``
with ``open()`` and no encoding. On cp1252 a volume name containing
U+201D raises ``UnicodeDecodeError`` before provenance is checked. A
UTF-8 BOM is invalid JSON. An ASCII cache still loads, and a malformed
file still raises.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "vesuvius"
    / "neural_tracing"
    / "tracers"
    / "fibers"
    / "data"
    / "replay.py"
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
        "vesuvius.neural_tracing.tracers.shared.annotations",
        "vesuvius.neural_tracing.tracers.fibers",
        "vesuvius.neural_tracing.tracers.fibers.data",
    ):
        _package(name)
    _stub(
        "vesuvius.neural_tracing.tracers.shared.annotations.traced",
        fiber_manifest=None,
    )
    _stub(
        "vesuvius.neural_tracing.tracers.fibers.data.tasks",
        TRAVEL_STRATA=(0.0,),
        stable_key=None,
    )
    _stub(
        "vesuvius.neural_tracing.tracers.fibers.data.state_labels",
        DEPARTURE_PATIENCE=3,
        facts=None,
        REPLAY_CLASS={},
        REPLAY_CLASSES=(),
    )
    spec = importlib.util.spec_from_file_location("replay_metadata_utf8", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


replay = _reader()


def _open_text_as_cp1252(real_open):
    def wrapped(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, *args, **kwargs):
        if isinstance(mode, str) and "b" not in mode and encoding in (None, "locale"):
            encoding = "cp1252"
        return real_open(file, mode, buffering, encoding, errors, newline, *args, **kwargs)

    return wrapped


def _empty_arrays():
    arrays = {}
    schema = {**replay.OnPolicyStates.FIELDS, **replay.OnPolicyStates.TRACK}
    for key, (dtype, trailing) in schema.items():
        shape = [0, *(0 if dim is None else dim for dim in trailing)]
        arrays[key] = np.empty(shape, dtype=np.dtype(dtype))
    return arrays


def _write_metadata(path: Path, volume: str, *, bom: bool = False) -> None:
    raw = json.dumps(
        {"fibers": [], "provenance": {"step": "collect", "volume": volume}},
        ensure_ascii=False,
    ).encode("utf-8")
    if bom:
        raw = b"\xef\xbb\xbf" + raw
    path.write_bytes(raw)


def _write_directory(directory: Path, volume: str, *, bom: bool = False, malformed: bool = False) -> None:
    directory.mkdir()
    if malformed:
        (directory / "metadata.json").write_bytes(b"{")
    else:
        _write_metadata(directory / "metadata.json", volume, bom=bom)
    for name, array in _empty_arrays().items():
        np.save(directory / f"{name}.npy", array)


def test_legacy_locale_keeps_unicode_volume(tmp_path, monkeypatch):
    directory = tmp_path / "cache"
    _write_directory(directory, _VOLUME)
    monkeypatch.setattr("builtins.open", _open_text_as_cp1252(open))

    loaded = replay.OnPolicyStates.load(directory)

    assert loaded.provenance["volume"] == _VOLUME
    assert loaded.provenance["step"] == "collect"


def test_utf8_bom_keeps_unicode_volume(tmp_path):
    directory = tmp_path / "cache"
    _write_directory(directory, _VOLUME, bom=True)

    loaded = replay.OnPolicyStates.load(directory)

    assert loaded.provenance["volume"] == _VOLUME


def test_legacy_locale_reads_unicode_sidecar_on_npz_refresh(tmp_path, monkeypatch):
    cache = tmp_path / "cache.npz"
    states = replay.OnPolicyStates(
        manifest=[],
        provenance={"step": "collect", "volume": "plain"},
        **_empty_arrays(),
    )
    states.save(cache)
    sidecar = tmp_path / "cache_mmap"
    sidecar.mkdir()
    _write_metadata(sidecar / "metadata.json", _VOLUME)
    monkeypatch.setattr("builtins.open", _open_text_as_cp1252(open))

    loaded = replay.OnPolicyStates.load(cache)

    assert loaded.provenance["volume"] == "plain"
    assert loaded.provenance["step"] == "collect"


def test_ascii_volume_still_loads(tmp_path):
    directory = tmp_path / "cache"
    _write_directory(directory, "scroll")

    loaded = replay.OnPolicyStates.load(directory)

    assert loaded.provenance["volume"] == "scroll"
    assert len(loaded) == 0


def test_malformed_metadata_still_raises(tmp_path):
    directory = tmp_path / "cache"
    _write_directory(directory, "scroll", malformed=True)

    with pytest.raises(json.JSONDecodeError):
        replay.OnPolicyStates.load(directory)
