"""The replay index is UTF-8. A legacy text open must not drop it.

``FollowDataset.refresh_replay`` reads the index with ``open()`` and no
encoding. On cp1252 a cache path containing U+201D raises
``UnicodeDecodeError`` before any cache is loaded. A UTF-8 BOM is
invalid JSON. An ASCII index still loads, a missing index is skipped,
and a malformed file still raises.
"""
from __future__ import annotations

import importlib.util
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
    / "fibers"
    / "data"
    / "follow_dataset.py"
)
_PATH = "scroll \u201d.npz"


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
        "vesuvius.neural_tracing.tracers.fibers",
        "vesuvius.neural_tracing.tracers.fibers.data",
        "vesuvius.neural_tracing.tracers.fibers.frames",
    ):
        _package(name)
    _stub("vesuvius.neural_tracing.tracers.shared.geometry", crop_local_grid=None)
    _stub("vesuvius.neural_tracing.tracers.shared.crop_sampling.volume", CTVolume=object)
    _stub("vesuvius.neural_tracing.tracers.shared.reference", SEED_FIELDS=())
    _stub(
        "vesuvius.neural_tracing.tracers.fibers.data.episodes",
        episode_decisions=None,
        episode_tensors=None,
        mark_episode_identity=None,
    )
    _stub(
        "vesuvius.neural_tracing.tracers.fibers.data.replay",
        EVENT_TABLE_SIZE=1,
        ReplayIndex=object,
        load_replay=None,
        replay_facts=None,
        replay_labeler_state=None,
        usable_replay=None,
    )
    _stub(
        "vesuvius.neural_tracing.tracers.fibers.data.samples",
        STARTUP_CATEGORIES=(),
        make_sample=None,
    )
    _stub(
        "vesuvius.neural_tracing.tracers.fibers.data.targets",
        label_state=None,
        resolve_trace_seed=None,
    )
    _stub(
        "vesuvius.neural_tracing.tracers.fibers.data.tasks",
        FALLBACKS=(),
        SOURCE={},
        TASK={},
        TASKS=(),
        TaskBudget=object,
        stable_key=None,
    )
    _stub(
        "vesuvius.neural_tracing.tracers.fibers.data.state_labels",
        RECOVERABLE=0,
        FOLLOWING=1,
    )
    _stub(
        "vesuvius.neural_tracing.tracers.fibers.frames.ct",
        FRAME_POLICY=None,
        frame_prefetch_bounds=None,
    )
    _stub("vesuvius.neural_tracing.tracers.shared.ct_structure", SeedHeadingError=type("SeedHeadingError", (Exception,), {}))
    spec = importlib.util.spec_from_file_location("replay_index_utf8", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


follow = _reader()


def _open_text_as_cp1252(real_open):
    def wrapped(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, *args, **kwargs):
        if isinstance(mode, str) and "b" not in mode and encoding in (None, "locale"):
            encoding = "cp1252"
        return real_open(file, mode, buffering, encoding, errors, newline, *args, **kwargs)

    return wrapped


def _write(path: Path, cache: str, *, bom: bool = False) -> None:
    raw = json.dumps([cache], ensure_ascii=False).encode("utf-8")
    if bom:
        raw = b"\xef\xbb\xbf" + raw
    path.write_bytes(raw)


def _refresh(path: Path | None):
    seen = {}

    def load_replay(paths):
        seen["paths"] = list(paths)
        return []

    follow.load_replay = load_replay
    host = types.SimpleNamespace(
        replay_index=None if path is None else str(path),
        _replay_paths=None,
        _set_replay=lambda caches: seen.setdefault("caches", caches),
    )
    follow.FollowDataset.refresh_replay(host)
    return host, seen


def test_legacy_locale_keeps_unicode_path(tmp_path, monkeypatch):
    path = tmp_path / "replay.json"
    _write(path, _PATH)
    monkeypatch.setattr("builtins.open", _open_text_as_cp1252(open))

    host, seen = _refresh(path)

    assert seen["paths"] == [_PATH]
    assert host._replay_paths == [_PATH]


def test_utf8_bom_keeps_unicode_path(tmp_path):
    path = tmp_path / "replay.json"
    _write(path, _PATH, bom=True)

    host, seen = _refresh(path)

    assert seen["paths"] == [_PATH]
    assert host._replay_paths == [_PATH]


def test_ascii_index_still_loads(tmp_path):
    path = tmp_path / "replay.json"
    _write(path, "scroll.npz")

    host, seen = _refresh(path)

    assert seen["paths"] == ["scroll.npz"]
    assert host._replay_paths == ["scroll.npz"]


def test_missing_index_is_skipped(tmp_path):
    host, seen = _refresh(tmp_path / "missing.json")

    assert "paths" not in seen
    assert host._replay_paths is None


def test_malformed_index_still_raises(tmp_path):
    path = tmp_path / "replay.json"
    path.write_bytes(b"{")

    with pytest.raises(json.JSONDecodeError):
        _refresh(path)
