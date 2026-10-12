"""Dataset JSON is UTF-8. A legacy text open must not drop it.

``read_dataset_config`` and the AFV replay index in ``build_mixed_dataset``
both call ``Path.read_text()`` with no encoding. On cp1252 a name or cache
path containing U+201D raises ``UnicodeDecodeError`` before the document
is checked. A UTF-8 BOM is invalid JSON. ASCII still loads, a missing
replay index is skipped, and a malformed file still raises.
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2] / "src" / "vesuvius" / "neural_tracing" / "tracers"
_DATASETS = _ROOT / "fibers" / "data" / "datasets.py"
_SPLITS = _ROOT / "shared" / "annotations" / "splits.py"
_PATHS = _ROOT / "shared" / "paths.py"
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


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _Fibers:
    metadata = {"fiber_count": 2}

    def __len__(self):
        return 1


class _FollowDataset:
    def __init__(self, *args, **kwargs):
        self.chunk = kwargs["chunk"]
        self.onpolicy = kwargs["onpolicy"]
        self.replay_index = kwargs["replay_index"]


def _reader():
    for name in (
        "vesuvius",
        "vesuvius.neural_tracing",
        "vesuvius.neural_tracing.tracers",
        "vesuvius.neural_tracing.tracers.shared",
        "vesuvius.neural_tracing.tracers.shared.annotations",
        "vesuvius.neural_tracing.tracers.shared.crop_sampling",
        "vesuvius.neural_tracing.tracers.fibers",
        "vesuvius.neural_tracing.tracers.fibers.data",
        "vesuvius.neural_tracing.tracers.fibers.frames",
    ):
        _package(name)
    splits = _load(_SPLITS, "dataset_json_splits")
    paths = _load(_PATHS, "dataset_json_paths")
    _stub("vesuvius.neural_tracing.tracers.shared.annotations.afv", AFVFibers=object, file_sha256=None)
    _stub(
        "vesuvius.neural_tracing.tracers.shared.annotations.splits",
        heldout_ids=splits.heldout_ids,
        validate_split_policy=splits.validate_split_policy,
    )
    _stub("vesuvius.neural_tracing.tracers.shared.annotations.traced", fiber_manifest=None)
    _stub("vesuvius.neural_tracing.tracers.shared.annotations.vc3d", load_fibers=None)
    _stub(
        "vesuvius.neural_tracing.tracers.fibers.data.follow_dataset",
        FollowDataset=_FollowDataset,
    )
    _stub(
        "vesuvius.neural_tracing.tracers.shared.paths",
        moved_path=paths.moved_path,
        canonical_ct_url=paths.canonical_ct_url,
    )
    _stub(
        "vesuvius.neural_tracing.tracers.shared.crop_sampling.volume",
        CTVolumeSpec=object,
        CTVolume=object,
    )
    _stub(
        "vesuvius.neural_tracing.tracers.fibers.data.observations",
        IdentityObservationBuilder=lambda *args, **kwargs: None,
    )
    _stub(
        "vesuvius.neural_tracing.tracers.fibers.data.afv_neighbors",
        AFVBank=lambda fibers: None,
    )
    _stub(
        "vesuvius.neural_tracing.tracers.fibers.data.seeds",
        make_seeds=None,
        read_manifest=None,
    )
    _stub("vesuvius.neural_tracing.tracers.shared.io", jsonable=None)
    _stub(
        "vesuvius.neural_tracing.tracers.fibers.frames.ct",
        SEED_HEADING_POLICY=None,
        TRACE_HEADING_POLICY=None,
        FRAME_POLICY=None,
    )
    _stub("vesuvius.neural_tracing.tracers.shared.crop_sampling.normalization", volume_key=None)
    _stub(
        "vesuvius.neural_tracing.tracers.fibers.data.replay",
        load_replay=lambda replay_paths: list(replay_paths),
        own_replay_paths=lambda replay_paths, dagger: list(replay_paths),
        usable_replay=lambda caches, fibers, n_history, grid_scale: caches,
    )
    _stub("vesuvius.neural_tracing.tracers.fibers.data.episodes", loader_chunk=lambda cfg, batch: 1)
    return _load(_DATASETS, "dataset_json_utf8")


datasets = _reader()


def _open_afv(source, cache_dir, normalization=None):
    spec = types.SimpleNamespace(grid_scale=float(source["grid_scale"]))
    return _Fibers(), _Fibers(), spec, "abc"


datasets.open_afv_source = _open_afv
datasets.validation_manifest = lambda *args, **kwargs: {"frozen": True}


def _open_text_as_cp1252(real_open):
    def wrapped(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, *args, **kwargs):
        if isinstance(mode, str) and "b" not in mode and encoding in (None, "locale"):
            encoding = "cp1252"
        return real_open(file, mode, buffering, encoding, errors, newline, *args, **kwargs)

    return wrapped


def _config(name: str) -> dict:
    return {
        "version": 1,
        "cache_dir": "cache",
        "sources": [
            {
                "name": name,
                "weight": 1,
                "kind": "paris4",
                "validation": {"strategy": "fiber_hash", "fraction": 0.1, "seed": 1},
                "fibers": "fibers.json",
                "ct": "https://example.test/vol",
                "manifest": "manifest.json",
                "val_z": 0,
            }
        ],
    }


def _write_config(path: Path, name: str, *, bom: bool = False) -> None:
    raw = json.dumps(_config(name), ensure_ascii=False).encode("utf-8")
    if bom:
        raw = b"\xef\xbb\xbf" + raw
    path.write_bytes(raw)


def _write_replay(directory: Path, cache: str, *, bom: bool = False) -> None:
    raw = json.dumps([cache], ensure_ascii=False).encode("utf-8")
    if bom:
        raw = b"\xef\xbb\xbf" + raw
    path = directory / "dagger" / "afv" / "replay.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(raw)


def _mix(out: Path, *, resume: bool = True):
    document = {
        "cache_dir": str(out / "cache"),
        "sources": [
            {
                "name": "afv",
                "kind": "afv",
                "weight": 1.0,
                "grid_scale": 1.0,
                "coordinate_space": "scroll",
                "validation": {"seed": 1},
            }
        ],
    }
    args = types.SimpleNamespace(batch=1, worker_cache_gb=1, afv_length_power=1.0)
    mixed, provenance = datasets.build_mixed_dataset(
        None,
        document,
        object(),
        types.SimpleNamespace(n_history=4),
        object(),
        args,
        seed=0,
        budget=object(),
        out=out,
        resume=resume,
    )
    return mixed.datasets[0], provenance


def test_legacy_locale_keeps_unicode_source_name(tmp_path, monkeypatch):
    path = tmp_path / "dataset.json"
    _write_config(path, _VOLUME)
    monkeypatch.setattr(io, "open", _open_text_as_cp1252(io.open))

    document, digest = datasets.read_dataset_config(path)

    assert document["sources"][0]["name"] == _VOLUME
    assert document["sources"][0]["ct"] == "https://example.test/vol"
    assert len(digest) == 64


def test_utf8_bom_keeps_unicode_source_name(tmp_path):
    path = tmp_path / "dataset.json"
    _write_config(path, _VOLUME, bom=True)

    document, _digest = datasets.read_dataset_config(path)

    assert document["sources"][0]["name"] == _VOLUME


def test_ascii_source_name_still_loads(tmp_path):
    path = tmp_path / "dataset.json"
    _write_config(path, "scroll")

    document, _digest = datasets.read_dataset_config(path)

    assert document["sources"][0]["name"] == "scroll"
    assert document["sources"][0]["ct"].endswith("/vol")


def test_malformed_config_still_raises(tmp_path):
    path = tmp_path / "dataset.json"
    path.write_bytes(b"{")

    with pytest.raises(json.JSONDecodeError):
        datasets.read_dataset_config(path)


def test_legacy_locale_keeps_unicode_replay_path(tmp_path, monkeypatch):
    _write_replay(tmp_path, _VOLUME + ".npz")
    monkeypatch.setattr(io, "open", _open_text_as_cp1252(io.open))

    dataset, provenance = _mix(tmp_path)

    assert dataset.onpolicy == [_VOLUME + ".npz"]
    assert provenance[0]["kind"] == "afv"


def test_utf8_bom_keeps_unicode_replay_path(tmp_path):
    _write_replay(tmp_path, _VOLUME + ".npz", bom=True)

    dataset, _provenance = _mix(tmp_path)

    assert dataset.onpolicy == [_VOLUME + ".npz"]


def test_ascii_replay_path_still_loads(tmp_path):
    _write_replay(tmp_path, "scroll.npz")

    dataset, _provenance = _mix(tmp_path)

    assert dataset.onpolicy == ["scroll.npz"]


def test_missing_replay_index_is_skipped(tmp_path):
    dataset, _provenance = _mix(tmp_path)

    assert dataset.onpolicy == []


def test_malformed_replay_index_still_raises(tmp_path):
    path = tmp_path / "dagger" / "afv" / "replay.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"{")

    with pytest.raises(json.JSONDecodeError):
        _mix(tmp_path)
