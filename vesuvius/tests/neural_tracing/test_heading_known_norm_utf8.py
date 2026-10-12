"""The heading model's saved CT normalization is UTF-8.

``load_sources`` reads that JSON with ``Path.read_text()`` and no
encoding, before ``prepare_normalization`` runs. On cp1252 a volume key
containing U+201D raises ``UnicodeDecodeError``. A UTF-8 BOM is invalid
JSON. An ASCII document still reaches the normalizer, a missing path
argument skips the read, and a malformed file still raises.
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
    / "fibers"
    / "heading_model"
    / "data.py"
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
        "vesuvius.neural_tracing.tracers.fibers",
        "vesuvius.neural_tracing.tracers.fibers.data",
        "vesuvius.neural_tracing.tracers.fibers.heading_model",
        "vesuvius.neural_tracing.tracers.fibers.frames",
    ):
        _package(name)
    seen = {}

    def read_dataset_config(path):
        seen["dataset"] = path
        return {"sources": []}, "digest"

    def prepare_normalization(out, specs, *, known=None):
        seen["known"] = known
        seen["specs"] = list(specs)
        return {"method": "crop_zscore_v1"}

    _stub(
        "vesuvius.neural_tracing.tracers.fibers.heading_model.model",
        HeadingConfig=object,
        model_inputs=None,
        prior_frames=None,
    )
    _stub("vesuvius.neural_tracing.tracers.fibers.heading_model.targets", in_crop_heading=None)
    _stub(
        "vesuvius.neural_tracing.tracers.fibers.heading_model.frames",
        family_ids=None,
        smoothed_tangent=None,
        fiber_corrected_normal=None,
    )
    _stub(
        "vesuvius.neural_tracing.tracers.fibers.data.datasets",
        WeightedDatasets=object,
        load_primary_dataset=None,
        open_afv_source=None,
        primary_source_spec=None,
        read_dataset_config=read_dataset_config,
    )
    _stub("vesuvius.neural_tracing.tracers.fibers.data.follow_dataset", FollowDataset=object)
    _stub(
        "vesuvius.neural_tracing.tracers.fibers.data.samples",
        SampleConfig=object,
        simulated_trace=None,
    )
    _stub(
        "vesuvius.neural_tracing.tracers.shared.geometry",
        traversal_curve=None,
        arclength=None,
        interp_at=None,
        normalize=None,
    )
    _stub("vesuvius.neural_tracing.tracers.shared.crop_sampling.crops", tight_block=None)
    _stub(
        "vesuvius.neural_tracing.tracers.fibers.frames.ct",
        linear12_heading=None,
    )
    _stub("vesuvius.neural_tracing.tracers.shared.crop_sampling.volume", CTVolume=object)
    _stub(
        "vesuvius.neural_tracing.tracers.shared.crop_sampling.normalization",
        prepare_normalization=prepare_normalization,
    )
    spec = importlib.util.spec_from_file_location("heading_known_norm_utf8", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, seen


heading, _seen = _reader()


def _open_text_as_cp1252(real_open):
    def wrapped(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, *args, **kwargs):
        if isinstance(mode, str) and "b" not in mode and encoding in (None, "locale"):
            encoding = "cp1252"
        return real_open(file, mode, buffering, encoding, errors, newline, *args, **kwargs)

    return wrapped


def _write(path: Path, volume: str, *, bom: bool = False) -> None:
    raw = json.dumps(
        {"method": "crop_zscore_v1", "volumes": {volume: {"volume": volume}}},
        ensure_ascii=False,
    ).encode("utf-8")
    if bom:
        raw = b"\xef\xbb\xbf" + raw
    path.write_bytes(raw)


def _load(tmp_path: Path, ct_normalization):
    _seen.clear()
    document, digest, sources, normalization = heading.load_sources(
        tmp_path / "dataset.json",
        tmp_path / "out",
        ct_normalization=ct_normalization,
    )
    assert document == {"sources": []}
    assert digest == "digest"
    assert sources == []
    assert normalization == {"method": "crop_zscore_v1"}
    assert _seen["specs"] == []
    return _seen["known"]


def test_legacy_locale_keeps_unicode_volume(tmp_path, monkeypatch):
    path = tmp_path / "ct_normalization.json"
    _write(path, _VOLUME)
    monkeypatch.setattr(io, "open", _open_text_as_cp1252(io.open))

    known = _load(tmp_path, path)

    assert _VOLUME in known["volumes"]


def test_utf8_bom_keeps_unicode_volume(tmp_path):
    path = tmp_path / "ct_normalization.json"
    _write(path, _VOLUME, bom=True)

    known = _load(tmp_path, path)

    assert _VOLUME in known["volumes"]


def test_ascii_volume_still_loads(tmp_path):
    path = tmp_path / "ct_normalization.json"
    _write(path, "scroll")

    known = _load(tmp_path, path)

    assert known["volumes"]["scroll"]["volume"] == "scroll"


def test_missing_argument_skips_the_read(tmp_path):
    known = _load(tmp_path, None)

    assert known is None


def test_malformed_document_still_raises(tmp_path):
    path = tmp_path / "ct_normalization.json"
    path.write_bytes(b"{")

    with pytest.raises(json.JSONDecodeError):
        _load(tmp_path, path)
