"""Neural-tracing inference JSON is UTF-8, including on a legacy locale.

``_read_tifxyz_scale_from_meta``, the volume ``meta.json`` voxel-size read,
and the extrapolation config open still use the process code page. On cp1252
a byte that code page does not define aborts the load before scale or
voxelsize is used. A UTF-8 BOM makes ``json.load`` fail on a UTF-8 locale.
"""
from __future__ import annotations

import builtins
import json
from types import SimpleNamespace

from vesuvius.neural_tracing.inference.common import (
    _read_tifxyz_scale_from_meta,
    resolve_extrapolation_settings,
    resolve_tifxyz_params,
)


def _open_text_as_cp1252(real_open):
    def wrapped(file, mode="r", *args, **kwargs):
        if isinstance(mode, str) and "b" not in mode and "encoding" not in kwargs:
            kwargs = dict(kwargs)
            kwargs["encoding"] = "cp1252"
        return real_open(file, mode, *args, **kwargs)

    return wrapped


def _write(path, payload: dict, bom: bool = False) -> None:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + raw)


def test_legacy_locale_reads_scale_and_voxelsize(tmp_path, monkeypatch):
    segment = tmp_path / "segment"
    segment.mkdir()
    # U+201D is UTF-8 e2 80 9d. Byte 0x9d is undefined in cp1252.
    _write(
        segment / "meta.json",
        {
            "scale": [0.1, 0.1],
            "voxelsize": 7.91,
            "name": "scroll \u201d label",
        },
    )
    monkeypatch.setattr(builtins, "open", _open_text_as_cp1252(builtins.open))

    assert _read_tifxyz_scale_from_meta(segment) == (0.1, 0.1)
    step, voxel, scale = resolve_tifxyz_params(
        SimpleNamespace(
            tifxyz_step_size=None,
            tifxyz_voxel_size_um=None,
            tifxyz_path=str(segment),
            volume_path=str(segment),
        ),
        model_config={},
        volume_scale=0,
    )
    assert (step, voxel, scale) == (10, 7.91, (0.1, 0.1))


def test_utf8_bom_does_not_hide_scale_or_config(tmp_path):
    segment = tmp_path / "segment"
    segment.mkdir()
    _write(segment / "meta.json", {"scale": [0.1, 0.1], "voxelsize": 7.91}, bom=True)
    config = tmp_path / "extrap.json"
    _write(config, {"extrap_degrade_prob": 0.25, "note": "scroll \u201d"}, bom=True)

    assert _read_tifxyz_scale_from_meta(segment) == (0.1, 0.1)
    settings = resolve_extrapolation_settings(
        SimpleNamespace(
            extrapolation_method="identity",
            config_path=str(config),
            checkpoint_path=None,
        )
    )
    assert settings["method"] == "identity"
    assert settings["degrade_prob"] == 0.25


def test_legacy_locale_reads_extrapolation_config(tmp_path, monkeypatch):
    config = tmp_path / "extrap.json"
    _write(config, {"extrap_degrade_prob": 0.25, "note": "scroll \u201d"})
    monkeypatch.setattr(builtins, "open", _open_text_as_cp1252(builtins.open))

    settings = resolve_extrapolation_settings(
        SimpleNamespace(
            extrapolation_method="identity",
            config_path=str(config),
            checkpoint_path=None,
        )
    )
    assert settings["degrade_prob"] == 0.25
