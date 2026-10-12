"""Intensity-property JSON is UTF-8. A legacy text open must not drop it.

``load_intensity_props_formatted`` and ``load_intensity_properties`` catch
every exception and treat it as a missing file. On cp1252 the U+201D byte
is undefined, so a valid file becomes "failed to parse". A UTF-8 BOM is
the same miss.
"""
from __future__ import annotations

import builtins
import importlib.util
import json
from pathlib import Path

# datasets/__init__.py imports ZarrDataset, which imports SimpleITK. Load the
# reader module by path so this test does not need that stack.
_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "vesuvius"
    / "models"
    / "datasets"
    / "intensity_properties.py"
)
_spec = importlib.util.spec_from_file_location("intensity_properties_utf8", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
_intensity = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_intensity)
load_intensity_properties = _intensity.load_intensity_properties
load_intensity_props_formatted = _intensity.load_intensity_props_formatted


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


def _props() -> dict:
    return {
        "mean": 10.0,
        "std": 2.0,
        "min": 0.0,
        "max": 20.0,
        "median": 9.0,
        "percentile_00_5": 1.0,
        "percentile_99_5": 19.0,
        "note": "scroll \u201d",
    }


def test_formatted_legacy_locale_keeps_the_stats(tmp_path, monkeypatch):
    path = tmp_path / "intensity_props.json"
    _write(path, _props())
    monkeypatch.setattr(builtins, "open", _open_text_as_cp1252(builtins.open))
    loaded = load_intensity_props_formatted(path)
    assert loaded is not None
    assert loaded["mean"] == 10.0


def test_formatted_utf8_bom_keeps_the_stats(tmp_path):
    path = tmp_path / "intensity_props.json"
    _write(path, _props(), bom=True)
    loaded = load_intensity_props_formatted(path)
    assert loaded is not None
    assert loaded["mean"] == 10.0


def test_cache_legacy_locale_keeps_the_stats(tmp_path, monkeypatch):
    path = tmp_path / "intensity_properties.json"
    _write(path, {"intensity_properties": _props(), "normalization_scheme": "ct"})
    monkeypatch.setattr(builtins, "open", _open_text_as_cp1252(builtins.open))
    loaded = load_intensity_properties(tmp_path)
    assert loaded is not None
    props, scheme = loaded
    assert scheme == "ct"
    assert props["mean"] == 10.0


def test_cache_utf8_bom_keeps_the_stats(tmp_path):
    path = tmp_path / "intensity_properties.json"
    _write(path, {"intensity_properties": _props(), "normalization_scheme": "ct"}, bom=True)
    loaded = load_intensity_properties(tmp_path)
    assert loaded is not None
    props, scheme = loaded
    assert scheme == "ct"
    assert props["mean"] == 10.0
