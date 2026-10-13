"""A native phase cache reads zarr.json with the process locale.

Under cp1252, U+201D in the sampling name raises UnicodeDecodeError
before the cache geometry is returned, and a UTF-8 BOM raises
JSONDecodeError. The sampling name should still be kept. A wrong
artifact type still raises. A missing file still raises.
Related to #1524.
"""
import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
SCRIPT = ROOT / "src/vesuvius/neural_tracing/winding_models/infer_winding_volume.py"
spec = importlib.util.spec_from_file_location("phase_cache_zarr_utf8_under_test", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def cp1252_text(monkeypatch):
    real = io.open

    def fake(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, *args, **kwargs):
        if "b" not in str(mode) and encoding in (None, "locale"):
            encoding = "cp1252"
        return real(file, mode, buffering, encoding, errors, newline, *args, **kwargs)

    monkeypatch.setattr(io, "open", fake)


def write_cache(directory, sampling="trilinear", *, bom=False, raw=None, artifact="winding_native_phase_cache"):
    directory.mkdir(parents=True, exist_ok=True)
    if raw is None:
        document = {
            "attributes": {
                "artifact_type": artifact,
                "ray_length": 32,
                "transverse_size": 8,
                "column_stride": 2,
                "spacing": 1.0,
                "sampling": sampling,
                "crossing_sigma_wv": 1.5,
            }
        }
        payload = json.dumps(document, ensure_ascii=False).encode("utf-8")
    else:
        payload = raw
    if bom:
        payload = b"\xef\xbb\xbf" + payload
    (directory / "zarr.json").write_bytes(payload)


def test_unicode_sampling_is_kept(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    write_cache(tmp_path, "scroll \u201d")
    cfg = mod._native_phase_cache_model_cfg(tmp_path)
    assert cfg["sampling"] == "scroll \u201d"
    assert cfg["ray_length"] == 32


def test_utf8_bom_is_kept(tmp_path):
    write_cache(tmp_path, "trilinear", bom=True)
    assert mod._native_phase_cache_model_cfg(tmp_path)["sampling"] == "trilinear"


def test_ascii_sampling_is_kept(tmp_path):
    write_cache(tmp_path, "trilinear")
    cfg = mod._native_phase_cache_model_cfg(tmp_path)
    assert cfg["sampling"] == "trilinear"
    assert cfg["crossing_sigma_wv"] == 1.5


def test_wrong_artifact_still_raises(tmp_path):
    write_cache(tmp_path, artifact="other")
    with pytest.raises(ValueError, match="not a native winding phase cache"):
        mod._native_phase_cache_model_cfg(tmp_path)


def test_malformed_zarr_json_still_raises(tmp_path):
    write_cache(tmp_path, raw=b"{")
    with pytest.raises(json.JSONDecodeError):
        mod._native_phase_cache_model_cfg(tmp_path)


def test_missing_zarr_json_still_raises(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    with pytest.raises(FileNotFoundError):
        mod._native_phase_cache_model_cfg(tmp_path)
