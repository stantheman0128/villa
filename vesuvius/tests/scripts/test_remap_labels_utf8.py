"""load_and_validate_remap reads remap.json with the process locale.

Under cp1252, U+201D in a padded label name raises UnicodeDecodeError
before the name is matched, and a UTF-8 BOM raises JSONDecodeError. The
padded label should still be accepted. A missing file is still reported.
Malformed JSON is still reported. An unknown label is still rejected. An
ASCII padded label is still kept. 3d_ink_params.json is unchanged.
Related to #1524.
"""
import builtins
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


def _package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
    return module


for _name in (
    "cv2",
    "fastremap",
    "tifffile",
    "zarr",
    "numcodecs",
    "scipy",
    "scipy.ndimage",
    "tqdm",
    "vesuvius",
    "vesuvius.tifxyz",
):
    _package(_name)
sys.modules["numcodecs"].Blosc = object
sys.modules["scipy.ndimage"].label = lambda *args, **kwargs: None
sys.modules["scipy.ndimage"].find_objects = lambda *args, **kwargs: None
sys.modules["scipy.ndimage"].distance_transform_edt = lambda *args, **kwargs: None
sys.modules["tqdm"].tqdm = lambda *args, **kwargs: args[0] if args else None
sys.modules["vesuvius.tifxyz"].read_tifxyz = lambda *args, **kwargs: None

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "src/vesuvius/scripts/labels_to_zarr.py"
spec = importlib.util.spec_from_file_location("remap_labels_utf8_under_test", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def cp1252_text(monkeypatch):
    real = builtins.open

    def fake(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
        if "b" not in str(mode) and encoding in (None, "locale"):
            encoding = "cp1252"
        return real(file, mode, buffering, encoding, errors, newline, closefd, opener)

    monkeypatch.setattr(builtins, "open", fake)


def write_json(path, document, *, bom=False, raw=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = raw if raw is not None else json.dumps(document, ensure_ascii=False).encode("utf-8")
    if bom:
        payload = b"\xef\xbb\xbf" + payload
    path.write_bytes(payload)


def document(name="ink"):
    return {
        "mappings": {"0": 0, "255": 1, "100": 2},
        "padded_labels": [name],
        "expand_labels": [],
    }


def labels(name="ink"):
    return {name: "255", "background": "100"}


def test_unicode_padded_label_is_accepted(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    path = tmp_path / "remap.json"
    write_json(path, document("scroll \u201d"))
    mappings, padded, expand, fill = mod.load_and_validate_remap(path, labels("scroll \u201d"))
    assert padded == {255}
    assert mappings[255] == 1
    assert mappings[0] == 0
    assert expand == []
    assert fill == 0


def test_utf8_bom_keeps_the_padded_label(tmp_path):
    path = tmp_path / "remap.json"
    write_json(path, document("ink"), bom=True)
    mappings, padded, expand, fill = mod.load_and_validate_remap(path, labels("ink"))
    assert padded == {255}
    assert mappings[255] == 1
    assert fill == 0
    assert expand == []


def test_ascii_padded_label_is_kept(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    path = tmp_path / "remap.json"
    write_json(path, document("ink"))
    mappings, padded, expand, fill = mod.load_and_validate_remap(path, labels("ink"))
    assert padded == {255}
    assert mappings[255] == 1
    assert mappings[100] == 2
    assert fill == 0
    assert expand == []


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        mod.load_and_validate_remap(tmp_path / "remap.json", labels("ink"))


def test_unknown_padded_label_still_raises(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    path = tmp_path / "remap.json"
    write_json(path, document("dust"))
    with pytest.raises(ValueError, match="dust"):
        mod.load_and_validate_remap(path, labels("ink"))


def test_malformed_json_still_raises(tmp_path):
    path = tmp_path / "remap.json"
    write_json(path, None, raw=b"{")
    with pytest.raises(json.JSONDecodeError):
        mod.load_and_validate_remap(path, labels("ink"))
