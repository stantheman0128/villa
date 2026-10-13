"""read_3d_ink_params reads 3d_ink_params.json with the process locale.

Under cp1252, U+201D in a label name raises UnicodeDecodeError before the
labels are returned, and a UTF-8 BOM raises JSONDecodeError. The label
name should still be returned. A missing file still returns an empty
document. Malformed JSON is still reported. An ASCII label name is still
kept. remap.json is unchanged.
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
spec = importlib.util.spec_from_file_location("ink_params_utf8_under_test", SCRIPT)
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
    return {"labels": {name: "255", "background": "100"}}


def test_unicode_label_is_returned(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    write_json(tmp_path / "3d_ink_params.json", document("scroll \u201d"))
    labels = mod.read_3d_ink_params(tmp_path)["labels"]
    assert labels["scroll \u201d"] == "255"
    assert labels["background"] == "100"


def test_utf8_bom_keeps_the_label(tmp_path):
    write_json(tmp_path / "3d_ink_params.json", document("ink"), bom=True)
    labels = mod.read_3d_ink_params(tmp_path)["labels"]
    assert labels["ink"] == "255"
    assert labels["background"] == "100"


def test_ascii_label_is_kept(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    write_json(tmp_path / "3d_ink_params.json", document("ink"))
    labels = mod.read_3d_ink_params(tmp_path)["labels"]
    assert labels["ink"] == "255"
    assert labels["background"] == "100"


def test_missing_file_returns_empty(tmp_path):
    assert mod.read_3d_ink_params(tmp_path) == {}


def test_document_without_labels_is_returned(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    write_json(tmp_path / "3d_ink_params.json", {"note": "plain"})
    assert mod.read_3d_ink_params(tmp_path) == {"note": "plain"}


def test_malformed_json_still_raises(tmp_path):
    write_json(tmp_path / "3d_ink_params.json", None, raw=b"{")
    with pytest.raises(json.JSONDecodeError):
        mod.read_3d_ink_params(tmp_path)
