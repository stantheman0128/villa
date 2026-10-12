"""The volume tracer reads an existing run.json with the process locale.

Under cp1252, U+201D in the fiber prefix raises UnicodeDecodeError before
the resume check compares settings, and a UTF-8 BOM raises JSONDecodeError.
The document should still be kept. A different prefix still rejects the
resume, and a device change is still allowed.
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
SCRIPT = ROOT / "src/vesuvius/neural_tracing/tracers/fibers/tracing/volume.py"
spec = importlib.util.spec_from_file_location("trace_volume_run_utf8_under_test", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


class Stop(Exception):
    """Raised once the settings comparison has accepted the existing run."""


class _Vol:
    shape = (4, 4, 4)


class _Tracer:
    checkpoint_path = "plain/ckpt.pt"


class _Spec:
    grid_scale = 8.0
    ct_grid_scale = 4.0
    ct_zarr = "plain/ct"


class _Policy:
    def to_dict(self):
        return {"confidence": 0.5, "n_commit": 8, "gate": "full"}


class _Zarr:
    @staticmethod
    def open(*_args, **_kwargs):
        return _Vol()


def _build_tracer(_args):
    return _Tracer(), _Vol(), _Spec(), _Policy()


def _load_done(_out, _owned):
    raise Stop("past the settings comparison")


mod.build_tracer = _build_tracer
mod.zarr = _Zarr()
mod.file_sha256 = lambda _path: "abc"
mod.load_done = _load_done


def argv(out, prefix=None):
    args = ["--presence", str(out / "presence"), "--out", str(out), "--box", "0,0,0,4,4,4"]
    if prefix is not None:
        args.extend(["--prefix", prefix])
    return args


def cp1252_text(monkeypatch):
    real = io.open

    def fake(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, *args, **kwargs):
        if "b" not in str(mode) and encoding in (None, "locale"):
            encoding = "cp1252"
        return real(file, mode, buffering, encoding, errors, newline, *args, **kwargs)

    monkeypatch.setattr(io, "open", fake)


def write_canonical(out):
    with pytest.raises(Stop):
        mod.main(argv(out))
    return json.loads((out / "run.json").read_text(encoding="utf-8"))


def test_unicode_prefix_is_kept(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    record = write_canonical(tmp_path)
    record["prefix"] = "scroll \u201d"
    (tmp_path / "run.json").write_bytes(json.dumps(record, ensure_ascii=False).encode("utf-8"))
    with pytest.raises(Stop):
        mod.main(argv(tmp_path, prefix="scroll \u201d"))


def test_utf8_bom_is_kept(tmp_path):
    record = write_canonical(tmp_path)
    raw = json.dumps(record).encode("utf-8")
    (tmp_path / "run.json").write_bytes(b"\xef\xbb\xbf" + raw)
    with pytest.raises(Stop):
        mod.main(argv(tmp_path))


def test_different_prefix_still_rejects(tmp_path):
    (tmp_path / "run.json").write_text('{"prefix": "other_"}', encoding="utf-8")
    with pytest.raises(SystemExit, match="different settings"):
        mod.main(argv(tmp_path))


def test_malformed_run_still_raises(tmp_path):
    (tmp_path / "run.json").write_text("{", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        mod.main(argv(tmp_path))


def test_speed_only_device_difference_is_allowed(tmp_path):
    record = write_canonical(tmp_path)
    record["device"] = "cpu"
    (tmp_path / "run.json").write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(Stop):
        mod.main(argv(tmp_path))


def test_ascii_threshold_mismatch_still_rejects(tmp_path):
    record = write_canonical(tmp_path)
    record["threshold"] = 10.0
    (tmp_path / "run.json").write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(SystemExit, match="threshold"):
        mod.main(argv(tmp_path))
