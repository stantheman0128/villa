"""plot_curves reads a training log with the process locale.

Under cp1252, U+201D in a dataset name raises UnicodeDecodeError before
the curves are drawn, and a UTF-8 BOM raises JSONDecodeError. The log
should still be plotted.
Related to #1524.
"""
import importlib.util
import io
import json
import sys
import types
from pathlib import Path

import pytest

mpl = types.ModuleType("matplotlib")
mpl.use = lambda *args, **kwargs: None
pyplot = types.ModuleType("matplotlib.pyplot")
mpl.pyplot = pyplot
sys.modules["matplotlib"] = mpl
sys.modules["matplotlib.pyplot"] = pyplot
sys.modules["matplotlib.patheffects"] = types.ModuleType("matplotlib.patheffects")


class _Ax:
    def __init__(self):
        self.calls = []

    def plot(self, xs, ys, label=None, **kwargs):
        self.calls.append((list(xs), label))

    def legend(self, **kwargs):
        return None

    def set_xlabel(self, *args, **kwargs):
        return None

    def set_ylim(self, *args, **kwargs):
        return None

    def get_legend_handles_labels(self):
        return ([object()], ["x"]) if self.calls else ([], [])


class _Fig:
    def tight_layout(self):
        return None

    def savefig(self, path, dpi=None):
        Path(path).write_bytes(b"png")


_drawn = {}


def _subplots(*args, **kwargs):
    count = args[1] if len(args) > 1 else 1
    axes = [_Ax() for _ in range(count)]
    _drawn["axes"] = axes
    return _Fig(), axes


pyplot.subplots = _subplots
pyplot.close = lambda *args, **kwargs: None

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
SCRIPT = ROOT / "src/vesuvius/neural_tracing/tracers/fibers/evaluation/monitor/plots.py"
spec = importlib.util.spec_from_file_location("plot_curves_utf8_under_test", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def write_jsonl(path, rows, bom=False):
    raw = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
    if bom:
        raw = b"\xef\xbb\xbf" + raw
    path.write_bytes(raw)


def cp1252_text(monkeypatch):
    real = io.open

    def fake(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, *args, **kwargs):
        if "b" not in str(mode) and encoding in (None, "locale"):
            encoding = "cp1252"
        return real(file, mode, buffering, encoding, errors, newline, *args, **kwargs)

    monkeypatch.setattr(io, "open", fake)


def rollout(dataset):
    return {
        "step": 1,
        "flow": 0.5,
        "confidence_loss": 0.1,
        "dataset": dataset,
        "split": "monitor",
        "threshold": 1,
        "roll_coverage": 0.2,
        "roll_precision": 0.3,
        "roll_diverged": 0.0,
    }


def draw(path, out):
    mod.plot_curves(path, out)
    return _drawn["axes"]


def test_unicode_dataset_reaches_the_plot(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    path = tmp_path / "log.jsonl"
    write_jsonl(path, [rollout("scroll \u201d")])
    axes = draw(path, tmp_path / "curves.png")
    labels = [label for ax in axes for _xs, label in ax.calls]
    assert any(label and "scroll \u201d" in label for label in labels)


def test_utf8_bom_reaches_the_plot(tmp_path):
    path = tmp_path / "log.jsonl"
    write_jsonl(path, [rollout("paris")], bom=True)
    axes = draw(path, tmp_path / "curves.png")
    labels = [label for ax in axes for _xs, label in ax.calls]
    assert any(label and "paris" in label for label in labels)


def test_malformed_line_still_raises(tmp_path, monkeypatch):
    cp1252_text(monkeypatch)
    path = tmp_path / "log.jsonl"
    path.write_bytes(b"{\n")
    with pytest.raises(json.JSONDecodeError):
        draw(path, tmp_path / "curves.png")


def test_resume_still_drops_later_steps(tmp_path):
    path = tmp_path / "log.jsonl"
    write_jsonl(path, [
        {"step": 5, "flow": 1.0, "confidence_loss": 0.2},
        {"step": 2, "event": "resume_configuration", "checkpoint": "plain/last.pt"},
        {"step": 3, "flow": 0.4, "confidence_loss": 0.1},
    ])
    axes = draw(path, tmp_path / "curves.png")
    assert axes[0].calls[0][0] == [3]


def test_row_without_loss_is_not_a_training_point(tmp_path):
    path = tmp_path / "log.jsonl"
    row = rollout("paris")
    del row["flow"]
    del row["confidence_loss"]
    write_jsonl(path, [row])
    axes = draw(path, tmp_path / "curves.png")
    assert axes[0].calls[0][0] == []
    labels = [label for _xs, label in axes[-1].calls]
    assert any(label and "paris" in label for label in labels)


def test_ascii_dataset_is_plotted(tmp_path):
    path = tmp_path / "log.jsonl"
    write_jsonl(path, [rollout("paris")])
    axes = draw(path, tmp_path / "curves.png")
    assert axes[0].calls[0][0] == [1]
    labels = [label for ax in axes for _xs, label in ax.calls]
    assert any(label and label.startswith("paris monitor") for label in labels)
