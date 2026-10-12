"""Volume(type="segment", segment_id=...) must resolve the scroll from scrolls.yaml.

The call in the Volume failure hint and in docs/accessing_data.md omits
scroll_id. Init then looks up energy by scroll None.
"""

from __future__ import annotations

import builtins
from pathlib import Path

import pytest
import yaml
import zarr

from vesuvius.data.volume import Volume


DOCUMENTED_SEGMENT = 20230827161847


def _install_local_loaders(monkeypatch, tmp_path: Path) -> None:
    """Stop init before it opens the remote segment store."""
    import vesuvius.data.volume as volume_module

    monkeypatch.setattr(volume_module, "is_aws_ec2_instance", lambda: False)
    array = zarr.open(
        str(tmp_path / "segment.zarr"),
        mode="w",
        shape=(2, 2, 2),
        chunks=(2, 2, 2),
        dtype="u1",
    )

    monkeypatch.setattr(Volume, "load_data", lambda self: array)
    monkeypatch.setattr(Volume, "load_ome_metadata", lambda self: {})
    monkeypatch.setattr(
        Volume, "download_inklabel", lambda self, save_path=None: None
    )


def _volume_with_config(path: Path) -> Volume:
    volume = Volume.__new__(Volume)
    volume.configs = str(path)
    return volume


def test_documented_segment_call_reads_scroll_from_config(monkeypatch, tmp_path):
    _install_local_loaders(monkeypatch, tmp_path)
    segment = Volume(type="segment", segment_id=DOCUMENTED_SEGMENT)
    assert str(segment.scroll_id) == "1"
    assert segment.energy == 54
    assert segment.resolution == 7.91
    assert "20230827161847.zarr" in segment.url


def test_explicit_energy_and_resolution_are_kept(monkeypatch, tmp_path):
    _install_local_loaders(monkeypatch, tmp_path)
    seen = {}

    def capture_url(self):
        seen["scroll_id"] = self.scroll_id
        seen["energy"] = self.energy
        seen["resolution"] = self.resolution
        return "memory://segment"

    monkeypatch.setattr(Volume, "get_url_from_yaml", capture_url)
    segment = Volume(
        type="segment",
        segment_id=DOCUMENTED_SEGMENT,
        energy=70,
        resolution=1.25,
    )
    assert str(seen["scroll_id"]) == "1"
    assert seen["energy"] == 70
    assert seen["resolution"] == 1.25
    assert segment.energy == 70
    assert segment.resolution == 1.25


def test_explicit_scroll_id_is_not_replaced(monkeypatch, tmp_path):
    _install_local_loaders(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match=r"scroll=2"):
        Volume(type="segment", segment_id=DOCUMENTED_SEGMENT, scroll_id=2)


def test_unknown_segment_names_the_id(monkeypatch, tmp_path):
    _install_local_loaders(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match=r"Segment 99999999999999"):
        Volume(type="segment", segment_id=99999999999999)


def test_inference_uses_the_config_entry_not_the_canonical_map(tmp_path):
    config = tmp_path / "scrolls.yaml"
    config.write_text(
        "\n".join(
            [
                '"1":',
                '  "54":',
                '    "7.91":',
                "      segments:",
                '        "111": "http://example.invalid/111.zarr"',
                '"9":',
                '  "12":',
                '    "1.25":',
                "      segments:",
                '        "222": "http://example.invalid/222.zarr"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    volume = _volume_with_config(config)
    assert volume._infer_scroll_from_segment(111) == ("1", 54, 7.91)
    assert volume._infer_scroll_from_segment("222") == ("9", 12, 1.25)


@pytest.mark.parametrize("text", ["", "   \n\n", "null\n"])
def test_empty_config_is_invalid(tmp_path, text):
    config = tmp_path / "scrolls.yaml"
    config.write_text(text, encoding="utf-8")
    volume = _volume_with_config(config)
    with pytest.raises(ValueError, match="empty or invalid YAML"):
        volume._infer_scroll_from_segment(111)


def test_malformed_config_propagates(tmp_path):
    config = tmp_path / "scrolls.yaml"
    config.write_text(":\n  - [\n", encoding="utf-8")
    volume = _volume_with_config(config)
    with pytest.raises(yaml.YAMLError):
        volume._infer_scroll_from_segment(111)


def test_missing_config_is_file_not_found(tmp_path):
    volume = _volume_with_config(tmp_path / "missing.yaml")
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        volume._infer_scroll_from_segment(111)


def test_inference_keeps_non_ascii_segment_id_under_legacy_locale(tmp_path, monkeypatch):
    config = tmp_path / "scrolls.yaml"
    config.write_text(
        "\n".join(
            [
                '"1":',
                '  "54":',
                '    "7.91":',
                "      segments:",
                '        "café": "http://example.invalid/segment.zarr"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    real_open = builtins.open

    def open_text_as_cp1252(file, mode="r", *args, **kwargs):
        if isinstance(mode, str) and "b" not in mode and "encoding" not in kwargs:
            kwargs = dict(kwargs)
            kwargs["encoding"] = "cp1252"
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", open_text_as_cp1252)
    volume = _volume_with_config(config)
    assert volume._infer_scroll_from_segment("café") == ("1", 54, 7.91)
