"""Volume config YAML is UTF-8, including on a legacy locale.

open() without an encoding uses the process code page. On cp1252 that
turns a non-ASCII path into a different string before PyYAML parses it.
"""
from __future__ import annotations

import builtins

import vesuvius.data.volume as volume_mod
from vesuvius.data.volume import Cube, Volume


def _open_text_as_cp1252(real_open):
    def wrapped(file, mode="r", *args, **kwargs):
        if isinstance(mode, str) and "b" not in mode and "encoding" not in kwargs:
            kwargs = dict(kwargs)
            kwargs["encoding"] = "cp1252"
        return real_open(file, mode, *args, **kwargs)

    return wrapped


def test_list_files_keeps_non_ascii_under_legacy_locale(tmp_path, monkeypatch):
    root = tmp_path / "prefix"
    configs = root / "vesuvius" / "install" / "configs"
    configs.mkdir(parents=True)
    (configs / "scrolls.yaml").write_text("café: {note: ok}\n", encoding="utf-8")
    monkeypatch.setattr(volume_mod, "get_installation_path", lambda: str(root))
    monkeypatch.setattr(builtins, "open", _open_text_as_cp1252(builtins.open))

    data = volume_mod.list_files()

    assert data["café"]["note"] == "ok"


def test_volume_url_keeps_non_ascii_under_legacy_locale(tmp_path, monkeypatch):
    path = tmp_path / "scrolls.yaml"
    path.write_text(
        '"1":\n'
        '  "54":\n'
        '    "7.91":\n'
        '      volume: "https://example.test/café.zarr"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(builtins, "open", _open_text_as_cp1252(builtins.open))
    vol = Volume.__new__(Volume)
    vol.type = "scroll"
    vol.configs = str(path)
    vol.path = ""
    vol.scroll_id = "1"
    vol.energy = "54"
    vol.resolution = "7.91"

    assert Volume.get_url_from_yaml(vol) == "https://example.test/café.zarr"


def test_cube_urls_keep_non_ascii_under_legacy_locale(tmp_path, monkeypatch):
    path = tmp_path / "cubes.yaml"
    path.write_text(
        "1:\n"
        "  54:\n"
        "    8:\n"
        "      '00010_00020_00030': 'https://example.test/café'\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(builtins, "open", _open_text_as_cp1252(builtins.open))
    cube = Cube.__new__(Cube)
    cube.configs = str(path)
    cube.scroll_id = 1
    cube.energy = 54
    cube.resolution = 8
    cube.z = 10
    cube.y = 20
    cube.x = 30

    volume_url, mask_url = Cube.get_url_from_yaml(cube)

    assert volume_url == "https://example.test/café/00010_00020_00030_volume.nrrd"
    assert mask_url == "https://example.test/café/00010_00020_00030_mask.nrrd"
