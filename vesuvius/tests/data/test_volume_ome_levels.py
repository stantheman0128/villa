"""HTTP stores cannot list a zarr group, so level discovery must use OME metadata.

Volume found resolution levels by iterating group.keys(). A store served
over https has no LIST, so keys() and len() are empty and construction
raises "No arrays found" even when .zattrs names the levels.
"""

from __future__ import annotations

import numpy as np
import zarr

from vesuvius.data.volume import Volume

_ZARR_V3 = int(zarr.__version__.split(".", 1)[0]) >= 3


def _write_group(path: str, names_and_shapes):
    if _ZARR_V3:
        root = zarr.open_group(store=path, mode="w", zarr_format=2)
        create = root.create_array
    else:
        root = zarr.open_group(store=path, mode="w")
        create = root.create_dataset
    for name, shape in names_and_shapes:
        array = create(name, shape=shape, chunks=tuple(min(8, n) for n in shape), dtype="u1")
        array[:] = 1
    return root


def _hide_listing(monkeypatch, path: str) -> None:
    group = zarr.open_group(path, mode="r")
    group_type = type(group)
    monkeypatch.setattr(group_type, "keys", lambda self: [])
    monkeypatch.setattr(group_type, "__len__", lambda self: 0)
    monkeypatch.setattr(group_type, "__iter__", lambda self: iter(()))


def test_unlistable_group_reads_levels_from_multiscales(tmp_path, monkeypatch):
    path = str(tmp_path / "remote.zarr")
    root = _write_group(path, [("0", (8, 8, 8)), ("1", (4, 4, 4))])
    root.attrs["multiscales"] = [{"datasets": [{"path": "0"}, {"path": "1"}]}]
    _hide_listing(monkeypatch, path)

    volume = Volume(type="zarr", path=path, normalization_scheme="none")
    assert volume._num_levels() == 2
    assert np.asarray(volume[0:4, 0:4, 0:4]).shape == (4, 4, 4)
    assert np.asarray(volume[0:2, 0:2, 0:2, 1]).shape == (2, 2, 2)


def test_non_numeric_multiscale_path_is_level_zero(tmp_path, monkeypatch):
    path = str(tmp_path / "named.zarr")
    root = _write_group(path, [("full", (6, 6, 6))])
    root.attrs["multiscales"] = [{"datasets": [{"path": "full"}]}]
    _hide_listing(monkeypatch, path)

    volume = Volume(type="zarr", path=path, normalization_scheme="none")
    assert volume._num_levels() == 1
    assert tuple(volume._level(0).shape) == (6, 6, 6)
    assert np.asarray(volume[0:2, 0:2, 0:2]).shape == (2, 2, 2)
