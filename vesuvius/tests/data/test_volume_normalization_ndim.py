"""Instance normalization of a read without a channel axis uses the whole read.

A 2D read used to treat each row as its own channel. A 1D read normalized
each voxel against itself and returned zeros.
"""

import numpy as np
import zarr


def _write(path, array):
    major = int(str(zarr.__version__).split(".", 1)[0])
    if major >= 3:
        store = zarr.open_array(
            str(path),
            mode="w",
            shape=array.shape,
            chunks=array.shape,
            dtype=array.dtype,
            zarr_format=2,
        )
    else:
        store = zarr.open(
            str(path),
            mode="w",
            shape=array.shape,
            chunks=array.shape,
            dtype=array.dtype,
        )
    store[:] = array
    return str(path)


def _gradient_volume(tmp_path):
    rng = np.random.default_rng(0)
    volume = (
        np.arange(16)[None, :, None] * 400
        + rng.integers(0, 300, size=(8, 16, 16))
    ).astype("uint16")
    return _write(tmp_path / "gradient.zarr", volume), volume


def _instance(array, scheme):
    values = array.astype(np.float32)
    if scheme == "instance_zscore":
        return (values - values.mean()) / max(float(values.std()), 1e-8)
    if scheme == "instance_minmax":
        return (values - values.min()) / max(float(values.max() - values.min()), 1e-8)
    lower, upper = np.percentile(values, (1.0, 99.0))
    span = float(upper - lower)
    if span <= 1e-8:
        return np.zeros_like(values)
    return (np.clip(values, lower, upper) - lower) / span


def test_2d_read_uses_the_whole_slice(tmp_path):
    from vesuvius.data.volume import Volume

    path, volume = _gradient_volume(tmp_path)
    reader = Volume(type="zarr", path=path, normalization_scheme="instance_zscore")
    got = np.asarray(reader[4, 0:16, 0:16])
    expected = _instance(volume[4], "instance_zscore")

    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, atol=1e-5)


def test_2d_and_single_plane_3d_reads_match(tmp_path):
    from vesuvius.data.volume import Volume

    path, _volume = _gradient_volume(tmp_path)
    reader = Volume(type="zarr", path=path, normalization_scheme="instance_zscore")
    plane = np.asarray(reader[4, 0:16, 0:16])
    slab = np.asarray(reader[4:5, 0:16, 0:16])[0]
    np.testing.assert_allclose(plane, slab, atol=1e-5)


def test_1d_read_is_not_all_zeros(tmp_path):
    from vesuvius.data.volume import Volume

    path, volume = _gradient_volume(tmp_path)
    reader = Volume(type="zarr", path=path, normalization_scheme="instance_minmax")
    got = np.asarray(reader[4, 7, 0:16])
    expected = _instance(volume[4, 7], "instance_minmax")

    assert got.shape == expected.shape
    assert len(np.unique(np.round(got, 5))) > 1
    np.testing.assert_allclose(got, expected, atol=1e-5)


def test_percentile_minmax_2d_uses_the_whole_slice(tmp_path):
    from vesuvius.data.volume import Volume

    path, volume = _gradient_volume(tmp_path)
    reader = Volume(type="zarr", path=path, normalization_scheme="percentile_minmax")
    got = np.asarray(reader[4, 0:16, 0:16])
    expected = _instance(volume[4], "percentile_minmax")

    np.testing.assert_allclose(got, expected, atol=1e-5)


def test_global_zscore_2d_read_keeps_its_shape(tmp_path):
    from vesuvius.data.volume import Volume

    path, volume = _gradient_volume(tmp_path)
    reader = Volume(
        type="zarr",
        path=path,
        normalization_scheme="global_zscore",
        global_mean=100.0,
        global_std=25.0,
    )
    got = np.asarray(reader[4, 0:16, 0:16])
    expected = (volume[4].astype(np.float32) - 100.0) / 25.0

    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, atol=1e-5)


def test_channel_axis_stays_per_channel(tmp_path):
    from vesuvius.data.volume import Volume

    rng = np.random.default_rng(1)
    array = np.stack([
        rng.integers(0, 100, size=(4, 8, 8)),
        rng.integers(500, 900, size=(4, 8, 8)),
    ]).astype("uint16")
    path = _write(tmp_path / "channels.zarr", array)
    reader = Volume(type="zarr", path=path, normalization_scheme="instance_zscore")
    got = np.asarray(reader[:, :, :, :])
    expected = np.stack([_instance(array[c], "instance_zscore") for c in range(2)])

    assert got.shape == array.shape
    np.testing.assert_allclose(got, expected, atol=1e-5)
