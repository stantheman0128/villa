# PHerc0358 object-store reproduction

Public PHerc0358 volume `20250821151737`, tracks plus Lasagna fields only, from the downloads through a short CPU fit. The tree under test is upstream `main` at `f29334cd8` (2026-11-16, includes `#2564` and `#2585`), which is a descendant of `e413746da`. Commands below use `$DATA` for a downloads directory outside the repository and `$BUILD` for the standalone object-store build directory, also outside the repository.

The finished path is a 100-voxel window, z `[10400, 10500)`, on a store that holds only the tracks whose packed bounds lie in that window. `vc_object_ingest` ingested 114451 of those tracks, the input report printed that count, and `fit_spiral.py` exited 0 after 300 steps. The first 300-step attempt exited 1 in 7 s because the virtualenv still had an `e413746da` extension, and the rerun after rebuilding that module against this checkout is the one timed. The full 12256170-track ingest was terminated mid-write and was not restarted.

## Environment

| Item | Value |
| --- | --- |
| OS | Ubuntu 24.04.5 LTS, Linux 6.12.113+, x86_64 |
| CPU | 4 vCPU, Intel Xeon (the guest reports the model as `Intel(R) Xeon(R) Processor`) |
| RAM | MemTotal 16387328 kB (15.6 GiB) |
| Disk | 254 GB volume, about 220 GB free before the track download |
| C compiler | gcc 13.3.0 |
| C++ compiler | g++ 13.3.0; `/usr/bin/c++` is Clang 18.1.3 |
| CMake / Ninja | 3.28.3 / 1.11.1 |
| Python | system 3.12.3; Spiral virtualenv 3.14.10 via uv 0.13.8 |
| GPU | none used; the fit was started with `FIT_SPIRAL_DEVICE=cpu` |

No NVIDIA device was used. The fit log says `flow field: no fused kernel on cpu`.

## Commands

Packages the object-store README does not name, installed before the build would configure. The first documented cmake failed before this install.

```sh
sudo apt-get install -y ninja-build build-essential pkg-config \
  libsqlite3-dev libssl-dev nlohmann-json3-dev libzstd-dev libopencv-dev \
  libboost-dev libomp-dev libcurl4-openssl-dev zlib1g-dev libblosc-dev \
  libblosc2-dev liblz4-dev libtiff-dev libpng-dev libjpeg-dev \
  libconcurrentqueue-dev python3-dev
sudo apt-get install -y g++-14 libstdc++-14-dev
```

`libblosc2-dev` was a second install: the reader configure asks for blosc2 >= 2.13. `g++-14` and `libstdc++-14-dev` were installed after Clang, as `/usr/bin/c++`, failed to link a compiler test (`cannot find -lstdc++`) because this image's Clang selects GCC 14's library directory while only the GCC 13 development package was present. `g++` itself linked a test program without that extra package.

Documented standalone build (`vc_object_store/README.md`):

```sh
cmake -S vc_object_store -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DVC_OBJECT_STORE_BUILD_ADAPTERS=ON \
  -DVC_OBJECT_STORE_BUILD_LINKING=OFF -DVC_OBJECT_STORE_BUILD_TESTS=ON
cmake --build "$BUILD" -j 4 --target vc_object_ingest
cmake --build "$BUILD" -j 4
```

The targeted build linked `$BUILD/bin/vc_object_ingest` (exit 0). The full documented build then failed while compiling `test_object_store_hosted_cache.cpp` (finding 5). `vc_object_ingest --help` works.

Spiral environment, from `spiral-fitting/`, as the README's "Running without CUDA" section:

```sh
uv sync --extra cpu
```

That created the 3.14.10 virtualenv and installed the CPU extra (Kimimaro). It does not build `vc_object_ingest`; the README says so. After fast-forwarding onto `f29334cd8` (`#2564` changes the native module), the editable `vc-object-store` module was rebuilt against that commit and checked with `import vc_object_store` plus `vc_spiral.track_crossings`.

Catalog, umbilicus, and Lasagna metadata, from the public bucket, into `$DATA`:

```sh
# metadata.json (catalog; the response is gzip), the umbilicus JSON,
# and PHerc0358.lasagna.json plus group-2 .zattrs/.zarray/.zgroup
# for PHerc0358_nx, _ny, and _grad_mag
```

Volume `20250821151737`: shape `[14744, 7783, 7783]`, `pixel_size_um` 9.362, `z_direction_is_top_to_bottom` true, `left_handed_coordinates` false, so the manifest sense is `ACW`. Umbilicus control points span z 1216..14064, which covers the fit window. Group `2` `.zarray` is shape `[3686, 1946, 1946]`, chunks `[32, 32, 32]`, dimension separator `/`. Its multiscale `.zattrs` scale is `[4.0, 4.0, 4.0]`. The Lasagna JSON says `scaledown` 2 and `source_to_base` 1.0.

OME-Zarr chunks are omitted when empty (HTTP 404), so a full 61 by 61 grid was not downloaded. Listed keys only, group 2, z rows 80-83, all three stores:

```sh
# list-type=2 for each store/2/{80,81,82,83}/, then curl those keys
```

That is the z window plus the 160-voxel margin. Group 2 is 4 times below the fit volume, so fit z `[10400, 10500)` is array z 2600..2624. Chunk row 81 is array z 2592-2623 and chunk row 82 is 2624-2655. One row of all three stores is on the order of 100 MB; four rows completed. The packer later reported 743 of 6017 union chunks missing and read them as zero. Those are holes in the published stores, not a failed download of a listed key. The fit continued.

Tracks. The public directory publishes the DBM (5.6 GiB), a packed `.vctracks` mirror, and a `.crossings.npz` (not used; `#2564` computes crossings in the fitter). The packed header is `VCTRK01`, 12256170 tracks, 469798434 points. `coordinates.i32` is 5.3 GiB and was downloaded. The DBM itself was not: a sparse file of the published size was enough to reproduce the stale-signature error, and ingest was given the `.vctracks` directory.

There is no z-window flag on `convert_track_store.py` or `vc_object_ingest`. `build_track_crossings.py --z-min/--z-max` exists, and the README now says the fit does not read that sidecar. Tracks whose packed z bounds lie entirely in `[10400, 10500)` were copied into a subset `.vctracks` directory: 114451 tracks, 4118215 points. Measured entirely-inside counts for nearby windows: `[10000, 10100)` 97709, `[10500, 10600)` 123473, `[10000, 11000)` 1459649.

Stale-signature check, DBM path, size and second-resolution mtime matching the published `Last-Modified` but not the stored nanosecond stamp:

```sh
"$BUILD/bin/vc_object_ingest" --db "$DATA/stores/stale.vcobj" \
  --author "$USER" --volume-uuid 20250821151737 --scale-factor 1 \
  --tracks "$DATA/stale-repro/PHerc0358_20250821151737_surface_m7_L0_th0.2.dbm"
```

Exit 1. Setting the mtime to the exact signature nanoseconds got past that check and then failed on the incomplete mirror (`invalid packed track header`), which shows the comparison really is name + size + nanosecond mtime.

`convert_track_store.py` when the destination mirror already exists, on a tiny DBM created for this check:

```sh
python convert_track_store.py "$DATA/convert-exists/tiny.dbm"
```

Exit 1. See finding 2. The public PHerc0358 mirror is already present, so the README's convert step hits the same exception.

Ingest of the z-window directory (the path that skips the DBM signature check, `PackedTrackStore.cpp` line 31):

```sh
"$BUILD/bin/vc_object_ingest" --db "$DATA/stores/pherc0358-z10400-10500.vcobj" \
  --author "$USER" --volume-uuid 20250821151737 --scale-factor 1 \
  --tracks "$DATA/tracks/z10400-10500.vctracks"
```

Exit 0. `114451 ingested, 0 problems`. A 200-track prefix of the same subset was ingested first (exit 0, `200 ingested, 0 problems`) to see the rate. Scale factor 1: `extract.json` has `downsample_factor` 2, but the extractor stores full-resolution coordinates, and the volume is the 9.362 um fit volume.

Input report, from `spiral-fitting/` on current main, with the rebuilt module:

```sh
python object_store_inputs.py "$DATA/stores/pherc0358-z10400-10500.vcobj" \
  --z-begin 10400 --z-end 10500
```

Exit 0. The report includes `tracks: 114451 selected (all active); 114451 at scale 1` and `constraints: ... tracks 114451`. It also computed crossings: `59207 computed from 114451 tracks in 1.2 s`.

Lasagna scale, not set in the manifest, so the fitter derives it:

```text
group 2, scaledown 2, source_to_base 1.0
lasagna_scale_for -> 4
"4 base voxels per array voxel, 1 per fit-volume voxel"
```

The fit log says `fiber-patch normals: lasagna at scale 4`. `scaledown` 2 with matching shapes is 4 full-resolution voxels per array voxel (`2**scaledown / base_per_volume_voxel`, and `base_per_volume_voxel` is 1 because the fit volume and the manifest `base_shape_zyx` are the same grid). `lasagna_scale_for` in `spiral-fitting/lasagna_source.py` returns that pair `(4, "4 base voxels per array voxel, 1 per fit-volume voxel")`, and `resolve_lasagna_scale` (`fit_session.py`) stores 4. Calling both on this manifest printed that pair. The packer line is `array 2`: group `2` is the OME-Zarr path whose `.zattrs` scale is `[4.0, 4.0, 4.0]`, not a scale of 2.

Documented pack, before the fit, with `--as` set to the open-data manifest URL so the identity is `url-e395b76c07f414b9db3512fcdee307b2`:

```sh
python pack_resident_pools.py "$DATA/PHerc0358.lasagna.json" \
  --as "<open-data PHerc0358.lasagna.json URL>" \
  --cache "$DATA/spiral-cache" --what normals,grad_mag
```

Exit 0. Wall time 2.9 s, peak RSS 130868 kB. It packed 6017 normal chunks and 5274 `grad_mag` chunks, and warned that one channel's group directory `2` was missing 743 of 6017 union chunks (the extra z=78 files present only under `nx`); those read back as zero. The fit above did not use that URL identity. Its manifest names the local Lasagna file without `original_url`, so the fitter packed again under `sha256-fefa7e2e…` and still logged scale 4.

Manifest (`schema_version` 1, `volume_uuid` `20250821151737`, voxel size 9.362, base shape as above, sense `ACW`, umbilicus `coordinate_scale` 1, `lasagna.manifest` pointing at the local `PHerc0358.lasagna.json`). Short fit, from `spiral-fitting/`:

```sh
FIT_SPIRAL_DEVICE=cpu WANDB_MODE=disabled \
FIT_SPIRAL_OUT_DIR="$DATA/fit-out" \
FIT_SPIRAL_CONFIG_OVERRIDES='{"z_begin":10400,"z_end":10500,"optimizer_num_training_steps":40,"input_use_tracks":true,"input_use_verified_patches":false,"input_use_fibers":false,"input_use_fiber_directions":false,"input_use_pcl_absolute":false,"input_use_pcl_relative":false,"input_use_pcl_same_winding":false,"input_use_pcl_drawn_control_points":false,"input_use_patch_overlaps":false,"input_use_winding_inference":false,"input_use_outer_shell":false,"dense_spacing_mode":"grad_mag","loss_weight_shell_outer":0.0,"loss_start_track_dt":0}' \
  .venv/bin/python fit_spiral.py \
  --inputs "$DATA/fit/spiral-inputs.json" \
  --object-store "$DATA/stores/pherc0358-z10400-10500.vcobj" \
  --object-store-author "$USER" --cache "$DATA/fit-cache"
```

`loss_start_track_dt` defaults to 25000, which a 40-step run would never reach, so it was set to 0. The other overrides are the old-flow starting point (tracks on, patches, fibers, PCLs, winding inference and outer shell off, `dense_spacing_mode` `grad_mag`, shell loss weight 0). Exit 0. The training loop runs `range(0, optimizer_num_training_steps)` and only then prints `Saving final checkpoint`; that line was reached and `checkpoint_fitted.ckpt` was written. Step 0 included `track_radius = 2394.6`. Final satisfaction: 13201/114451 tracks, 1462744/4118215 points. A warning during the final track evaluation did not fail the run: `tracks.py:1782` `UserWarning: The given NumPy array is not writable` while wrapping coordinates for PyTorch.

The same command with `optimizer_num_training_steps` 300, and `FIT_SPIRAL_OUT_DIR` set to a separate output directory, is the short fit. The first 300-step attempt exited 1 in 7 s because the virtualenv still had an `e413746da` extension, and the rerun after rebuilding that module against this checkout (`uv sync --extra cpu` from `spiral-fitting/`) is the one timed. No code fix. The rebuilt `selected_inputs` accepts `relationship_refs` (`selected_inputs(self, refs: list, position: dict, relationship_refs: object | None = None) -> dict`). Exit 0. Wall time 235.1 s, peak RSS 3957716 kB (3.77 GiB). Progress lines are sampled about every 5 s, so the last optimizing line is `PROGRESS Optimizing — 296/300` and the next line is `Saving final checkpoint`, which `fit_spiral.py` prints only after `range(0, optimizer_num_training_steps)` returns. `checkpoint_fitted.ckpt` was written in that separate output directory. Step 0 loss was 2460.9 with `track_radius = 2394.6`; step 200 loss was 1721.7. Final satisfaction: 16753/114451 tracks, 1801421/4118215 points. The same `tracks.py:1782` non-writable-array warning appeared and did not fail the run. The log says `fiber-patch normals: lasagna at scale 4`.

## Wall time and peak memory

Timed with `resource.getrusage(RUSAGE_CHILDREN).ru_maxrss` (Linux, kilobytes) on the child process. `/usr/bin/time` is not installed; an earlier `time -v` wrapper exited 127 before the ingest started.

| Step | Wall time | Peak RSS |
| --- | --- | --- |
| Ingest, 114451 tracks, z `[10400, 10500)` | 224.5 s | 90364 kB (88 MiB) |
| Fit, 40-step probe, same window, CPU | 438.2 s, exit 0, checkpoint written | 4013292 kB (3.83 GiB) |
| Fit, 300 steps, same window, CPU, module rebuilt against this checkout | 235.1 s, exit 0, checkpoint written | 3957716 kB (3.77 GiB) |
| Full-catalog ingest, stopped at 1180192 committed tracks | 3055 s of progress logged; process then gone | 649976 kB observed peak |

The 200-track probe was 0.322 s and 25232 kB. The input report, which also builds the crossing CSR, was 54.7 s and 1966712 kB (1.88 GiB). The URL-identity pack above was 2.9 s and 130868 kB.

## Full-catalog ingest, stopped

A second ingest, of the whole packed directory (12256170 tracks) into `$DATA/scroll.vcobj`, was running beside the window store. It was terminated: the ingest process and its wrapper were gone, and the log has no `INGEST_EXIT` line. The last progress line is:

```text
Ingest track: ingesting tracks [##----------------------] 9.6% 1180113/12256170 ETA 28672s elapsed 3055s — 889058235617
```

After the process was gone, `tracks` and `store_commits` each count 1180192. The store file is 6685097984 bytes, with a WAL still beside it. The highest RSS sampled while it ran was 649976 kB. `vc_object_ingest --help` has no resume switch (surfaces and tracks commit individually, and a repeated import of an existing identity is an error). The database was mid-write, so this ingest was not started over. The 100-voxel window inside z 10000-11000 that was actually fit is `[10400, 10500)` on the finished window store. The preferred `[10240, 10340)` window sits in scale-4 chunk row 80 (`z // (32 * 4)` with `.zarray` chunks 32 and `.zattrs` scale 4.0) and was not read from the partial full store.

## Findings

Ranked by how many people following this README path would hit them. The Clang failure is last because `g++` only warns and the ingest binary can still link; the others stop a fresh run of the documented commands. The first 300-step attempt exited 1 in 7 s because the virtualenv still had an `e413746da` extension. The rerun after rebuilding against this checkout exited 0. No code fix.

| Rank | What stops | Where | Proposed fix |
| --- | --- | --- | --- |
| 1 | Fresh configure has no package list, and this image's Clang cannot link libstdc++ | `vc_object_store/README.md:125` | Name the apt/brew packages beside the cmake snippet, including a libstdc++ the selected `c++` can link |
| 2 | `convert_track_store.py` refuses a mirror that is already there | `spiral-fitting/tracks.py:410` | If the packed mirror exists and its signature matches, skip it; document `--force` |
| 3 | Ingest of a downloaded DBM rejects the published mirror | `vc_object_store/src/PackedTrackStore.cpp:52` | Accept a whole-second mtime the way `tracks.py` already does, or document passing the `.vctracks` directory |
| 4 | No z window, and one SQLite commit per track | `GeometryIngest.cpp:261`, `object_store_inputs.py:1955` | Filter ingest and the crossing kernel to a z window, and commit tracks in batches |
| 5 | Documented `cmake --build` fails on Clang | `test_object_store_hosted_cache.cpp:605` | Use int coordinates in those two fixtures (fixed on this branch) |

### 1. The documented build does not say what to install

Command:

```sh
cmake -S vc_object_store -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DVC_OBJECT_STORE_BUILD_ADAPTERS=ON \
  -DVC_OBJECT_STORE_BUILD_LINKING=OFF -DVC_OBJECT_STORE_BUILD_TESTS=ON
```

Verbatim, before any packages were installed:

```text
CMake Error: CMake was unable to find a build program corresponding to "Ninja".  CMAKE_MAKE_PROGRAM is not set.  You probably need to select a different build tool.
CMake Error: CMAKE_C_COMPILER not set, after EnableLanguage
CMake Error: CMAKE_CXX_COMPILER not set, after EnableLanguage
```

After Ninja and the libraries above, the Clang link test failed with `/usr/bin/ld: cannot find -lstdc++`. `g++` linked the same program. Installing `libstdc++-14-dev` let `/usr/bin/c++` link, and configure then succeeded. From the first failure to that configure was about 4 minutes (the gap is the missing documentation, not the compile). `vc_object_store/README.md:125-128` is the command. Upstream search of issues and pull requests, open and closed, found no request to list these packages.

### 2. The README's convert step fails when the mirror shipped

Command: `python convert_track_store.py "$DATA/convert-exists/tiny.dbm"` with a real tiny DBM and an existing `.vctracks` directory. The public PHerc0358 listing already contains that directory. The README (`spiral-fitting/README.md:26-27` and `:341-342`) says to run convert, then ingest the DBM.

Verbatim:

```text
File "spiral-fitting/tracks.py", line 410, in write_packed_track_store
    raise FileExistsError(destination)
FileExistsError: tiny.dbm.vctracks
```

`spiral-fitting/tracks.py:409-410`. No matching issue or pull request. Proposed fix: if the destination exists and its signature matches the DBM, return it; keep `--force` for a rebuild.

### 3. A downloaded DBM is always "stale" to `vc_object_ingest`

Command: the `vc_object_ingest --tracks DBM` invocation above, with the file size equal to the published size and the mtime equal to the HTTP `Last-Modified` second. The published `source_db_signature` stores a nanosecond mtime (`1785310500096948487`). HTTP `Last-Modified` cannot preserve that.

Verbatim:

```text
0 ingested, 1 problems
0 ingested, 0 already in the store, 1 problems
packed tracks are stale for <dbm>; rebuild with Spiral's convert_track_store.py --force
```

`vc_object_store/src/PackedTrackStore.cpp:52-55`. Passing the `.vctracks` directory skips the check (line 31) and is how the window was ingested. No issue or pull request reports the C++ check. Proposed fix: treat a whole-second mtime as a match, which `spiral-fitting/tracks.py` `_db_signatures_match` already does, or make the README say to pass the directory.

### 4. A short window still means the whole track database

`vc_object_ingest` and `convert_track_store.py` have no z bounds. Each track is its own commit:

```cpp
auto one = [&](Prepared p) {
    std::vector<Prepared> batch;
    batch.push_back(std::move(p));
    commitBatch(target, std::move(batch), report, coordinateContext);
};
```

`vc_object_store/src/geometry/GeometryIngest.cpp:261-264`, called from the track loop at lines 292-309. The public set is 12256170 tracks. At the measured 511 tracks/s and about 5.6 kB of store per track, the full ingest is several hours and on the order of 70 GB. After `#2564` the fitter computes crossings from every selected track, including tracks outside the z window (`spiral-fitting/object_store_inputs.py:1955-1957`). The README's `--z-min/--z-max` applies only to `build_track_crossings.py`, which the fit no longer reads. No issue or pull request asks for a z window on ingest. Proposed fix: accept a z window on ingest and run the crossing kernel on that subset, and commit many tracks per transaction. The 114451-track ingest above is that subset, built outside the tools.

### 5. Clang rejects the hosted-cache track fixtures

Command: `cmake --build "$BUILD" -j 4` with `/usr/bin/c++` = Clang 18. `g++` 13 and `g++-14` only warn (`-Wnarrowing`, exit 0). Clang exits 1.

Verbatim:

```text
vc_object_store/tests/cpp/test_object_store_hosted_cache.cpp:605:57: error: type 'double' cannot be narrowed to 'int' in initializer list [-Wc++11-narrowing]
            vc::geometry::Track track; track.points = {{double(seed), double(i), 1}, {double(seed), double(i), 2}};
vc_object_store/tests/cpp/test_object_store_hosted_cache.cpp:637:57: error: type 'double' cannot be narrowed to 'int' in initializer list [-Wc++11-narrowing]
            vc::geometry::Track track; track.points = {{double(i), 1, 2}, {double(i), 3, 4}, {double(i), 5, 6}};
```

`TrackPoint` is `std::array<int32_t, 3>` (`vc_object_store/include/vc_object_store/geometry/Geometry.hpp:15`). The same file already builds other tracks with integer literals. No open or closed issue or pull request names this test. Pull request `#571` fixed other C++11 narrowing errors years ago and does not touch this file. macOS `c++` is Clang, so the documented build fails there too.

Fixed on this branch: both lists use ints. Rebuilding that target exited 0, and the test printed `object_store_hosted_cache: ok`.

## Known upstream

- [PR 2020](https://github.com/ScrollPrize/villa/pull/2020), merged 2026-10-21, `spiral-fitting: accept shipped track caches after download (mtime precision)`. The Python loader compares mtime at whole-second resolution when one side has no sub-second part. The C++ check in finding 3 does not. Still present on `f29334cd8`.
- [PR 2564](https://github.com/ScrollPrize/villa/pull/2564), merged 2026-11-16, `Spiral: compute track crossings in the fitter instead of reading them from the store`. On current main, and it is the code this fit ran. It removes the store's track-track refresh from the fit. It does not add a z window; crossings are still computed for every selected track. Finding 4 stands.
- [PR 571](https://github.com/ScrollPrize/villa/pull/571), closed, older C++11 narrowing fixes elsewhere. Not this test.
- [Issue 2320](https://github.com/ScrollPrize/villa/issues/2320), open, `vc_object_ingest` refuses published fiber annotations. Not on the tracks-and-Lasagna path run here.

Searches that returned nothing relevant, open and closed: `source_db_signature`, `packed tracks are stale`, `convert_track_store` together with an existing mirror, and a z window on `vc_object_ingest`.
