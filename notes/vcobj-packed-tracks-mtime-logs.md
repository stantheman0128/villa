# Packed-track mtime logs

`cursor/vcobj-packed-tracks-mtime` is `a37c981bff20113555ee4d2a0d6ac42a9c4f1e84`, one commit on `74cba7a61f5e927c01b8cc0085434ea307d841aa`.

```
 vc_object_store/src/PackedTrackStore.cpp           |  49 ++++++++-
 vc_object_store/tests/cpp/CMakeLists.txt           |   5 +
 .../tests/cpp/test_packed_track_store.cpp          | 115 +++++++++++++++++++++
 3 files changed, 168 insertions(+), 1 deletion(-)
```

`$MAIN_BUILD` is a build of upstream `main` at `74cba7a61`. `$FIX_BUILD` is a build of `a37c981b`. `$DATA` is a downloads directory outside the repository. `<temp>` is a throwaway directory created by the test.

The 5.98 GB DBM was not on disk. The stand-in was `$DATA/PHerc0358_20250821151737_surface_m7_L0_th0.2.dbm`: size `5977288704`, allocated blocks `0`, `mtime_ns` `1785310500000000000`. The adjacent `.vctracks` directory contained only `metadata.json`.

## Before: ingest from upstream main

```
command: $MAIN_BUILD/bin/vc_object_ingest --db $DATA/mtime-ingest-before.vcobj --author ubuntu --volume-uuid 20250821151737 --scale-factor 1 --tracks $DATA/PHerc0358_20250821151737_surface_m7_L0_th0.2.dbm
dbm size: 5977288704
dbm mtime_ns: 1785310500000000000
dbm blocks: 0
adjacent vctracks: $DATA/PHerc0358_20250821151737_surface_m7_L0_th0.2.dbm.vctracks
adjacent vctracks entries: metadata.json
exit: 1
stdout:
0 ingested, 1 problems
0 ingested, 0 already in the store, 1 problems
stderr:
Ingest track: starting | elapsed 0s
Ingest track: starting / elapsed 0s (finished)
$DATA/PHerc0358_20250821151737_surface_m7_L0_th0.2.dbm: packed tracks are stale for $DATA/PHerc0358_20250821151737_surface_m7_L0_th0.2.dbm; rebuild with Spiral's convert_track_store.py --force
```

## After: ingest from the fixed binary

```
command: $FIX_BUILD/bin/vc_object_ingest --db $DATA/mtime-ingest-after.vcobj --author ubuntu --volume-uuid 20250821151737 --scale-factor 1 --tracks $DATA/PHerc0358_20250821151737_surface_m7_L0_th0.2.dbm
dbm size: 5977288704
dbm mtime_ns: 1785310500000000000
dbm blocks: 0
adjacent vctracks: $DATA/PHerc0358_20250821151737_surface_m7_L0_th0.2.dbm.vctracks
adjacent vctracks entries: metadata.json
exit: 1
stdout:
0 ingested, 1 problems
0 ingested, 0 already in the store, 1 problems
stderr:
Ingest track: starting | elapsed 0s
Ingest track: starting / elapsed 0s (finished)
$DATA/PHerc0358_20250821151737_surface_m7_L0_th0.2.dbm: invalid packed track header: $DATA/PHerc0358_20250821151737_surface_m7_L0_th0.2.dbm.vctracks
```

## Test before the fix

```
compiler: g++
build: $MAIN_BUILD
source: upstream/main 74cba7a61f5e927c01b8cc0085434ea307d841aa (PackedTrackStore.cpp without the whole-second fix; test source is the amended test_packed_track_store.cpp)
command: $MAIN_BUILD/bin/test_packed_track_store
exit: 1
stdout:

stderr:
whole-second copy: packed tracks are stale for <temp>/tracks.dbm; rebuild with Spiral's convert_track_store.py --force
command: ctest --test-dir $MAIN_BUILD --output-on-failure -R '^object_store_packed_track_store$'
exit: 8
stdout:
Internal ctest changing into directory: $MAIN_BUILD
Test project $MAIN_BUILD
    Start 25: object_store_packed_track_store
1/1 Test #25: object_store_packed_track_store ...***Failed    0.00 sec
whole-second copy: packed tracks are stale for <temp>/tracks.dbm; rebuild with Spiral's convert_track_store.py --force


0% tests passed, 1 tests failed out of 1

Total Test time (real) =   0.01 sec

The following tests FAILED:
	 25 - object_store_packed_track_store (Failed)
stderr:
Errors while running CTest
```

## Test after the fix

```
compiler: g++
build: $FIX_BUILD
source: cursor/vcobj-packed-tracks-mtime (amended commit)
command: $FIX_BUILD/bin/test_packed_track_store
exit: 0
stdout:
packed track signature tests passed
stderr:

command: $FIX_BUILD/bin/test_packed_store
exit: 0
stdout:
packed VFS tests passed
stderr:

command: ctest --test-dir $FIX_BUILD --output-on-failure -R '^object_store_packed(_track_store)?$'
exit: 0
stdout:
Test project $FIX_BUILD
    Start 24: object_store_packed
1/2 Test #24: object_store_packed ...............   Passed    0.67 sec
    Start 25: object_store_packed_track_store
2/2 Test #25: object_store_packed_track_store ...   Passed    0.00 sec

100% tests passed, 0 tests failed out of 2

Total Test time (real) =   0.67 sec
stderr:
```
