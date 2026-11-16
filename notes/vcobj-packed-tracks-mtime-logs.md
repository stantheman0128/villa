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

# Real files

`$DATA` is a fresh directory outside the repository. Free space before the download was 206 GB of 254 GB, above the 20 GB stop line. The DBM download succeeded on the first attempt. No stop rule was hit.

Upstream `main`, fetched before the run, is `74cba7a61f5e927c01b8cc0085434ea307d841aa`. `$MAIN_BUILD` is the existing g++ build of that commit. Its `PackedTrackStore.cpp` does not contain `samePackedDbSignature`. `$FIX_BUILD` is the existing g++ build of `a37c981bff20113555ee4d2a0d6ac42a9c4f1e84`. Neither binary was rebuilt.

The published tracks listing is `https://dl.ash2txt.org/datasets/spiral_datasets/PHerc0358/20250821151737/tracks/`. The mirror beside the DBM is `PHerc0358_20250821151737_surface_m7_L0_th0.2.dbm.vctracks` and contains every name in that mirror's listing. Sizes below are the published byte sizes. `coordinates.i32`, `family_codes.i8`, `header.bin`, `metadata.json`, `offsets.i64`, `source_ids.u64`, and `z_bounds.i32` were hardlinked from an existing download after each published size matched that file byte for byte. `arclengths.f64` and `tortuosities.f64` were not in that download and were fetched with wget. The mirror was not regenerated.

```
command: cd $DATA && wget --tries=1 --timeout=180 --read-timeout=180 --progress=dot:giga https://dl.ash2txt.org/datasets/spiral_datasets/PHerc0358/20250821151737/tracks/PHerc0358_20250821151737_surface_m7_L0_th0.2.dbm
wget exit: 0
saved: 5977288704/5977288704
command: cd $DATA/PHerc0358_20250821151737_surface_m7_L0_th0.2.dbm.vctracks && wget --tries=1 --timeout=120 --read-timeout=120 --progress=dot:mega https://dl.ash2txt.org/datasets/spiral_datasets/PHerc0358/20250821151737/tracks/PHerc0358_20250821151737_surface_m7_L0_th0.2.dbm.vctracks/arclengths.f64 https://dl.ash2txt.org/datasets/spiral_datasets/PHerc0358/20250821151737/tracks/PHerc0358_20250821151737_surface_m7_L0_th0.2.dbm.vctracks/tortuosities.f64
wget exit: 0
saved: arclengths.f64 98049360/98049360, tortuosities.f64 98049360/98049360
```

DBM `stat`:

```
size: 5977288704
mtime_ns: 1785310500000000000
blocks: 11674400
modify: 2026-07-29 07:35:00.000000000 +0000
```

`blocks` is the `stat` allocated-block count (512-byte units) and is non-zero. The file is fully allocated.

Mirror file list:

```
arclengths.f64 98049360
coordinates.i32 5637581208
family_codes.i8 12256170
header.bin 64
metadata.json 219
offsets.i64 98049368
source_ids.u64 98049360
tortuosities.f64 98049360
z_bounds.i32 98049360
```

## Before: real DBM, upstream main

```
command: $MAIN_BUILD/bin/vc_object_ingest --db $DATA/mtime-ingest-before.vcobj --author ubuntu --volume-uuid 20250821151737 --scale-factor 1 --tracks $DATA/PHerc0358_20250821151737_surface_m7_L0_th0.2.dbm
source: upstream/main 74cba7a61f5e927c01b8cc0085434ea307d841aa
compiler: g++
dbm size: 5977288704
dbm mtime_ns: 1785310500000000000
dbm blocks: 11674400
adjacent vctracks: $DATA/PHerc0358_20250821151737_surface_m7_L0_th0.2.dbm.vctracks
wall_sec: 0.086841
exit: 1
stdout:
0 ingested, 1 problems
0 ingested, 0 already in the store, 1 problems
stderr:
Ingest track: starting | elapsed 0s
Ingest track: starting / elapsed 0s (finished)
$DATA/PHerc0358_20250821151737_surface_m7_L0_th0.2.dbm: packed tracks are stale for $DATA/PHerc0358_20250821151737_surface_m7_L0_th0.2.dbm; rebuild with Spiral's convert_track_store.py --force
```

## After: real DBM, fixed binary, stopped past 100000 tracks

The same command, a new empty store, and `$FIX_BUILD`. Stderr was not a terminal, so progress is the newline-delimited log. The process was stopped with SIGTERM to its PID after the log had passed 100000 tracks. It was not run through all 12256170 tracks. Stdout is empty because the process was stopped before the summary. `exit: 143` is that signal, not an ingest error.

```
command: $FIX_BUILD/bin/vc_object_ingest --db $DATA/mtime-ingest-after.vcobj --author ubuntu --volume-uuid 20250821151737 --scale-factor 1 --tracks $DATA/PHerc0358_20250821151737_surface_m7_L0_th0.2.dbm
source: a37c981bff20113555ee4d2a0d6ac42a9c4f1e84
compiler: g++
dbm size: 5977288704
dbm mtime_ns: 1785310500000000000
dbm blocks: 11674400
adjacent vctracks: $DATA/PHerc0358_20250821151737_surface_m7_L0_th0.2.dbm.vctracks
wall_sec: 241.822751
exit: 143
stdout:

stderr:
Ingest track: starting | elapsed 0s
Ingest track: ingesting tracks [------------------------] 0.0% 2683/12256170 ETA 21541s elapsed 5s — 2683
Ingest track: ingesting tracks [------------------------] 0.0% 5366/12256170 ETA 22184s elapsed 10s — 4294969028
Ingest track: ingesting tracks [------------------------] 0.0% 5366/12256170 ETA 22184s elapsed 10s — 4294969028
Ingest track: ingesting tracks [------------------------] 0.1% 7926/12256170 ETA 22743s elapsed 15s — 8589934992
Ingest track: ingesting tracks [------------------------] 0.1% 10507/12256170 ETA 22980s elapsed 20s — 8589937573
Ingest track: ingesting tracks [------------------------] 0.1% 13016/12256170 ETA 23250s elapsed 25s — 12884903203
Ingest track: ingesting tracks [------------------------] 0.1% 15611/12256170 ETA 23301s elapsed 30s — 12884905798
Ingest track: ingesting tracks [------------------------] 0.1% 18084/12256170 ETA 23495s elapsed 35s — 17179871008
Ingest track: ingesting tracks [------------------------] 0.2% 20476/12256170 ETA 23734s elapsed 40s — 17179873400
Ingest track: ingesting tracks [------------------------] 0.2% 22920/12256170 ETA 23868s elapsed 45s — 21474838382
Ingest track: ingesting tracks [------------------------] 0.2% 25427/12256170 ETA 23915s elapsed 50s — 21474840889
Ingest track: ingesting tracks [------------------------] 0.2% 27906/12256170 ETA 23977s elapsed 55s — 25769805945
Ingest track: ingesting tracks [------------------------] 0.2% 30364/12256170 ETA 24045s elapsed 60s — 30064771576
Ingest track: ingesting tracks [------------------------] 0.3% 32888/12256170 ETA 24054s elapsed 65s — 30064774100
Ingest track: ingesting tracks [------------------------] 0.3% 35384/12256170 ETA 24079s elapsed 70s — 34359739227
Ingest track: ingesting tracks [------------------------] 0.3% 37671/12256170 ETA 24235s elapsed 75s — 34359741514
Ingest track: ingesting tracks [------------------------] 0.3% 37671/12256170 ETA 24235s elapsed 75s — 34359741514
Ingest track: ingesting tracks [------------------------] 0.3% 40108/12256170 ETA 24281s elapsed 80s — 38654706953
Ingest track: ingesting tracks [------------------------] 0.3% 42376/12256170 ETA 24418s elapsed 85s — 38654709221
Ingest track: ingesting tracks [------------------------] 0.3% 42376/12256170 ETA 24418s elapsed 85s — 38654709221
Ingest track: ingesting tracks [------------------------] 0.4% 44645/12256170 ETA 24541s elapsed 90s — 42949674175
Ingest track: ingesting tracks [------------------------] 0.4% 46835/12256170 ETA 24692s elapsed 95s — 42949676365
Ingest track: ingesting tracks [------------------------] 0.4% 49178/12256170 ETA 24753s elapsed 100s — 47244641743
Ingest track: ingesting tracks [------------------------] 0.4% 51495/12256170 ETA 24819s elapsed 105s — 47244644060
Ingest track: ingesting tracks [------------------------] 0.4% 53792/12256170 ETA 24889s elapsed 110s — 51539609653
Ingest track: ingesting tracks [------------------------] 0.4% 53792/12256170 ETA 24889s elapsed 110s — 51539609653
Ingest track: ingesting tracks [------------------------] 0.5% 56198/12256170 ETA 24905s elapsed 115s — 55834575656
Ingest track: ingesting tracks [------------------------] 0.5% 58567/12256170 ETA 24934s elapsed 120s — 55834578025
Ingest track: ingesting tracks [------------------------] 0.5% 60864/12256170 ETA 24990s elapsed 125s — 60129543636
Ingest track: ingesting tracks [------------------------] 0.5% 63209/12256170 ETA 25023s elapsed 130s — 60129545981
Ingest track: ingesting tracks [------------------------] 0.5% 65515/12256170 ETA 25068s elapsed 135s — 64424511421
Ingest track: ingesting tracks [------------------------] 0.6% 67797/12256170 ETA 25119s elapsed 140s — 64424513703
Ingest track: ingesting tracks [------------------------] 0.6% 70124/12256170 ETA 25149s elapsed 145s — 68719478873
Ingest track: ingesting tracks [------------------------] 0.6% 72536/12256170 ETA 25148s elapsed 150s — 73014444163
Ingest track: ingesting tracks [------------------------] 0.6% 74994/12256170 ETA 25131s elapsed 155s — 73014446621
Ingest track: ingesting tracks [------------------------] 0.6% 77383/12256170 ETA 25138s elapsed 160s — 77309411772
Ingest track: ingesting tracks [------------------------] 0.7% 79700/12256170 ETA 25166s elapsed 165s — 77309414089
Ingest track: ingesting tracks [------------------------] 0.7% 82083/12256170 ETA 25172s elapsed 170s — 81604379255
Ingest track: ingesting tracks [------------------------] 0.7% 84412/12256170 ETA 25194s elapsed 175s — 81604381584
Ingest track: ingesting tracks [------------------------] 0.7% 86755/12256170 ETA 25210s elapsed 180s — 85899347069
Ingest track: ingesting tracks [------------------------] 0.7% 89168/12256170 ETA 25205s elapsed 185s — 85899349482
Ingest track: ingesting tracks [------------------------] 0.7% 91556/12256170 ETA 25208s elapsed 190s — 90194314550
Ingest track: ingesting tracks [------------------------] 0.8% 93928/12256170 ETA 25214s elapsed 195s — 90194316922
Ingest track: ingesting tracks [------------------------] 0.8% 96208/12256170 ETA 25243s elapsed 200s — 94489282006
Ingest track: ingesting tracks [------------------------] 0.8% 98568/12256170 ETA 25251s elapsed 205s — 94489284366
Ingest track: ingesting tracks [------------------------] 0.8% 100924/12256170 ETA 25259s elapsed 210s — 98784249642
Ingest track: ingesting tracks [------------------------] 0.8% 103293/12256170 ETA 25263s elapsed 215s — 103079215354
Ingest track: ingesting tracks [------------------------] 0.8% 103293/12256170 ETA 25263s elapsed 215s — 103079215354
Ingest track: ingesting tracks [------------------------] 0.9% 105684/12256170 ETA 25262s elapsed 220s — 103079217745
Ingest track: ingesting tracks [------------------------] 0.9% 108081/12256170 ETA 25259s elapsed 225s — 107374183615
Ingest track: ingesting tracks [------------------------] 0.9% 110487/12256170 ETA 25253s elapsed 230s — 107374186021
Ingest track: ingesting tracks [------------------------] 0.9% 112904/12256170 ETA 25246s elapsed 235s — 111669151638
Ingest track: ingesting tracks [------------------------] 0.9% 115256/12256170 ETA 25252s elapsed 240s — 111669153990
```

The last progress line is `Ingest track: ingesting tracks [------------------------] 0.9% 115256/12256170 ETA 25252s elapsed 240s — 111669153990`. The line at elapsed 210s had already shown `100924/12256170`.

```
command: sqlite3 $DATA/mtime-ingest-after.vcobj "SELECT count(*) FROM tracks;"
tracks: 116130
command: sqlite3 $DATA/mtime-ingest-after.vcobj "SELECT count(*) FROM objects WHERE kind='track';"
objects: 116130
```

The stored count is ahead of the last printed `115256` because the progress line is throttled and commits continued until the signal arrived.
