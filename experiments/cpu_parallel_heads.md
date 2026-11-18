# Exact water-head reconstruction on CPU

This probe compares the production breadth-first scan with water spans on the same 3 cm fixtures. It advances five simulated seconds, excludes the first six 30 Hz input frames, and measures batches of three engine steps (1/60 simulated second). The mixed fixture contains 296,352 water cells, 108,864 sand grains, an outlet and a breakable wall. The water-only fixture uses the same tank and outlet.

Each face-connected water component gets the head of its highest cell. Ordinary reconstruction can compute the same result by joining overlapping X runs. The probe groups runs into full chunk-aligned Z bands, performs interior unions on persistent worker threads, joins adjacent bands in a serial seam pass, then updates heads and wakes chunks in parallel. Pressure-balancing steps still use the original breadth-first scan, including its route and transfer order.

## Results

Ryzen 9 8945HX, Windows, optimized Clang build from revision `fd55ba3`. Times are milliseconds per batch.

| Method | Mixed head mean | Mixed total P95 | Water-only head mean | Water-only total P95 |
| --- | ---: | ---: | ---: | ---: |
| Breadth-first scan | 5.29–5.33 | 50.99–52.93 | 4.35 | 8.19 |
| Spans, 1 worker | 3.19 | 48.43 | 2.64 | 6.99 |
| Spans, 4 workers | 2.85–2.95 | 51.47–51.49 | 2.30 | 6.97 |
| Spans, 8 workers | 2.73 | 48.01 | 2.42 | 7.19 |

In the mixed fixture the span variants processed 375 reconstructions as runs and 75 with the original scan. All variants matched three checkpoint digests per fixture. The digest includes voxel occupancy, water heads, awake chunks, and fine-sand position and motion. All runs preserved mass and discharged 3,726 water cells through the outlet. Each mixed run launched 900 grains and fractured 64 stone cells. The span solver also matched an independent breadth-first reference on 16 reproducible randomized 3D volumes with bridges across band boundaries.

Changing the traversal did most of the work. Increasing from one to eight workers saved another 0.46 ms in mean head time in the mixed fixture. The changes to mixed total P95 were small relative to run-to-run variation. The 31.25 MiB scratch cost mostly comes from the dense row directory. A sparse directory or active-region allocation would lower that cost.

## Reproduce

The build script uses Python 3.12 or later and C++17 threading. It instruments a Git snapshot under the ignored build directory.

`python experiments/cpu_parallel_heads_build.py --revision fd55ba3 --refinement 6 --cxx clang++`

Run `build/cpu-parallel-research/r6/head_bench 0 mixed`, then repeat with `1`, `4` and `8` workers and with the `water` fixture. A worker count of `0` selects the breadth-first baseline. Windows builds with a MinGW compiler can require its runtime DLL directory on `PATH`.
