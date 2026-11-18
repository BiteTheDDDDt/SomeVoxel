# CPU grain state precision

The source snapshot comes from SomeVoxel 21b1418. The driver uses the existing force field, sequential face sweep, granular contact, water pressure and terrain fracture. The 3 cm scene has 108,864 grains. The mixed fixture adds 296,352 water cells. Every batch advances 1/60 second at three 180 Hz steps. Two runs per implementation use alternating order. The first 16 frames are excluded.

| 3 cm fixture | FP32 state | FP16 travel only | FP16 travel and velocity |
| --- | ---: | ---: | ---: |
| Mixed P95 | 52.5–55.2 ms | 54.9–55.8 ms | 53.9–54.8 ms |
| Dry P95 | 43.4–48.2 ms | 44.1–45.1 ms | 42.4–45.6 ms |
| Passive P95 | 10.8–11.0 ms | 10.9–12.7 ms | 11.4–11.5 ms |
| Bytes per grain | 40 | 36 | 28 |
| Allocated fine sand MiB | 5.0 | 4.5 | 3.5 |
| Peak wall impulse (N·s) | 1.358 | 1.340 | 1.366 |
| Fractured wall cells | 64 | 57 | 57 |
| Escaped water cells | 3,726 | 3,726 | 3,727 |

CPU was AMD Ryzen 9 8945HX. Clang 23 compiled all three with the same flags including AVX2 and F16C. Travel is decompressed to FP32 once inside the face sweep and compressed when the sweep returns. The third variant saves velocity components as FP16 and uses FP32 calculations with quantization at storage boundaries. Earlier and later runs produced identical positions within each implementation. Final sand centroid differed by at most 2.4 mm across storage modes. The 12 m/s isolated packed projectile met a one-cell travel tolerance and vertical velocity error of at most 0.012 m/s. All scenes preserved sand mass and water volume.

Half storage is a footprint experiment. Compression saved 1.5 MiB of sand capacity in the combined case while the dense grid and pressure data occupy hundreds of MiB. Simulated-frame latency did not improve reliably and the fracture pattern changed.

## Vector arithmetic

The manual SIMD sweep evaluates a single grain's three velocity and displacement axes together. It returns early when the grain crosses no face and has no retained contact. The sequential face walk and collision response still run as before. A separate build leaves source arithmetic unchanged and targets Zen 4. Both comparisons disable FP contraction.

| 3 cm fixture | Scalar P95 | SIMD axis arithmetic P95 | Zen 4 compiler target P95 |
| --- | ---: | ---: | ---: |
| Mixed | 47.5–50.7 ms | 50.9–52.6 ms | 51.9 ms |
| Dry | 39.0–39.3 ms | 39.4–41.2 ms | 41.1 ms |

Scalar and SIMD ran twice per fixture. The Zen 4 build ran once between them. Grain positions, velocities, sub-cell travel, wall impulse, fracture count and water totals matched across builds in both fixtures. Lane setup and extraction preserved the trajectory but did not improve latency. Wider parallel work across particles or independent regions remains a separate experiment.

## Reproduce

From the repository root, run `python experiments/cpu_grain_precision/prepare.py build/grain-precision` and build inside `build/grain-precision`. The preparation script copies the current core by default. Pass `--source` with the path to the core of a pinned checkout to compare revisions. Use a Clang toolchain and x86 CPU with F16C and AVX2 support.

```sh
clang++ -std=c++17 -O3 -DNDEBUG -DSOMEVOXEL_REFINEMENT=6 -mf16c -mavx2 core/world.cpp bench.cpp -o fp32
clang++ -std=c++17 -O3 -DNDEBUG -DSOMEVOXEL_REFINEMENT=6 -mf16c -mavx2 -DSOMEVOXEL_HALF_TRAVEL core/world.cpp bench.cpp -o half-travel
clang++ -std=c++17 -O3 -DNDEBUG -DSOMEVOXEL_REFINEMENT=6 -mf16c -mavx2 -DSOMEVOXEL_HALF_TRAVEL -DSOMEVOXEL_HALF_VELOCITY core/world.cpp bench.cpp -o half-state
```

Each executable accepts `mixed`, `dry` or `resting`.

To reproduce the vector arithmetic comparison, add `-ffp-contract=off` to the FP32 command. Add `-DSOMEVOXEL_SIMD_SWEEP` for the manual SIMD build. Replace `-mf16c -mavx2` with `-march=znver4` for the host-specific scalar build.
