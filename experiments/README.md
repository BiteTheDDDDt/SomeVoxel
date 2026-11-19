# Parallel simulation experiments

These probes run independently of the production solver. Measurements below used a Ryzen 9 8945HX and a GeForce RTX 5070 Ti Laptop GPU.

[Follow-up experiments on rate, CPU reconstruction and numeric precision](performance_precision.md)

### Physical-unit replay

Build `somevoxel_rate_bench` with `-DSOMEVOXEL_BUILD_EXPERIMENTS=ON` and the grid and tick-rate settings to compare. Run `launch-budget` to check the selected loose-sand mass against the 131.22 kg reference with a full supply. Run `wall-sweep` to check that a shot that clears the wall plane in one tick without an obstacle still stops at the face of a one-cell wall. An acceptance check exits with a nonzero status on failure.

`mixed` replays five simulated seconds of reservoir flow and a charged shot. The report gives the mass at release in kilograms, the impulse of the fastest qualifying sand contact in N·s and N·s per square meter of the contact face, and stone fracture volume in liters. As an estimate of the impulse applied at release, `cast_impulse_ns` multiplies released mass by the mass-weighted velocity increment. Use `water`, `water-mirrored`, `water-narrow`, and `water-narrow-mirrored` to compare tank outlets. `water_net_flow_l_s` is the final water volume beyond the fixed outlet plane divided by elapsed simulated seconds. It measures net discharge over the test window.

`paged_material_grid.py` checks initial occupancy on CUDA for fixed scene
bounds. It accepts JSON lines with a `layout` identifier and flat
`boxes` records in `[x0,y0,z0,x1,y1,z1,kind]` order. Uniform static pages
use a single directory word. Mixed pages and pages containing a dynamic
material have storage. The probe checks material lookup against the
original boxes. Its memory figures cover initial occupancy only.

[CUDA material movement and cross-page updates](paged_material_motion.md)

[Exact Vulkan water heads from immutable packed CPU occupancy](gpu_water_heads/README.md)

## Matched sand fixture

Sand starts above a floor and beside an internal wall. A local spring acceleration field pulls it toward a moving point for 0.4 seconds. The field then switches off for another 0.8 seconds. The 3 cm fixtures use the same initial grain positions across solvers. Timings are P95 per **1/60 second of simulated time**, excluding the first eight frames. CUDA timings include submission and a completion fence. Rendering, setup and validation are outside the timings. Contact order, material behavior and velocity limits differ between models.

| 3 cm fixture | 109,200 grains | 1,055,832 grains |
| --- | ---: | ---: |
| CPU tiles, regroup every sweep, 8 threads | 5.8–5.9 ms | 46.3–47.3 ms |
| CPU tiles, persistent membership, 8 threads | 2.2–2.3 ms | 17.7–17.9 ms |
| CPU tiles, persistent membership, 16 threads | 1.8–1.9 ms | 15.3–16.2 ms |
| CUDA lattice, impulses, friction and slip | 0.39 ms | 3.6–4.3 ms |
| CUDA APIC/MPM, 6 cm transfer grid, 2 substeps | 1.1 ms | 21–22 ms |
| CUDA APIC/MPM, 6 cm transfer grid, 4 substeps | 3.5 ms | 42 ms |
| CUDA APIC/MPM, 6 cm transfer grid, 8 substeps | 4.0 ms | 82 ms |

### CPU tile migration

The persistent mode keeps tile membership across sweeps. After the sorted boundary pass, it removes departing IDs and merges sorted arrivals in affected tiles. Retaining the same ID order preserves the collision order. Two runs at each of 1, 4, 8 and 16 threads produced identical final hashes in both modes for each fixture and particle count. The hash includes grain IDs, occupancy, velocity and subcell travel. The mean time spent on grouping or migration in the dense field test at 8 threads fell from 27.5–29.6 ms to 1.3–1.4 ms per frame.

```sh
somevoxel_parallel_tiles 8 dense regroup pull
somevoxel_parallel_tiles 8 dense persistent pull
```

### CUDA lattice with contact

The lattice keeps velocity and subcell travel for every grain. Disjoint pairs transfer equal-mass momentum with a Coulomb-limited tangential impulse. Downward-moving grains blocked by a neighbor can propose a slip into a clear downhill diagonal. Repeated runs with and without per-frame readback produced the same final hash. Checks covered unique grain IDs, occupancy, static floor and wall, and cleared diagonal reservations.

- The isolated 2.4 m/s impact matched its free-flight position within 2.1 mm at the first checkpoint and rebounded at −0.60 m/s.
- The oblique two-grain test yielded normal velocities of 1.20 and 1.20 m/s, and tangential velocities of 0.66 and 0.54 m/s, matching the equal-mass and friction calculation.
- A passive 12 × 44 × 12-cell column spread to a central span of 40 × 6 × 41 cells in two seconds, measured using the 5th and 95th percentiles per axis. Including ejecta, its bounds were 85 × 9 × 79 cells.
- In the sparse fixture, attraction moved the mean X position 0.55 m farther than the gravity-only control. Mean Z velocity was 0.70 m/s at release and 0.69 m/s 0.2 seconds later. The gravity-only control had zero wall collisions.
- The 12 m/s isolated impact rebounded at −3 m/s with eight substeps. Advancing the full dense fixture at eight substeps measured 14.4 ms P95. The two-substep speed cap is 3.6 m/s per axis.
- Reading back all 1,055,832 positions every frame raised the combined P95 to 7.2 ms. In one instrumented frame, paired contact and transport took 2.52 of the 3.58 ms across the CUDA kernels. Fusing contact and transport removed six dispatches per substep, with overlapping dense timings across both schedules.

| CUDA lattice with contact, friction and slip | Pitch | Substeps | P95 |
| --- | ---: | ---: | ---: |
| Smaller block at finer pitch | 1.5 cm | 4 | 6.5 ms at 873,600 grains |
| Dense block at finer pitch | 1.5 cm | 4 | 78.8 ms at 8,446,656 grains |
| Smaller block at finer pitch | 1 cm | 6 | 30.3 ms at 2,948,400 grains |
| Local 1.0 × 0.6 × 1.4 m block | 1 cm | 6 | 10.1–11.1 ms at 840,000 grains |

The fine-pitch tests retain the same 3.6 m/s per-axis cap by increasing the substep count. Occupancy and slip reservations are dense buffers. The prototype covers single-resolution sand and static bounds.

```sh
python -m pip install warp-lang==1.19.0
python experiments/matched_cuda.py lattice dense --contact --slip --fused --friction 0.45
python experiments/matched_cuda.py lattice dense --contact --slip --fused --friction 0.45 --readback
python experiments/matched_cuda.py lattice local --scale 3 --substeps 6 --contact --slip --fused --friction 0.45
```

Use `--passive` to disable attraction and `--profile` to time individual kernels for an additional frame outside the benchmark interval.

### CUDA APIC/MPM

This solver uses quadratic APIC transfers, compressive pressure, scalar irreversible compaction and projected static collisions. Positions stayed finite, grid mass matched particle mass, and isolated rebound passed. The material model has no shear yield. At four substeps, changing the transfer grid pitch from 6 cm to 3 cm or 12 cm measured 37.6 ms and 42.2 ms P95. In an instrumented two-substep frame, particle-to-grid transfer took 18.32 of 18.88 ms across the measured stages. The direct scatter updates mass and momentum at 27 grid points for every particle.

```sh
python experiments/matched_cuda.py mpm dense --substeps 2 --profile
python experiments/matched_cuda.py mpm dense --substeps 4 --grid-factor 1
python experiments/matched_cuda.py mpm dense --substeps 8
```

### Engineering decision

Persistent tiles are the next CPU change to validate against the production solver. The CUDA results favor a GPU lattice for dense active granular matter, with active spatial tiles and GPU-resident draw data. Fine pitch needs a spatial budget, since the fully active dense volume exceeds the 60 Hz budget at 1.5 cm. APIC/MPM remains suitable for experiments with a separate material model and a smaller active population.

## Earlier solver-specific fixtures

The following timings use distinct workloads and should be compared within each probe.

## CPU tile ownership

Each sweep groups particles by tile. Workers integrate and commit local moves inside exclusive tiles. Cross-tile attempts are sorted by grain ID before the serial boundary pass. The probe covers gravity, a horizontal field, occupied destinations and wall contacts. It uses one crossing per axis per tick, with inelastic blocking and a 120 Hz timestep.

```sh
cmake -S . -B build/experiments -DSOMEVOXEL_BUILD_EXPERIMENTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/experiments --config Release
```

Run `somevoxel_parallel_tiles <threads> <sparse|dense>` from the build output. Compare the final hash, which covers particle IDs, positions, velocities and subcell travel. Occupancy and wall checks are included.

| Particles | 1 thread P95 | 2 threads | 4 threads | 8 threads |
| ---: | ---: | ---: | ---: | ---: |
| 109,200 | 2.36–3.12 ms | 2.90 ms | 2.47 ms | 2.61 ms |
| 1,055,832 | 28.75–29.25 ms | 24.46 ms | 19.94 ms | 18.61 ms |

All thread counts produced identical hashes for each fixture. For the dense fixture, the worker phase fell from 11.1 to 3.9 ms on average while serial regrouping cost 12.0 ms on eight threads. Boundary resolution took 0.23 ms on average.

## GPU block cellular automaton

The retired SomeAvatar F10 solver and its replay fixtures are archived alongside this experiment in `gpu_ca/`. To replay an old case, import that Godot project and run `godot --path experiments/gpu_ca --script res://gpu_world_smoke.gd`.

The Vulkan compute probe uses four shifted partitions of disjoint 2×2×2 blocks per tick. Its cells carry IDs. Gravity and the field choose local swaps. Momentum, damage and cross-material exchange are outside the probe's model.

```sh
godot --path experiments/gpu_ca --script res://bench.gd -- small
godot --path experiments/gpu_ca --script res://bench.gd -- large
```

Use a rendering backend with compute support. P95 for four dispatches and a GPU fence was 0.28 ms at 163,392 grains and 2.43 ms at 1,307,136 grains. Both runs preserved every ID and the barrier. Timings exclude initialization and validation readback.

## GPU material point method

The Warp probe uses quadratic APIC transfers, compressive pressure, irreversible volume change, gravity and floor friction. Eight substeps advance 1/60 of a simulated second. The probe uses a scalar compaction model. Shear yield, internal walls and gameplay contacts are outside its scope.

```sh
python -m pip install warp-lang==1.19.0
python experiments/gpu_mpm.py --particles 109200 --grid 48 --substeps 8
python experiments/gpu_mpm.py --particles 1055832 --grid 96 --substeps 8
```

The CUDA probe measured 4.28 and 55.17 ms P95 per batch, respectively. Positions remained finite and grid mass differed from particle mass by less than 0.00003% in both measurements. Timings include submission and completion, with JIT compilation and validation readback excluded.

References. [Space subdivision and parallel contact processing](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda) · [Block cellular automata](https://futhark-lang.org/blog/2016-12-04-diving-beet.html) · [MLS-MPM and grid coupling](https://yuanming.taichi.graphics/publication/2018-mlsmpm/mls-mpm-cpic.pdf)
