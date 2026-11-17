# Parallel simulation experiments

These probes run independently of the production solver. Timings represent different workloads and should be compared within each probe. Example measurements below used a Ryzen 9 8945HX and a GeForce RTX 5070 Ti Laptop GPU.

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

All thread counts produced identical hashes for each fixture. For the dense fixture, the worker phase fell from 11.1 to 3.9 ms on average while serial regrouping cost 12.0 ms on eight threads. Boundary resolution took 0.23 ms on average. Keeping tile membership across sweeps is a candidate for the next iteration.

## GPU block cellular automaton

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
