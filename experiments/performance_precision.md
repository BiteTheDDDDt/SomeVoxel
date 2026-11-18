# Simulation rate and storage precision experiments

The fixtures ran on a Ryzen 9 8945HX and a GeForce RTX 5070 Ti Laptop GPU. Each timing sample advances 1/60 simulated second. CPU timing covers native solver steps. GPU timing covers its lattice probe and completion fence. Both exclude rendering. The GPU fixture uses its own sand contact and slip implementation.

## Update rates at 3 cm

The mixed fixture contains a tank with an outlet and a sand pile that is pulled and released into a wall. It runs five simulated seconds with a 30 Hz input schedule. The simulation rate experiment preserves the physical speed cap by allowing more swept faces per step. It keeps reaction and pressure-balancing periods in seconds and adjusts the per-step ground retention coefficient. Bulk water still advances with the existing local cell transfers on its scheduled ticks.

| Simulation / bulk water | P95 | Water past outlet after 5 s | Peak sand impulse |
| --- | ---: | ---: | ---: |
| 180 / 90 Hz (control) | 53.3–61.9 ms | 100.6 L | 1.358 N·s |
| 180 / 60 Hz | 61.3 ms | 347.3 L | 1.358 N·s |
| 180 / 45 Hz | 48.2 ms | 47.0 L | 1.358 N·s |
| 120 / 60 Hz | 38.8–41.9 ms | 63.1 L | 1.366 N·s |
| 90 / 45 Hz | 39.4 ms | 42.1 L | 1.364 N·s |
| 60 / 30 Hz | 20.7 ms | 19.2 L | 1.376 N·s |

All runs retained sand, water and stone inventories and recorded sand-wall impacts. At 120/60 Hz, the position of the held group at release moved about 4.2 mm relative to the control and peak impact impulse changed by 0.6%. Water discharge changed by 37%. Changing only the bulk-water rate at 180 Hz left the measured sand outcomes identical. Discharge changed substantially, with a nonmonotonic response. Step-rate tuning for bulk water requires a transfer model calibrated in physical time and validation across different outlet geometries. The 2.25 cm fixture at 120/60 Hz still took 108.1 ms P95.

```sh
cmake -S . -B build/rate120 -DCMAKE_BUILD_TYPE=Release -DSOMEVOXEL_BUILD_EXPERIMENTS=ON -DSOMEVOXEL_REFINEMENT=6 -DSOMEVOXEL_EXPERIMENTAL_SIMULATION_HZ=120 -DSOMEVOXEL_EXPERIMENTAL_WATER_HZ=60
cmake --build build/rate120 --config Release --target somevoxel_rate_bench
```

Run `somevoxel_rate_bench` with `mixed`, `sand`, `water` or `water-mirrored` from its build output. Set the rate overrides to 180 and 90 for the control. Overrides are opt-in.

### Sweep orientation and water

The water-only fixture can be reflected along X without changing the tank volume, outlet dimensions or gravity. At 180/90 Hz, the default water ticks all have an even simulation tick index, so the shared sweep always scans X in the same direction on those ticks. The experimental sweep phase uses the ordinal of the bulk-water tick on those sweeps, rotating both horizontal directions. The phase affects the whole shared sweep on bulk-water ticks, including sand.

| Simulation / bulk water | Phase source | Water through original outlet | Water through mirrored outlet |
| --- | --- | ---: | ---: |
| 180 / 90 Hz | Simulation tick | 100.6 L | 2,078.7 L |
| 180 / 60 Hz | Simulation tick | 347.8 L | 345.0 L |
| 180 / 90 Hz | Bulk-water tick on water sweeps | 547.9 L | 543.7 L |
| 120 / 60 Hz | Bulk-water tick on water sweeps | 349.2 L | 346.2 L |
| 120 / 90 Hz | Bulk-water tick on water sweeps | 545.7 L | 549.2 L |

The rotating phase reduced the mirrored difference at 180/90 Hz to under 1% of the pair's average. This fixture does not establish the physical target flow rate. Slowing water to 60 Hz still reduced discharge by about 36% between the rotating-phase profiles. Their mixed-fixture P95 timings were 51.7 and 37.4–39.3 ms, respectively. Both held material inventories and registered sand impacts. Use `-DSOMEVOXEL_EXPERIMENTAL_WATER_SWEEP_PHASE=ON` to reproduce.

At 120/90 Hz, a cumulative schedule makes three water updates for every four engine ticks. Keeping the water rate at 90 Hz kept the mirrored tests within 1% of the 180/90 Hz rotating-phase results. The mixed fixture discharged 541.6 L and recorded a 1.350 N·s peak impact.

## CPU parallel water-head reconstruction

An exact span-based probe reduced mean reconstruction in the mixed fixture from 5.29–5.33 ms to 3.19 ms on one worker and 2.73 ms on eight. Whole-solver P95 improved from 51.0–52.9 ms to 48.0 ms in the eight-worker run, with overlapping timings across worker configurations. All checkpoint digests matched. The prototype currently allocates 31.25 MiB of additional scratch. See [method, randomized tests and reproduction](cpu_parallel_heads.md).

## Numeric precision

CPU FP16 travel and velocity lowered grain storage from 40 to 28 bytes, saving about 1.5 MiB in the mixed fixture. Its mixed P95 overlapped FP32 and the number of fractured cells changed from 64 to 57. CPU storage conversion uses FP32 for calculations. See [the isolated CPU grain experiments](cpu_grain_precision/README.md).

Vectorizing three components of a single grain's sweep produced identical motion and occupancy hashes. In its paired mixed runs, explicit SIMD measured 50.9–52.6 ms P95 and a Zen 4 compiler target measured 51.9 ms. The scalar control measured 47.5–50.7 ms. Independent particles and tiles provide a more promising target for parallel work.

On the GPU, integer coordinate packing and deterministic slip arbitration maintained the FP32 control's final state hash. At 1 cm and 840,000 active grains, the median P95 over four runs fell from 11.17 to 9.77 ms, and tracked buffer usage fell from 1,965 to 1,003 MiB. The remaining dense occupancy allocation scales with world volume. FP16 velocity storage changed the bulk motion in this fixture, so preserving full precision near contacts is the safer starting point. See [reproduction and per-format results](cuda_motion_precision.md).
