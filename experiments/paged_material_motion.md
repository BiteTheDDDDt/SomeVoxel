# CUDA updates on a paged material grid

The probe extends the paged occupancy check to 48 updates on two 1 cm fixture layouts with 1296 × 648 × 1008 logical cells. The two fixtures contain 5.10–5.31 million sand cells and 8.33–8.40 million water cells. Material storage covers an initial halo around the mobile tiles. Immobile homogeneous tiles use a directory entry. Each of the three axis sweeps has two parity passes, so neighboring pairs can exchange cells across a tile boundary with one writer per cell. An additional support snapshot precedes each horizontal axis so water considers the same support and overhead state throughout that sweep.

The simplified rules move sand toward a moving target, allow denser sand to exchange with water, and let water move toward drop-offs or away from a filled column. The probe stores material occupancy. It leaves velocity, stone fracture, role contacts and pressure balancing to later prototypes. The initial halo reserves all pages available during a run.

## Results

RTX 5070 Ti Laptop GPU, Warp 1.19.0. CUDA event timings start after four warmup updates. Tracked buffers include the grid, directory, scheduled pages, water flags and counters. Drawing and verification are outside the timing interval.

| Halo | Fixture | Tracked buffers | Sand cross-page moves | Water cross-page moves | CUDA update P95 |
| --- | --- | ---: | ---: | ---: | ---: |
| 3 tiles | A | 163.6 MiB | 5,963 | 15,486 | 4.25–4.32 ms |
| 3 tiles | B | 162.6 MiB | 5,677 | 135,058 | 4.24 ms isolated, 27.4–27.7 ms in sequence |
| 6 tiles | A | 203.3 MiB | 5,963 | 15,486 | 5.76–6.11 ms |
| 6 tiles | B | 200.3 MiB | 5,677 | 135,058 | 5.95–6.05 ms |

All runs conserved material counts and retained the static cells. Logical coordinate hashes matched across both halo sizes and both submission methods. The second fixture's P95 spike persisted with CUDA Graph replay when run after the first fixture. The isolated run produced the same state with a 4.24 ms P95. The source of that timing variance remains unverified. The large fixtures kept the test sand and water apart. The layered contact fixture recorded 1,918 direct sand/water exchanges and retained both inventories.

## Godot Vulkan port

The Vulkan probe uses the same atlas, update order, material rules and target path. Each fixture ran in its own Godot 4.7.2 process on the same GPU.

| Fixture | Vulkan P50 | Vulkan P95 | Tracked storage | Material hash |
| --- | ---: | ---: | ---: | --- |
| A | 3.14 ms | 3.26 ms | 256.7 MiB | Matches CUDA |
| B | 3.07 ms | 3.14 ms | 255.1 MiB | Matches CUDA |

Vulkan timing covers command recording, submission and the completion wait, with the first four updates excluded. Vulkan stores support flags as 32-bit words and omits the CUDA per-page event counters. The verifier reads back once after all 48 updates and checks material counts, static occupancy and the logical coordinate hash. The probe uses a local rendering device with no scene drawing.

## Reproduce

```sh
python -m pip install warp-lang==1.19.0
python experiments/paged_material_motion.py experiments/fixtures/layered_material_contact.jsonl --bounds 48 40 48 --frames 36 --halo 3
python experiments/paged_material_motion.py layouts.jsonl --bounds 1296 648 1008 --frames 48 --halo 3 --graph --trace
```

The input accepts one object per line with a `layout` identifier and a flat array of `[x0,y0,z0,x1,y1,z1,material]` boxes. The bounds are exclusive and coordinates are in cells. IDs 0–3 represent air, static solid, sand and water. An optional `focus` field provides the initial attraction target. `--graph` uses a prerecorded graph for each update. `--trace` reports the individual CUDA event times.

```sh
python experiments/paged_material_motion.py layouts.jsonl --bounds 1296 648 1008 --halo 3 --graph > build/cuda-motion.log
python experiments/paged_vulkan_fixture.py prepare layouts.jsonl build/vulkan-motion --bounds 1296 648 1008 --halo 3
godot --path experiments/gpu_ca --script res://paged_motion.gd -- /absolute/path/to/build/vulkan-motion/0
godot --path experiments/gpu_ca --script res://paged_motion.gd -- /absolute/path/to/build/vulkan-motion/1
python experiments/paged_vulkan_fixture.py verify build/cuda-motion.log build/vulkan-motion/0 build/vulkan-motion/1
```
