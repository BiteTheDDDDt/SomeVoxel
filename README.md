# SomeVoxel

SomeVoxel is a C++17 library for simulating deformable 3D voxel materials. It provides fixed-step sand and liquid motion, material contacts, fracture, structural support, heat transfer, and phase changes. The optional Godot GDExtension registers `VoxelWorld`.

## Build and test the core

```sh
cmake -S . -B build/core -DCMAKE_BUILD_TYPE=Release
cmake --build build/core --config Release
ctest --test-dir build/core -C Release --output-on-failure
```

In a CMake consumer, use `add_subdirectory` and link `SomeVoxel::Core`. Include `core/world.hpp` and construct a `somevoxel::World`. Pass `load_boxes` an array and its element count. Each box is encoded as seven integers: inclusive minimum X/Y/Z, exclusive maximum X/Y/Z, and a material ID from `World::Kind`. Position queries use metres. `status().step_seconds` gives the simulated duration of one `step()`.

The default grid has 4.5 cm cells and runs at 120 Hz. Its dimensions and refinement are build settings. The API is experimental.

Configure `SOMEVOXEL_PARALLEL_WATER_WORKERS` at build time to enable the parallel water-head solver. At runtime, `set_water_workers(n)` selects a limit per world, with 0 for all detected logical processors and −1 for the build default. `water_worker_count()` reports the configured count, including the calling thread. Ordinary water-head rebuilds use the pool. Pressure checks can reuse the packed component surfaces until the water topology changes. When a connected surface can discharge, the solver uses the original sequential parcel routes.

Enable `set_sand_sweep_profiling(true)` to collect the latest step's row visits, loose-sand integration count, face attempts, face crossings, contact count and sweep time. Read the counters with `sand_sweep_stats()`. The Godot bridge returns the same fields as a dictionary.

Alternative CPU and GPU solvers can be tested independently using the [simulation experiments](experiments/README.md).

`spawn_packed_particle` inserts a cube of packed material with an edge of 2, 4, or 8 cells and an optional impulse in newton-seconds. It returns a handle on success or zero if the volume is unavailable. Use `packed_particle_state` to query its grid position and velocity. Reloading the world or fracturing the packed cube invalidates its handle.

## Build the Godot extension

The extension build requires Python in addition to CMake and a C++ compiler. CMake fetches and verifies a pinned archive of the Godot 4.6 bindings. Build and install the addon into a Godot project:

```sh
cmake -S . -B build/godot -DSOMEVOXEL_BUILD_GODOT=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/godot --config Release
cmake --install build/godot --config Release --prefix /path/to/godot-project
```

Open or reimport the Godot project after installation. The addon is installed under `addons/somevoxel`. Example smoke check:

```gdscript
var world = ClassDB.instantiate("VoxelWorld")
assert(world.load_boxes(PackedInt32Array([0, 0, 0, 16, 1, 16, 1, 4, 5, 4, 8, 9, 8, 2])))
for tick in 120:
    world.step()
assert(world.status().sand_mass == 64)
```

### Rendering sand from the CPU world

`VoxelWorld.sand_snapshot(cull, previous_rows)` packs the CPU world's current and previous cell indices into an RGBA32F texture, using 16 bytes per slot. It returns the visible instance count, width, height, texture bytes, and the three bit shifts for decoding coordinates. The installed `presentation/sand_instances.gdshader` decodes the positions, interpolates in the vertex shader, and samples the colors returned by `sand_palette()`. Use an origin-aligned MultiMesh with a custom AABB that covers the world. Update the snapshot when the full `particle_motion(2)` value or visibility settings change and supply the render interpolation fraction every frame. The regular `particle_batch()` output remains available for conventional rendering.

`VoxelWorld.take_sand_tiles(cull)` exposes the same texture format as stateful deltas for a single renderer. The result contains a reset flag, the coordinate shifts, the texture width, and an array of changed 64³-cell tiles. Each nonempty tile includes its ID, instance count, padded texture bytes, minimum and exclusive maximum bounds in cells, and whether its endpoints need interpolation. An empty tile returns its ID and an instance count of zero. Drop cached tiles when the reset flag is set. Use the returned bounds for each origin-aligned MultiMesh. Poll when the full motion value or culling setting changes.

### Vulkan water heads in the CPU world

For builds with the parallel water-head solver enabled, create a `RefCounted` from `addons/somevoxel/gpu/connected_heads.gd` and keep a reference to it while its `solve` callable is registered with `VoxelWorld.set_water_head_accelerator()`. The callback receives an immutable, cropped occupancy snapshot and returns a face-connected water head for each occupied X run. It runs synchronously on the simulation caller and must avoid accessing or mutating the world. Clear it with an empty `Callable()`.

The solver attempts acceleration at one million or more water cells. The packed box is limited to two million rows. The CPU solver handles pressure transfer, small workloads, and failed acceleration. `status()` reports rebuilds completed by the accelerator, attempts, and cumulative microseconds for those attempts including packing, execution, and publication. The [workgroup benchmark](experiments/gpu_water_heads/README.md) separates GPU computation and readback.

## Godot GPU material world

The addon also includes a sparse compute backend under `addons/somevoxel/gpu` for Godot 4.6+ with Vulkan and Forward+. It uses a local RenderingDevice. The host uploads ordered boxes, and the device owns subsequent material state, page allocation, and dispatch sizing.

```gdscript
const GpuWorld = preload("res://addons/somevoxel/gpu/paged_world.gd")

var world := GpuWorld.new()
assert(world.load_boxes(Vector3i(32, 32, 32), PackedInt32Array([
    0, 0, 0, 32, 1, 32, GpuWorld.Kind.STONE,
    8, 12, 8, 16, 20, 16, GpuWorld.Kind.SAND,
])))
assert(world.submit_steps(4, Vector3(16, 20, 12), 12.0))
# CPU work can run before the next synchronization.
var surface := world.surface_cells(Vector3i(0, 0, 0), Vector3i(32, 32, 32))
world.close()
```

- Call the backend from one thread. The caller controls the fixed update rate and the conversion between cell coordinates and metres. `step()` waits for the device. `submit_steps()` returns after submission. Queries, edits, and readbacks synchronize with pending steps.
- Air, stone, sand, and water share the CPU material IDs 0–3. Use `configure_motion()` to set the cell size in metres and the fixed step in seconds. Sand uses velocity integration with the CPU solver's gravity and attraction coefficients. Controlled water shares the velocity integrator, and bulk water spreads over local support. Velocity and subcell travel use thirty-seconds of a cell per step. Fast material can traverse multiple faces per step, capped at 511/32 cells per step per axis. Each swept segment checks the occupancy along its path. Enable `granular_sliding` for sideways settling on slopes.
- The GPU allocates 8³ pages at a moving frontier. Set a memory budget with `max_resident_pages`. The default reserves extra space based on the initial layout. Pages remain resident until reload. `status()` reports resident and active pages and allocation misses. A full pool blocks movement into missing pages.
- Set `connected_water_pressure = true` before loading to enable hydrostatic redistribution. At 15 Hz the GPU finds face-connected runs of passive water and transfers up to 48 cells from higher exposed sites to lower supported sites. The pressure pass uses extra memory and bounds its endpoint search. Rates during redistribution may differ from the CPU solver. `status().pressure_transfers` reports committed transfers.
- `sample_cells()` reads materials at explicit coordinates. `set_cells()` applies a batch of unique cell edits. A failed edit preserves materials but may have reserved pages. `surface_cells()` returns exposed cells and the total count, so callers can detect a truncated readback. Each record contains X, Y, Z, and a packed integer with the material in the low byte and face bits in the next byte. Face order is −X, +X, −Y, +Y, −Z, +Z.
- `ray_hit()` traverses the live GPU grid with a material mask. `column_map()` returns an RGBA32F snapshot of the highest one or two contiguous sand and water runs in sampled X/Z columns, with inclusive base and exclusive top in cell units. For two runs, the lower layer follows the upper layer in the returned image. The snapshot is intended for presentation.
- `set_kinematic_box()` defines one square-footprint obstacle. Material already inside gains outward motion and can drain through an open side over subsequent steps. The caller handles the obstacle's motion and solid-terrain collision. Fracture, thermal and phase simulation, general rigid-body contacts, and the full liquid solver are available in the C++ backend.

To run the GPU smoke test, copy `tests/gpu_world_smoke.gd` into the root of a Forward+ project containing the installed addon. Import the shader in the editor, then run:

```sh
godot --path /path/to/godot-project --rendering-driver vulkan --script res://gpu_world_smoke.gd
```

Pass `-- --boxes /path/to/layouts.jsonl` to time full box layouts. Each JSON line takes `bounds` as an XYZ array and `boxes` as the flat box array. Optional inputs are `max_pages`, `focus` as an XYZ array, and `field_radius`. The benchmark reports load time, fenced P50 and P95 for simulation steps, buffer bytes, and page counts before and after the run.
