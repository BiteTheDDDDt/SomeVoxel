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

Settled liquids share one cell-transfer pass. Each material supplies the flow cadence, downhill reach and pressure approximation. The connected-head profile uses the existing parallel water-head backend. The local-column profile tests the material immediately above the cell. Active parcels use the common swept integrator with their own force and drag settings.

Compaction consumes a finite work budget. Applied pressure and eligible high-energy contacts pass work into the same packing check. Contact and pressure-pulse fracture accumulate work against material resistance. The solver queues packed fractures until the sweep finishes, then preserves mass and parent momentum during subdivision.

Sustained compression samples the dominant applied acceleration against a bearing face before the motion solver cancels blocked travel. Loose sand uses a fixed physical bearing depth as a local pressure approximation. Packed bodies use their mass and face area. Work above sand's yield pressure accumulates in a bounded regional cache and decays when the load stops. A completed region uses the same donor and occupancy check as a short impact.

Sand forms can recruit loose grains and packed bodies. Packed bodies consume their occupied volume in the form budget and retain one binding at their anchor. Packing a region mostly held by a form transfers its binding to the new body. Fracture passes the binding to the resulting pieces within the motion-record budget.

Fresh Windows, Linux and macOS builds default to four workers for water-head rebuilds and sand rendering. At refinement 18 and higher, water-head rebuilds default to eight. Override `SOMEVOXEL_PARALLEL_WATER_WORKERS` and `SOMEVOXEL_PARALLEL_RENDER_WORKERS` at configure time, or set either to an empty string for its serial implementation. At runtime, `set_water_workers(n)` selects a limit per world, with 0 for all detected logical processors and −1 for the build default. `water_worker_count()` includes the calling thread. Pressure checks can reuse the packed component surfaces until the water topology changes. When a connected surface can discharge, the solver uses the original sequential parcel routes.

Enable `set_sand_sweep_profiling(true)` to collect the latest step's row visits, loose-sand integration count, face attempts, face crossings, contact count and sweep time. Read the counters with `sand_sweep_stats()`. The Godot bridge returns the same fields as a dictionary.

Alternative CPU and GPU solvers can be tested independently using the [simulation experiments](experiments/README.md).

`spawn_packed_particle` inserts a cube of packed material with an edge of 2, 4, or 8 cells and an optional impulse in newton-seconds. It returns a handle on success or zero if the volume is unavailable. Use `packed_particle_state` to query its grid position and velocity. Reloading the world or fracturing the packed cube invalidates its handle.

`free_sand_throw()` and `launch_free_sand()` select up to 131.22 kg of loose sand within the active focus, rounded down to whole cells. Packed bodies have a separate selection limit and contribute their actual mass to the charge work calculation. The [rate and precision fixtures](experiments/performance_precision.md) report outflow, impact impulse and fracture volume in physical units.

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

For builds with the parallel water-head solver enabled, create a `RefCounted` from `addons/somevoxel/gpu/connected_heads.gd` and register its `solve_compact` callable with `VoxelWorld.set_water_head_accelerator()`. Keep a reference while registered. The callback receives an immutable, cropped occupancy snapshot and returns the exact height of the face-connected component containing each occupied X run. It runs synchronously on the simulation caller. Clear it with an empty `Callable()`.

In water volumes under 255 cells tall, `solve_compact` first reads one height byte per eight-cell row. When separate runs in a row have different heights, it reads four individual height bytes per row. Taller volumes and the original `solve` method return four little-endian 16-bit heights per row. The CPU validates every occupied run before applying the result to the world.

Register the callback with `set_water_head_accelerator(solve, true)` for automatic selection. On successive ordinary rebuilds the world measures both backends, including upload, readback and publication. It excludes the first run on each backend, compares the median of three runs and reuses the faster backend. It samples again after sustained work or a substantial change in the water volume or topology. `status()` reports the last backend and the most recent CPU and GPU times. During map loading, `ConnectedHeads.prepare()` can warm the Vulkan device and pipeline before the first interactive frame.

The solver attempts acceleration at one million or more water cells. The packed box is limited to two million rows. The CPU solver handles pressure transfer, small workloads, and failed acceleration. `status()` reports rebuilds completed by the accelerator, attempts, and cumulative microseconds for those attempts including packing, execution, and publication. The [workgroup benchmark](experiments/gpu_water_heads/README.md) separates GPU computation and readback.
