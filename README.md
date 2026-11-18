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
- Air, stone, sand, and water share the CPU material IDs 0–3. Sand and water exchange places under gravity, and water spreads over local support. Sand responds to a radial acceleration with independent strength and radius, and to spherical impulses. Velocity and subcell travel use sixteenths of a cell per step. The controlled solver caps each component to 15/16 cells per step.
- The GPU allocates 8³ pages at a moving frontier. Set a memory budget with `max_resident_pages`. The default reserves extra space based on the initial layout. Pages remain resident until reload. `status()` reports resident and active pages and allocation misses. A full pool blocks movement into missing pages.
- `sample_cells()` reads materials at explicit coordinates. `set_cells()` applies a batch of unique cell edits. A failed edit preserves materials but may have reserved pages. `surface_cells()` returns exposed cells and the total count, so callers can detect a truncated readback. Each record contains X, Y, Z, and a packed integer with the material in the low byte and face bits in the next byte. Face order is −X, +X, −Y, +Y, −Z, +Z.
- The GPU backend currently covers only this material subset. Fracture, thermal and phase simulation, rigid-body and character contacts, and the full liquid solver belong to the C++ backend. The GPU surface API produces a CPU snapshot for consumers.

To run the GPU smoke test, copy `tests/gpu_world_smoke.gd` into the root of a Forward+ project containing the installed addon. Import the shader in the editor, then run:

```sh
godot --path /path/to/godot-project --rendering-driver vulkan --script res://gpu_world_smoke.gd
```

Pass `-- --boxes /path/to/layouts.jsonl` to time full box layouts. Each JSON line takes `bounds` as an XYZ array and `boxes` as the flat box array. Optional inputs are `max_pages`, `focus` as an XYZ array, and `field_radius`. The benchmark reports load time, fenced P50 and P95 for simulation steps, buffer bytes, and page counts before and after the run.
