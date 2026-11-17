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
