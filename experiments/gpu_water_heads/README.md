# Exact Vulkan water-head reconstruction

An ordinary head rebuild assigns each face-connected water component the height of its highest cell. The Vulkan kernel joins occupied X runs and computes the component maximum. It returns one 16-bit height per contiguous run. The CPU reference uses the current four-worker span solver with packed row masks. An independent breadth-first scan supplies the expected output.

This probe measures a single immutable occupancy snapshot. It covers two reservoir-sized volumes separated by a partition, the same volumes connected by a one-cell opening, and a fragmented case with diagonal neighbors, four runs per row, and bridges across row and page boundaries. All runs use a 320 × 112 × 256 grid. Every compared output matched the breadth-first reference byte for byte.

| Fixture | Water cells | CPU row masks P50 / P95 | Vulkan fenced P50 / P95 | Vulkan roundtrip P50 / P95 |
| --- | ---: | ---: | ---: | ---: |
| Closed partition | 6,393,996 | 8.65 / 9.74 ms | 2.13 / 2.33 ms | 4.77 / 9.45 ms |
| Open partition | 6,393,997 | 8.78 / 9.97 ms | 1.94 / 2.44 ms | 4.58 / 7.55 ms |
| Fragmented | 129,067 | 0.87 / 1.66 ms | 0.15 / 0.17 ms | 2.58 / 3.19 ms |

Measured on Windows with a Ryzen 9 8945HX, an RTX 5070 Ti Laptop GPU, an optimized Clang build, and Godot 4.7.2 Vulkan. The first four repetitions warm up each run. The remaining twelve provide the timing samples. Fenced timings start with the input in device memory and end after synchronization. Roundtrip timings include the upload of 1.09 MiB in occupancy masks and the readback of 8.75 MiB in head values. Snapshot extraction and application to the live simulation are outside both CPU and GPU timings.

The CPU is faster for the fragmented fixture once the GPU result must return to the CPU. Large connected volumes have room for improvement. A production path would need to track dirty rows and measure the cost of applying changed heads before the CPU sweep. Pressure-balancing ticks use route order and parcel-state advection. Both stay with the authoritative solver in this experiment.

## Reproduce

Stage the project in an ignored build directory. The C++ program writes a packed occupancy snapshot and its expected heads into the current directory.

```sh
mkdir -p build/gpu-water-heads
cp experiments/gpu_water_heads/project.godot experiments/gpu_water_heads/bench.gd \
    experiments/gpu_water_heads/head.glsl build/gpu-water-heads/
clang++ -std=c++17 -O3 -pthread -Icore experiments/gpu_water_heads/reference.cpp -o build/gpu-water-heads/reference
cd build/gpu-water-heads
godot --headless --import
for fixture in closed open fragmented; do
    ./reference "$fixture" || exit 1
    godot --rendering-driver vulkan --script res://bench.gd || exit 1
done
```

On Windows use the Godot console executable and the equivalent native Clang build. Run the Vulkan benchmark with rendering enabled. The process exits nonzero if its output differs from the reference.
