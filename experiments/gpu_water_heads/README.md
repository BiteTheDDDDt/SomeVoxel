# Exact Vulkan water-head reconstruction

An ordinary head rebuild assigns each face-connected water component the height of its highest cell. The Vulkan kernel joins occupied X runs and computes the component maximum. It returns one 16-bit height per contiguous run. The CPU reference uses the current four-worker span solver with packed row masks. An independent breadth-first scan supplies the expected output.

This probe measures a single immutable occupancy snapshot. It covers two reservoir-sized volumes separated by a partition, the same volumes connected by a one-cell opening, and a fragmented case with diagonal neighbors, four runs per row, and bridges across row and page boundaries. All runs use a 320 × 112 × 256 grid. Every compared output matched the breadth-first reference byte for byte.

| Fixture | Water cells | CPU row masks | Vulkan 64 | Vulkan 128 | Vulkan 256 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Closed partition | 6,393,996 | 8.65 / 9.74 | 4.12 / 4.75 | 4.19 / 4.95 | 4.17 / 4.73 |
| Open partition | 6,393,997 | 8.78 / 9.97 | 3.94 / 4.31 | 4.02 / 4.52 | 4.06 / 4.53 |
| Fragmented | 129,067 | 0.87 / 1.66 | 2.41 / 2.68 | 2.42 / 2.86 | 2.45 / 3.40 |

The table gives P50 / P95 wall time in milliseconds. Vulkan columns include uploading 1.09 MiB of occupancy masks, all four compute passes, and reading back 8.75 MiB of head values. Measured on Windows with a Ryzen 9 8945HX, an RTX 5070 Ti Laptop GPU, an optimized Clang build, and Godot 4.7.2 Vulkan. The CPU benchmark collects 12 samples after four warmups. Each GPU series collects 40 samples after six warmups. Snapshot extraction and application to the live simulation are outside both timings.

At 128 threads per workgroup, the dense fixtures took 1.59–1.73 ms P50 with the occupancy masks already on the device. Timestamp passes put the union at 1.29–1.44 ms and head publication at about 0.02 ms. Using ordinary reads of the finalized parent links lowered publication from 0.24–0.43 ms with the same workgroup size. Readback alone took 2.23–2.27 ms on the host. It remained larger than the entire fenced solve for the dense fixtures. The three workgroup sizes had similar roundtrip timings in the table. An earlier closed-partition sweep had P50 times of 6.41–6.57 ms for 64 and 128 before a repeat returned the figures above. The sweep does not establish one consistent winner.

The benchmark reports `fenced` for dispatch through synchronization with device-resident input. A second series reports `roundtrip`, host upload enqueue, submit and wait, and blocking readback. `upload_fenced` times an upload followed by a dependent four-byte buffer copy and synchronization. GPU stage timings come from a separate series with timestamps between compute lists. Its timings can include transitions between passes. These series measure different submissions and their percentiles should be read separately. [Godot RenderingDevice documentation](https://docs.godotengine.org/en/stable/classes/class_renderingdevice.html)

The CPU is faster for the fragmented fixture once the GPU result must return to the CPU. Large connected volumes have room for improvement. A production path would need to track dirty rows and measure the cost of applying changed heads before the CPU sweep. Pressure-balancing ticks use route order and parcel-state advection. Both stay with the authoritative solver in this experiment.

## Reproduce

Stage the project in an ignored build directory. The C++ program writes a packed occupancy snapshot and its expected heads into the current directory.

```sh
mkdir -p build/gpu-water-heads
cp experiments/gpu_water_heads/project.godot experiments/gpu_water_heads/bench.gd \
    experiments/gpu_water_heads/head.glsl build/gpu-water-heads/
clang++ -std=c++17 -O3 -pthread -Icore experiments/gpu_water_heads/reference.cpp -o build/gpu-water-heads/reference
cd build/gpu-water-heads
for fixture in closed open fragmented; do
    ./reference "$fixture" || exit 1
    godot --rendering-driver vulkan --script res://bench.gd || exit 1
done
```

On Windows use the Godot console executable and the equivalent native Clang build. Run the Vulkan benchmark with rendering enabled. By default it runs all three workgroup sizes. Append `-- --group-size 128` to measure one size in isolation. The process exits nonzero if its output differs from the reference.
