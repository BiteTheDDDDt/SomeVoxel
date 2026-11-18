"""Build the liquid head comparison from an isolated copy of a Git revision."""
import argparse
import io
import os
from pathlib import Path
import subprocess
import tarfile


PROBE_FIELDS = """
    std::unique_ptr<WaterSpans<nx, ny, nz>> research_spans;
    uint64_t research_pressure_ns = 0, research_span_rebuilds = 0, research_bfs_rebuilds = 0;
    struct PressureClock {
        uint64_t &total_ns;
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        ~PressureClock() {
            total_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start).count();
        }
    };
public:
    void research_head_workers(int threads) {
        if (threads > 0) research_spans = std::make_unique<WaterSpans<nx, ny, nz>>(threads);
    }
    uint64_t research_head_nanos() const { return research_pressure_ns; }
    uint64_t research_span_count() const { return research_span_rebuilds; }
    uint64_t research_bfs_count() const { return research_bfs_rebuilds; }
    size_t research_scratch_bytes() const { return research_spans ? research_spans->scratch_bytes() : 0; }
    uint64_t research_head_digest() const {
        uint64_t hash = 14695981039346656037ull;
        const auto add = [&](uint32_t value) { hash = (hash ^ value) * 1099511628211ull; };
        const auto add_float = [&](float value) {
            uint32_t representation;
            std::memcpy(&representation, &value, sizeof(value));
            add(representation);
        };
        for (size_t i = 0; i < voxels.size(); ++i) {
            const Voxel &voxel = voxels[i];
            if (voxel.kind == Empty) continue;
            add(uint32_t(i));
            add(voxel.kind);
            add(voxel.occupant);
            add(voxel.stamp);
            if (voxel.kind == Water) add(liquid_head.empty() ? 0 : liquid_head[i]);
        }
        for (const Grain &grain : grains) {
            add(grain.position);
            for (int axis = 0; axis < 3; ++axis) {
                add_float(grain.velocity[axis]);
                add_float(grain.travel[axis]);
            }
        }
        for (bool active : awake) add(active);
        return hash;
    }
private:
"""

PROBE_ROUTE = """
        PressureClock clock{research_pressure_ns};
        if (research_spans && !balance) {
            ++research_span_rebuilds;
            research_spans->rebuild(water_positions, [&](int i, int head) {
                if (head == liquid_head[i]) return;
                liquid_head[i] = LiquidHead(head);
                const I3 point = decode(i);
                awake[chunk_id({point.x / block, point.y / block, point.z / block})] = true;
            });
            return;
        }
        ++research_bfs_rebuilds;
"""


def replace_once(path, marker, replacement):
    original = path.read_text(encoding="utf-8")
    if original.count(marker) != 1:
        raise RuntimeError(f"Expected one {marker!r} in {path}")
    path.write_text(original.replace(marker, replacement, 1), encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="clang++", help="C++17 compiler with C++ threads")
    parser.add_argument("--refinement", type=int, default=6)
    parser.add_argument("--revision", default="HEAD")
    args = parser.parse_args()
    experiments = Path(__file__).resolve().parent
    root = experiments.parent
    out = root / "build" / "cpu-parallel-research" / f"r{args.refinement}"
    out.mkdir(parents=True, exist_ok=True)
    archive = subprocess.check_output(["git", "archive", args.revision, "core", "tests"], cwd=root)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(out, filter="data")
    include = '#include "types.hpp"'
    replace_once(out / "core" / "world.hpp", include,
                 '#include <chrono>\n#include <cstring>\n#include "voxel_water_spans.hpp"\n' + include)
    include = '#include "detail/voxel_liquid_pressure.hpp"'
    replace_once(out / "core" / "world.hpp", include, include + "\n" + PROBE_FIELDS)
    marker = "        // Only occupied water cells participate"
    replace_once(out / "core" / "detail" / "voxel_liquid_pressure.hpp", marker, PROBE_ROUTE + marker)
    for source, name in [(experiments / "cpu_parallel_heads_bench.cpp", "head_bench"),
                         (out / "tests" / "world_smoke.cpp", "world_smoke")]:
        exe = out / (name + (".exe" if os.name == "nt" else ""))
        subprocess.run([args.cxx, "-std=c++17", "-O3", "-DNDEBUG", "-pthread",
                        f"-DSOMEVOXEL_REFINEMENT={args.refinement}", "-I", str(out), "-I", str(root / "core" / "detail"),
                        str(source), str(out / "core" / "world.cpp"), "-o", str(exe)], check=True)
        print(exe)


if __name__ == "__main__":
    main()
