#include "detail/voxel_water_spans.hpp"
#include <chrono>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

constexpr int X = 320, Y = 112, Z = 256, RX = X / 8, SLICE_ROWS = RX * Z;
using Clock = std::chrono::steady_clock;
using Heads = std::vector<uint16_t>;
using Rows = std::vector<uint8_t>;

static void store_head(const Rows &rows, Heads &heads, int id, int height) {
    unsigned starts = rows[id / 8] & ~(unsigned(rows[id / 8]) << 1);
    if (!(starts & (1u << (id % 8)))) return;
    int index = __builtin_popcount(starts & ((1u << (id % 8)) - 1u));
    heads[id / 8 * 4 + index] = static_cast<uint16_t>(height);
}

static Heads breadth_first(const Rows &rows, const std::vector<int> &water) {
    Heads heads(rows.size() * 4);
    auto unvisited = rows;
    std::vector<int> queue;
    for (int cell : water) {
        if (!(unvisited[cell / 8] & (1u << (cell % 8)))) continue;
        const auto visit = [&](int id) {
            uint8_t face = uint8_t(1u << (id % 8));
            if (unvisited[id / 8] & face) {
                unvisited[id / 8] &= uint8_t(~face);
                queue.push_back(id);
            }
        };
        queue.clear();
        visit(cell);
        int head = 0;
        for (size_t i = 0; i < queue.size(); ++i) {
            int id = queue[i], x = id % X, y = id / (X * Z), z = id / X % Z;
            head = std::max(head, y + 1);
            if (x > 0) visit(id - 1);
            if (x + 1 < X) visit(id + 1);
            if (z > 0) visit(id - X);
            if (z + 1 < Z) visit(id + X);
            if (y > 0) visit(id - X * Z);
            if (y + 1 < Y) visit(id + X * Z);
        }
        for (int id : queue) store_head(rows, heads, id, head);
    }
    return heads;
}

// Diagonal-only neighbors, multiple runs in one row, sparse stepwise
// bridges, and corners on opposite sides of the flat index boundaries.
static bool fragmented(int x, int y, int z) {
    if (x < 112 && y < 27 && z < 64) return (x + y + z) % 2 == 0;
    if (x >= 18 && x < 251 && z >= 26 && z < 189 && y == 41) return (x * z) % 13 != 0;
    if (x == 160 && z == 83 && y >= 41 && y < 73) return true;
    if (x >= 160 && x <= 176 && y == 72 && z == 83) return true;
    return (x == 0 && y == Y - 1 && z == Z - 1) || (x == X - 1 && y == Y - 1 && z == 0)
        || (x == 0 && y == 0 && z == 0);
}

static bool reservoir(int x, int y, int z, bool open) {
    if (x < 5 || x > 310 || y >= 108 || z < 2 || z >= 254) return false;
    if (x == 149) return open && y == 0 && z == 126;
    return x < 149 || y < 61;
}

int main(int argc, char** argv) {
    const std::string fixture = argc == 2 ? argv[1] : "closed";
    const bool open = fixture == "open", scattered = fixture == "fragmented";
    if (!open && !scattered && fixture != "closed") return 2;
    std::vector<int> water;
    Rows rows(SLICE_ROWS * Y);
    int height = 0;
    for (int y = 0; y < Y; ++y)
        for (int z = 0; z < Z; ++z)
            for (int x = 0; x < X; ++x) {
                if (!(scattered ? fragmented(x,y,z) : reservoir(x,y,z,open))) continue;
                const int id = x + (z + Z * y) * X;
                rows[id / 8] |= uint8_t(1u << (id % 8));
                water.push_back(id);
                height = y + 1;
            }
    const auto write = [&](const auto& data, const char* path) {
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(data.front()));
        if (!file) throw std::runtime_error(path);
    };
    write(rows, "rows.u8");
    const auto expected = breadth_first(rows, water);
    write(expected, "expected.u16");
    int failures = 0;
    const std::vector<std::pair<int,int>> moving;
    for (bool direct_rows : {false, true}) {
        somevoxel::WaterSpans<X, Y, Z> cpu(4);
        Heads output(expected.size());
        std::vector<float> samples;
        int mismatches = 0;
        for (int trial = 0; trial < 16; ++trial) {
            const auto apply = [&](int id, int top) { store_head(rows, output, id, top); };
            const auto start = Clock::now();
            if (direct_rows)
                cpu.rebuild_rows(rows, moving, height, apply);
            else
                cpu.rebuild(water, apply);
            const float elapsed = std::chrono::duration<float, std::milli>(Clock::now() - start).count();
            if (trial >= 4) samples.push_back(elapsed);
            if (output != expected) ++mismatches;
        }
        std::sort(samples.begin(), samples.end());
        std::printf("{\"fixture\":\"%s\",\"direct_rows\":%d,\"water\":%zu,\"rows\":%zu,\"mismatch_trials\":%d,"
            "\"cpu_p50_ms\":%.3f,\"cpu_p95_ms\":%.3f}\n", fixture.c_str(), direct_rows, water.size(), rows.size(),
            mismatches, samples[5], samples.back());
        failures += mismatches;
    }
    return failures ? 1 : 0;
}
