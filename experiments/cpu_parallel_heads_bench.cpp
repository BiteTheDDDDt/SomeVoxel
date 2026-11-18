#include "core/world.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>

using namespace somevoxel;
using Clock = std::chrono::steady_clock;

static float percentile(std::vector<float> data, float p) {
    std::sort(data.begin(), data.end());
    return data[size_t(std::ceil(p * data.size())) - 1];
}

static bool check_span_heads(int threads) {
    constexpr int x_size = 24, y_size = 18, z_size = 32, slice = x_size * z_size, volume = slice * y_size;
    WaterSpans<x_size, y_size, z_size> solver(threads);
    std::mt19937 rng(2019);
    std::vector<int> cells, visited, stack;
    std::vector<int> expected(volume), actual(volume);
    for (int trial = 0; trial < 16; ++trial) {
        visited.assign(volume, 0);
        expected.assign(volume, 0);
        actual.assign(volume, 0);
        cells.clear();
        for (int i = 0; i < volume; ++i) {
            if (rng() % 100 < unsigned(6 * trial + 4) || (i % x_size == 8 && i / slice == 8)) {
                cells.push_back(i);
                visited[i] = 1;
            }
        }
        std::shuffle(cells.begin(), cells.end(), rng);
        for (int start : cells) {
            if (!visited[start]) continue;
            stack.clear();
            stack.push_back(start);
            visited[start] = 0;
            int top = 0;
            for (size_t cursor = 0; cursor < stack.size(); ++cursor) {
                const int i = stack[cursor], x = i % x_size, z = i / x_size % z_size, y = i / slice;
                top = std::max(top, y + 1);
                const auto enqueue = [&](int j) {
                    if (visited[j]) { visited[j] = 0; stack.push_back(j); }
                };
                if (x > 0) enqueue(i - 1);
                if (x + 1 < x_size) enqueue(i + 1);
                if (z > 0) enqueue(i - x_size);
                if (z + 1 < z_size) enqueue(i + x_size);
                if (y > 0) enqueue(i - slice);
                if (y + 1 < y_size) enqueue(i + slice);
            }
            for (int i : stack) expected[i] = top;
        }
        solver.rebuild(cells, [&](int i, int head) { actual[i] = head; });
        if (actual != expected) return false;
    }
    return true;
}

int main(int argc, char **argv) {
    const int threads = argc > 1 ? std::atoi(argv[1]) : 0;
    const bool water_only = argc > 2 && std::string(argv[2]) == "water";
    const int frames = argc > 3 ? std::atoi(argv[3]) : 150;
    if (threads < 0 || threads > 16 || frames <= 6) return 2;
    if (threads && !check_span_heads(threads)) return 4;

    using Box = std::array<int32_t, 7>;
    constexpr int r = SOMEVOXEL_REFINEMENT;
    const Box sand{12, 1, 12, 24, 4, 24, World::Sand};
    const Box crest{15, 4, 15, 21, 6, 21, World::Sand};
    constexpr std::array<Box, 9> stationary{{
        {0, 0, 0, 72, 1, 56, World::Stone}, {32, 1, 12, 33, 13, 25, World::Stone},
        {6, 1, 32, 24, 8, 34, World::Stone}, {6, 1, 48, 24, 8, 50, World::Stone},
        {6, 1, 34, 8, 8, 48, World::Stone}, {22, 1, 34, 24, 8, 39, World::Stone},
        {22, 1, 43, 24, 8, 48, World::Stone}, {22, 4, 39, 24, 8, 43, World::Stone},
        {8, 1, 34, 22, 8, 48, World::Water},
    }};
    std::vector<int32_t> layout;
    const auto append = [&](const Box &box) {
        for (int i = 0; i < 6; ++i) layout.push_back(box[i] * r);
        layout.push_back(box.back());
    };
    for (const Box &box : stationary) append(box);
    if (!water_only) { append(sand); append(crest); }

    auto world = std::make_unique<World>();
    world->research_head_workers(threads);
    if (!world->load_boxes(layout.data(), layout.size())) return 3;
    world->set_cast_profile(0.85f, 1.75f);
    const auto before = world->status();
    const Vec3 start(3.24f, 1.12f, 3.24f), end(4.66f, 1.12f, 3.24f);
    std::vector<float> batch_times, head_times;
    std::vector<uint64_t> digests;
    int launched = 0;
    for (int frame = 0; frame < frames; ++frame) {
        if (!water_only && frame >= 15 && frame < 66) {
            const float progress = float(frame - 15) / 51.f;
            const float t = std::clamp((progress - 0.25f) / 0.65f, 0.f, 1.f);
            world->set_charge_gather(progress);
            world->set_field(start + (end - start) * (t * t * (3.f - 2.f * t)), true);
        } else if (!water_only && frame == 66) {
            launched = world->launch_free_sand(end, {1.f, 0.f, 0.f}, 3.f, 1.f);
            world->set_field(end, false);
            world->set_charge_gather(0.f);
        }
        for (int batch = 0; batch < 2; ++batch) {
            const auto old_head = world->research_head_nanos();
            const auto begin = Clock::now();
            for (int tick = 0; tick < r / 2; ++tick) world->step();
            const float elapsed = std::chrono::duration<float, std::milli>(Clock::now() - begin).count();
            if (frame >= 6) {
                batch_times.push_back(elapsed);
                head_times.push_back(float(world->research_head_nanos() - old_head) / 1e6f);
            }
        }
        if (frame == frames / 3 || frame == frames * 2 / 3 || frame == frames - 1)
            digests.push_back(world->research_head_digest());
    }
    const auto after = world->status();
    const int spilled = world->range_count(World::Water, {24 * r, r, 0}, {72 * r, 36 * r, 56 * r});
    const bool conserved = before.sand_mass == after.sand_mass && before.water_volume == after.water_volume;
    float total_head_ms = 0;
    for (float ms : head_times) total_head_ms += ms;
    std::cout << std::fixed << std::setprecision(3)
        << "{\"workers\":" << threads << ",\"fixture\":\"" << (water_only ? "water" : "mixed")
        << "\",\"frames\":" << frames << ",\"batch_p50_ms\":" << percentile(batch_times, 0.50f)
        << ",\"batch_p95_ms\":" << percentile(batch_times, 0.95f)
        << ",\"head_mean_ms\":" << total_head_ms / head_times.size()
        << ",\"head_p95_ms\":" << percentile(head_times, 0.95f)
        << ",\"head_span_rebuilds\":" << world->research_span_count()
        << ",\"head_bfs_rebuilds\":" << world->research_bfs_count()
        << ",\"span_scratch_bytes\":" << world->research_scratch_bytes()
        << ",\"conserved\":" << conserved << ",\"spilled\":" << spilled
        << ",\"launched\":" << launched << ",\"fractures\":" << after.contact_stone_fractures << ",\"digests\":[";
    for (size_t i = 0; i < digests.size(); ++i) std::cout << (i ? ",\"" : "\"") << digests[i] << '\"';
    std::cout << "]}\n";
    return conserved && spilled > 0 && (water_only || frames <= 66 || (launched > 0 && after.contact_stone_fractures > 0)) ? 0 : 1;
}
