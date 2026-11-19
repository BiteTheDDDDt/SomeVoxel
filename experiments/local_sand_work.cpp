#include "core/world.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>

int main() {
    using somevoxel::Cell;
    using somevoxel::Vec3;
    using somevoxel::World;
    auto world = std::make_unique<World>();
    const float pitch = world->status().cell_size;
    if (std::abs(pitch - 0.01f) > 1e-6f) return 1;
    const auto cell = [=](float v) { return int(std::lround(v / pitch)); };
    const int authored[][7] = {
        {0, 0, 0, 36, 1, 36, World::Stone},
        {17, 1, 17, 28, 2, 26, World::Sand},
        {17, 2, 17, 27, 4, 26, World::Sand},
        {17, 4, 17, 26, 6, 26, World::Sand},
        {18, 6, 19, 24, 8, 24, World::Sand},
        {26, 1, 20, 27, 15, 24, World::Wood},
    };
    std::vector<int32_t> layout;
    for (const auto &box : authored) {
        for (int i = 0; i < 6; ++i) layout.push_back(cell(0.18f * box[i]));
        layout.push_back(box[6]);
    }
    if (!world->load_boxes(layout.data(), layout.size())) return 2;
    const int64_t mass = world->status().sand_mass;
    world->set_sand_sweep_profiling(true);
    std::cout << std::fixed << std::setprecision(6);
    bool conserved = true;
    const auto run = [&](const char *label, int steps) {
        std::vector<double> times;
        int64_t scanned = 0, skipped = 0, integrated = 0, tested = 0, contacts = 0;
        for (int i = 0; i < steps; ++i) {
            const auto start = std::chrono::steady_clock::now();
            world->step();
            times.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count());
            const auto &stats = world->sand_sweep_stats();
            scanned += stats.form_scanned;
            skipped += stats.settled_skipped;
            integrated += stats.integrated;
            tested += stats.tested_faces;
            contacts += stats.contacts;
        }
        std::sort(times.begin(), times.end());
        const auto state = world->status();
        conserved = conserved && state.sand_mass == mass;
        const int below = world->range_count(World::Gravel, {320, 94, 330}, {438, 122, 435});
        const int above = world->range_count(World::Gravel, {320, 122, 330}, {438, 164, 435});
        const auto impact = world->sand_impact_peak(World::Wood);
        std::cout << label << " steps=" << steps << " p95_ms=" << times[(95 * steps + 99) / 100 - 1]
            << " scanned=" << scanned << " settled_skipped=" << skipped
            << " integrated=" << integrated << " tested_faces=" << tested << " contacts=" << contacts
            << " mass=" << state.sand_mass << " packed=" << state.packed_sand_blocks
            << " below_above_cells=" << below << '/' << above
            << " impact=" << impact->impulse << " wood_fractures=" << state.contact_wood_fractures
            << std::endl;
    };
    run("settle", 4);
    world->set_cast_profile(0.25f, 10.f);
    world->set_charge_gather(1.f);
    world->set_field({3.80f, 1.22f, 3.82f}, true);
    run("pull", 40);
    world->set_field({}, false);
    world->set_cast_profile(1.f, 8.f);
    world->set_particle_form({3.82f, 1.48f, 3.82f}, {1.f, 0.f, 0.f}, World::Sand, 1, 2.4f, 8.f);
    run("gather", 52);
    const int launched = world->launch_particle_form({1.f, 0.f, 0.f}, 4.f, 1.f);
    std::cout << "launched=" << launched << std::endl;
    run("impact", 32);
    return conserved && launched > 0 && world->sand_impact_peak(World::Wood)->impulse > 0.f ? 0 : 3;
}
