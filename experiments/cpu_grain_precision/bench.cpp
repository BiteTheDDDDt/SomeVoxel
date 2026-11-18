#include "core/world.hpp"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

namespace somevoxel {
struct GrainPrecisionProbe {
    static void print(const World &world) {
        double px = 0, py = 0, pz = 0, energy = 0;
        uint64_t hash = 14695981039346656037ull, motion_hash = hash;
        const auto append = [&](float value) {
            uint32_t bits;
            std::memcpy(&bits, &value, sizeof(bits));
            motion_hash = (motion_hash ^ bits) * 1099511628211ull;
        };
        for (const auto &grain : world.grains) {
            const Vec3 p = World::center(World::decode(grain.position)) + Vec3(grain.travel);
            px += p.x;
            py += p.y;
            pz += p.z;
            const Vec3 velocity = grain.velocity, travel = grain.travel;
            energy += velocity.length_squared() * .5;
            for (int axis = 0; axis < 3; ++axis) {
                append(velocity[axis]);
                append(travel[axis]);
            }
            hash = (hash ^ uint64_t(grain.position)) * 1099511628211ull;
        }
        const double n = double(world.grains.size());
        std::cout << "\"center\":[" << px/n << "," << py/n << "," << pz/n << "],\"mean_energy_per_kg\":" << energy/n
                  << ",\"cells_hash\":\"" << hash << "\",\"motion_hash\":\"" << motion_hash
                  << "\",\"sand_storage\":"
                  << (world.grains.capacity() * sizeof(World::Grain));
    }

    static bool verify_ballistic() {
        World world;
        constexpr float launch_speed = 12.0f;
        constexpr int extent = 2;
        auto handle = world.spawn_packed_particle({1.20f, 2.20f, 1.20f}, extent,
            {launch_speed * World::packed_particle_mass(extent), 0, 0});
        const auto initial = world.packed_particle_state(handle);
        if (!initial) return false;
        for (int i = 0; i < 27; ++i) world.step();
        const auto current = world.packed_particle_state(handle);
        if (!current) return false;
        const float t = World::dt * 27;
        const float expected_x = initial->position.x + t * launch_speed;
        const float expected_vy = -World::gravity * t;
        return std::abs(current->position.x - expected_x) <= World::scale &&
            std::abs(current->velocity.y - expected_vy) <= .012f;
    }
};
}

int main(int argc, char **argv) {
    using somevoxel::World;
    using somevoxel::Vec3;
    using somevoxel::Cell;
    using Clock = std::chrono::steady_clock;
    using Probe = somevoxel::GrainPrecisionProbe;
    if (!Probe::verify_ballistic()) {
        std::cerr << "Ballistic probe failed\n";
        return 2;
    }
    const std::string scenario = argc > 1 ? argv[1] : "mixed";
    const bool wet = scenario == "mixed";
    const bool resting = scenario == "resting";
    constexpr int refinement = SOMEVOXEL_REFINEMENT;
    constexpr int frames = 300;
    constexpr int steps_per_frame = refinement / 2;
    const Vec3 start{3.24f, 1.12f, 3.24f};
    const Vec3 end{4.66f, 1.12f, 3.24f};
    World world;
    std::vector<int32_t> voxel_boxes;
    const auto box = [&](int x0, int y0, int z0, int x1, int y1, int z1, int kind) {
        for (int coord : {x0, y0, z0, x1, y1, z1}) voxel_boxes.push_back(refinement * coord);
        voxel_boxes.push_back(kind);
    };
    box(0, 0, 0, 72, 1, 56, World::Stone);
    box(12, 1, 12, 24, 4, 24, World::Sand);
    box(15, 4, 15, 21, 6, 21, World::Sand);
    box(32, 1, 12, 33, 13, 25, World::Stone);
    if (wet) {
        box(6, 1, 32, 24, 8, 34, World::Stone);
        box(6, 1, 48, 24, 8, 50, World::Stone);
        box(6, 1, 34, 8, 8, 48, World::Stone);
        box(22, 1, 34, 24, 8, 39, World::Stone);
        box(22, 1, 43, 24, 8, 48, World::Stone);
        box(22, 4, 39, 24, 8, 43, World::Stone);
        box(8, 1, 34, 22, 8, 48, World::Water);
    }
    if (!world.load_boxes(voxel_boxes.data(), voxel_boxes.size())) return 3;
    const auto old = world.status();
    world.set_cast_profile(0.85f, 1.75f);
    somevoxel::ThrowInfo release{};
    int launched = 0;
    std::vector<double> timings, active;
    for (int frame = 0; frame < frames; ++frame) {
        if (!resting && frame >= 30 && frame <= 130 && !(frame & 1)) {
            const float progress = float(frame - 30) / 102.f;
            const float t = std::clamp((progress - .25f) / (.90f - .25f), 0.f, 1.f);
            world.set_charge_gather(progress);
            world.set_field(start.lerp(end, t * t * (3.f - 2.f * t)), true);
        } else if (!resting && frame == 132) {
            release = world.free_sand_throw(end, 3.0f, 1.0f);
            launched = world.launch_free_sand(end, Vec3(1, 0, 0), 3.0f, 1.0f);
            world.set_field(end, false);
            world.set_charge_gather(0.f);
        }
        const auto t0 = Clock::now();
        for (int step = 0; step < steps_per_frame; ++step) world.step();
        const double millis = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        if (frame >= 16) timings.push_back(millis);
        if (frame >= 30 && frame < 162) active.push_back(millis);
    }
    const auto current = world.status();
    const auto impact = world.sand_impact_peak(World::Stone).value_or(somevoxel::ImpactInfo{});
    const Cell world_bounds(72 * refinement, 36 * refinement, 56 * refinement);
    const int escaped = world.range_count(World::Water, {24 * refinement, refinement, 0}, world_bounds);
    const auto p95 = [](std::vector<double> &times) {
        std::sort(times.begin(), times.end());
        return times[size_t(std::ceil(times.size() * .95)) - 1];
    };
    const bool conserved = old.sand_mass == current.sand_mass && old.water_volume == current.water_volume;
    std::cout << std::setprecision(9) << "{\"scenario\":\"" << scenario
              << "\",\"refinement\":" << current.refinement << ",\"grain_bytes\":" << current.grain_bytes
              << ",\"p95_ms\":" << p95(timings) << ",\"active_p95_ms\":" << p95(active)
              << ",\"sand\":" << current.sand_mass << ",\"water\":" << current.water_volume << ",\"escaped_water\":" << escaped
              << ",\"launched\":" << launched << ",\"launch_speed\":" << release.speed
              << ",\"impact\":" << impact.impulse << ",\"fractures\":" << current.contact_stone_fractures
              << ",\"conserved\":" << (conserved ? "true" : "false") << ",";
    Probe::print(world);
    std::cout << "}\n";
    return conserved && (resting || launched > 0 && impact.impulse > 0) && (!wet || escaped > 0) ? 0 : 1;
}
