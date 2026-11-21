#include "core/world.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string_view>
#include <vector>

namespace {
using somevoxel::Vec3;
using somevoxel::World;
using Clock = std::chrono::steady_clock;

struct Scene {
    Vec3 start, target;
    int sand, gravel, water;
};

// Inputs and motion from SomeAvatar b9ecd51, lab_layouts.gd and voxel_lab_bench.gd.
constexpr std::array<Scene, 2> scenes{{
    {{3.95f, 1.35f, 3.87f}, {6.42f, 0.47f, 3.87f}, 142312, 5107, 308448},
    {{3.30f, 1.45f, 6.57f}, {3.30f, 0.47f, 4.40f}, 124768, 6364, 311040}
}};
constexpr int batches = 240, warmup = 45;

bool load_layouts(const char *path, std::array<std::vector<int32_t>, 2> &layouts) {
    std::ifstream input(path);
    if (!input) return false;
    while (input >> std::ws && !input.eof()) {
        int map;
        std::array<int32_t, 7> box;
        if (!(input >> map) || map < 1 || map > 2) return false;
        for (auto &value : box)
            if (!(input >> value)) return false;
        auto &layout = layouts[map - 1];
        layout.insert(layout.end(), box.begin(), box.end());
    }
    return input.eof() && !layouts[0].empty() && !layouts[1].empty();
}

void checkpoint(const World &world, int map, int tick) {
    const auto state = world.status();
    uint64_t occupancy = 14695981039346656037ull;
    const int refinement = int(state.refinement);
    for (int y = 0; y < 36 * refinement; ++y)
        for (int z = 0; z < 56 * refinement; ++z)
            for (int x = 0; x < 72 * refinement; ++x) {
                const Vec3 position = Vec3(x + 0.5f, y + 0.5f, z + 0.5f) * state.cell_size;
                occupancy = (occupancy ^ uint64_t(world.sample(position))) * 1099511628211ull;
            }
    const auto peak = world.sand_impact_peak(World::Stone).value();
    std::cout << "map=" << map << " tick=" << tick << " material_hash=" << std::hex << occupancy << std::dec
        << " sand=" << state.sand << " gravel=" << state.gravel << " sand_mass=" << state.sand_mass
        << " sand_volume=" << state.sand_volume << " water=" << state.water << " active_water=" << state.active_water
        << " moved_sand=" << state.moved << " moved_gravel=" << state.moved_gravel
        << " stone_fractures=" << state.contact_stone_fractures << " peak_speed=" << std::hexfloat << peak.speed
        << " peak_impulse=" << peak.impulse << std::defaultfloat << '\n';
}

bool replay(World &world, int map, const std::vector<int32_t> &layout, bool check, int count) {
    const Scene &scene = scenes[map - 1];
    if (!world.load_boxes(layout.data(), layout.size())) return false;
    const auto initial = world.status();
    if (initial.sand != scene.sand || initial.gravel != scene.gravel || initial.water != scene.water) {
        std::cerr << "Map " << map << " failed its initial material inventory check.\n";
        return false;
    }
    std::vector<double> samples;
    samples.reserve(count);
    for (int frame = 0; frame < count; ++frame) {
        const double t = std::clamp((frame - 72) / 93.0, 0.0, 1.0);
        world.set_field(scene.start.lerp(scene.target, float(t * t * (3.0 - 2.0 * t))), frame >= 34 && frame < 180);
        const auto start = Clock::now();
        world.step();
        world.step();
        if (check) {
            // Sample settling, the start and end of the pull, travel and the released pile.
            if (frame == 0 || frame == 34 || frame == 119 || frame == 179 || frame == count - 1)
                checkpoint(world, map, (frame + 1) * 2);
        } else if (frame >= warmup) {
            samples.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
        }
    }
    const auto final = world.status();
    if (initial.sand_mass != final.sand_mass || initial.water_volume != final.water_volume) {
        std::cerr << "Map " << map << " did not conserve sand or water.\n";
        return false;
    }
    if (!check) {
        std::sort(samples.begin(), samples.end());
        const auto percentile = [&](int percent) { return samples[(samples.size() * percent + 99) / 100 - 1]; };
        std::cout << "map=" << map << " steps_per_batch=2 samples=" << samples.size()
            << " p50_ms=" << percentile(50) << " p95_ms=" << percentile(95) << '\n';
    }
    return true;
}
}

int main(int argc, char **argv) {
    const std::string_view mode = argc > 2 ? argv[2] : "";
    int count = batches;
    if (argc == 4) {
        const std::string_view limit(argv[3]);
        const auto parsed = std::from_chars(limit.data(), limit.data() + limit.size(), count);
        if (parsed.ec != std::errc{} || parsed.ptr != limit.data() + limit.size()) count = 0;
    }
    if ((argc != 3 && argc != 4) || (mode != "check" && mode != "timing")
            || count < 1 || count > batches || (mode == "timing" && count <= warmup)) {
        std::cerr << "Usage: somevoxel_sand_world_bench sand_world_3cm.txt check|timing [batches, 1..240; timing >45]\n";
        return 2;
    }
    std::array<std::vector<int32_t>, 2> layouts;
    if (!load_layouts(argv[1], layouts)) {
        std::cerr << "Expected map ID followed by seven integer box fields in " << argv[1] << '\n';
        return 2;
    }
    auto world = std::make_unique<World>();
    const auto config = world->status();
    if (config.refinement != 6 || std::abs(120.f * config.step_seconds - 1.f) > 1e-6f || config.bulk_water_hz != 90) {
        std::cerr << "Use the 3 cm, 120 Hz physics, 90 Hz water configuration.\n";
        return 2;
    }
    for (int map = 1; map <= 2; ++map)
        if (!replay(*world, map, layouts[map - 1], mode == "check", count)) return 1;
    return 0;
}
