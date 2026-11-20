#include "core/world.hpp"
#include "core/detail/voxel_water_spans.hpp"

#include <array>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string_view>

namespace {
using somevoxel::Cell;
using somevoxel::HeadWorkers;
using somevoxel::Vec3;
using somevoxel::World;
using Clock = std::chrono::steady_clock;

constexpr int domain_count = 4;
constexpr int batches = 60;
constexpr int32_t boxes[][7] = {
    {0, 0, 0, 26, 1, 23, World::Stone},
    {2, 2, 2, 9, 6, 8, World::Sand},
    {3, 6, 3, 8, 8, 7, World::Sand},
    {13, 1, 0, 17, 13, 15, World::Stone},
};
constexpr std::array<Cell, domain_count> origins{{{2, 0, 2}, {40, 0, 2}, {2, 0, 31}, {40, 0, 31}}};

struct Result {
    uint64_t occupancy = 1469598103934665603ull;
    int64_t mass = 0, volume = 0, launched = 0, fractures = 0;
    int64_t dense_bytes = 0;
    float speed = 0.f, impulse = 0.f;
    double p95_ms = 0.;
};

std::vector<int32_t> layout_for(int first, int last, int refinement) {
    std::vector<int32_t> layout;
    for (int i = first; i < last; ++i)
        for (const auto &box : boxes) {
            for (int j = 0; j < 6; ++j) layout.push_back((box[j] + origins[i][j % 3]) * refinement);
            layout.push_back(box[6]);
        }
    return layout;
}

int launch(World &world, int domain) {
    const Vec3 focus = (Vec3(origins[domain]) + Vec3(7.f, 3.f, 5.f)) * 0.18f;
    world.set_field(focus, true);
    const int released = world.launch_free_sand(focus, {1.f, 0.f, 0.f}, 3.f, 1.f);
    world.set_field(focus, false);
    return released;
}

void collect(Result &result, const World &world) {
    const auto state = world.status();
    result.mass += state.sand_mass;
    result.volume += state.sand_volume;
    result.fractures += state.contact_stone_fractures;
    result.dense_bytes += state.dense_grid_bytes;
    const auto peak = world.sand_impact_peak(World::Stone);
    if (peak->speed > result.speed) {
        result.speed = peak->speed;
        result.impulse = peak->impulse;
    }
}

void hash_domain(Result &result, const World &world, int domain, int refinement, float pitch) {
    const Cell lo = (origins[domain] - Cell(2, 0, 2)) * refinement;
    const Cell hi = (origins[domain] + Cell(32, 18, 25)) * refinement;
    for (int y = lo.y; y < hi.y; ++y)
        for (int z = lo.z; z < hi.z; ++z)
            for (int x = lo.x; x < hi.x; ++x) {
                const auto kind = world.sample(Vec3(x + 0.5f, y + 0.5f, z + 0.5f) * pitch);
                result.occupancy = (result.occupancy ^ uint64_t(kind)) * 1099511628211ull;
            }
}

Result replay(int workers) {
    const bool partitioned = workers > 0;
    const int count = partitioned ? domain_count : 1;
    std::array<std::unique_ptr<World>, domain_count> worlds;
    std::array<int, domain_count> launched{};
    int refinement = 0, steps_per_sample = 0;
    float pitch = 0.f;
    for (int i = 0; i < count; ++i) {
        worlds[i] = std::make_unique<World>();
        const auto state = worlds[i]->status();
        refinement = int(state.refinement);
        pitch = state.cell_size;
        steps_per_sample = int(std::lround(1.f / (60.f * state.step_seconds)));
        const auto layout = layout_for(i, partitioned ? i + 1 : domain_count, refinement);
        if (!worlds[i]->load_boxes(layout.data(), layout.size())) std::abort();
        worlds[i]->set_cast_profile(1.f, 2.f);
    }
    HeadWorkers pool(std::max(1, workers));
    Result result;
    std::vector<double> times;
    for (int batch = 0; batch < batches; ++batch) {
        if (batch == 15)
            for (int domain = 0; domain < domain_count; ++domain)
                launched[domain] = launch(*worlds[partitioned ? domain : 0], domain);
        const auto start = Clock::now();
        pool.run([&](int worker) {
            for (int i = worker; i < count; i += pool.count)
                for (int step = 0; step < steps_per_sample; ++step) worlds[i]->step();
        });
        if (batch >= 4)
            times.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
    }
    std::sort(times.begin(), times.end());
    result.p95_ms = times[size_t(std::ceil(times.size() * 0.95)) - 1];
    for (int n : launched) result.launched += n;
    for (int i = 0; i < count; ++i) collect(result, *worlds[i]);
    for (int i = 0; i < domain_count; ++i) hash_domain(result, *worlds[partitioned ? i : 0], i, refinement, pitch);
    return result;
}

void report(int workers, const Result &result) {
    std::cout << "workers=" << workers << " p95_ms=" << result.p95_ms << " sand_mass=" << result.mass
        << " sand_volume=" << result.volume << " launched=" << result.launched
        << " fractures=" << result.fractures << " peak_speed=" << result.speed
        << " peak_impulse=" << result.impulse << " occupancy=" << result.occupancy
        << " dense_bytes=" << result.dense_bytes << std::endl;
}

bool same_state(const Result &a, const Result &b) {
    return a.mass == b.mass && a.volume == b.volume && a.launched == b.launched && a.fractures == b.fractures &&
        a.speed == b.speed && a.impulse == b.impulse && a.occupancy == b.occupancy;
}
}

int main(int argc, char **argv) {
    const bool require_merged_parity = argc == 2 && std::string_view(argv[1]) == "--require-merged-parity";
    if (argc > 1 && !require_merged_parity) return 2;
    std::cout << std::fixed << std::setprecision(6);
    const auto merged = replay(0);
    report(0, merged);
    const auto reference = replay(1);
    report(1, reference);
    const bool merged_parity = same_state(merged, reference);
    std::cout << "merged_parity=" << std::boolalpha << merged_parity << '\n';
    bool valid = merged.mass == reference.mass && reference.launched > 0 && reference.impulse > 0.f &&
        reference.fractures > 0 && (!require_merged_parity || merged_parity);
    for (int workers : {2, 4}) {
        const auto result = replay(workers);
        report(workers, result);
        valid = valid && same_state(result, reference);
    }
    return valid ? 0 : 1;
}
