#include "core/world.hpp"
#include "core/detail/voxel_water_spans.hpp"

#include <array>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string_view>

namespace somevoxel {
class SweepReplay {
public:
    using Worlds = std::array<std::unique_ptr<World>, 4>;
    using Accepted = std::array<int, 4>;
    struct Admission {
        int candidates;
        Accepted counts;
        Cell first_rejected;
    };

    static void gather(World &world) {
        world.replay_candidates.clear();
        world.replay_pressure_order.clear();
        world.replay_order = 0;
        world.record_replay_candidates = true;
        world.advance_step();
        world.integrate_contact_pressure();
        world.record_replay_candidates = false;
    }

    static Admission admit(Worlds &worlds) {
        struct Event {
            World::SweepCandidate candidate;
            int domain;
        };
        std::vector<Event> events;
        for (int domain = 0; domain < int(worlds.size()); ++domain)
            for (const auto &candidate : worlds[domain]->replay_candidates)
                events.push_back({candidate, domain});
        std::stable_sort(events.begin(), events.end(), [](const Event &a, const Event &b) {
            return a.candidate.order < b.candidate.order;
        });
        constexpr size_t budget = World::compaction_job_limit;
        Accepted accepted{};
        for (size_t i = 0; i < std::min(budget, events.size()); ++i) {
            const auto &[event, domain] = events[i];
            auto &world = *worlds[domain];
            world.compaction_jobs.emplace_back(event.key, event.work);
            if (event.pressure)
                world.contact_compression.at(event.key).retry_after = world.tick + std::max(1, World::simulation_hz / 15);
            ++accepted[domain];
        }
        const Cell first_rejected = events.size() > budget ? World::decode(events[budget].candidate.contact) : Cell(-1, -1, -1);
        return {int(events.size()), accepted, first_rejected};
    }

    static void finish(World &world) { world.finish_step(); }

    static void enable(World &world) { world.replay_track_gravel = true; }

    static void order_gravel(Worlds &worlds, std::vector<uint64_t> &global_order) {
        std::vector<World::ReplayGravelEvent> changes;
        for (auto &world : worlds) {
            changes.insert(changes.end(), world->replay_gravel_events.begin(), world->replay_gravel_events.end());
            world->replay_gravel_events.clear();
        }
        if (changes.empty()) return;
        std::stable_sort(changes.begin(), changes.end(), [](auto a, auto b) { return a.order < b.order; });
        // Recreate the serial gravel vector's swap-erases and appends.
        // Filtering that order gives each worker its next integration order.
        for (auto event : changes) {
            if (event.erase) {
                const auto item = std::find(global_order.begin(), global_order.end(), event.identity);
                if (item == global_order.end()) std::abort();
                *item = global_order.back();
                global_order.pop_back();
            } else {
                global_order.push_back(event.identity);
            }
        }
        std::unordered_map<uint64_t, unsigned> ranks;
        for (unsigned i = 0; i < global_order.size(); ++i) ranks.emplace(global_order[i], i);
        for (auto &world : worlds) {
            std::vector<int> old_ids;
            for (int id = 0; id < int(world->gravel.size()); ++id) {
                old_ids.push_back(id);
                world->replay_gravel_ranks[id] = ranks.at(world->replay_gravel_ids[id]);
            }
            std::sort(old_ids.begin(), old_ids.end(), [&](int a, int b) {
                return world->replay_gravel_ranks[a] < world->replay_gravel_ranks[b];
            });
            std::vector<World::Grain> ordered;
            decltype(world->packed_strain) strains;
            std::vector<uint64_t> identities;
            std::vector<unsigned> ordered_ranks;
            ordered.reserve(old_ids.size());
            for (int old : old_ids) {
                const int next = int(ordered.size());
                const auto &grain = world->gravel[old];
                ordered.push_back(grain);
                identities.push_back(world->replay_gravel_ids[old]);
                ordered_ranks.push_back(world->replay_gravel_ranks[old]);
                World::each_cube(grain.position, grain.extent, [&](int i) { world->voxels[i].occupant = next; });
                if (auto strain = world->packed_strain.find(old); strain != world->packed_strain.end())
                    strains.emplace(next, strain->second);
            }
            world->gravel.swap(ordered);
            world->replay_gravel_ids.swap(identities);
            world->replay_gravel_ranks.swap(ordered_ranks);
            world->packed_strain.swap(strains);
        }
    }

    static std::vector<uint8_t> terrain(const World &world) {
        std::vector<uint8_t> kinds;
        kinds.reserve(world.voxels.size());
        for (const auto &voxel : world.voxels) kinds.push_back(voxel.kind);
        return kinds;
    }

    static int compare(const World &merged, const Worlds &worlds, const std::vector<uint8_t> &terrain) {
        int differences = 0;
        for (size_t i = 0; i < terrain.size(); ++i) {
            const auto baseline = terrain[i];
            auto composed = baseline;
            bool overlap = false;
            for (const auto &world : worlds) {
                const auto kind = uint8_t(world->voxels[i].kind);
                if (kind == baseline) continue;
                if (composed != baseline) overlap = true;
                composed = kind;
            }
            differences += overlap || composed != merged.voxels[i].kind;
        }
        return differences;
    }
};
}

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

bool shared_budget_replay(int workers) {
    using somevoxel::SweepReplay;
    auto merged = std::make_unique<World>();
    SweepReplay::Worlds worlds;
    const auto profile = merged->status();
    const int refinement = int(profile.refinement);
    if (refinement != 6 || std::abs(profile.step_seconds - 1.f / 120.f) > 1e-6f) {
        std::cerr << "This parity fixture uses 3 cm cells at 120 Hz.\n";
        return false;
    }
    std::vector<int32_t> terrain_layout;
    for (const auto &origin : origins)
        for (const auto &box : boxes) {
            if (box[6] != World::Stone) continue;
            for (int i = 0; i < 6; ++i) terrain_layout.push_back((box[i] + origin[i % 3]) * refinement);
            terrain_layout.push_back(box[6]);
        }
    if (!merged->load_boxes(terrain_layout.data(), terrain_layout.size())) return false;
    const auto terrain_kinds = SweepReplay::terrain(*merged);
    const auto whole = layout_for(0, domain_count, refinement);
    if (!merged->load_boxes(whole.data(), whole.size())) return false;
    merged->set_cast_profile(1.f, 2.f);
    for (int domain = 0; domain < domain_count; ++domain) {
        worlds[domain] = std::make_unique<World>();
        auto layout = terrain_layout;
        const auto contents = layout_for(domain, domain + 1, refinement);
        for (size_t i = 0; i < contents.size(); i += 7)
            if (contents[i + 6] == World::Sand)
                layout.insert(layout.end(), contents.begin() + i, contents.begin() + i + 7);
        if (!worlds[domain]->load_boxes(layout.data(), layout.size())) return false;
        worlds[domain]->set_cast_profile(1.f, 2.f);
        SweepReplay::enable(*worlds[domain]);
    }
    HeadWorkers pool(workers);
    std::vector<uint64_t> gravel_order;
    int first_volume_difference = 0;
    std::vector<double> samples;
    SweepReplay::Admission launch_admission{};
    bool valid = true;
    for (int tick = 1; tick <= 120; ++tick) {
        if (tick == 31)
            for (int domain = 0; domain < domain_count; ++domain)
                valid = (launch(*merged, domain) == launch(*worlds[domain], domain)) && valid;
        merged->step();
        const auto start = Clock::now();
        pool.run([&](int worker) {
            for (int domain = worker; domain < domain_count; domain += pool.count) SweepReplay::gather(*worlds[domain]);
        });
        const auto admission = SweepReplay::admit(worlds);
        pool.run([&](int worker) {
            for (int domain = worker; domain < domain_count; domain += pool.count) SweepReplay::finish(*worlds[domain]);
        });
        SweepReplay::order_gravel(worlds, gravel_order);
        samples.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
        if (tick == 31) launch_admission = admission;
        int64_t volume = 0;
        for (const auto &world : worlds) volume += world->status().sand_volume;
        if (!first_volume_difference && volume != merged->status().sand_volume) first_volume_difference = tick;
    }
    Result expected, actual;
    collect(expected, *merged);
    for (const auto &world : worlds) collect(actual, *world);
    const int differences = SweepReplay::compare(*merged, worlds, terrain_kinds);
    std::vector<double> frames;
    for (size_t i = 8; i < samples.size(); i += 2) frames.push_back(samples[i] + samples[i + 1]);
    std::sort(frames.begin(), frames.end());
    const auto &[candidates, accepted, rejected] = launch_admission;
    std::cout << "workers=" << workers << " tick31_candidates=" << candidates << " admissions=";
    for (int n : accepted) std::cout << ' ' << n;
    std::cout << " first_rejected=" << rejected.x << ',' << rejected.y << ',' << rejected.z
        << " first_volume_difference=" << first_volume_difference
        << " fractures=" << expected.fractures << '/' << actual.fractures
        << " grid_cells=" << terrain_kinds.size() << " full_grid_differences=" << differences << " p95_ms=" << frames[size_t(std::ceil(frames.size() * .95)) - 1]
        << std::endl;
    return valid && candidates == 817 && accepted == SweepReplay::Accepted{8, 12, 24, 20} &&
        rejected == Cell(57, 14, 51) && first_volume_difference == 0 &&
        expected.mass == actual.mass && expected.impulse == actual.impulse &&
        expected.fractures == 180 && expected.fractures == actual.fractures && differences == 0;
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
    if (argc == 2 && std::string_view(argv[1]) == "--shared-budget") {
        const bool one = shared_budget_replay(1);
        const bool four = shared_budget_replay(4);
        return one && four ? 0 : 1;
    }
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
