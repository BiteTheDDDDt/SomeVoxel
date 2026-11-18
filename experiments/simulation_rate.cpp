#include "core/world.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string_view>

using somevoxel::Cell;
using somevoxel::Vec3;
using somevoxel::World;
using Clock = std::chrono::steady_clock;

namespace {
constexpr int input_hz = 30, batch_hz = 60, duration = 5;
constexpr int charge_start = 15, release = 66;
constexpr int width = 72, height = 36, depth = 56, outlet_x = 24;
constexpr Vec3 field_start(3.24f, 1.12f, 3.24f);
constexpr Vec3 field_end(4.66f, 1.12f, 3.24f);
constexpr int32_t boxes[][7] = {
    {0, 0, 0, 72, 1, 56, World::Stone},
    {12, 1, 12, 24, 4, 24, World::Sand},
    {15, 4, 15, 21, 6, 21, World::Sand},
    {32, 1, 12, 33, 13, 25, World::Stone},
    {6, 1, 32, 24, 8, 34, World::Stone},
    {6, 1, 48, 24, 8, 50, World::Stone},
    {6, 1, 34, 8, 8, 48, World::Stone},
    {22, 1, 34, 24, 8, 39, World::Stone},
    {22, 1, 43, 24, 8, 48, World::Stone},
    {22, 4, 39, 24, 8, 43, World::Stone},
    {8, 1, 34, 22, 8, 48, World::Water},
};

int rock_total(const World &world, int refinement) {
    const Cell bounds = Cell(width, height, depth) * refinement;
    int total = 0;
    for (auto kind : {World::Stone, World::Rock, World::Bedrock})
        total += world.range_count(kind, {}, bounds);
    return total;
}

double percentile(const std::vector<double> &sorted, double rank) {
    return sorted[size_t(std::ceil(sorted.size() * rank)) - 1];
}
}

int main(int argc, char **argv) {
    const std::string_view scenario = argc > 1 ? argv[1] : "mixed";
    if (scenario != "mixed" && scenario != "sand" && scenario != "water" && scenario != "water-mirrored") return 2;
    const bool mirrored = scenario == "water-mirrored";
    const bool has_sand = scenario == "mixed" || scenario == "sand";
    const bool has_water = scenario != "sand";
    World world;
    const auto setup = world.status();
    const int refinement = int(setup.refinement);
    const int hz = int(std::lround(1.f / setup.step_seconds));
    const float cell_volume = setup.cell_size * setup.cell_size * setup.cell_size;
#ifdef SOMEVOXEL_EXPERIMENTAL_WATER_HZ
    const int water_hz = SOMEVOXEL_EXPERIMENTAL_WATER_HZ;
#else
    const int water_hz = refinement < 4 ? hz : hz / 2;
#endif
    std::vector<int32_t> layout;
    for (const auto &box : boxes) {
        if ((box[6] == World::Sand && !has_sand) || (box[6] == World::Water && !has_water)) continue;
        std::array<int32_t, 6> transformed;
        std::copy_n(box, transformed.size(), transformed.begin());
        if (mirrored) {
            transformed[0] = width - box[3];
            transformed[3] = width - box[0];
        }
        for (int axis = 0; axis < 6; ++axis) layout.push_back(transformed[axis] * refinement);
        layout.push_back(box[6]);
    }
    if (!world.load_boxes(layout.data(), layout.size())) return 1;
    const auto initial = world.status();
    const int rock_before = rock_total(world, refinement);
    world.set_cast_profile(0.85f, 1.75f);

    std::vector<double> samples;
    somevoxel::ThrowInfo held;
    int launched = 0;
    for (int batch = 0; batch < batch_hz * duration; ++batch) {
        const int frame = batch / (batch_hz / input_hz);
        if (has_sand && batch % (batch_hz / input_hz) == 0) {
            if (frame >= charge_start && frame < release) {
                const float charge = float(frame - charge_start) / (release - charge_start);
                const float t = std::clamp((charge - 0.25f) / 0.65f, 0.f, 1.f);
                world.set_charge_gather(charge);
                world.set_field(field_start.lerp(field_end, t * t * (3.f - 2.f * t)), true);
            } else if (frame == release) {
                held = world.free_sand_throw(field_end, 3.f, 1.f);
                launched = world.launch_free_sand(field_end, {1.f, 0.f, 0.f}, 3.f, 1.f);
                world.set_field(field_end, false);
                world.set_charge_gather(0.f);
            }
        }
        const auto start = Clock::now();
        for (int tick = batch * hz / batch_hz; tick < (batch + 1) * hz / batch_hz; ++tick)
            world.step();
        samples.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
    }
    std::sort(samples.begin(), samples.end());
    const auto final = world.status();
    const auto impact = world.sand_impact_peak(World::Stone);
    const Cell outlet_lo(mirrored ? 0 : outlet_x, 1, 0);
    const Cell outlet_hi(mirrored ? width - outlet_x : width, height, depth);
    const int escaped = world.range_count(World::Water, outlet_lo * refinement, outlet_hi * refinement);
    const bool conserved = final.sand_mass == initial.sand_mass && final.water_volume == initial.water_volume
        && rock_total(world, refinement) == rock_before;
#ifdef SOMEVOXEL_EXPERIMENTAL_WATER_SWEEP_PHASE
    constexpr bool water_phase = true;
#else
    constexpr bool water_phase = false;
#endif
    std::cout << std::fixed << std::setprecision(6) << std::boolalpha
        << "{\"scenario\":\"" << scenario << "\",\"refinement\":" << refinement << ",\"sim_hz\":" << hz
        << ",\"water_hz\":" << water_hz << ",\"water_phase\":" << water_phase
        << ",\"p50_ms\":" << percentile(samples, 0.50) << ",\"p95_ms\":" << percentile(samples, 0.95)
        << ",\"p99_ms\":" << percentile(samples, 0.99) << ",\"conserved\":" << conserved
        << ",\"sand_mass_cells\":" << final.sand_mass << ",\"water_cells\":" << final.water_volume
        << ",\"water_out_m3\":" << escaped * cell_volume << ",\"launched\":" << launched
        << ",\"cast_mass_kg\":" << held.mass << ",\"cast_speed_m_s\":" << held.speed
        << ",\"cast_center\":[" << held.position.x << ',' << held.position.y << ',' << held.position.z << ']'
        << ",\"peak_speed_m_s\":" << (impact ? impact->speed : 0.f)
        << ",\"peak_impulse_ns\":" << (impact ? impact->impulse : 0.f)
        << ",\"fractured_m3\":" << final.contact_stone_fractures * cell_volume << "}\n";
    return !conserved || (has_sand && (launched == 0 || !impact)) || (has_water && escaped == 0);
}
