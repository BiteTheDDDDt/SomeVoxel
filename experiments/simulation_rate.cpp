#include "core/world.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
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
constexpr int32_t narrow_outlet[][7] = {
    {22, 1, 39, 24, 4, 40, World::Stone},
    {22, 1, 42, 24, 4, 43, World::Stone},
};

void append_box(std::vector<int32_t> &layout, const int32_t (&box)[7], int refinement, bool mirrored = false) {
    std::array<int32_t, 6> transformed;
    std::copy_n(box, transformed.size(), transformed.begin());
    if (mirrored) {
        transformed[0] = width - box[3];
        transformed[3] = width - box[0];
    }
    for (int axis = 0; axis < 6; ++axis) layout.push_back(transformed[axis] * refinement);
    layout.push_back(box[6]);
}

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

int launch_budget(World &world, int refinement, double cell_volume) {
    constexpr int32_t supply[7] = {15, 1, 15, 21, 7, 21, World::Sand};
    constexpr Vec3 focus(3.24f, 0.72f, 3.24f);
    constexpr double reference_kg = 131.22; // 900 loose 4.5 cm grains at 1600 kg/m³.
    const double grain_kg = 1600.0 * cell_volume;
    std::vector<int32_t> layout;
    append_box(layout, supply, refinement);
    if (!world.load_boxes(layout.data(), layout.size())) return 1;
    const auto initial = world.status();
    world.set_field(focus, true);
    const auto preview = world.free_sand_throw(focus, 3.f, 1.f);
    const int released = world.launch_free_sand(focus, {1.f, 0.f, 0.f}, 3.f, 1.f);
    const bool sufficient = double(initial.sand_mass) * grain_kg > reference_kg;
    const bool conserved = world.status().sand_mass == initial.sand_mass;
    const bool within_budget = preview.mass <= reference_kg + 0.0001 &&
        preview.mass >= reference_kg - grain_kg - 0.0001;
    std::cout << "{\"scenario\":\"launch-budget\",\"refinement\":" << refinement
        << ",\"target_mass_kg\":" << reference_kg << ",\"cast_mass_kg\":" << preview.mass
        << ",\"cast_speed_m_s\":" << preview.speed << ",\"cast_impulse_ns\":" << double(preview.mass) * preview.speed
        << ",\"grain_mass_kg\":" << grain_kg << ",\"available_mass_kg\":" << initial.sand_mass * grain_kg
        << ",\"launched\":" << released << ",\"preview_count\":" << preview.count
        << ",\"conserved\":" << conserved << ",\"mass_within_one_grain\":" << within_budget << "}\n";
    return !(sufficient && conserved && within_budget && released > 0 && released == preview.count);
}

int wall_sweep(World &world, int refinement, float cell_size) {
    const int wall_x = 23 * refinement;
    const int shot_y = 5 * refinement, shot_z = 19 * refinement;
    const Cell grain(wall_x - 1, shot_y, shot_z);
    const Cell bounds = Cell(width, height, depth) * refinement;
    std::vector<int32_t> layout;
    append_box(layout, boxes[0], refinement);
    layout.insert(layout.end(), {grain.x, grain.y, grain.z, grain.x + 1, grain.y + 1, grain.z + 1, World::Sand});
    const auto fire = [&]() {
        if (!world.load_boxes(layout.data(), layout.size())) return false;
        const Vec3 focus = (Vec3(grain) + Vec3(0.5f, 0.5f, 0.5f)) * cell_size;
        world.set_cast_profile(1.f, 2.f);
        world.set_field(focus, true);
        if (world.launch_free_sand(focus, {1.f, 0.f, 0.f}, 3.f, 1.f) != 1) return false;
        world.set_field(focus, false);
        world.step();
        return world.status().sand_mass == 1;
    };
    const Cell far_side(wall_x + 1, 0, 0);
    if (!fire()) return 1;
    const int free_crossings = world.range_count(World::Sand, far_side, bounds);
    // One anchored voxel spans the fast path. The control must cross the
    // wall's entire position before the blocked shot can exercise the sweep.
    layout.insert(layout.end(), {wall_x, refinement, shot_z - refinement, wall_x + 1,
        shot_y + refinement, shot_z + refinement, World::Stone});
    if (!fire()) return 1;
    const int blocked_crossings = world.range_count(World::Sand, far_side, bounds);
    const bool hit_face = world.range_count(World::Sand, grain, grain + Cell(1, 1, 1)) == 1;
    const bool passed = free_crossings == 1 && blocked_crossings == 0 && hit_face;
    std::cout << "{\"scenario\":\"wall-sweep\",\"refinement\":" << refinement
        << ",\"wall_width_m\":" << cell_size << ",\"control_crossings\":" << free_crossings
        << ",\"blocked_crossings\":" << blocked_crossings << ",\"stopped_at_face\":" << hit_face
        << ",\"no_tunneling\":" << passed << "}\n";
    return !passed;
}
}

int main(int argc, char **argv) {
    const std::string_view scenario = argc > 1 ? argv[1] : "mixed";
    if (scenario != "mixed" && scenario != "sand" && scenario != "water" && scenario != "water-mirrored" &&
            scenario != "water-narrow" && scenario != "water-narrow-mirrored" && scenario != "launch-budget" &&
            scenario != "wall-sweep") return 2;
    const bool narrow = scenario == "water-narrow" || scenario == "water-narrow-mirrored";
    const bool mirrored = scenario == "water-mirrored" || scenario == "water-narrow-mirrored";
    const bool has_sand = scenario == "mixed" || scenario == "sand";
    const bool has_water = scenario != "sand";
    auto storage = std::make_unique<World>();
    World &world = *storage;
    const auto setup = world.status();
    const int refinement = int(setup.refinement);
    const int hz = int(std::lround(1.f / setup.step_seconds));
    const double cell = setup.cell_size;
    const double cell_volume = cell * cell * cell;
    std::cout << std::fixed << std::setprecision(6) << std::boolalpha;
    if (scenario == "launch-budget") return launch_budget(world, refinement, cell_volume);
    if (scenario == "wall-sweep") return wall_sweep(world, refinement, setup.cell_size);

    std::vector<int32_t> layout;
    for (const auto &box : boxes) {
        if ((box[6] == World::Sand && !has_sand) || (box[6] == World::Water && !has_water)) continue;
        append_box(layout, box, refinement, mirrored);
    }
    if (narrow)
        for (const auto &box : narrow_outlet) append_box(layout, box, refinement, mirrored);
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
    const double water_out_l = escaped * cell_volume * 1000.;
    const double water_net_flow_l_s = water_out_l / duration;
    const bool conserved = final.sand_mass == initial.sand_mass && final.water_volume == initial.water_volume
        && rock_total(world, refinement) == rock_before;
#ifdef SOMEVOXEL_EXPERIMENTAL_WATER_SWEEP_PHASE
    constexpr bool water_phase = true;
#else
    constexpr bool water_phase = false;
#endif
    std::cout << "{\"scenario\":\"" << scenario << "\",\"refinement\":" << refinement << ",\"sim_hz\":" << hz
        << ",\"water_hz\":" << setup.bulk_water_hz << ",\"water_phase\":" << water_phase << ",\"simulated_s\":" << duration
        << ",\"p50_ms\":" << percentile(samples, 0.50) << ",\"p95_ms\":" << percentile(samples, 0.95)
        << ",\"p99_ms\":" << percentile(samples, 0.99) << ",\"conserved\":" << conserved
        << ",\"sand_mass_cells\":" << final.sand_mass << ",\"water_cells\":" << final.water_volume
        << ",\"water_out_m3\":" << escaped * cell_volume << ",\"water_out_l\":" << water_out_l
        << ",\"water_net_flow_l_s\":" << water_net_flow_l_s << ",\"launched\":" << launched
        << ",\"cast_mass_kg\":" << held.mass << ",\"cast_speed_m_s\":" << held.speed
        << ",\"cast_impulse_ns\":" << double(held.mass) * held.speed
        << ",\"cast_center\":[" << held.position.x << ',' << held.position.y << ',' << held.position.z << ']'
        << ",\"peak_speed_m_s\":" << (impact ? impact->speed : 0.f)
        << ",\"peak_impulse_ns\":" << (impact ? impact->impulse : 0.f)
        << ",\"peak_impulse_ns_m2\":" << (impact ? impact->impulse / (cell * cell) : 0.)
        << ",\"fractured_m3\":" << final.contact_stone_fractures * cell_volume
        << ",\"fractured_l\":" << final.contact_stone_fractures * cell_volume * 1000. << "}\n";
    return !conserved || (has_sand && (launched == 0 || launched != held.count || !impact)) ||
        (has_water && water_net_flow_l_s <= 0.);
}
