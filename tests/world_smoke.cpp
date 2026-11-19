#include "core/world.hpp"
#include <iostream>

static bool water_head_selection() {
    const auto check = [](bool faster_gpu) {
        somevoxel::detail::WaterHeadSelection selector;
        const int cpu = faster_gpu ? 400 : 100, gpu = faster_gpu ? 100 : 400;
        for (int run = 0; run < 8; ++run) {
            const bool device = selector.choose_gpu(1000, 100, 0);
            if (device != bool(run & 1)) return false;
            // Cold-start cost and an isolated outlier cannot pick the winner.
            selector.measured(run < 2 || run == 6 || run == 7 ? 5000 : (device ? gpu : cpu));
        }
        for (int run = 0; run < 16; ++run) {
            if (selector.choose_gpu(1000, 100, 0) != faster_gpu) return false;
            selector.measured(1);
        }
        if (selector.choose_gpu(2200, 100, 0)) return false;
        selector.measured(cpu);
        if (!selector.choose_gpu(2200, 100, 0)) return false;
        selector.gpu_failed();
        for (int run = 0; run < 96; ++run) {
            if (selector.choose_gpu(2200, 100, 0)) return false;
            selector.measured(cpu);
        }
        if (selector.choose_gpu(2200, 100, 0)) return false;
        selector.measured(cpu);
        return selector.choose_gpu(2200, 100, 0);
    };
    return check(true) && check(false);
}

static bool communicating_vessels(somevoxel::World &world) {
    using somevoxel::World;
    const int stone = World::Stone, water = World::Water;
    const int32_t layout[] = {
        0, 0, 0, 208, 4, 144, stone,
        59, 4, 49, 60, 24, 57, stone,
        73, 4, 49, 74, 24, 57, stone,
        60, 4, 49, 73, 24, 50, stone,
        60, 4, 56, 73, 24, 57, stone,
        66, 4, 50, 67, 24, 56, stone,
        60, 4, 50, 66, 20, 56, water,
        // A separate reservoir keeps the fixture large enough for the
        // packed pressure preflight and must keep its original inventory.
        128, 4, 64, 129, 24, 129, stone,
        193, 4, 64, 194, 24, 129, stone,
        129, 4, 64, 193, 24, 65, stone,
        129, 4, 128, 193, 24, 129, stone,
        129, 4, 65, 193, 18, 128, water,
    };
    if (!world.load_boxes(layout, std::size(layout))) return false;
    const auto start = world.status();
    const auto chamber_volume = [&](int bottom) {
        return world.range_count(water, {67, bottom, 50}, {73, 24, 56});
    };
    for (int i = 0; i < 16; ++i) world.step();
    if (chamber_volume(4) != 0 || world.status().water_volume != start.water_volume) return false;
    const float pitch = start.cell_size;
    if (world.carve({66.f * pitch, 4.5f * pitch, 52.5f * pitch}, 0.92f * pitch) != 1) return false;
    for (int i = 0; i < 480; ++i) world.step();
    return chamber_volume(9) > 0 && world.status().water_volume == start.water_volume &&
        world.range_count(water, {129, 4, 65}, {193, 24, 128}) == 64 * 14 * 63;
}

static bool loaded_sand_fractures(somevoxel::World &world) {
    using somevoxel::World;
    const int32_t layout[] = {
        0, 0, 0, 140, 4, 140, World::Stone,
        70, 4, 48, 72, 32, 72, World::Stone,
    };
    if (!world.load_boxes(layout, std::size(layout))) return false;
    const float pitch = world.status().cell_size;
    const float mass = World::packed_particle_mass(2);
    const auto id = world.spawn_packed_particle({52.f * pitch, 15.f * pitch, 60.f * pitch},
        2, {mass * 22.f, 0.f, 0.f});
    if (!id) return false;
    const auto before = world.status();
    for (int i = 0; i < 24; ++i) world.step();
    const auto after = world.status();
    return !world.packed_particle_state(id) && after.sand == 8 &&
        after.sand_mass == before.sand_mass && after.sand_volume == before.sand_volume;
}

static bool loaded_sand_compacts(somevoxel::World &world) {
    using somevoxel::World;
    const int32_t layout[] = {
        0, 0, 0, 120, 4, 120, World::Stone,
        60, 4, 60, 62, 6, 62, World::Sand,
        62, 4, 60, 63, 6, 61, World::Sand,
        63, 4, 59, 66, 9, 64, World::Stone,
        59, 4, 59, 63, 7, 60, World::Stone,
        59, 4, 62, 63, 7, 63, World::Stone,
    };
    const float pitch = world.status().cell_size;
    const float mass = World::packed_particle_mass(2);
    for (const bool strong : {false, true}) {
        if (!world.load_boxes(layout, std::size(layout))) return false;
        const float speed = strong ? 22.f : 2.f;
        if (!world.spawn_packed_particle({58.5f * pitch, 5.f * pitch, 61.f * pitch},
                2, {mass * speed, 0.f, 0.f})) return false;
        const auto before = world.status();
        world.step();
        const auto after = world.status();
        if (after.sand_mass != before.sand_mass || after.packed_sand_blocks != int(strong) ||
                after.sand_volume != before.sand_volume - 2 * int(strong)) return false;
    }
    return true;
}

static bool sustained_compression(somevoxel::World &world) {
    using somevoxel::Vec3;
    using somevoxel::World;
    const int32_t layout[] = {
        0, 0, 0, 130, 4, 130, World::Stone,
        53, 4, 53, 66, 12, 66, World::Sand,
        66, 4, 51, 68, 16, 68, World::Stone,
        53, 4, 51, 66, 15, 53, World::Stone,
        53, 4, 66, 66, 15, 68, World::Stone,
    };
    const Vec3 focus = Vec3(65.5f, 8.5f, 59.5f) * world.status().cell_size;
    for (const bool form : {false, true}) {
        for (const bool strong : {false, true}) {
            if (!world.load_boxes(layout, std::size(layout))) return false;
            const auto initial = world.status();
            const float strength = strong ? 5.f : 0.1f;
            world.set_cast_profile(1.f, strength);
            if (form) world.set_particle_form(focus, {1.f, 0.f, 0.f}, World::Sand, 1, 0.72f, strength);
            else world.set_field(focus, true);
            world.set_charge_gather(1.f);
            bool compacted = false;
            for (int i = 0; i < 180; ++i) {
                world.step();
                const auto state = world.status();
                if (state.sand_mass != initial.sand_mass) return false;
                if (state.packed_sand_blocks > 0 && state.sand_volume < initial.sand_volume) {
                    if (i < 8) return false;
                    compacted = true;
                    break;
                }
            }
            if (compacted != strong) return false;
        }
    }
    return true;
}

static bool packed_sand_forms(somevoxel::World &world) {
    using somevoxel::Vec3;
    using somevoxel::World;
    const int32_t floor[] = {0, 0, 0, 120, 4, 120, World::Stone};
    for (int extent : {2, 4, 8}) {
        if (!world.load_boxes(floor, std::size(floor))) return false;
        const Vec3 source(3.f, 1.5f, 3.f);
        const auto handle = world.spawn_packed_particle(source, extent);
        if (!handle) return false;
        const auto initial = world.packed_particle_state(handle);
        world.set_particle_form(source, {1, 0, 0}, World::Sand, 1, 2.4f, 4.f);
        world.step();
        if (world.particle_form_count() != 1) return false;
        world.set_particle_form(source + Vec3(0, 0.65f, 0), {1, 0, 0}, World::Sand, 1, 2.4f, 4.f);
        for (int tick = 0; tick < 42; ++tick) world.step();
        const auto held = world.packed_particle_state(handle);
        const auto preview = world.particle_form_throw(1.f, 1.f);
        if (!held || held->position.y < initial->position.y + 0.09f || preview.blocks != 1 || preview.count != 1 ||
                std::abs(preview.mass - World::packed_particle_mass(extent)) > 0.0001f) return false;
        if (world.launch_particle_form({1, 0, 0}, 1.f, 1.f) != 1 ||
                world.packed_particle_state(handle)->velocity.x <= held->velocity.x) return false;
    }

    const int32_t pile[] = {
        0, 0, 0, 120, 4, 120, World::Stone,
        60, 4, 60, 62, 6, 62, World::Sand,
        62, 4, 60, 63, 6, 61, World::Sand,
    };
    if (!world.load_boxes(pile, std::size(pile))) return false;
    const Vec3 center = Vec3(61, 5, 61) * world.status().cell_size;
    world.set_particle_form(center, {1, 0, 0}, World::Sand, 1, 1.4f, 2.f);
    world.step();
    const auto before = world.particle_form_throw();
    if (before.count != 10 || world.compact(center, 0.14f, 2) != 1) return false;
    const auto packed = world.particle_form_throw();
    if (packed.count != 1 || packed.blocks != 1 || std::abs(packed.mass - before.mass) > 0.0001f) return false;
    if (world.impact(center, 0.14f, 1.f) != 1) return false;
    const auto loose = world.particle_form_throw();
    return loose.count == 10 && loose.blocks == 0 && std::abs(loose.mass - before.mass) <= 0.0001f;
}

int main() {
    if (!water_head_selection()) {
        std::cerr << "Water head backend selection or GPU failure recovery failed\n";
        return 1;
    }
    using somevoxel::World;
    World world;
    const int32_t floor[] = {0, 0, 0, 32, 1, 32, World::Stone};
    if (!world.load_boxes(floor, sizeof(floor) / sizeof(floor[0]))) return 1;
    const auto dry_grid = world.status().dense_grid_bytes;
    world.step();
    world.step();
    if (world.status().dense_grid_bytes != dry_grid) {
        std::cerr << "A dry step allocated liquid pressure storage\n";
        return 1;
    }
    const int32_t layout[] = {
        0, 0, 0, 32, 1, 32, World::Stone,
        4, 8, 4, 8, 12, 8, World::Sand,
        16, 8, 4, 20, 12, 8, World::Water,
    };
    if (!world.load_boxes(layout, sizeof(layout) / sizeof(layout[0]))) {
        std::cerr << "Could not load the test scene\n";
        return 1;
    }
    const auto initial = world.status();
    const int ticks = int(1.f / initial.step_seconds);
    for (int tick = 0; tick < ticks; ++tick) world.step();
    const auto current = world.status();
    const int falling_sand = world.range_count(World::Sand, {0, 1, 0}, {32, 8, 32});
    const int falling_water = world.range_count(World::Water, {0, 1, 0}, {32, 8, 32});
    if (initial.sand_mass != 64 || initial.water_volume != 64 ||
            current.sand_mass != initial.sand_mass || current.water_volume != initial.water_volume ||
            falling_sand == 0 || falling_water == 0) {
        std::cerr << "Gravity or material conservation failed\n";
        return 1;
    }
    std::cout << "SomeVoxel Core preserved sand and water after settling\n";

    constexpr int extent = 4;
    const float mass = World::packed_particle_mass(extent);
    const auto target = world.spawn_packed_particle({1.02f, 1.f, 0.45f}, extent);
    const auto projectile = world.spawn_packed_particle({0.55f, 1.f, 0.45f}, extent, {mass * 5.f, 0, 0});
    if (!target || !projectile || world.spawn_packed_particle({0.55f, 1.f, 0.45f}, extent) != 0) {
        std::cerr << "Packed particle insertion did not respect occupancy\n";
        return 1;
    }
    const auto before_contact = world.packed_particle_state(target);
    for (int tick = 0; tick < 24; ++tick) world.step();
    const auto after_contact = world.packed_particle_state(target);
    if (!before_contact || !after_contact || after_contact->position.x <= before_contact->position.x ||
            after_contact->velocity.x <= 0 || world.status().sand_mass != initial.sand_mass + 2 * 64) {
        std::cerr << "Packed particles did not transfer momentum or conserve material\n";
        return 1;
    }
    if (!world.load_boxes(layout, sizeof(layout) / sizeof(layout[0])) || world.packed_particle_state(target)) {
        std::cerr << "A world reload must invalidate packed particle handles\n";
        return 1;
    }
    std::cout << "Tracked packed particles transferred momentum and conserved material\n";
    if (!communicating_vessels(world)) {
        std::cerr << "Opening a pressure gate failed to raise the water level or changed the total volume\n";
        return 1;
    }
    std::cout << "Communicating vessels conserved water and raised the receiving chamber\n";
    if (!loaded_sand_fractures(world)) {
        std::cerr << "A packed body failed to fracture from contact or lost mass\n";
        return 1;
    }
    std::cout << "A contact fractured a packed projectile and conserved its sand\n";
    if (!loaded_sand_compacts(world)) {
        std::cerr << "Compaction did not distinguish weak and strong contact loads\n";
        return 1;
    }
    std::cout << "Only the strong contact compacted sand, preserving its mass\n";
    if (!sustained_compression(world)) {
        std::cerr << "Sustained contact failed to distinguish gentle and strong compression\n";
        return 1;
    }
    std::cout << "Sustained bearing loads compacted held sand without changing its mass\n";
    if (!packed_sand_forms(world)) {
        std::cerr << "Mixed-size forms lost control, mass or release momentum\n";
        return 1;
    }
    std::cout << "Forms lifted and launched every packed size and retained control across packing and fracture\n";
}
