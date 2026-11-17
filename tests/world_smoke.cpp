#include "core/world.hpp"
#include <iostream>

int main() {
    using somevoxel::World;
    World world;
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
}
