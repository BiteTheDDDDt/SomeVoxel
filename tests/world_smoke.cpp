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
}
