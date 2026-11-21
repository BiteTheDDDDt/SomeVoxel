#include "core/world.hpp"

#include <cmath>
#include <iostream>
#include <iterator>
#include <memory>

using somevoxel::World;

static bool check_shaft(World &world, bool together) {
    constexpr int lower_x = 40;
    const int upper_x = together ? lower_x : 48;
    // The lower grain rests at y=7, immediately below the eight-cell chunk boundary.
    const int32_t layout[] = {
        39, 0, 39, 42, 15, 42, World::Stone,
        47, 0, 39, 50, 15, 42, World::Stone,
        40, 7, 40, 41, 15, 41, World::Empty,
        48, 7, 40, 49, 15, 41, World::Empty,
        lower_x, 7, 40, lower_x + 1, 8, 41, World::Sand,
        upper_x, 10, 40, upper_x + 1, 11, 41, World::Sand,
    };
    if (!world.load_boxes(layout, std::size(layout))) return false;

    const auto state = world.status();
    const auto sand_at = [&](int x, int y) {
        return world.range_count(World::Sand, {x, y, 40}, {x + 1, y + 1, 41}) == 1;
    };
    const int rest_y = together ? 8 : 7;
    int previous_y = 10;
    bool passed_gap = false;
    if (!sand_at(lower_x, 7) || !sand_at(upper_x, previous_y)) return false;
    const int steps = int(std::ceil(0.5f / state.step_seconds));
    for (int tick = 1; tick <= steps; ++tick) {
        world.step();
        int upper_y = rest_y;
        while (upper_y <= previous_y && !sand_at(upper_x, upper_y)) ++upper_y;
        if (!sand_at(lower_x, 7) || upper_y > previous_y || world.status().sand_mass != 2) {
            std::cerr << "Lost a grain or reversed its fall at tick " << tick << '\n';
            return false;
        }
        if (upper_y > rest_y && upper_y < 10) passed_gap = true;
        previous_y = upper_y;
    }
    return passed_gap && previous_y == rest_y;
}

int main() {
    auto world = std::make_unique<World>();
    if (!check_shaft(*world, true)) {
        std::cerr << "Sand in one shaft failed to stack across the chunk boundary\n";
        return 1;
    }
    if (!check_shaft(*world, false)) {
        std::cerr << "Sand in separate shafts failed to settle independently\n";
        return 1;
    }
    return 0;
}
