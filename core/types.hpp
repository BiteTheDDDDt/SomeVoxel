#pragma once

#include <cstdint>
#include <utility>
#include <vector>
#include "vector.hpp"

namespace somevoxel {

struct SandSweepStats {
    int64_t empty_rows = 0;
    int64_t settled_skipped = 0;
    int64_t form_scanned = 0;
    int64_t water_form_scanned = 0;
    int64_t integrated = 0;
    int64_t tested_faces = 0;
    int64_t crossed_faces = 0;
    int64_t contacts = 0;
    int64_t time_us = 0;
};

struct WorldStats {
    int64_t refinement = 0;
    float cell_size = 0;
    float step_seconds = 0;
    int64_t bulk_water_hz = 0;
    int64_t voxel_bytes = 0;
    int64_t grain_bytes = 0;
    int64_t dense_grid_bytes = 0;
    int64_t sand = 0;
    int64_t gravel = 0;
    int64_t sand_volume = 0;
    int64_t sand_mass = 0;
    int64_t packed_sand_blocks = 0;
    int64_t blocks_9 = 0;
    int64_t blocks_18 = 0;
    int64_t blocks_36 = 0;
    int64_t displaced_water = 0;
    int64_t water = 0;
    int64_t active_water = 0;
    int64_t water_volume = 0;
    int64_t ice = 0;
    int64_t wood = 0;
    int64_t grass = 0;
    int64_t ash = 0;
    int64_t steam = 0;
    int64_t air_water = 0;
    int64_t air_collected = 0;
    int64_t steam_collected = 0;
    int64_t air_recharged = 0;
    int64_t lava = 0;
    int64_t vine = 0;
    int64_t mud = 0;
    int64_t char_cells = 0;
    int64_t burning = 0;
    int64_t moved = 0;
    int64_t moved_gravel = 0;
    int64_t chunks = 0;
    int64_t chunk_capacity = 0;
    int64_t stone_version = 0;
    int64_t water_version = 0;
    int64_t elements_version = 0;
    int64_t contact_stone_fractures = 0;
    int64_t contact_wood_fractures = 0;
    int64_t contact_ice_fractures = 0;
    int64_t contact_char_fractures = 0;
};

struct ThrowInfo {
    Vec3 position, velocity;
    float speed = 0, mass = 0;
    int count = 0;
    int blocks = 0;
};

struct EarthWaveInfo {
    Vec3 position, direction;
    bool active = false;
    int ended = 0;
    float strength = 0, radius = 0, impulse = 0, travelled = 0, range = 0;
    int64_t serial = 0, ejected = 0, disturbed = 0, contact_count = 0;
    std::vector<Vec3> trail;
};

struct ImpactInfo {
    float speed, impulse;
};

struct PackedParticleState {
    Vec3 position, previous_position, velocity;
    int extent = 0;
};

using TargetContacts = std::vector<std::pair<int, float>>;

} // namespace somevoxel
