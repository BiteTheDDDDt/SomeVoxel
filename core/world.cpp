#include "world.hpp"
#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace somevoxel {

// Each occupied row has at most eight bits. Callers skip empty masks.
static int next_row_cell(unsigned row, bool descending) {
#ifdef _MSC_VER
    unsigned long index;
    if (descending) _BitScanReverse(&index, row);
    else _BitScanForward(&index, row);
    return int(index);
#else
    return descending ? 31 - __builtin_clz(row) : __builtin_ctz(row);
#endif
}

bool World::load_boxes(const int32_t *layout, size_t length) {
    if (length % 7 != 0 || (length != 0 && !layout)) return false;
    for (size_t i = 0; i < length; i += 7) {
        const I3 lo(layout[i], layout[i + 1], layout[i + 2]);
        const I3 hi(layout[i + 3], layout[i + 4], layout[i + 5]);
        const int kind = layout[i + 6];
        const int extent = preset_extent(kind);
        if (!inside(lo) || hi.x <= lo.x || hi.y <= lo.y || hi.z <= lo.z ||
                !inside(hi - I3(1, 1, 1)) ||
                (extent == 0 && (kind < Empty || (kind > Water && !element_kind(kind))))) return false;
        if (extent > 0 && ((lo.x | lo.y | lo.z | hi.x | hi.y | hi.z) & (extent - 1))) return false;
    }
    std::fill(voxels.begin(), voxels.end(), Voxel{});
    std::fill(mobile_rows.begin(), mobile_rows.end(), 0);
    std::fill(bulk_water_rows.begin(), bulk_water_rows.end(), 0);
    std::fill(liquid_head.begin(), liquid_head.end(), 0);
    grains.clear();
    gravel.clear();
    tracked_packed_particles.clear();
    rock_fragments.clear();
    water_positions.clear();
    liquid_motion.clear();
    reset_particle_forms();
    charge_gather = 0.f;
    for (auto &positions : element_positions) positions.clear();
    for (auto &particles : element_grains) particles.clear();
    detached_vines.clear();
    vine_roots.clear();
    stone_support_pending.clear();
    earth_wave = {};
    airborne_earth_ejecta.clear();
    pulling = false;
    remove_body();
    water_displaced = 0;
    tick = voxel_stamp = 0;
    for (size_t i = 0; i < length; i += 7)
        for (int y = layout[i + 1]; y < layout[i + 4]; ++y)
            for (int z = layout[i + 2]; z < layout[i + 5]; ++z)
                for (int x = layout[i]; x < layout[i + 3]; ++x)
                    voxels[flat({x, y, z})].kind = Kind(layout[i + 6]);
    for (int i = 0; i < int(voxels.size()); ++i) {
        if (i < slice && voxels[i].kind == Stone) voxels[i].kind = Bedrock;
        const Kind authored = voxels[i].kind;
        const int extent = preset_extent(authored);
        if (extent > 0) {
            if (voxels[i].occupant >= 0) continue;
            const I3 p = decode(i);
            const int align = ~(extent - 1);
            const int origin = flat({p.x & align, p.y & align, p.z & align});
            if (i == origin) {
                bool full = true;
                each_cube(i, extent, [&](int cell) { full &= voxels[cell].kind == authored; });
                if (full) {
                    const int id = int(gravel.size());
                    gravel.push_back({i, i, 0, extent});
                    each_cube(i, extent, [&](int cell) { voxels[cell] = {Gravel, 0, id}; });
                    continue;
                }
            }
            // An overlapping layout box may leave an incomplete cobble.
            // Keep its remaining material as individual sand voxels.
            voxels[i].kind = Sand;
        }
        if (voxels[i].kind == Water) {
            voxels[i].occupant = int(water_positions.size());
            water_positions.push_back(i);
            sync_bulk_water(i);
        } else if (voxels[i].kind == Sand) {
            voxels[i].occupant = int(grains.size());
            grains.push_back({i, i, 0});
        } else if (element_kind(voxels[i].kind)) {
            attach_element(i, voxels[i].kind);
        } else {
            continue;
        }
        set_mobile(i, initially_mobile(voxels[i].kind));
    }
    settle_unanchored_layout();
    for (int i : element_cells(Vine))
        vine_roots[voxels[i].occupant] = vine_touches_anchor(decode(i));
    awake.fill(true);
    reset_terrain_fractures();
    reset_elements();
    reset_fire_jet();
    reset_humidity();
    element_support_dirty = true;
    refresh_element_support();
    refresh_vine_support();
    for (auto &revision : element_versions) ++revision;
    ++elements_version;
    ++stone_version;
    stone_dirty.fill(true);
    ++water_version;
    ++water_topology_version;
    water_dirty.fill(true);
    active_count = chunk_count;
    moved_grains = 0;
    moved_gravel = 0;
    moved_rock = 0;
    ++sand_revision;
    ++rock_revision;
    return true;
}

float World::packed_particle_mass(int extent) {
    if (extent != 2 && extent != 4 && extent != 8) return 0.f;
    return particle_mass(Gravel, extent);
}

int64_t World::spawn_packed_particle(Vec3 position, int extent, Vec3 impulse) {
    const float mass = packed_particle_mass(extent);
    if (mass == 0.f || !position.is_finite() || !std::isfinite(impulse.length_squared()) ||
            next_packed_handle == std::numeric_limits<int64_t>::max()) return 0;
    const float offset = (extent - 1) * scale * 0.5f;
    const I3 origin = grid(position - V3(offset, offset, offset));
    const I3 edge(extent - 1, extent - 1, extent - 1);
    if (!inside(origin) || !inside(origin + edge)) return 0;
    const int index = flat(origin);
    bool free = true;
    each_cube(index, extent, [&](int i) { free &= voxels[i].kind == Empty && !body_at(i); });
    if (!free) return 0;

    const int slot = int(gravel.size());
    gravel.emplace_back(index, index, tick, extent, (impulse / mass).limit_length(maximum_swept_speed),
        position - particle_center(origin, extent));
    each_cube(index, extent, [&](int i) { voxels[i] = {Gravel, voxel_stamp, slot}; });
    const int64_t handle = next_packed_handle++;
    tracked_packed_particles.emplace(handle, slot);
    wake_box(origin, origin + edge, awake);
    ++sand_revision;
    return handle;
}

std::optional<PackedParticleState> World::packed_particle_state(int64_t handle) const {
    const auto entry = tracked_packed_particles.find(handle);
    if (entry == tracked_packed_particles.end()) return std::nullopt;
    const Grain &grain = gravel[entry->second];
    const Vec3 center = particle_center(decode(grain.position), grain.extent);
    const Vec3 previous = grain.moved == tick ? particle_center(decode(grain.previous), grain.extent) : center;
    return PackedParticleState{center, previous, grain.velocity, grain.extent};
}

void World::step() {
    ++tick;
    // This tag only distinguishes changes within the current step. Clear
    // stale marks before reusing a tag, leaving the physics clock intact.
    if (voxel_stamp == voxel_stamp_max) {
        for (Voxel &voxel : voxels) voxel.stamp = 0;
        voxel_stamp = 0;
    }
    ++voxel_stamp;
    // Separate update rates retain responsive fine sand without evaluating
    // the dense reservoir's connected water head twice per rendered frame.
    // A water update occurs when the cumulative schedule crosses a whole tick.
    // This also supports ratios such as 90 updates in 120 simulation ticks.
    const uint64_t water_phase = uint64_t(tick) * bulk_water_hz;
    water_tick = water_phase % simulation_hz < bulk_water_hz;
    uint32_t sweep_phase = tick;
#ifdef SOMEVOXEL_EXPERIMENTAL_WATER_SWEEP_PHASE
    // A fixed stride can sample the same orientation on every water tick.
    // This probe advances the shared sweep using the water tick index.
    if (water_tick) sweep_phase = uint32_t(water_phase / simulation_hz);
#endif
    const bool reverse_x = (sweep_phase & 1u) != 0;
    const bool reverse_z = (sweep_phase & 2u) != 0;
    const bool defer_bulk_water = !water_tick && !pulling_water && !winding;
    moved_grains = 0;
    moved_gravel = 0;
    moved_rock = 0;
    moved_water_form = 0;
    debris_moved.fill(false);
    next_awake.fill(false);
    prepare_particle_forms();
    if (pulling) wake_field();
    if (water_tick) refresh_liquid_head();
    active_count = 0;
    step_gravel();
    // Reuse the horizontal worklist for all eight voxel rows of a slab.
    // The traversal within each row stays bottom-up and alternates X/Z.
    std::array<I3, bx * bz> slab_chunks;
    for (int cy = 0; cy < by; ++cy) {
        int count = 0;
        for (int zi = 0; zi < bz; ++zi) {
            const int cz = reverse_z ? bz - 1 - zi : zi;
            for (int xi = 0; xi < bx; ++xi) {
                const int cx = reverse_x ? bx - 1 - xi : xi;
                const I3 chunk(cx, cy, cz);
                if (awake[chunk_id(chunk)]) slab_chunks[count++] = chunk;
            }
        }
        active_count += count;
        for (int y = cy * block; y < std::min(ny, (cy + 1) * block); ++y) {
            for (int entry = 0; entry < count; ++entry) {
                const I3 chunk = slab_chunks[entry];
                for (int z = 0; z < block; ++z) {
                    const I3 row(chunk.x * block, y, chunk.z * block + (reverse_z ? block - 1 - z : z));
                    const int row_id = flat(row) / block;
                    unsigned mask = mobile_rows[row_id];
                    if (defer_bulk_water && bulk_water_rows[row_id]) {
                        // These cells can only keep this chunk scheduled
                        // until the next bulk update. Moving water is still integrated.
                        next_awake[chunk_id(chunk)] = true;
                        mask &= ~unsigned(bulk_water_rows[row_id]);
                    }
                    while (mask) {
                        const int x = next_row_cell(mask, reverse_x);
                        mask &= ~(1u << x);
                        update_voxel({row.x + x, row.y, row.z});
                    }
                }
            }
        }
    }
    awake = next_awake;
    finish_terrain_fractures();
    step_earth_wave();
    step_fire_jet();
    step_elements();
    step_humidity();
    refresh_element_support();
    refresh_vine_support();
    if (moved_grains || moved_gravel) ++sand_revision;
    if (moved_rock) ++rock_revision;
    if (moved_water_form) ++water_form_revision;
}

} // namespace somevoxel
