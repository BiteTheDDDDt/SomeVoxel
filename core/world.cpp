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

void World::recruit_form_particles() {
    int available_cells = form_capacity() - held_form_cells();
    const int limit = std::min({available_cells, int(form_motion_limit - form_motion.size()), 120});
    if (limit <= 0) return;

    form_candidates.clear();
    const float reach_squared = particle_form.radius * particle_form.radius;
    const bool whip = particle_form.style == WhipForm;
    const V3 line = particle_form.point - particle_form.handle;
    const float inverse_length = whip ? 1.f / std::max(0.000001f, line.length_squared()) : 0.f;
    const auto consider = [&](int i, int extent, float search_squared) {
        if (profile_sand_sweep && particle_form.material == Sand) ++sand_sweep.form_scanned;
        if (voxels[i].formed || cube_cells(extent) > available_cells) return;
        const V3 sample = particle_center(decode(i), extent);
        const float nearest = whip ? std::clamp((sample - particle_form.handle).dot(line) * inverse_length, 0.f, 1.f) : 1.f;
        const float distance = sample.distance_squared_to(particle_form.handle + line * nearest);
        if (distance > search_squared) return;
        form_candidates.emplace_back(distance, i);
    };
    if (particle_form.material == Water) {
        for (int i : water_positions) consider(i, 1, reach_squared);
    } else {
        if (grains.size() <= 4096) {
            for (const Grain &grain : grains) consider(grain.position, 1, reach_squared);
        } else {
            // Fine candidates outside the current sphere cannot outrank
            // the full shortlist. Packed anchors are considered separately.
            for (float radius = std::min(particle_form.radius, block * scale);;) {
                form_candidates.clear();
                const V3 reach(radius, radius, radius);
                const I3 lo = grid(particle_form.point - reach);
                const I3 hi = grid(particle_form.point + reach);
                const float search_squared = radius * radius;
                for (int cy = std::max(0, lo.y) / block; cy <= std::min(ny - 1, hi.y) / block; ++cy)
                    for (int cz = std::max(0, lo.z) / block; cz <= std::min(nz - 1, hi.z) / block; ++cz)
                        for (int cx = std::max(0, lo.x) / block; cx <= std::min(nx - 1, hi.x) / block; ++cx) {
                            const I3 chunk(cx, cy, cz);
                            if (!sand_chunk_counts[chunk_id(chunk)] ||
                                    chunk_distance_squared(chunk, particle_form.point) > search_squared) continue;
                            const I3 start = chunk * block;
                            for (int y = start.y; y < std::min(ny, start.y + block); ++y)
                                for (int z = start.z; z < start.z + block; ++z) {
                                    const int base = flat({start.x, y, z});
                                    for (unsigned bits = mobile_rows[base / block]; bits; bits &= bits - 1) {
                                        const int id = base + next_row_cell(bits, false);
                                        if (voxels[id].kind == Sand) consider(id, 1, search_squared);
                                    }
                                }
                        }
                if (int(form_candidates.size()) >= limit || radius == particle_form.radius) break;
                radius = std::min(radius * 2.f, particle_form.radius);
            }
        }
        for (const Grain &grain : gravel) consider(grain.position, grain.extent, reach_squared);
    }
    if (int(form_candidates.size()) > limit) {
        std::nth_element(form_candidates.begin(), form_candidates.begin() + limit, form_candidates.end());
        form_candidates.resize(limit);
    }
    if (particle_form.material == Sand && !gravel.empty())
        std::sort(form_candidates.begin(), form_candidates.end());
    for (const auto &candidate : form_candidates) {
        const int i = candidate.second;
        const int cells = voxels[i].kind == Gravel ? cube_cells(gravel[voxels[i].occupant].extent) : 1;
        if (cells > available_cells) continue;
        FormMotion motion;
        motion.anchor = form_anchor(uint32_t(i) + tick * 0x9e3779b9u, particle_form.style);
        motion.style = particle_form.style;
        if (particle_form.material == Water) ensure_liquid_motion(i);
        attach_form_motion(i, motion);
        available_cells -= cells;
    }
}

World::FreeSandCandidates World::free_sand_candidates(V3 point) const {
    FreeSandCandidates candidates;
    if (!pulling || !point.is_finite() || point.distance_squared_to(field) > 4.f * scale * scale)
        return candidates;

    constexpr double nominal_pitch = layout_cell_size / refinement;
    constexpr double grain_kg = sand_density * nominal_pitch * nominal_pitch * nominal_pitch;
    // Round down to complete grains. The tolerance absorbs binary rounding
    // when the physical budget fits an exact number of cells.
    constexpr size_t fine_limit = std::max(size_t(1), size_t(free_sand_mass_budget_kg / grain_kg + 1e-7));
    constexpr size_t block_limit = 32;
    const float reach = free_sand_focus_reach();
    const float reach_squared = reach * reach;
    const auto trim = [](auto &nearest, size_t limit) {
        if (nearest.size() <= limit) return;
        std::nth_element(nearest.begin(), nearest.begin() + limit, nearest.end());
        nearest.resize(limit);
    };
    const auto consider = [&](auto &nearest, const Voxel &voxel, I3 position, int extent) {
        if (voxel.formed) return;
        const float distance = free_sand_focus_distance_squared(field, particle_center(position, extent));
        if (distance < reach_squared) nearest.emplace_back(distance, voxel.occupant);
    };

    // The bounds include each candidate center before the exact ellipsoid
    // test. The world can be mostly empty or contain distant reservoirs of sand.
    const V3 bounds(reach, reach / 1.4f, reach);
    const I3 low = grid(field - bounds) - I3(1, 1, 1);
    const I3 high = grid(field + bounds) + I3(1, 1, 1);
    const I3 first(std::max(0, low.x), std::max(0, low.y), std::max(0, low.z));
    const I3 last(std::min(nx - 1, high.x), std::min(ny - 1, high.y), std::min(nz - 1, high.z));
    if (first.x > last.x || first.y > last.y || first.z > last.z) return candidates;
    const size_t row_count = size_t(last.x / block - first.x / block + 1) *
        (last.y - first.y + 1) * (last.z - first.z + 1);

    candidates.fine.reserve(std::min(fine_limit, grains.size()));
    if (refinement <= 4 || grains.size() <= row_count) {
        for (const Grain &grain : grains)
            consider(candidates.fine, voxels[grain.position], decode(grain.position), grain.extent);
    } else {
        const auto scan = [&](int begin_y, int end_y, auto emit) {
            for (int y = begin_y; y < end_y; ++y) {
                for (int z = first.z; z <= last.z; ++z) {
                    const int start = flat({first.x / block * block, y, z});
                    for (int base = start, x = first.x / block * block; x <= last.x; base += block, x += block) {
                        for (unsigned bits = mobile_rows[base / block]; bits != 0; bits &= bits - 1) {
                            const int lane = next_row_cell(bits, false);
                            const Voxel &voxel = voxels[base + lane];
                            if (voxel.kind != Sand || voxel.formed) continue;
                            const float distance = free_sand_focus_distance_squared(field, particle_center({x + lane, y, z}, 1));
                            if (distance < reach_squared) emit(distance, voxel.occupant);
                        }
                    }
                }
            }
        };
        const auto append = [&](float distance, int id) {
            candidates.fine.emplace_back(distance, id);
            if (candidates.fine.size() == 2 * fine_limit) trim(candidates.fine, fine_limit);
        };
#ifdef SOMEVOXEL_PARALLEL_WATER_WORKERS
        if (row_count >= 32768 && water_worker_count() > 1) {
            auto &pool = water_solver().worker_pool();
            static constexpr int buckets = 128;
            const float to_bucket = buckets / reach_squared;
            const auto bucket = [to_bucket](float distance) { return std::min(buckets - 1, int(distance * to_bucket)); };
            std::vector<std::array<size_t, buckets>> histogram(pool.count);
            const int layers = last.y - first.y + 1;
            const auto visit_band = [&](int worker, auto emit) {
                scan(first.y + layers * worker / pool.count, first.y + layers * (worker + 1) / pool.count, emit);
            };
            pool.run([&](int worker) {
                visit_band(worker, [&](float distance, int) { ++histogram[worker][bucket(distance)]; });
            });
            // Distances in lower buckets are always nearer. Only the bucket
            // containing the cutoff needs selection by the exact (distance, id).
            size_t nearer = 0;
            int cutoff = 0;
            for (; cutoff < buckets; ++cutoff) {
                size_t count = 0;
                for (const auto &counts : histogram) count += counts[cutoff];
                if (nearer + count > fine_limit) break;
                nearer += count;
            }
            std::vector<size_t> interior_offsets(pool.count + 1), boundary_offsets(pool.count + 1);
            for (int k = 0; k < pool.count; ++k) {
                interior_offsets[k + 1] = interior_offsets[k];
                for (int b = 0; b < cutoff; ++b) interior_offsets[k + 1] += histogram[k][b];
                boundary_offsets[k + 1] = boundary_offsets[k] + (cutoff < buckets ? histogram[k][cutoff] : 0);
            }
            candidates.fine.resize(nearer);
            decltype(candidates.fine) boundary(boundary_offsets.back());
            pool.run([&](int worker) {
                size_t near_index = interior_offsets[worker], boundary_index = boundary_offsets[worker];
                visit_band(worker, [&](float distance, int id) {
                    const int index = bucket(distance);
                    if (index < cutoff) candidates.fine[near_index++] = {distance, id};
                    else if (index == cutoff) boundary[boundary_index++] = {distance, id};
                });
            });
            trim(boundary, fine_limit - nearer);
            candidates.fine.insert(candidates.fine.end(), boundary.begin(), boundary.end());
        } else
#endif
        {
            scan(first.y, last.y + 1, append);
        }
    }
    trim(candidates.fine, fine_limit);

    // Packed bodies keep their separate selection rule and contribute their
    // actual packed mass to the charge energy budget.
    candidates.blocks.reserve(std::min(block_limit, gravel.size()));
    for (const Grain &grain : gravel)
        consider(candidates.blocks, voxels[grain.position], decode(grain.position), grain.extent);
    trim(candidates.blocks, block_limit);
    return candidates;
}

bool World::load_boxes(const int32_t *layout, size_t length) {
    if (length % 7 != 0 || (length != 0 && !layout)) return false;
    int layout_height = 0;
    for (size_t i = 0; i < length; i += 7) {
        const I3 lo(layout[i], layout[i + 1], layout[i + 2]);
        const I3 hi(layout[i + 3], layout[i + 4], layout[i + 5]);
        const int kind = layout[i + 6];
        const int extent = preset_extent(kind);
        if (!inside(lo) || hi.x <= lo.x || hi.y <= lo.y || hi.z <= lo.z ||
                !inside(hi - I3(1, 1, 1)) ||
                (extent == 0 && (kind < Empty || (kind > Water && !element_kind(kind))))) return false;
        if (extent > 0 && ((lo.x | lo.y | lo.z | hi.x | hi.y | hi.z) & (extent - 1))) return false;
        if (kind != Empty) layout_height = std::max(layout_height, hi.y);
    }
    std::fill(voxels.begin(), voxels.end(), Voxel{});
    std::fill(mobile_rows.begin(), mobile_rows.end(), 0);
    std::fill(mobile_row_groups.begin(), mobile_row_groups.end(), 0);
    std::fill(bulk_water_rows.begin(), bulk_water_rows.end(), 0);
    std::fill(water_chunk_counts.begin(), water_chunk_counts.end(), 0);
    std::fill(sand_chunk_counts.begin(), sand_chunk_counts.end(), 0);
    std::fill(settled_sand_chunks.begin(), settled_sand_chunks.end(), 0);
    liquid_head.clear();
    grains.clear();
    gravel.clear();
    reset_sand_tiles();
    tracked_packed_particles.clear();
    rock_fragments.clear();
    water_positions.clear();
#ifdef SOMEVOXEL_PARALLEL_WATER_WORKERS
    water_scan_end = 0;
    water_spans.reset();
    pressure_surfaces.clear();
    pressure_surface_version = -1;
    accelerated_head_rebuilds = 0;
    accelerator_attempts = 0;
    accelerator_time_us = 0;
    head_selection = {};
#endif
    head_times = {head_times.automatic};
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
    support_seen.clear();
    support_epoch = 0;
    water_displaced = 0;
    tick = voxel_stamp = 0;
    for (size_t i = 0; i < length; i += 7)
        for (int y = layout[i + 1]; y < layout[i + 4]; ++y)
            for (int z = layout[i + 2]; z < layout[i + 5]; ++z)
                for (int x = layout[i]; x < layout[i + 3]; ++x)
                    voxels[flat({x, y, z})].kind = Kind(layout[i + 6]);
    // The grid was cleared before applying the boxes, so all cells above
    // the highest authored material remain empty even on a world reload.
    const int layout_end = layout_height * slice;
    for (int i = 0; i < layout_end; ++i) {
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
            ++water_chunk_counts[chunk_at(decode(i))];
            sync_bulk_water(i);
        } else if (voxels[i].kind == Sand) {
            voxels[i].occupant = int(grains.size());
            add_sand_grain({i, i, 0});
        } else if (element_kind(voxels[i].kind)) {
            attach_element(i, voxels[i].kind);
        } else {
            continue;
        }
        set_mobile(i, initially_mobile(voxels[i].kind));
    }
    settle_unanchored_layout(layout_end);
    // Loaded sand starts at rest. Later certifications inspect the live
    // grains because collisions and forces can disturb an interior.
    std::transform(sand_chunk_counts.begin(), sand_chunk_counts.end(), settled_sand_chunks.begin(),
        [](uint16_t count) { return count == block * block * block; });
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
    mark_gravel_tile(origin);
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
    sand_sweep = {};
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
    const bool passive_bulk_water = !pulling_water && !winding;
    const bool defer_bulk_water = !water_tick && passive_bulk_water;
    constexpr unsigned full_row = (1u << block) - 1;
    moved_grains = 0;
    moved_gravel = 0;
    moved_rock = 0;
    moved_water_form = 0;
    debris_moved.fill(false);
    next_awake.fill(false);
    prepare_particle_forms();
    if (pulling) wake_field();
    if (water_tick) refresh_liquid_head();
    sample_sand_field_pressure();
    sample_sand_pressure_pulses();
    active_count = 0;
    step_gravel();
    const auto sweep_start = profile_sand_sweep ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    // Reuse the horizontal worklist for all eight voxel rows of a slab.
    // The traversal within each row stays bottom-up and alternates X/Z.
    std::array<I3, bx * bz> slab_chunks;
    const bool allow_settled_sand = !winding && airborne_earth_ejecta.empty();
    for (int cy = 0; cy < by; ++cy) {
        int count = 0;
        for (int zi = 0; zi < bz; ++zi) {
            const int cz = reverse_z ? bz - 1 - zi : zi;
            for (int xi = 0; xi < bx; ++xi) {
                const int cx = reverse_x ? bx - 1 - xi : xi;
                const I3 chunk(cx, cy, cz);
                if (!awake[chunk_id(chunk)]) continue;
                if (allow_settled_sand) certify_settled_sand(chunk);
                slab_chunks[count++] = chunk;
            }
        }
        active_count += count;
        for (int y = cy * block; y < std::min(ny, (cy + 1) * block); ++y) {
            for (int entry = 0; entry < count; ++entry) {
                const I3 chunk = slab_chunks[entry];
                const int chunk_index = chunk_id(chunk);
                if (settled_sand_chunks[chunk_index]) {
                    if (allow_settled_sand && full_sand_neighborhood(chunk)) {
                        if (profile_sand_sweep) sand_sweep.settled_skipped += block * block;
                        continue;
                    }
                    settled_sand_chunks[chunk_index] = false;
                }
                if (!mobile_row_groups[(y * bz + chunk.z) * bx + chunk.x]) {
                    if (profile_sand_sweep) sand_sweep.empty_rows += block;
                    continue;
                }
                const bool full_sand_candidate = sand_chunk_counts[chunk_index] == block * block * block;
                for (int z = 0; z < block; ++z) {
                    const I3 row(chunk.x * block, y, chunk.z * block + (reverse_z ? block - 1 - z : z));
                    const int row_start = flat(row);
                    const int row_id = row_start / block;
                    unsigned mask = mobile_rows[row_id];
                    if (!mask) {
                        if (profile_sand_sweep) ++sand_sweep.empty_rows;
                        continue;
                    }
                    if (passive_bulk_water && bulk_water_rows[row_id]) {
                        if (defer_bulk_water) {
                            // These cells can only keep this chunk scheduled
                            // until the next bulk update. Moving water is still integrated.
                            next_awake[chunk_id(chunk)] = true;
                            mask &= ~unsigned(bulk_water_rows[row_id]);
                        } else if (bulk_water_rows[row_id] == full_row &&
                                row.x > 0 && row.x + block < nx && row.z > 0 && row.z < nz - 1 && row.y > 0 &&
                                (bulk_water_rows[row_id - 1] & (1u << (block - 1))) &&
                                (bulk_water_rows[row_id + 1] & 1u) &&
                                bulk_water_rows[row_id - bx] == full_row &&
                                bulk_water_rows[row_id + bx] == full_row &&
                                bulk_water_rows[row_id - slice / block] == full_row) {
                            // Every possible flow face is blocked by water.
                            // Read live rows so moves earlier in the sweep can open a face.
                            continue;
                        }
                    }
                    const int y_lane = y & (block - 1);
                    if (full_sand_candidate && z > 0 && z < block - 1 && y_lane > 0 && y_lane < block - 1) {
                        const int edge = reverse_x ? block - 1 : 0;
                        if (mask & (1u << edge)) {
                            mask &= ~(1u << edge);
                            update_voxel({row.x + edge, row.y, row.z}, row_start + edge, false);
                        }
                        // The leading edge can open a hole. Once confirmed
                        // full, no interior grain can leave before the far edge.
                        if (sand_chunk_counts[chunk_index] == block * block * block) {
                            constexpr unsigned edges = 1u | (1u << (block - 1));
                            for (unsigned inside = mask & ~edges; inside;) {
                                const int x = next_row_cell(inside, reverse_x);
                                inside &= ~(1u << x);
                                update_voxel({row.x + x, row.y, row.z}, row_start + x, true);
                            }
                            mask &= edges;
                        }
                    }
                    while (mask) {
                        const int x = next_row_cell(mask, reverse_x);
                        mask &= ~(1u << x);
                        update_voxel({row.x + x, row.y, row.z}, row_start + x, false);
                    }
                }
            }
        }
    }
    if (profile_sand_sweep) {
        sand_sweep.time_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - sweep_start).count();
    }
    awake = next_awake;
    finish_packed_fractures();
    finish_contact_compaction();
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
