#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <queue>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <optional>
#include "types.hpp"
#include "detail/paged_field.hpp"
#if defined(SOMEVOXEL_PARALLEL_WATER_WORKERS) || defined(SOMEVOXEL_PARALLEL_RENDER_WORKERS)
#include "detail/voxel_water_spans.hpp"
#endif

namespace somevoxel {

namespace detail {
// Use successive required rebuilds to compare backends. The first run on
// each side warms allocations and device pipelines before measuring costs.
class WaterHeadSelection {
    static constexpr int probes = 8, reuse = 96;
    int probe = 0, remaining = 0;
    int64_t initial_cells = 0, initial_height = 0, initial_topology = 0;
    std::array<std::array<int64_t, 3>, 2> samples{};
    bool prefer_gpu = false;

public:
    bool choose_gpu(int64_t cells, int height, int64_t topology) {
        if (probe == probes) {
            const bool changed = std::abs(cells - initial_cells) * 2 > initial_cells ||
                std::abs(height - initial_height) * 2 > initial_height ||
                topology - initial_topology > initial_cells;
            if (remaining > 0 && (remaining > reuse - 16 || !changed)) {
                --remaining;
                return prefer_gpu;
            }
            probe = 0;
        }
        if (probe == 0) {
            initial_cells = cells;
            initial_height = height;
            initial_topology = topology;
        }
        return (probe & 1) != 0;
    }

    void measured(int64_t microseconds) {
        if (probe == probes) return;
        if (probe >= 2) samples[probe % 2][(probe - 2) / 2] = microseconds;
        if (++probe == probes) {
            for (auto &costs : samples) std::sort(costs.begin(), costs.end());
            prefer_gpu = samples[1][1] < samples[0][1];
            remaining = reuse;
        }
    }

    void gpu_failed() {
        prefer_gpu = false;
        probe = probes;
        remaining = reuse;
    }
};
} // namespace detail

class World {
    friend class WorldPresentation;

    using V3 = Vec3;
    using I3 = Cell;
#ifndef SOMEVOXEL_REFINEMENT
#define SOMEVOXEL_REFINEMENT 4
#endif
    static constexpr int refinement = SOMEVOXEL_REFINEMENT;
    static constexpr int block = 8;
    static constexpr int nx = 72 * refinement, ny = 36 * refinement, nz = 56 * refinement;
    static constexpr int slice = nx * nz;
    static constexpr int bx = nx / block, by = (ny + block - 1) / block, bz = nz / block;
    static constexpr int chunk_count = bx * by * bz;
    static_assert(nx % block == 0 && nz % block == 0);
    static constexpr double layout_cell_size = 0.18;
    static constexpr float scale = float(layout_cell_size) / refinement;
    static constexpr int reference_hz = 30 * refinement;
#ifdef SOMEVOXEL_EXPERIMENTAL_SIMULATION_HZ
    static constexpr int simulation_hz = SOMEVOXEL_EXPERIMENTAL_SIMULATION_HZ;
#else
    static constexpr int simulation_hz = reference_hz;
#endif
#ifdef SOMEVOXEL_EXPERIMENTAL_WATER_HZ
    static constexpr int bulk_water_hz = SOMEVOXEL_EXPERIMENTAL_WATER_HZ;
#else
    static constexpr int bulk_water_hz = refinement < 4 ? simulation_hz : simulation_hz / 2;
#endif
    static_assert(simulation_hz >= 60 && simulation_hz % 30 == 0);
    static_assert(bulk_water_hz > 0 && bulk_water_hz <= simulation_hz && bulk_water_hz % 15 == 0);
    static constexpr float dt = 1.f / simulation_hz;
    static constexpr float maximum_particle_speed = scale / (1.f / reference_hz);
    static constexpr int reference_faces_per_axis = 6;
    static constexpr int max_faces_per_axis = (reference_faces_per_axis * reference_hz + simulation_hz - 1) / simulation_hz;
    static constexpr float maximum_swept_speed = maximum_particle_speed * reference_faces_per_axis;
    static constexpr float gravity = 13.f;
    static constexpr float max_cast_strength = 10.f;
    static constexpr float sand_density = 1600.f;
    static constexpr int sand_packing_percent = 125;
    static constexpr float rock_density = 2600.f;
    static constexpr int patch_width = 24;
    static constexpr int patch_x = (nx + patch_width - 1) / patch_width;
    static constexpr int patch_z = (nz + patch_width - 1) / patch_width;
    static constexpr int stone_patch_width = 48;
    static constexpr int stone_patch_x = (nx + stone_patch_width - 1) / stone_patch_width;
    static constexpr int stone_patch_z = (nz + stone_patch_width - 1) / stone_patch_width;
public:
#ifdef SOMEVOXEL_WIDE_GRID
    enum Kind : uint8_t {
#else
    enum Kind : uint32_t {
#endif
        Empty, Stone, Sand, Water, Gravel, Rock, Bedrock, Preset18, Preset36,
        Ice, Wood, Grass, Ash, Steam, Lava, Vine, Mud, Char
    };
private:
    static_assert(Char < (1u << 5));

#ifdef SOMEVOXEL_WIDE_GRID
    static constexpr uint32_t voxel_stamp_max = std::numeric_limits<uint32_t>::max();
    using SupportStamp = uint32_t;
    struct Voxel {
        Kind kind = Empty;
        bool formed = false;
        bool liquid_moving = false;
        uint32_t stamp = 0;
        int occupant = -1;

        Voxel() = default;
        Voxel(Kind material, uint32_t updated, int id) : kind(material), stamp(updated), occupant(id) {}
    };
    static_assert(sizeof(Voxel) == 12);
#else
    static constexpr int voxel_stamp_bits = 25;
    static constexpr uint32_t voxel_stamp_max = (1u << voxel_stamp_bits) - 1;
    using SupportStamp = uint16_t;
    struct Voxel {
        Kind kind : 5;
        uint32_t formed : 1, liquid_moving : 1, stamp : voxel_stamp_bits;
        int occupant;

        Voxel() : Voxel(Empty, 0, -1) {}
        Voxel(Kind material, uint32_t updated, int id) :
            kind(material), formed(false), liquid_moving(false), stamp(updated), occupant(id) {}
    };
    static_assert(sizeof(Voxel) == 8);
#endif
    struct Grain {
        int previous;
        int position;
        uint32_t moved = 0;
        uint8_t extent = 1;
        uint8_t blocked_faces = 0;
        uint16_t sand_units = 0; // Zero denotes the uncompressed mass for this footprint.
        V3 velocity;
        V3 travel;

        Grain() = default;
        Grain(int from, int to, uint32_t updated, int width = 1, V3 speed = V3(), V3 offset = V3()) :
            previous(from), position(to), moved(updated), extent(uint8_t(width)), velocity(speed), travel(offset) {}
    };
    struct MaterialLoad {
        float work;
        V3 impulse;
    };
    static_assert(sizeof(Grain) == 16 + 2 * sizeof(V3));
    struct GravelLiquidMove {
        std::array<int, 8 * 8> sources, exits;
        std::vector<uint8_t> seen;
        std::vector<int> frontier, outlets;
        int count = 0;
        int axis = 0;
        int direction = 0;
    };
    enum FormStyle : uint8_t { NoForm, ShotForm, BladeForm, WhipForm };
    struct FormMotion {
        V3 anchor;
        V3 forward;
        FormStyle style = NoForm;
        uint32_t expires = 0;
        uint32_t last_contact = 0;
    };
    struct GranularMaterial {
        float density;
        float attraction_range;
        float ground_retention;
        bool granular_repose = true;
    };
    std::vector<Voxel> voxels = std::vector<Voxel>(nx * ny * nz);
    // One bit per moving material. Each byte describes an eight-cell X row.
    std::vector<uint8_t> mobile_rows = std::vector<uint8_t>(nx * ny * nz / block);
    // Eight adjacent X rows in Z share one summary byte at each Y layer.
    std::vector<uint8_t> mobile_row_groups = std::vector<uint8_t>(size_t(ny) * bx * bz);
    std::vector<uint8_t> bulk_water_rows = std::vector<uint8_t>(nx * ny * nz / block);
    // Includes moving and formed water. One count covers an 8-cubed chunk.
    std::vector<uint16_t> water_chunk_counts = std::vector<uint16_t>(chunk_count);
    std::vector<uint16_t> sand_chunk_counts = std::vector<uint16_t>(chunk_count);
    std::vector<uint8_t> settled_sand_chunks = std::vector<uint8_t>(chunk_count);
    std::vector<Grain> grains;
    std::vector<Grain> gravel;
    std::unordered_map<int64_t, int> tracked_packed_particles;
    int64_t next_packed_handle = 1;
    GravelLiquidMove gravel_liquid_scratch;
    std::vector<Grain> rock_fragments;
    std::array<std::vector<int>, Char - Ice + 1> element_positions;
    std::array<int64_t, Char - Ice + 1> element_versions{};
    std::vector<uint8_t> detached_vines;
    std::vector<uint8_t> vine_roots;
    bool vine_support_dirty = false;
    // Loose ash and detached structural voxels share the granular integrator.
    // The indices match the corresponding entries in element_positions.
    std::array<std::vector<Grain>, 8> element_grains;
    std::array<int64_t, 3> debris_revisions{};
    std::array<bool, 3> debris_moved{};
    std::vector<int> stone_support_pending;
    bool element_support_dirty = false;
    int64_t elements_version = 0;
    std::vector<int> visible_sand;
    int64_t visibility_revision = -1;
    std::vector<int> water_positions;
#ifdef SOMEVOXEL_PARALLEL_WATER_WORKERS
    int water_scan_end = 0;
#endif
    // Current and proposed occupancy use one two-bit field per voxel.
    PagedField<uint8_t, (size_t(nx) * ny * nz + 3) / 4, true> body_mask;
    std::vector<int> body_cells;
    std::vector<int> body_next;
    V3 body_feet;
    float body_radius = 0;
    float body_height = 0;
    int water_displaced = 0;
    PagedField<SupportStamp, size_t(nx) * ny * nz> support_seen;
    SupportStamp support_epoch = 0;
    // Pad every X row so head workers that own different Z chunks never
    // update the same word, even when the world width is not a multiple of 32.
    class ChunkActivity {
        static constexpr int words_per_row = (bx + 31) / 32;
        std::vector<uint32_t> rows = std::vector<uint32_t>(size_t(by) * bz * words_per_row);

    public:
        static constexpr int row_words = words_per_row;
        static constexpr size_t bytes = size_t(by) * bz * words_per_row * sizeof(uint32_t);

        void set(I3 chunk) {
            auto &row = rows[(chunk.y * bz + chunk.z) * words_per_row + chunk.x / 32];
            const uint32_t flag = uint32_t(1) << (chunk.x % 32);
            if (!(row & flag)) row |= flag;
        }

        void set_for_cell(I3 p) { set({p.x / block, p.y / block, p.z / block}); }

        bool test(I3 chunk) const {
            return (word(chunk.y, chunk.z, chunk.x / 32) & (uint32_t(1) << (chunk.x % 32))) != 0;
        }

        uint32_t word(int y, int z, int index) const {
            return rows[(y * bz + z) * words_per_row + index];
        }

        void clear() { std::fill(rows.begin(), rows.end(), 0); }

        void set_all() {
            std::fill(rows.begin(), rows.end(), ~uint32_t(0));
            if constexpr (bx % 32 != 0) {
                for (size_t i = words_per_row - 1; i < rows.size(); i += words_per_row)
                    rows[i] = (uint32_t(1) << (bx % 32)) - 1;
            }
        }

        void swap(ChunkActivity &other) { rows.swap(other.rows); }
    };
    ChunkActivity awake, next_awake;
    std::array<bool, patch_x * patch_z> water_dirty{};
    std::array<bool, stone_patch_x * stone_patch_z> stone_dirty{};
    bool track_water_patches = true;
    uint32_t tick = 0;
    uint32_t voxel_stamp = 0;
    V3 field;
    bool pulling = false;
    float cast_radius_scale = 1.f;
    float cast_strength = 1.f;
    float charge_gather = 0.f;
    bool water_tick = true;
    int stone_version = 0;
    int64_t water_version = 0;
    int64_t water_topology_version = 0;
    int active_count = 0;
    int moved_grains = 0;
    bool profile_sand_sweep = false;
    SandSweepStats sand_sweep;
    int moved_gravel = 0;
    int moved_rock = 0;
    int64_t sand_revision = 0;
    int64_t rock_revision = 0;
    int64_t head_version = -1;

    static std::array<I3, 4> cardinal() {
        return { I3(1, 0, 0), I3(0, 0, 1), I3(-1, 0, 0), I3(0, 0, -1) };
    }
    static int flat(I3 p) { return p.x + nx * (p.z + nz * p.y); }
    static I3 decode(int i) { return { i % nx, i / (nx * nz), i / nx % nz }; }
    static bool inside(I3 p) {
        return p.x >= 0 && p.x < nx && p.y >= 0 && p.y < ny && p.z >= 0 && p.z < nz;
    }
    static int grid_axis(float coordinate, int cells) {
        const float cell = coordinate / scale;
        if (cell < 0.f) return -1;
        return cell < cells ? int(cell) : cells;
    }
    static I3 grid(V3 p) {
        return {grid_axis(p.x, nx), grid_axis(p.y, ny), grid_axis(p.z, nz)};
    }
    static V3 center(I3 p) { return (V3(p) + V3(0.5f, 0.5f, 0.5f)) * scale; }
    static int chunk_id(I3 p) { return p.x + bx * (p.z + bz * p.y); }
    static int chunk_at(I3 p) { return chunk_id({p.x / block, p.y / block, p.z / block}); }

    void change_sand_count(int id, int delta) {
        sand_chunk_counts[id] = uint16_t(sand_chunk_counts[id] + delta);
        settled_sand_chunks[id] = false;
    }

    void add_sand_grain(Grain grain) {
        change_sand_count(chunk_at(decode(grain.position)), 1);
        grains.push_back(grain);
    }

    bool full_sand_neighborhood(I3 chunk) const {
        if (chunk.x == 0 || chunk.x + 1 == bx || chunk.y == 0 || chunk.y + 1 == by ||
                chunk.z == 0 || chunk.z + 1 == bz) return false;
        constexpr int full = block * block * block;
        const int id = chunk_id(chunk);
        return sand_chunk_counts[id] == full &&
            sand_chunk_counts[id - 1] == full && sand_chunk_counts[id + 1] == full &&
            sand_chunk_counts[id - bx] == full && sand_chunk_counts[id + bx] == full &&
            sand_chunk_counts[id - bx * bz] == full && sand_chunk_counts[id + bx * bz] == full;
    }

    static float chunk_distance_squared(I3 chunk, V3 point) {
        const V3 low = V3(chunk) * (block * scale);
        const V3 high = low + V3(block * scale, block * scale, block * scale);
        return point.distance_squared_to(point.clamp(low, high));
    }

    void certify_settled_sand(I3 chunk) {
        const int id = chunk_id(chunk);
        // A missed certification just runs the regular sweep. Stagger
        // retries across chunks so moving interiors are not reread every step.
        if (settled_sand_chunks[id] || ((uint32_t(id) ^ tick) & 7u) ||
                sand_chunk_counts[id] != block * block * block ||
                !full_sand_neighborhood(chunk)) return;
        const float reach = 2.4f * cast_radius_scale + 1.75f * scale;
        if (pulling && chunk_distance_squared(chunk, field) <= reach * reach) return;
        const I3 start = chunk * block;
        for (int y = 0; y < block; ++y)
            for (int z = 0; z < block; ++z) {
                const int row = flat(start + I3(0, y, z));
                for (int x = 0; x < block; ++x) {
                    const Voxel &cell = voxels[row + x];
                    const Grain &grain = grains[cell.occupant];
                    if (cell.formed || grain.velocity != V3() || grain.travel.y != 0.f ||
                            (grain.blocked_faces & grain_axis_faces(1))) return;
                }
            }
        settled_sand_chunks[id] = true;
    }
    static uint32_t hash(uint32_t key) {
        key ^= key >> 16;
        key *= 0x7feb352du;
        key ^= key >> 15;
        key *= 0x846ca68bu;
        return key ^ (key >> 16);
    }
    Kind kind_at(I3 p) const { return inside(p) ? voxels[flat(p)].kind : Stone; }

    static bool element_kind(int kind) { return kind >= Ice && kind <= Char; }
    static bool structure_kind(int kind) { return kind == Ice || kind == Wood || kind == Char; }
    static bool element_particle(Kind kind) {
        return kind == Mud || kind == Ash || kind == Steam || kind == Lava || kind == Vine || structure_kind(kind);
    }
    static bool initially_mobile(Kind kind) {
        return kind == Sand || kind == Rock || kind == Water || kind == Ash ||
            kind == Steam || kind == Lava || kind == Vine || kind == Mud;
    }
    static bool passable_body(Kind kind) {
        return kind == Empty || kind == Water || kind == Steam || kind == Grass;
    }
    static int element_grain_slot(Kind kind) {
        switch (kind) {
            case Ice: return 0;
            case Wood: return 1;
            case Char: return 2;
            case Mud: return 4;
            case Steam: return 5;
            case Lava: return 6;
            case Vine: return 7;
            default: return 3; // Ash
        }
    }
    const std::vector<int> &element_cells(Kind kind) const { return element_positions[kind - Ice]; }
    void mark_element(Kind kind) {
        if (!element_kind(kind)) return;
        ++element_versions[kind - Ice];
        ++elements_version;
    }
    bool mobile_at(int i) const { return (mobile_rows[i / block] & (1u << (i % block))) != 0; }
    void set_mobile(int i, bool mobile) {
        const auto bit = uint8_t(1u << (i % block));
        const int row = i / block;
        auto &cells = mobile_rows[row];
        const bool was_empty = cells == 0;
        if (mobile) cells |= bit;
        else cells &= uint8_t(~bit);
        if (was_empty == (cells == 0)) return;
        const int line = row / bx;
        auto &group = mobile_row_groups[line / block * bx + row % bx];
        const auto lane = uint8_t(1u << (line % block));
        if (was_empty) group |= lane;
        else group &= uint8_t(~lane);
    }

    void sync_bulk_water(int i) {
        const auto bit = uint8_t(1u << (i % block));
        const Voxel &cell = voxels[i];
#ifdef SOMEVOXEL_PARALLEL_WATER_WORKERS
        if (cell.kind == Water && i >= water_scan_end) water_scan_end = (i / slice + 1) * slice;
#endif
        if (cell.kind == Water && !cell.liquid_moving) bulk_water_rows[i / block] |= bit;
        else bulk_water_rows[i / block] &= uint8_t(~bit);
    }

    static int patch_id(I3 p) { return p.x / patch_width + (p.z / patch_width) * patch_x; }

    template <int Width, int Columns, size_t Count>
    static void dirty_surface(I3 p, std::array<bool, Count> &dirty) {
        const int id = p.x / Width + (p.z / Width) * Columns;
        dirty[id] = true;
        if (p.x > 0 && p.x % Width == 0) dirty[id - 1] = true;
        if (p.x + 1 < nx && (p.x + 1) % Width == 0) dirty[id + 1] = true;
        if (p.z > 0 && p.z % Width == 0) dirty[id - Columns] = true;
        if (p.z + 1 < nz && (p.z + 1) % Width == 0) dirty[id + Columns] = true;
    }

    void dirty_water(I3 p) {
        if (track_water_patches) dirty_surface<patch_width, patch_x>(p, water_dirty);
    }

    void wake_box(I3 first, I3 last, ChunkActivity &chunks) {
        const I3 lo(std::max(0, first.x - 1) / block, std::max(0, first.y - 1) / block, std::max(0, first.z - 1) / block);
        const I3 hi(std::min(nx - 1, last.x + 1) / block, std::min(ny - 1, last.y + 1) / block, std::min(nz - 1, last.z + 1) / block);
        for (int y = lo.y; y <= hi.y; ++y)
            for (int z = lo.z; z <= hi.z; ++z)
                for (int x = lo.x; x <= hi.x; ++x) chunks.set({x, y, z});
    }

    void wake_near(I3 p, ChunkActivity &chunks) {
        // An interior cell and all of its neighbors are in the same chunk.
        const int x = p.x & (block - 1), y = p.y & (block - 1), z = p.z & (block - 1);
        if (x > 0 && x < block - 1 && y > 0 && y < block - 1 && z > 0 && z < block - 1) {
            chunks.set_for_cell(p);
            return;
        }
        wake_box(p, p, chunks);
    }

    void wake_field() {
        // Account for a grain's remaining sub-cell travel before discarding
        // chunks that cannot meet the spherical force field.
        const float radius = 2.4f * cast_radius_scale + 1.75f * scale;
        const V3 reach(radius, radius, radius);
        const I3 low = grid(field - reach), high = grid(field + reach);
        if (low.x >= nx || low.y >= ny || low.z >= nz || high.x < 0 || high.y < 0 || high.z < 0) return;
        const I3 lo(std::max(0, low.x) / block, std::max(0, low.y) / block, std::max(0, low.z) / block);
        const I3 hi(std::min(nx - 1, high.x) / block, std::min(ny - 1, high.y) / block, std::min(nz - 1, high.z) / block);
        const float side = block * scale;
        const auto outside = [side](float value, int chunk) {
            const float near = chunk * side;
            return value - std::clamp(value, near, near + side);
        };
        for (int y = lo.y; y <= hi.y; ++y) {
            const float dy = outside(field.y, y);
            for (int z = lo.z; z <= hi.z; ++z) {
                const float dz = outside(field.z, z);
                const float x_budget = radius * radius - dy * dy - dz * dz;
                if (x_budget < 0.f) continue;
                for (int x = lo.x; x <= hi.x; ++x) {
                    const float dx = outside(field.x, x);
                    if (dx * dx <= x_budget) {
                        const int id = chunk_id({x, y, z});
                        awake.set({x, y, z});
                        settled_sand_chunks[id] = false;
                    }
                }
            }
        }
    }

    void move(I3 from, I3 to) {
        const int a = flat(from), b = flat(to);
        move_earth_ejecta(a, b);
        Voxel &source = voxels[a], &destination = voxels[b];
        if (source.kind == Water) {
            water_positions[source.occupant] = b;
        }
        if (destination.kind == Water) {
            water_positions[destination.occupant] = a;
        }
        const bool water_moved = source.kind == Water || destination.kind == Water;
        const bool water_surface_moved = (source.kind == Water && !source.formed) ||
            (destination.kind == Water && !destination.formed);
        if (water_moved) {
            ++water_topology_version;
            if (source.kind != destination.kind) {
                const int origin = chunk_at(from), target = chunk_at(to);
                if (origin != target) {
                    const bool from_water = source.kind == Water;
                    --water_chunk_counts[from_water ? origin : target];
                    ++water_chunk_counts[from_water ? target : origin];
                }
            }
        }
        if (water_surface_moved) {
            ++water_version;
            dirty_water(from);
            dirty_water(to);
        }
        // Water already dirtied both positions. Dry particles only affect its
        // mesh if they actually change which faces are occluded.
        if (!water_surface_moved && (rock_surface(source.kind) != rock_surface(destination.kind))) {
            const std::array<I3, 6> faces = {{{0,1,0}, {0,-1,0}, {1,0,0}, {-1,0,0}, {0,0,1}, {0,0,-1}}};
            for (I3 direction : faces) {
                if (!surface_water_at(from + direction) && !surface_water_at(to + direction)) continue;
                ++water_version;
                dirty_water(from);
                dirty_water(to);
                break;
            }
        }
        if (source.kind == Rock) record_grain_move(rock_fragments[source.occupant], b, moved_rock);
        if (source.kind == Sand) {
            record_grain_move(grains[source.occupant], b, moved_grains);
            mark_sand_tiles(from);
            mark_sand_tiles(to);
        }
        if (source.kind != destination.kind && (source.kind == Sand || destination.kind == Sand)) {
            const int origin = chunk_at(from), target = chunk_at(to);
            if (origin != target) {
                const int delta = source.kind == Sand ? -1 : 1;
                change_sand_count(origin, delta);
                change_sand_count(target, -delta);
            }
        }
        const auto shift_element = [&](const Voxel &cell, int target) {
            if (!element_kind(cell.kind)) return;
            element_positions[cell.kind - Ice][cell.occupant] = target;
            if (element_particle(cell.kind)) {
                const int slot = element_grain_slot(cell.kind);
                Grain &grain = element_grains[slot][cell.occupant];
                const bool debris = structure_kind(cell.kind) && mobile_at(grain.position);
                if (grain.moved != tick) grain.previous = grain.position;
                grain.position = target;
                grain.moved = tick;
                if (debris) {
                    ++debris_revisions[slot];
                    debris_moved[slot] = true;
                    return;
                }
            }
            mark_element(cell.kind);
        };
        if (element_kind(source.kind) || element_kind(destination.kind)) {
            if (source.kind == Vine || destination.kind == Vine) vine_support_dirty = true;
            if (source.kind == Mud || destination.kind == Mud) {
                invalidate_support_above(source.kind == Mud ? from : to, 1);
                invalidate_vine_support_around(from);
                invalidate_vine_support_around(to);
            }
            shift_element(source, b);
            shift_element(destination, a);
        }
        const bool source_mobile = mobile_at(a);
        const bool destination_mobile = mobile_at(b);
        std::swap(source, destination);
        source.stamp = voxel_stamp;
        destination.stamp = voxel_stamp;
        set_mobile(a, destination_mobile);
        set_mobile(b, source_mobile);
        if (water_moved) {
            sync_bulk_water(a);
            sync_bulk_water(b);
        }
        const bool wetting_moved = !water_positions.empty() && (water_moved ||
            (source.kind == Sand && (!wet_sand.empty() || water_in_contact_chunks(from))) ||
            (destination.kind == Sand && (!wet_sand.empty() || water_in_contact_chunks(to))));
        if (wetting_moved || track_fuel_wetting || !reaction_cells.empty()) move_element_state(a, b);
        if (source.liquid_moving || destination.liquid_moving) move_liquid_motion(a, b);
        if (source.formed || destination.formed) move_form_state(a, b);
        wake_near(from, next_awake);
        wake_near(to, next_awake);
    }

    void record_grain_move(Grain &grain, int destination, int &moved) {
        if (grain.moved != tick) grain.previous = grain.position;
        grain.position = destination;
        grain.moved = tick;
        ++moved;
    }

    static constexpr std::array<int, 8> gravel_offsets() {
        return {0, 1, nx, nx + 1, slice, slice + 1, slice + nx, slice + nx + 1};
    }

    template <typename Function>
    static void each_cube(int anchor, int extent, Function apply) {
        for (int y = 0; y < extent; ++y)
            for (int z = 0; z < extent; ++z)
                for (int x = 0; x < extent; ++x) apply(anchor + x + z * nx + y * slice);
    }

    static bool rock_surface(Kind kind) {
        return kind == Stone || kind == Rock || kind == Bedrock || structure_kind(kind) ||
            kind == Vine || kind == Mud || kind == Lava;
    }
    static bool intact_stone(Kind kind) { return kind == Stone || kind == Bedrock; }
    static int preset_extent(int kind) {
        if (kind == Gravel) return 2;
        if (kind == Preset18) return 4;
        if (kind == Preset36) return 8;
        return 0;
    }
    uint8_t body_flags(int i) const {
        return (body_mask.get(i / 4) >> ((i % 4) * 2)) & 3;
    }
    void set_body_flags(int i, uint8_t flags) {
        const int shift = (i % 4) * 2;
        const uint8_t lane = body_mask.get(i / 4);
        body_mask.set(i / 4, uint8_t((lane & ~(3u << shift)) | (flags << shift)));
    }
    bool body_at(int i) const { return !body_mask.empty() && body_flags(i) != 0; }

    void set_rock(int i) {
        const I3 p = decode(i);
        invalidate_vine_support_around(p);
        dirty_surface<stone_patch_width, stone_patch_x>(p, stone_dirty);
        voxels[i] = {Rock, voxel_stamp, int(rock_fragments.size())};
        rock_fragments.push_back({i, i, tick});
        set_mobile(i, true);
        wake_near(p, awake);
    }

    SupportStamp next_support_search(int marks = 1) {
        if (support_epoch > std::numeric_limits<SupportStamp>::max() - marks) {
            support_seen.clear();
            support_epoch = 0;
        }
        const SupportStamp first = support_epoch + 1;
        support_epoch += marks;
        return first;
    }

    void release_unsupported(const std::vector<int> &seeds) {
        // A completed search leaves only anchored Stone or detached Rock.
        // Reuse the visited mark for all seeds and never wrap mid-search.
        const SupportStamp anchored = next_support_search(2);
        const SupportStamp visited = anchored + 1;
        const std::array<I3, 6> directions = {{{0,-1,0}, {1,0,0}, {-1,0,0}, {0,0,1}, {0,0,-1}, {0,1,0}}};
        std::vector<int> component;
        for (int seed : seeds) {
            if (voxels[seed].kind != Stone || support_seen.get(seed) == anchored) continue;
            component.clear();
            component.push_back(seed);
            support_seen.set(seed, visited);
            bool supported = false;
            for (size_t cursor = 0; cursor < component.size() && !supported; ++cursor) {
                const I3 p = decode(component[cursor]);
                for (I3 d : directions) {
                    const I3 q = p + d;
                    if (!inside(q)) continue;
                    const int i = flat(q);
                    if (voxels[i].kind == Bedrock || (voxels[i].kind == Stone && support_seen.get(i) == anchored)) {
                        supported = true;
                        break;
                    }
                    if (voxels[i].kind != Stone || support_seen.get(i) == visited) continue;
                    support_seen.set(i, visited);
                    component.push_back(i);
                }
            }
            if (supported) {
                for (int i : component) support_seen.set(i, anchored);
            } else {
                for (int i : component) set_rock(i);
            }
        }
    }

    void settle_unanchored_layout(int layout_end) {
        const SupportStamp visited = next_support_search();
        std::vector<int> anchored;
        for (int i = slice; i < std::min(2 * slice, layout_end); ++i) {
            if (voxels[i].kind != Stone || voxels[i - slice].kind != Bedrock) continue;
            support_seen.set(i, visited);
            anchored.push_back(i);
        }
        const std::array<I3, 6> dirs = {{{0,1,0}, {0,-1,0}, {1,0,0}, {-1,0,0}, {0,0,1}, {0,0,-1}}};
        for (size_t cursor = 0; cursor < anchored.size(); ++cursor) {
            const I3 p = decode(anchored[cursor]);
            for (I3 direction : dirs) {
                const I3 target = p + direction;
                if (!inside(target)) continue;
                const int i = flat(target);
                if (voxels[i].kind != Stone || support_seen.get(i) == visited) continue;
                support_seen.set(i, visited);
                anchored.push_back(i);
            }
        }
        for (int i = slice; i < layout_end; ++i)
            if (voxels[i].kind == Stone && support_seen.get(i) != visited) set_rock(i);
    }

    void erase_sand(int id) {
        change_sand_count(chunk_at(decode(grains[id].position)), -1);
        if (!sand_tile_dirty.empty()) mark_sand_tiles(decode(grains[id].position));
        if (!airborne_earth_ejecta.empty()) airborne_earth_ejecta.erase(grains[id].position);
        if (!wet_sand.empty()) wet_sand.erase(grains[id].position);
        forget_particle_form(grains[id].position);
        const int last = int(grains.size()) - 1;
        if (id != last) {
            grains[id] = grains[last];
            voxels[grains[id].position].occupant = id;
            if (!sand_tile_dirty.empty()) mark_sand_tile(sand_tile_id(decode(grains[id].position)));
        }
        grains.pop_back();
    }

    void attach_element(int i, Kind kind) {
        auto &positions = element_positions[kind - Ice];
        voxels[i] = {kind, voxel_stamp, int(positions.size())};
        positions.push_back(i);
        if (kind == Vine) {
            detached_vines.push_back(0);
            vine_roots.push_back(0);
            vine_support_dirty = true;
        }
        if (element_particle(kind)) element_grains[element_grain_slot(kind)].push_back({i, i, tick});
    }

    void erase_element(const Voxel &cell) {
        auto &positions = element_positions[cell.kind - Ice];
        const int id = cell.occupant;
        const int last = int(positions.size()) - 1;
        if (id != last) {
            positions[id] = positions[last];
            voxels[positions[id]].occupant = id;
        }
        positions.pop_back();
        if (cell.kind == Vine) {
            if (id != last) {
                detached_vines[id] = detached_vines[last];
                vine_roots[id] = vine_roots[last];
            }
            detached_vines.pop_back();
            vine_roots.pop_back();
            vine_support_dirty = true;
        }
        if (element_particle(cell.kind)) {
            auto &particles = element_grains[element_grain_slot(cell.kind)];
            if (id != last) particles[id] = particles[last];
            particles.pop_back();
        }
    }

    bool anchored_ice_site(I3 point) const {
        for (I3 side : structural_neighbors()) {
            const I3 neighbor = point + side;
            if (!inside(neighbor)) continue;
            const int index = flat(neighbor);
            const Kind material = voxels[index].kind;
            if (intact_stone(material) || (structure_kind(material) && !mobile_at(index)) ||
                    (side.y < 0 && (material == Gravel || material == Mud))) return true;
        }
        return false;
    }

    bool replace_element_cell(int i, Kind next) {
        if (i < 0 || i >= int(voxels.size())) return false;
        const Voxel before = voxels[i];
        if (before.kind == next) return true;
        const auto supported_kind = [](Kind k) {
            return k == Empty || k == Stone || k == Sand || k == Water || element_kind(k);
        };
        if (!supported_kind(before.kind) || !supported_kind(next)) return false;
        if (body_at(i) && next != Empty && next != Steam && next != Grass) return false;
        if (!terrain_strain.empty()) terrain_strain.erase(i);
        const I3 p = decode(i);
        const bool retain_grain = element_particle(before.kind) && element_particle(next);
        Grain prior_motion{i, i, tick};
        const bool was_mobile = mobile_at(i);
        // Unsupported parcels keep their momentum when the material slot changes.
        // Water freezing against fixed terrain still joins the support graph.
        const bool from_liquid = before.kind == Water && before.liquid_moving &&
            (next == Steam || (next == Ice && !anchored_ice_site(p)));
        const bool to_liquid = (before.kind == Ice || before.kind == Steam) && next == Water &&
            was_mobile && liquid_motion_has_room();
        if (retain_grain || to_liquid)
            prior_motion = element_grains[element_grain_slot(before.kind)][before.occupant];
        if (from_liquid) prior_motion = liquid_particle(i);
        const bool handoff_motion = (from_liquid || to_liquid) && prior_motion.velocity.length_squared() > 0.0009f;
        if (handoff_motion) prior_motion.velocity *= particle_mass(before.kind, prior_motion) / particle_mass(next, prior_motion);
        if (before.kind == Water) {
            forget_particle_form(i);
            if (before.liquid_moving) remove_liquid_motion(i);
            const int tail = water_positions.back();
            water_positions[before.occupant] = tail;
            voxels[tail].occupant = before.occupant;
            water_positions.pop_back();
        } else if (before.kind == Sand) {
            erase_sand(before.occupant);
        } else if (element_kind(before.kind)) {
            erase_element(before);
        }

        voxels[i] = {next, voxel_stamp, -1};
        if (next == Water) {
            voxels[i].occupant = int(water_positions.size());
            water_positions.push_back(i);
            if (handoff_motion) ensure_liquid_motion(i) = prior_motion;
        } else if (next == Sand) {
            voxels[i].occupant = int(grains.size());
            add_sand_grain({i, i, tick});
            mark_sand_tiles(p);
        } else if (element_kind(next)) {
            attach_element(i, next);
            if (retain_grain || handoff_motion)
                element_grains[element_grain_slot(next)][voxels[i].occupant] = prior_motion;
        }
        set_mobile(i, initially_mobile(next) || (retain_grain && was_mobile) || handoff_motion);
        if (before.kind == Water || next == Water) sync_bulk_water(i);
        if (before.kind == Sand || next == Sand) ++sand_revision;
        mark_element(before.kind);
        mark_element(next);
        if (before.kind == Water || next == Water) {
            if (next == Water) ++water_chunk_counts[chunk_at(p)];
            else --water_chunk_counts[chunk_at(p)];
            ++water_topology_version;
        }
        if (before.kind == Water || next == Water || rock_surface(before.kind) != rock_surface(next)) {
            ++water_version;
            dirty_water(p);
        }
        if (before.kind == Stone || next == Stone) {
            ++stone_version;
            dirty_surface<stone_patch_width, stone_patch_x>(p, stone_dirty);
            if (next == Stone) stone_support_pending.push_back(i);
            if (before.kind == Stone) {
                for (I3 d : structural_neighbors()) {
                    const I3 q = p + d;
                    if (inside(q) && kind_at(q) == Stone) stone_support_pending.push_back(flat(q));
                }
            }
        }
        if (structure_kind(before.kind) || structure_kind(next) || before.kind == Stone || next == Stone)
            element_support_dirty = true;
        if (before.kind == Mud) invalidate_support_above(p, 1);
        if (next == Sand) queue_sand_contact(i);
        if (next == Water) queue_water_contact(i);
        wake_near(p, awake);
        if (vine_anchor_kind(before.kind) || vine_anchor_kind(next)) invalidate_vine_support_around(p);
        return true;
    }

    static std::array<I3, 6> structural_neighbors() {
        return {{{0,-1,0}, {0,1,0}, {1,0,0}, {-1,0,0}, {0,0,1}, {0,0,-1}}};
    }

    void detach_element(int i) {
        if (vine_anchor_kind(voxels[i].kind)) invalidate_vine_support_around(decode(i));
        set_mobile(i, true);
        mark_element(voxels[i].kind);
        wake_near(decode(i), awake);
    }

    static bool vine_anchor_kind(Kind kind) {
        return intact_stone(kind) || kind == Wood || kind == Mud;
    }

    bool vine_touches_anchor(I3 p) const {
        for (I3 d : structural_neighbors()) {
            const I3 q = p + d;
            if (!inside(q)) continue;
            const int i = flat(q);
            const Kind kind = voxels[i].kind;
            if (vine_anchor_kind(kind) && (!mobile_at(i) || (kind == Mud && !can_enter(q - I3(0, 1, 0), Mud)))) return true;
        }
        return false;
    }

    bool vine_has_anchor(I3 p) const {
        return vine_roots[voxels[flat(p)].occupant] && vine_touches_anchor(p);
    }

    void invalidate_vine_support_around(I3 p) {
        if (vine_support_dirty || element_cells(Vine).empty()) return;
        for (I3 d : structural_neighbors()) {
            if (kind_at(p + d) == Vine) {
                vine_support_dirty = true;
                return;
            }
        }
    }

    void refresh_vine_support() {
        if (!vine_support_dirty) return;
        vine_support_dirty = false;
        const auto &positions = element_cells(Vine);
        if (positions.empty()) return;
        const SupportStamp visited = next_support_search();
        std::vector<int> rooted;
        for (int i : positions) {
            auto &root = vine_roots[voxels[i].occupant];
            if (root && !vine_touches_anchor(decode(i))) root = 0;
            if (!root) continue;
            support_seen.set(i, visited);
            rooted.push_back(i);
        }
        for (size_t cursor = 0; cursor < rooted.size(); ++cursor) {
            const I3 p = decode(rooted[cursor]);
            for (I3 d : structural_neighbors()) {
                const I3 q = p + d;
                if (!inside(q)) continue;
                const int i = flat(q);
                if (voxels[i].kind != Vine || support_seen.get(i) == visited) continue;
                support_seen.set(i, visited);
                rooted.push_back(i);
            }
        }
        for (int i : positions) {
            const bool detached = support_seen.get(i) != visited;
            auto &flag = detached_vines[voxels[i].occupant];
            if (detached && !flag) wake_near(decode(i), awake);
            flag = detached;
        }
    }

    void invalidate_support_above(I3 base, int extent) {
        const int y = base.y + extent;
        if (element_support_dirty || y >= ny ||
                (element_cells(Ice).empty() && element_cells(Wood).empty() && element_cells(Char).empty())) return;
        for (int z = base.z; z < base.z + extent; ++z)
            for (int x = base.x; x < base.x + extent; ++x) {
                const int i = flat({x, y, z});
                if (structure_kind(voxels[i].kind) && !mobile_at(i)) {
                    element_support_dirty = true;
                    return;
                }
            }
    }

    void refresh_element_support() {
        if (!stone_support_pending.empty()) {
            const size_t old_rock_count = rock_fragments.size();
            release_unsupported(stone_support_pending);
            stone_support_pending.clear();
            if (rock_fragments.size() != old_rock_count) {
                ++stone_version;
                ++rock_revision;
            }
        }
        if (!element_support_dirty) return;
        element_support_dirty = false;
        if (element_cells(Ice).empty() && element_cells(Wood).empty() && element_cells(Char).empty()) return;
        const SupportStamp visited = next_support_search();
        std::vector<int> connected;
        for (Kind kind : {Ice, Wood, Char}) {
            for (int seed : element_cells(kind)) {
                if (mobile_at(seed) || support_seen.get(seed) == visited) continue;
                connected.clear();
                connected.push_back(seed);
                support_seen.set(seed, visited);
                bool supported = false;
                for (size_t cursor = 0; cursor < connected.size(); ++cursor) {
                    const I3 p = decode(connected[cursor]);
                    for (I3 d : structural_neighbors()) {
                        const I3 q = p + d;
                        if (!inside(q)) continue;
                        const int i = flat(q);
                        const Kind neighbor = voxels[i].kind;
                        if (intact_stone(neighbor) || (d.y < 0 && (neighbor == Gravel || neighbor == Mud)))
                            supported = true;
                        if (!structure_kind(neighbor) || mobile_at(i) || support_seen.get(i) == visited) continue;
                        support_seen.set(i, visited);
                        connected.push_back(i);
                    }
                }
                if (!supported) for (int i : connected) detach_element(i);
            }
        }
    }

    void erase_gravel(int id) {
        forget_particle_form(gravel[id].position);
        if (!sand_tile_dirty.empty()) mark_gravel_tile(decode(gravel[id].position));
        if (!airborne_earth_ejecta.empty()) airborne_earth_ejecta.erase(gravel[id].position);
        const int last = int(gravel.size()) - 1;
        if (!packed_strain.empty()) {
            packed_strain.erase(id);
            if (id != last) {
                auto entry = packed_strain.extract(last);
                if (entry) {
                    entry.key() = id;
                    packed_strain.insert(std::move(entry));
                }
            }
        }
        for (auto it = tracked_packed_particles.begin(); it != tracked_packed_particles.end();) {
            if (it->second == id) {
                it = tracked_packed_particles.erase(it);
            } else {
                if (it->second == last) it->second = id;
                ++it;
            }
        }
        if (id != last) {
            gravel[id] = gravel[last];
            each_cube(gravel[id].position, gravel[id].extent, [&](int i) { voxels[i].occupant = id; });
            if (!sand_tile_dirty.empty()) mark_gravel_tile(decode(gravel[id].position));
        }
        gravel.pop_back();
    }

    bool fits_gravel(I3 target, int id, int *blocking_voxel = nullptr) const {
        const Grain &grain = gravel[id];
        const int extent = grain.extent;
        if (!inside(target) || !inside(target + I3(extent - 1, extent - 1, extent - 1))) return false;
        const I3 source = decode(grain.position);
        for (int y = target.y; y < target.y + extent; ++y) {
            for (int z = target.z; z < target.z + extent; ++z) {
                const bool same_row = y >= source.y && y < source.y + extent && z >= source.z && z < source.z + extent;
                for (int x = target.x; x < target.x + extent; ++x) {
                    if (same_row && x >= source.x && x < source.x + extent) {
                        x = source.x + extent - 1;
                        continue;
                    }
                    const int i = flat({x, y, z});
                    if (body_at(i)) return false;
                    if (voxels[i].kind != Empty) {
                        if (blocking_voxel) *blocking_voxel = i;
                        return false;
                    }
                }
            }
        }
        return true;
    }

    bool gravel_awake(I3 from, int extent) const {
        const I3 lo(from.x / block, from.y / block, from.z / block);
        const I3 hi((from.x + extent - 1) / block, (from.y + extent - 1) / block, (from.z + extent - 1) / block);
        for (int y = lo.y; y <= hi.y; ++y)
            for (int z = lo.z; z <= hi.z; ++z)
                for (int x = lo.x; x <= hi.x; ++x)
                    if (awake.test({x, y, z})) return true;
        return false;
    }

    void move_gravel(int id, I3 target, ChunkActivity &wake,
            const GravelLiquidMove *liquid = nullptr) {
        Grain &grain = gravel[id];
        const int origin = grain.position;
        auto form = voxels[origin].formed ? form_motion.extract(origin) : decltype(form_motion)::node_type{};
        const I3 from = decode(grain.position);
        const int destination = flat(target);
        move_earth_ejecta(grain.position, destination);
        invalidate_support_above(from, grain.extent);
        each_cube(grain.position, grain.extent, [&](int i) { voxels[i] = {}; });
        if (liquid) commit_gravel_liquid_move(grain, *liquid);
        each_cube(destination, grain.extent, [&](int i) { voxels[i] = {Gravel, voxel_stamp, id}; });
        if (form) {
            form.key() = destination;
            form_motion.insert(std::move(form));
            voxels[destination].formed = true;
        }
        record_grain_move(grain, destination, moved_gravel);
        mark_gravel_tile(from);
        mark_gravel_tile(target);
        const int edge = grain.extent - 1;
        const I3 corner(edge, edge, edge);
        wake_box(from, from + corner, wake);
        wake_box(target, target + corner, wake);
    }

    bool can_enter(I3 target, Kind moving, int *blocking_voxel = nullptr, bool rising_steam = false) const {
        if (!inside(target)) return false;
        const int index = flat(target);
        const Voxel &candidate = voxels[index];
        // Body occupancy can only coexist with materials it admits.
        if (passable_body(candidate.kind) && body_at(index)) return false;
        if (candidate.kind == Empty ||
                (candidate.stamp != voxel_stamp && (
                    ((moving == Sand || moving == Rock || moving == Mud) && candidate.kind == Water) ||
                    (moving == Water && (candidate.kind == Ash || candidate.kind == Steam)) ||
                    (rising_steam && candidate.kind == Water))))
            return true;
        if (blocking_voxel) *blocking_voxel = index;
        return false;
    }

    static V3 particle_center(I3 from, int extent) {
        const float half = extent * 0.5f;
        return (V3(from) + V3(half, half, half)) * scale;
    }

    static constexpr GranularMaterial granular_material(Kind kind) {
        if (kind == Water) return {1000.f, 0.f, 0.62f};
        if (kind == Rock) return {rock_density, 0.f, 0.82f};
        if (kind == Ice) return {917.f, 0.f, 0.96f};
        if (kind == Wood) return {580.f, 0.f, 0.85f};
        if (kind == Char) return {330.f, 0.f, 0.83f};
        if (kind == Ash) return {170.f, 0.f, 0.85f};
        if (kind == Mud) return {sand_density + granular_material(Water).density, 2.40f, 0.70f};
        // Each steam voxel carries the water equivalent of a liquid voxel.
        if (kind == Steam) return {1000.f, 0.f, 1.f, false};
        if (kind == Lava) return {2600.f, 1.45f, 0.72f, false};
        if (kind == Vine) return {800.f, 0.f, 0.85f, false};
        return {sand_density, kind == Gravel ? 1.85f : 2.40f, 0.83f};
    }

    static float step_ground_retention(Kind kind, float retention) {
        if constexpr (simulation_hz == reference_hz) return retention;
        static const auto coefficients = [] {
            std::array<float, Char + 1> values{};
            for (int i = 0; i <= Char; ++i)
                values[i] = std::pow(granular_material(Kind(i)).ground_retention, float(reference_hz) / simulation_hz);
            return values;
        }();
        return coefficients[kind];
    }

    static constexpr float particle_mass(Kind kind, int extent) {
        const float width = extent * scale;
        return granular_material(kind).density * width * width * width;
    }

    static int cube_cells(int extent) { return extent * extent * extent; }

    static int gravel_mass_units(const Grain &grain) {
        return grain.sand_units ? grain.sand_units : cube_cells(grain.extent);
    }

    static float gravel_density_ratio(const Grain &grain) {
        return grain.sand_units ? float(grain.sand_units) / cube_cells(grain.extent) : 1.f;
    }

    static float particle_mass(Kind kind, const Grain &grain) {
        return particle_mass(kind, grain.extent) * gravel_density_ratio(grain);
    }

    V3 pull_acceleration(V3 offset, V3 velocity, float radius, float mass_response, float stiffness, float damping) const {
        const float edge = std::clamp((1.f - offset.length_squared() / (radius * radius)) * 2.8f, 0.f, 1.f);
        const float response = edge * mass_response * cast_strength;
        return offset * (stiffness * response) - velocity * (damping * std::sqrt(response));
    }

    Grain *movable_particle_at(int i) {
        if (i < 0) return nullptr;
        const Voxel &cell = voxels[i];
        switch (cell.kind) {
            case Sand: return &grains[cell.occupant];
            case Rock: return &rock_fragments[cell.occupant];
            case Gravel: return &gravel[cell.occupant];
            case Water: return cell.liquid_moving ? &liquid_particle(i) : nullptr;
            case Vine:
                if (vine_has_anchor(decode(i)) || (!pulling_vines && !detached_vines[cell.occupant])) return nullptr;
                [[fallthrough]];
            case Ice: case Wood: case Char: case Ash: case Mud: case Steam: case Lava:
                return mobile_at(i) ? &element_grains[element_grain_slot(cell.kind)][cell.occupant] : nullptr;
            default: return nullptr;
        }
    }

    void share_sand_contact(Grain &grain, int axis, int direction, int obstacle, I3 attempted) {
        Grain &other = grains[voxels[obstacle].occupant];
        const float relative_speed = (grain.velocity[axis] - other.velocity[axis]) * direction;
        if (relative_speed > 0.f) {
            constexpr float mass = particle_mass(Sand, 1);
            if (relative_speed >= minimum_sand_compaction_speed)
                record_compaction_load(attempted, {0.125f * mass * relative_speed * relative_speed, {}});
            const float shared = (grain.velocity[axis] * mass + other.velocity[axis] * mass) / (mass + mass);
            grain.velocity[axis] = other.velocity[axis] = shared;
            const int chunk = chunk_at(attempted);
            next_awake.set_for_cell(attempted);
            settled_sand_chunks[chunk] = false;
        }
    }

    void resolve_granular_contact(Grain &grain, Kind kind, int axis, int direction, int blocking_voxel,
            float kinetic_velocity, bool density_exchange, I3 attempted) {
        if (profile_sand_sweep && kind == Sand) ++sand_sweep.contacts;
        if (kind == Sand && blocking_voxel >= 0 && voxels[blocking_voxel].kind == Sand) {
            share_sand_contact(grain, axis, direction, blocking_voxel, attempted);
            return;
        }
        const bool terrain_impact = !density_exchange && blocking_voxel >= 0 && kinetic_velocity * direction > 2.4f;
        const auto kinetic_projectile = [&]() {
            Grain projectile = grain;
            projectile.velocity[axis] = kinetic_velocity;
            return projectile;
        };
        if (terrain_impact && kind == Gravel &&
                record_gravel_terrain_contacts(attempted, kinetic_projectile(), axis, direction)) {
            const float momentum = particle_mass(kind, grain) * std::abs(kinetic_velocity);
            // The rigid side and the projectile each absorb a fraction of the
            // incident work. The remaining normal energy is dissipated.
            record_packed_load(voxels[grain.position].occupant, {0.02f * momentum * std::abs(kinetic_velocity), {}});
            grain.velocity[axis] = 0.f;
            return;
        }
        Grain *other = movable_particle_at(blocking_voxel);
        if (!other && density_exchange && voxels[blocking_voxel].kind == Water &&
                std::abs(grain.velocity[axis]) > 1.7f && liquid_motion_has_room())
            other = &ensure_liquid_motion(blocking_voxel);
        if (!other) {
            if (!density_exchange) {
                if (terrain_impact && kind != Gravel)
                    record_terrain_contact(blocking_voxel, kinetic_projectile(), kind, axis, direction);
                grain.velocity[axis] = 0.f;
            }
            return;
        }
        const float relative_speed = (grain.velocity[axis] - other->velocity[axis]) * direction;
        if (relative_speed <= 0.f) return;
        // Inelastic contact: share normal momentum and keep tangential motion.
        // Occupancy is still committed separately through the face test.
        const float mass = particle_mass(kind, grain);
        const float other_mass = particle_mass(voxels[blocking_voxel].kind, *other);
        const Kind other_kind = voxels[blocking_voxel].kind;
        if (kind == Gravel || other_kind == Gravel) {
            const float reduced_mass = mass * other_mass / (mass + other_mass);
            const float lost_energy = 0.5f * reduced_mass * relative_speed * relative_speed;
            const MaterialLoad fracture{0.12f * lost_energy, {}};
            if (kind == Gravel) record_packed_load(voxels[grain.position].occupant, fracture);
            if (other_kind == Gravel) record_packed_load(voxels[blocking_voxel].occupant, fracture);
            if (other_kind == Sand) record_compaction_load(decode(blocking_voxel), {0.50f * lost_energy, {}});
        }
        const float shared_speed = (grain.velocity[axis] * mass + other->velocity[axis] * other_mass)
            / (mass + other_mass);
        grain.velocity[axis] = shared_speed;
        other->velocity[axis] = shared_speed;
        if (other_kind == Sand) settled_sand_chunks[chunk_at(decode(blocking_voxel))] = false;
        const I3 p = kind == Gravel ? decode(blocking_voxel) : attempted;
        next_awake.set_for_cell(p);
    }

    struct ParticlePath {
        I3 target;
        I3 exchange_from;
        bool gravel_water_exchange = false;
    };

    void commit_particle_path(I3 from, const ParticlePath &path) {
        if (path.target == from) return;
        if (path.exchange_from != from) move(from, path.exchange_from);
        move(path.exchange_from, path.target);
    }

    void move_across_two_faces(I3 from, I3 via, I3 to) {
        if (kind_at(via) != Empty) {
            move(from, via);
            return;
        }
        commit_particle_path(from, {to, kind_at(to) == Empty ? from : via});
    }

    static uint8_t grain_face_bit(int axis, int direction) { return uint8_t(1u << (2 * axis + (direction > 0))); }
    static uint8_t grain_axis_faces(int axis) { return uint8_t(3u << (2 * axis)); }

    bool resting_sand(I3 p, int index, Grain &grain) {
        if (grain.velocity != V3() || winding || !airborne_earth_ejecta.empty() || p.y == 0) return false;
        const int below = index - slice;
        const Kind support = voxels[below].kind;
        if (support == Empty || support == Water) return false;
        if (pulling) {
            const float reach = granular_material(Sand).attraction_range * cast_radius_scale;
            const V3 offset = field - particle_center(p, grain.extent) - grain.travel;
            if (offset.length_squared() < reach * reach) return false;
        }
        for (I3 side : cardinal())
            if (can_enter(p + side, Sand) && can_enter(p + side + I3(0, -1, 0), Sand)) return false;

        // Apply the same support bookkeeping as the zero-velocity solver exit.
        grain.travel.y = 0.f;
        grain.blocked_faces &= uint8_t(~grain_axis_faces(1));
        if (retains_grain_face(below)) grain.blocked_faces |= grain_face_bit(1, -1);
        return true;
    }

    bool retains_grain_face(int obstacle) const {
        if (obstacle < 0) return false;
        const Kind surface = voxels[obstacle].kind;
        return (surface == Bedrock || terrain_toughness(surface) > 0.f) && !mobile_at(obstacle);
    }

    static float integrate_axis(Grain &grain, int axis, float acceleration, int &direction, float &interval) {
        float speed = std::clamp(grain.velocity[axis] + acceleration * dt, -maximum_swept_speed, maximum_swept_speed);
        if (acceleration == 0.f && std::abs(speed) < 0.03f) speed = 0.f;
        grain.velocity[axis] = speed;
        const float distance = speed * dt;
        const float previous = grain.travel[axis];
        grain.travel[axis] += distance;
        if (distance == 0.f) return std::numeric_limits<float>::infinity();
        direction = distance > 0.f ? 1 : -1;
        const bool resting_face = (grain.blocked_faces & grain_face_bit(axis, direction)) != 0;
        grain.blocked_faces &= uint8_t(~grain_axis_faces(axis));
        if (!resting_face && std::abs(grain.travel[axis]) < scale) return std::numeric_limits<float>::infinity();
        interval = scale / std::abs(distance);
        return resting_face ? 0.f : (direction * scale - previous) / distance;
    }

    template <typename Fits>
    ParticlePath sweep_particle(Grain &grain, I3 origin, Kind kind, V3 acceleration, Fits fits, FormMotion *form,
            GravelLiquidMove *liquid) {
        const float infinity = std::numeric_limits<float>::infinity();
        const V3 previous_velocity = grain.velocity;
        const V3 previous_travel = grain.travel;
        EarthImpactPeaks *earth_peaks = nullptr;
        if (!airborne_earth_ejecta.empty() && (kind == Sand || kind == Gravel || kind == Rock)) {
            const auto ejecta = airborne_earth_ejecta.find(grain.position);
            if (ejecta != airborne_earth_ejecta.end()) earth_peaks = ejecta->second.get();
        }
        V3 crossing(infinity, infinity, infinity), interval;
        I3 direction;
        uint8_t previous_faces = grain.blocked_faces;
        for (int axis = 0; axis < 3; ++axis)
            crossing[axis] = integrate_axis(grain, axis, acceleration[axis], direction[axis], interval[axis]);
        if (crossing == V3(infinity, infinity, infinity)) return {origin, origin};

        I3 position = origin;
        // The bound covers every face reached at the supported maximum
        // speed, plus a retained face at the start of the step.
        for (int face = 0; face < 3 * (max_faces_per_axis + 1); ++face) {
            int axis = 0;
            if (crossing.y < crossing[axis]) axis = 1;
            if (crossing.z < crossing[axis]) axis = 2;
            if (crossing[axis] > 1.000001f) break;

            const bool profile = profile_sand_sweep && kind == Sand;
            if (profile) ++sand_sweep.tested_faces;
            I3 next = position;
            next[axis] += direction[axis];
            int obstacle = -1;
            const bool open = fits(next, &obstacle);
            const bool wet = !open && liquid && gravel_can_enter_water(grain, obstacle);
            const uint8_t face_bit = grain_face_bit(axis, direction[axis]);
            const bool resting_face = crossing[axis] == 0.f && (previous_faces & face_bit) != 0;
            previous_faces &= uint8_t(~face_bit);
            if (resting_face && (open || wet)) {
                // A now-open neighbor must still take normal travel to enter.
                crossing[axis] = interval[axis] * (1.f - direction[axis] * previous_travel[axis] / scale);
                continue;
            }
            if (!open) {
                if (wet && plan_gravel_liquid_move(grain, position, next, axis, direction[axis], obstacle, *liquid)) {
                    grain.travel = {};
                    grain.blocked_faces = 0;
                    return {next, position, true};
                }
                if (retains_grain_face(obstacle)) grain.blocked_faces |= face_bit;
                // A body already resting against this face has no new free
                // travel to build impact speed. Momentum received before this
                // step still contributes, including pushes from other grains.
                const float kinetic_velocity = resting_face ? previous_velocity[axis] : grain.velocity[axis];
                resolve_granular_contact(grain, kind, axis, direction[axis], obstacle, kinetic_velocity, false, next);
            } else if ((form && contact_form_target(*form, kind, grain, position, next, grain.velocity, axis)) ||
                    (earth_peaks && axis == 1 && direction[axis] > 0 &&
                        record_earth_contact(grain, position, next, *earth_peaks))) {
                grain.velocity[axis] = 0.f;
            } else {
                if (kind != Gravel && kind_at(next) != Empty) {
                    const int displaced = flat(next);
                    resolve_granular_contact(grain, kind, axis, direction[axis], displaced, grain.velocity[axis], true, next);
                    // Commit the last empty cell first. The density exchange
                    // transfers the displaced material across this face only.
                    grain.travel = {};
                    grain.blocked_faces = 0;
                    if (profile) ++sand_sweep.crossed_faces;
                    return {next, position};
                }
                if (profile) ++sand_sweep.crossed_faces;
                position = next;
                grain.travel[axis] -= direction[axis] * scale;
                crossing[axis] += interval[axis];
                continue;
            }
            grain.travel[axis] = 0.f;
            crossing[axis] = infinity;
        }
        return {position, origin};
    }

    // Inside a solid volume of loose sand, all three attempted faces are
    // blocked for this entire grain update. Normal contact changes only its
    // own axis. Apply those contacts directly in the existing grain order.
    bool step_enclosed_sand(Grain &grain, I3 p, V3 acceleration, bool attracted, bool enclosed) {
        const int i = grain.position;
        if (!enclosed) {
            if (p.x == 0 || p.x == nx - 1 || p.y == 0 || p.y == ny - 1 || p.z == 0 || p.z == nz - 1)
                return false;
            if (voxels[i - 1].kind != Sand || voxels[i + 1].kind != Sand ||
                    voxels[i - nx].kind != Sand || voxels[i + nx].kind != Sand ||
                    voxels[i - slice].kind != Sand || voxels[i + slice].kind != Sand)
                return false;
        }

        if (!attracted && grain.velocity.y <= 0.f && acceleration.y <= 0.f) {
            grain.travel.y = 0.f;
            grain.blocked_faces &= uint8_t(~grain_axis_faces(1));
            if (grain.velocity.y < -0.3f) {
                if (profile_sand_sweep) ++sand_sweep.contacts;
                share_sand_contact(grain, 1, -1, i - slice, p + I3(0, -1, 0));
            } else {
                grain.velocity.y = 0.f;
            }
            acceleration.y = 0.f;
            const float retention = step_ground_retention(Sand, granular_material(Sand).ground_retention);
            grain.velocity.x *= retention;
            grain.velocity.z *= retention;
        }
        if (grain.velocity == V3() && acceleration == V3()) return true;
        constexpr int stride[] = {1, slice, nx};
        for (int axis = 0; axis < 3; ++axis) {
            int direction = 0;
            float interval;
            if (integrate_axis(grain, axis, acceleration[axis], direction, interval) > 1.000001f) continue;
            I3 attempted = p;
            attempted[axis] += direction;
            if (profile_sand_sweep) {
                ++sand_sweep.tested_faces;
                ++sand_sweep.contacts;
            }
            share_sand_contact(grain, axis, direction, i + stride[axis] * direction, attempted);
            grain.travel[axis] = 0.f;
        }
        if (grain.velocity.length_squared() > 0.0009f || (acceleration.length_squared() > 0.01f && attracted))
            next_awake.set_for_cell(p);
        return true;
    }

    V3 field_drive(const Grain &grain, I3 from, Kind kind, const FormMotion *form, bool &attracted,
            bool elastic_only = false) const {
        const GranularMaterial material = granular_material(kind);
        const float density_ratio = gravel_density_ratio(grain);
        if (form && form->expires == 0) {
            const V3 position = particle_center(from, grain.extent) + (elastic_only ? V3() : grain.travel);
            const V3 offset = form_destination(*form) - position;
            const bool lash = form->style == WhipForm && whip_is_lashing();
            const float response = form_body_speed(grain, particle_form.strength * particle_form.response_scale) *
                (lash ? 2.2f : 1.f);
            const float gather = form_confinement();
            V3 spring = offset;
            if (gather > 0.f) {
                V3 inward = particle_form.point - position;
                if (form->style == BladeForm)
                    inward -= particle_form.right * (0.85f * inward.dot(particle_form.right));
                spring += inward * gather;
            }
            attracted = true;
            const V3 elastic = spring * (110.f * response);
            return elastic_only ? elastic : elastic - grain.velocity * (12.f * std::sqrt(response * (1.f + gather)));
        } else if (!form && pulling && material.attraction_range > 0.f) {
            const V3 offset = field - particle_center(from, grain.extent) - (elastic_only ? V3() : grain.travel);
            const float reach = material.attraction_range * cast_radius_scale;
            if (offset.length_squared() < reach * reach) {
                float response = sand_density / (material.density * density_ratio * (1.f + 0.065f * (grain.extent - 1)));
                if (kind == Sand || kind == Gravel) response *= free_sand_pull_response(offset.length_squared());
                attracted = true;
                return pull_acceleration(offset, elastic_only ? V3() : grain.velocity, reach, response,
                    free_pull_stiffness, free_pull_damping);
            }
        }
        return {};
    }

    V3 wind_drive(const Grain &grain, I3 from, Kind kind) const {
        if (!winding) return {};
        const float pressure = wind_pressure(particle_center(from, grain.extent));
        if (pressure <= 0.f) return {};
        const float density = granular_material(kind).density * gravel_density_ratio(grain);
        const float drag = kind == Steam ? 2.5f : std::clamp(700.f / (density * grain.extent), 0.12f, 2.5f);
        return (wind_direction * 30.f - grain.velocity) * (2.f * drag * pressure);
    }

    // Integrate forces at a fixed rate. Sub-cell displacement accumulates
    // until a face crossing can be committed to the shared occupancy grid.
    template <typename Fits>
    ParticlePath advance_particle(Grain &grain, I3 from, Kind kind, Fits fits,
            FormMotion *form = nullptr, V3 external_acceleration = V3(), GravelLiquidMove *liquid = nullptr,
            bool enclosed = false) {
        const I3 down(0, -1, 0);
        const GranularMaterial material = granular_material(kind);
        V3 acceleration = V3(0, -gravity, 0) + external_acceleration;
        bool attracted = false;
        acceleration += field_drive(grain, from, kind, form, attracted);
        acceleration += wind_drive(grain, from, kind);
        if (kind != Sand || !attracted)
            sample_bearing_contact(grain, from, kind, acceleration, fits);
        if (kind == Sand && airborne_earth_ejecta.empty() && step_enclosed_sand(grain, from, acceleration, attracted, enclosed))
            return {from, from};
        I3 slope;
        int support = -1;
        // A driven grain uses swept contact against mobile material in
        // either direction. Resting grains and fixed supports can settle early.
        if (grain.velocity.y <= 0.f && acceleration.y <= 0.f && !fits(from + down, &support) &&
                !(liquid && gravel_can_enter_water(grain, support)) &&
                !(attracted && kind == Sand && support >= 0 &&
                    (voxels[support].kind == Sand || voxels[support].kind == Gravel))) {
            grain.travel.y = 0.f;
            grain.blocked_faces &= uint8_t(~grain_axis_faces(1));
            if (retains_grain_face(support)) grain.blocked_faces |= grain_face_bit(1, -1);
            if (grain.velocity.y < -0.3f)
                resolve_granular_contact(grain, kind, 1, -1, support, grain.velocity.y, false, from + down);
            else
                grain.velocity.y = 0.f;
            acceleration.y = 0.f;
            if (!attracted && kind != Water) {
                const auto sides = cardinal();
                const unsigned start = hash(uint32_t(grain.position)) & 3u;
                if (material.granular_repose) {
                    for (unsigned i = 0; i < sides.size(); ++i) {
                        const I3 side = sides[(start + i) & 3u];
                        if (!fits(from + side) || !fits(from + side + down)) continue;
                        slope = side;
                        acceleration += V3(side) * (1.55f * gravity);
                        break;
                    }
                }
                if (slope == I3()) {
                    const float retention = step_ground_retention(kind, material.ground_retention);
                    grain.velocity.x *= retention;
                    grain.velocity.z *= retention;
                }
            }
        }
        if (grain.velocity.y <= 0.f && acceleration.y <= 0.f && !airborne_earth_ejecta.empty() &&
                (kind == Sand || kind == Gravel || kind == Rock))
            airborne_earth_ejecta.erase(grain.position);
        if (grain.velocity == V3() && acceleration == V3()) return {from, from};
        const ParticlePath path = sweep_particle(grain, from, kind, acceleration, fits, form, liquid);
        if (grain.velocity.length_squared() > 0.0009f || (acceleration.length_squared() > 0.01f && attracted))
            next_awake.set_for_cell(path.target);
        return path;
    }

    void step_gravel() {
        GravelLiquidMove &liquid = gravel_liquid_scratch;
        for (int id = 0; id < int(gravel.size()); ++id) {
            Grain &grain = gravel[id];
            const I3 from = decode(grain.position);
            if (!gravel_awake(from, grain.extent)) continue;
            FormMotion *form = voxels[grain.position].formed ? &form_motion.at(grain.position) : nullptr;
            const ParticlePath path = advance_particle(grain, from, Gravel, [&](I3 candidate, int *contact = nullptr) {
                return fits_gravel(candidate, id, contact);
            }, form, V3(), &liquid);
            if (path.target == from) continue;
            if (path.gravel_water_exchange && path.exchange_from != from)
                move_gravel(id, path.exchange_from, next_awake);
            move_gravel(id, path.target, next_awake, path.gravel_water_exchange ? &liquid : nullptr);
        }
    }

    void step_element_particle(I3 p, const Voxel &voxel) {
        const Kind kind = voxel.kind;
        Grain &grain = element_grains[element_grain_slot(kind)][voxel.occupant];
        V3 acceleration;
        if (kind == Vine) {
            const V3 offset = vine_field - center(p) - grain.travel;
            const float reach = vine_pull_radius * cast_radius_scale;
            const bool pulled = pulling_vines && offset.length_squared() < reach * reach;
            if ((!pulled && !detached_vines[voxel.occupant]) || (pulled && vine_has_anchor(p))) {
                grain.velocity = V3();
                grain.travel = V3();
                grain.blocked_faces = 0;
                return;
            }
            if (pulled) acceleration = pull_acceleration(offset, grain.velocity, reach, 1.f, 94.f, 11.f);
        } else if (kind == Steam) {
            // Buoyancy and drag act continuously on the vapor parcel.
            const I3 up(0, 1, 0);
            acceleration = V3(0, gravity + 27.f, 0) - grain.velocity * 14.f;
            if (!can_enter(p + up, Steam, nullptr, true)) {
                for (I3 side : cardinal()) {
                    if (can_enter(p + side, Steam) && can_enter(p + side + up, Steam, nullptr, true))
                        acceleration += V3(side) * 11.f;
                }
            }
        }
        const ParticlePath path = advance_particle(grain, p, kind, [&](I3 candidate, int *contact = nullptr) {
            const bool rising = kind == Steam && candidate.y > p.y && candidate.x == p.x && candidate.z == p.z;
            return can_enter(candidate, kind, contact, rising);
        }, nullptr, acceleration);
        commit_particle_path(p, path);
    }

    void update_voxel(I3 p, int index, bool enclosed) {
        const Voxel &voxel = voxels[index];
        const Kind kind = voxel.kind;
        if (voxel.stamp == voxel_stamp || !mobile_at(index)) return;
        if (voxel.formed) {
            if (profile_sand_sweep && kind == Sand) ++sand_sweep.integrated;
            FormMotion &form = form_motion.at(index);
            Grain &grain = kind == Water ? liquid_particle(index) : grains[voxel.occupant];
            const ParticlePath path = advance_particle(grain, p, kind, [&](I3 candidate, int *contact = nullptr) {
                return can_enter(candidate, kind, contact);
            }, &form, V3(), nullptr, enclosed);
            commit_particle_path(p, path);
            return;
        }
        if (kind == Sand || kind == Rock) {
            Grain &grain = kind == Sand ? grains[voxel.occupant] : rock_fragments[voxel.occupant];
            if (kind == Sand && resting_sand(p, index, grain)) return;
            if (profile_sand_sweep && kind == Sand) ++sand_sweep.integrated;
            const ParticlePath path = advance_particle(grain, p, kind, [&](I3 candidate, int *contact = nullptr) {
                return can_enter(candidate, kind, contact);
            }, nullptr, V3(), nullptr, enclosed);
            commit_particle_path(p, path);
            return;
        }
        if (const auto *material = bulk_liquid_material(kind)) {
            step_liquid(p, index, voxel, *material);
        } else if (element_particle(kind)) {
            step_element_particle(p, voxel);
        }
    }

public:
    float gravity_acceleration() const { return gravity; }
    int64_t element_version(int kind) const { return element_kind(kind) ? element_versions[kind - Ice] : 0; }

    bool load_boxes(const int32_t *layout, size_t length);
    static float packed_particle_mass(int extent);
    int64_t spawn_packed_particle(Vec3 position, int extent, Vec3 impulse = {});
    std::optional<PackedParticleState> packed_particle_state(int64_t handle) const;

    void step();

    void set_sand_sweep_profiling(bool enabled) {
        profile_sand_sweep = enabled;
        sand_sweep = {};
    }
    const SandSweepStats &sand_sweep_stats() const { return sand_sweep; }

    // -1 uses the build setting and 0 uses all available logical processors.
    void set_water_workers(int limit) {
#ifdef SOMEVOXEL_PARALLEL_WATER_WORKERS
        limit = std::max(-1, limit);
        if (limit != requested_water_workers) {
            requested_water_workers = limit;
            water_spans.reset();
            head_selection = {};
        }
#else
        (void)limit;
#endif
    }

    int water_worker_count() const {
#ifdef SOMEVOXEL_PARALLEL_WATER_WORKERS
        const int detected = int(std::thread::hardware_concurrency());
        constexpr int configured = SOMEVOXEL_PARALLEL_WATER_WORKERS;
        const int available = detected > 0 ? detected : configured;
        if (requested_water_workers == 0) return available;
        const int limit = requested_water_workers < 0 ? configured : requested_water_workers;
        return std::min(available, limit);
#else
        return 1;
#endif
    }

    // A synchronous backend for ordinary water heads. The callback may only
    // read the packed snapshot and return its heads. It must not mutate this world.
    using WaterHeadAccelerator = std::function<bool(const std::vector<uint8_t> &, int, int, int,
        std::vector<uint8_t> &)>;

    struct WaterHeadTimes {
        bool automatic = false, last_gpu = false;
        int64_t cpu_us = 0, gpu_us = 0, first_cpu_us = 0, first_gpu_us = 0;
    };
    const WaterHeadTimes &water_head_times() const { return head_times; }

    void set_water_head_accelerator(WaterHeadAccelerator accelerator, bool auto_select = false) {
#ifdef SOMEVOXEL_PARALLEL_WATER_WORKERS
        water_head_accelerator = std::move(accelerator);
        head_times = {auto_select && bool(water_head_accelerator)};
        head_selection = {};
#else
        (void)accelerator;
        (void)auto_select;
#endif
    }

    int64_t water_head_accelerated_rebuilds() const {
#ifdef SOMEVOXEL_PARALLEL_WATER_WORKERS
        return accelerated_head_rebuilds;
#else
        return 0;
#endif
    }

    int64_t water_head_accelerator_attempts() const {
#ifdef SOMEVOXEL_PARALLEL_WATER_WORKERS
        return accelerator_attempts;
#else
        return 0;
#endif
    }

    int64_t water_head_accelerator_time_us() const {
#ifdef SOMEVOXEL_PARALLEL_WATER_WORKERS
        return accelerator_time_us;
#else
        return 0;
#endif
    }

    int64_t sand_motion() const {
        return (sand_revision << 1) | int(moved_grains != 0 || moved_gravel != 0);
    }

    int64_t rock_motion() const {
        return (rock_revision << 1) | int(moved_rock != 0);
    }

    int64_t particle_motion(int material) const {
        if (material == Water) return (water_form_revision << 1) | int(moved_water_form != 0);
        if (structure_kind(material)) {
            const int slot = element_grain_slot(Kind(material));
            return ((element_version(material) + debris_revisions[slot]) << 1) | int(debris_moved[slot]);
        }
        return material == Rock ? rock_motion() : material == Sand ? sand_motion() : 0;
    }

    void set_field(V3 point, bool active) {
        const bool next_pulling = active && point.is_finite();
        if (pulling && (!next_pulling || point != field)) {
            // When released or translated, wake previously supported sand.
            wake_field();
        }
        pulling = next_pulling;
        field = point;
    }

    void set_charge_gather(float progress) {
        charge_gather = std::isfinite(progress) ? std::clamp(progress, 0.f, 1.f) : 0.f;
    }

    void set_cast_profile(float radius_scale, float strength) {
        if (!std::isfinite(radius_scale) || !std::isfinite(strength)) return;
        const float next_radius = std::clamp(radius_scale, 0.10f, 2.50f);
        const float next_strength = std::clamp(strength, 0.10f, max_cast_strength);
        if (next_radius != cast_radius_scale || next_strength != cast_strength) {
            const float wake_radius = std::max(cast_radius_scale, next_radius);
            if (pulling) wake_field();
            if (pulling_water) wake_reaction_area(water_field, water_pull_radius * wake_radius);
            if (pulling_vines) wake_reaction_area(vine_field, vine_pull_radius * wake_radius);
            if (winding) wake_reaction_area(wind_point, wind_radius * wake_radius);
        }
        cast_radius_scale = next_radius;
        cast_strength = next_strength;
    }

    V3 wind_acceleration_at(V3 point) const {
        return point.is_finite() ? wind_direction * (20.f * wind_pressure(point)) : V3();
    }

    int sample(V3 point) const { return kind_at(grid(point)); }

    V3 ray_hit(V3 origin, V3 direction, float distance, bool include_water, bool ignore_formed = false) const {
        const float inf = std::numeric_limits<float>::infinity();
        const V3 miss(inf, inf, inf);
        if (!origin.is_finite() || !direction.is_finite() || !std::isfinite(distance) ||
                distance <= 0 || direction.length_squared() < 0.000001f) return miss;
        direction = direction.normalized();
        const V3 sizes(nx * scale, ny * scale, nz * scale);
        float enter = 0, leave = distance;
        for (int axis = 0; axis < 3; ++axis) {
            if (std::abs(direction[axis]) < 0.000001f) {
                if (origin[axis] < 0 || origin[axis] >= sizes[axis]) return miss;
                continue;
            }
            const float a = -origin[axis] / direction[axis];
            const float b = (sizes[axis] - origin[axis]) / direction[axis];
            enter = std::max(enter, std::min(a, b));
            leave = std::min(leave, std::max(a, b));
        }
        constexpr float epsilon = 0.000025f;
        if (enter + epsilon > leave) return miss;
        I3 cell = grid(origin + direction * (enter + epsilon));
        I3 step_dir;
        V3 crossing, interval;
        for (int axis = 0; axis < 3; ++axis) {
            if (std::abs(direction[axis]) < 0.000001f) {
                crossing[axis] = inf;
                interval[axis] = inf;
                continue;
            }
            step_dir[axis] = direction[axis] > 0 ? 1 : -1;
            const int edge = cell[axis] + (step_dir[axis] > 0);
            crossing[axis] = (edge * scale - origin[axis]) / direction[axis];
            interval[axis] = scale / std::abs(direction[axis]);
        }
        const float held_reach = free_sand_focus_reach();
        while (inside(cell) && enter <= leave) {
            const Voxel &voxel = voxels[flat(cell)];
            const Kind material = voxel.kind;
            bool held_sand = ignore_formed && voxel.formed;
            if (ignore_formed && (material == Gravel || (material == Sand && pulling))) {
                const Grain &grain = material == Sand ? grains[voxel.occupant] : gravel[voxel.occupant];
                // The aim ray can see through the cluster around the held
                // charge. The projectile still collides with those grains.
                held_sand = voxels[grain.position].formed || (pulling &&
                    free_sand_focus_distance_squared(field, particle_center(decode(grain.position), grain.extent)) <
                        held_reach * held_reach);
            }
            if (material != Empty && material != Steam && (include_water || material != Water) &&
                    !held_sand)
                return origin + direction * std::min(leave, enter + epsilon);
            enter = std::min({crossing.x, crossing.y, crossing.z});
            for (int axis = 0; axis < 3; ++axis) {
                if (crossing[axis] > enter + epsilon) continue;
                cell[axis] += step_dir[axis];
                crossing[axis] += interval[axis];
            }
        }
        return miss;
    }

    template <typename Contact>
    bool scan_body(V3 feet, float radius, float height, Contact contact) const {
        if (!feet.is_finite() || !std::isfinite(radius) || !std::isfinite(height) ||
                radius <= 0 || radius > 0.5f || height <= 0 || height > 3.f) return false;
        const I3 lo = grid(feet - V3(radius, -0.007f, radius));
        const I3 hi = grid(feet + V3(radius, height, radius));
        if (lo.x < 0 || lo.y < 0 || lo.z < 0 || hi.x >= nx || hi.z >= nz) return false;
        for (int z = lo.z; z <= hi.z; ++z) {
            const float dz = feet.z - std::clamp(feet.z, z * scale, (z + 1) * scale);
            for (int x = lo.x; x <= hi.x; ++x) {
                const float dx = feet.x - std::clamp(feet.x, x * scale, (x + 1) * scale);
                if (dx * dx + dz * dz >= radius * radius) continue;
                for (int y = lo.y; y <= std::min(hi.y, ny - 1); ++y) {
                    const int i = flat({x, y, z});
                    if (!contact(i, hi.y)) return false;
                }
            }
        }
        return true;
    }

    bool body_clear(V3 feet, float radius, float height) const {
        return scan_body(feet, radius, height, [&](int i, int) {
            return passable_body(voxels[i].kind);
        });
    }

    float push_body_up(V3 feet, float radius, float height, float speed, float body_mass) {
        if (!std::isfinite(speed) || !std::isfinite(body_mass) || speed <= 0.f || body_mass <= 0.f) return -1.f;
        std::unordered_set<int> planned;
        std::vector<int> upward_moves;
        const float energy = 0.5f * body_mass * speed * speed;
        float load_mass = 0.f;
        float load_momentum = 0.f;
        const auto plan = [&](const auto &self, int i, int minimum_y) -> bool {
            const Voxel &voxel = voxels[i];
            if (voxel.kind == Empty) return !body_at(i);
            const Grain *grain = movable_particle_at(i);
            if (!grain) return false;
            const int source = grain->position;
            const int extent = grain->extent;
            const int bottom = source / slice;
            if (bottom < minimum_y || bottom + extent >= ny) return false;
            if (!planned.insert(source).second) return true;
            const float mass = particle_mass(voxel.kind, *grain);
            load_mass += mass;
            load_momentum += mass * grain->velocity.y;
            if (load_mass * gravity * scale >= energy) return false;
            const int top = source + extent * slice;
            for (int z = 0; z < extent; ++z)
                for (int x = 0; x < extent; ++x)
                    if (!self(self, top + x + z * nx, minimum_y)) return false;
            upward_moves.push_back(source);
            return true;
        };
        // Each body substep crosses at most one voxel. Resolve the stack from
        // the top down, then commit only if every contact has somewhere to go.
        if (!scan_body(feet, radius, height, [&](int i, int head_y) {
                return passable_body(voxels[i].kind) ||
                    (i / slice == head_y && plan(plan, i, head_y));
            })) return -1.f;
        if (load_mass == 0.f) return speed;
        // Share the incoming vertical momentum before paying the work of
        // lifting the occupied column through a voxel.
        const float total_mass = body_mass + load_mass;
        const float contact_speed = (body_mass * speed + load_momentum) / total_mass;
        const float work = load_mass * gravity * scale;
        const float remaining = 0.5f * total_mass * contact_speed * contact_speed - work;
        if (contact_speed <= 0.f || remaining <= 0.f) return -1.f;
        const float shared_speed = std::sqrt(2.f * remaining / total_mass);
        bool sand_moved = false;
        bool rock_moved = false;
        for (int i : upward_moves) {
            const I3 from = decode(i);
            const I3 to = from + I3(0, 1, 0);
            const Kind kind = voxels[i].kind;
            Grain &grain = *movable_particle_at(i);
            grain.velocity.y = shared_speed;
            grain.travel.y = 0.f;
            grain.blocked_faces = 0;
            if (kind == Gravel) {
                move_gravel(voxels[i].occupant, to, awake);
                sand_moved = true;
            } else {
                move(from, to);
                wake_near(from, awake);
                wake_near(to, awake);
                rock_moved |= kind == Rock;
                sand_moved |= kind == Sand;
            }
        }
        if (sand_moved) ++sand_revision;
        if (rock_moved) ++rock_revision;
        return shared_speed;
    }

    void remove_body() {
        for (int i : body_cells) {
            set_body_flags(i, 0);
            wake_near(decode(i), awake);
        }
        body_cells.clear();
        body_radius = body_height = 0;
    }

    bool place_body(V3 feet, float radius, float height) {
        if (feet == body_feet && radius == body_radius && height == body_height && radius > 0.f) return true;
        body_next.clear();
        std::vector<int> displaced;
        if (!scan_body(feet, radius, height, [&](int i, int) {
                const Kind material = voxels[i].kind;
                if (!passable_body(material)) return false;
                body_next.push_back(i);
                if (material == Water) displaced.push_back(i);
                return true;
            })) return false;
        constexpr uint8_t current_body = 1;
        constexpr uint8_t proposed_body = 2;
        for (int i : body_next) set_body_flags(i, body_flags(i) | proposed_body);
        std::vector<int> destinations;
        if (!displaced.empty()) {
            struct WaterComponent {
                int parent;
                int demand = 1;
                int capacity = 0;
                int pending = 1;
            };
            struct WaterRoute {
                int cost;
                int component;
            };
            std::vector<WaterComponent> components;
            components.reserve(displaced.size());
            int unfilled = int(displaced.size());
            const auto root = [&](int component) {
                while (components[component].parent != component) {
                    const int parent = components[component].parent;
                    components[component].parent = components[parent].parent;
                    component = parent;
                }
                return component;
            };
            const auto deficit = [&](int component) {
                return std::max(0, components[component].demand - components[component].capacity);
            };
            const auto unite = [&](int a, int b) {
                a = root(a);
                b = root(b);
                if (a == b) return;
                if (components[a].demand < components[b].demand) std::swap(a, b);
                unfilled -= deficit(a) + deficit(b);
                components[b].parent = a;
                components[a].demand += components[b].demand;
                components[a].capacity += components[b].capacity;
                components[a].pending += components[b].pending;
                unfilled += deficit(a);
            };
            std::vector<int> exits;
            exits.reserve(displaced.size());
            using SearchEntry = std::pair<int, int>;
            std::priority_queue<SearchEntry, std::vector<SearchEntry>, std::greater<SearchEntry>> frontier;
            std::unordered_map<int, WaterRoute> routes;
            routes.reserve(displaced.size() * 3);
            const int highest_wet_cell = *std::max_element(displaced.begin(), displaced.end()) / slice;
            const int head = grid(feet + V3(0, height, 0)).y;
            const int lift_cost = 1 + std::clamp((head - highest_wet_cell) / refinement, 0, 7);
            const std::array<I3, 6> dirs = {{{0,-1,0}, {1,0,0}, {-1,0,0}, {0,0,1}, {0,0,-1}, {0,1,0}}};
            for (int source : displaced) {
                const int component = int(components.size());
                components.push_back({component});
                routes.emplace(source, WaterRoute{0, component});
                frontier.emplace(0, source);
            }
            int visited = 0;
            // A finer body voxelization can supply more source cells than
            // the entire search budget of the standard resolutions.
            constexpr int reference_search_budget = 24000;
            const size_t search_budget = refinement > 8 ?
                std::max<size_t>(reference_search_budget, displaced.size() * 12) : reference_search_budget;
            while (!frontier.empty() && unfilled > 0 && size_t(visited) < search_budget) {
                const auto [cost, i] = frontier.top();
                frontier.pop();
                const auto [best_cost, component] = routes.at(i);
                if (best_cost != cost) continue;
                ++visited;
                --components[root(component)].pending;
                // Open space inside the proposed cylinder can connect an
                // interior drop to its boundary. Only exterior cells store water.
                if (voxels[i].kind == Empty && !(body_flags(i) & proposed_body)) {
                    exits.push_back(i);
                    auto &group = components[root(component)];
                    if (group.capacity < group.demand) --unfilled;
                    ++group.capacity;
                    if (unfilled == 0) break;
                }
                const I3 point = decode(i);
                for (I3 direction : dirs) {
                    const I3 next = point + direction;
                    if (!inside(next)) continue;
                    const int neighbor = flat(next);
                    // The part of the body that was and will stay occupied
                    // cannot conduct flow across the character.
                    if (body_flags(neighbor) == (current_body | proposed_body)) continue;
                    const Kind kind = voxels[neighbor].kind;
                    if (kind != Empty && kind != Water) continue;
                    // Pressure climbs a wet column readily. Shallow water
                    // spreads sideways before filling a dry vertical column.
                    const int level_cost = kind == Water ? 4 : 1;
                    const int climb_cost = kind == Water ? 0 : lift_cost;
                    const int next_cost = cost + (direction.y > 0 ? climb_cost : level_cost);
                    auto [route, inserted] = routes.emplace(neighbor, WaterRoute{next_cost, component});
                    if (inserted) {
                        ++components[root(component)].pending;
                    } else {
                        if (route->second.component != component) unite(component, route->second.component);
                        if (route->second.cost <= next_cost) continue;
                        route->second.cost = next_cost;
                    }
                    frontier.emplace(next_cost, neighbor);
                }
                const int group = root(component);
                if (components[group].pending == 0 && deficit(group) > 0) break;
            }
            if (unfilled > 0) {
                for (int i : body_next) set_body_flags(i, 0);
                for (int i : body_cells) set_body_flags(i, current_body);
                return false;
            }
            // The whole move is feasible. Assign the closest visited exits to
            // their own connected sources, then commit the displacements.
            std::vector<int> first_source(displaced.size(), -1), next_source(displaced.size(), -1);
            destinations.resize(displaced.size());
            for (int i = int(displaced.size()) - 1; i >= 0; --i) {
                int &first = first_source[root(i)];
                next_source[i] = first;
                first = i;
            }
            for (int exit : exits) {
                int &first = first_source[root(routes.at(exit).component)];
                if (first < 0) continue;
                destinations[first] = exit;
                first = next_source[first];
            }
        }
        for (int i : body_cells) {
            if (body_flags(i) & proposed_body) continue;
            set_body_flags(i, 0);
            wake_near(decode(i), awake);
        }
        for (int i : body_next) set_body_flags(i, current_body);
        for (size_t i = 0; i < destinations.size(); ++i) {
            const I3 from = decode(displaced[i]), to = decode(destinations[i]);
            move(from, to);
            wake_near(from, awake);
            wake_near(to, awake);
        }
        water_displaced += int(destinations.size());
        body_cells.swap(body_next);
        body_feet = feet;
        body_radius = radius;
        body_height = height;
        return true;
    }

private:
    template <typename Accept>
    bool gather_gravel_shell(I3 origin, int extent, int needed, std::vector<int> &cells, Accept accepts) const {
        cells.clear();
        if (needed <= 0) return true;
        for (I3 face : structural_neighbors()) {
            const int axis = face.x != 0 ? 0 : face.y != 0 ? 1 : 2;
            const int u = (axis + 1) % 3;
            const int v = (axis + 2) % 3;
            I3 candidate = origin;
            candidate[axis] += face[axis] > 0 ? extent : -1;
            for (int a = 0; a < extent; ++a) {
                candidate[u] = origin[u] + a;
                for (int b = 0; b < extent; ++b) {
                    candidate[v] = origin[v] + b;
                    if (!inside(candidate) || !accepts(candidate)) continue;
                    cells.push_back(flat(candidate));
                }
            }
        }
        if (int(cells.size()) < needed) return false;
        const auto rank = [&](int cell) {
            const I3 offset = decode(cell) - origin;
            const I3 delta = offset + offset - I3(extent - 1, extent - 1, extent - 1);
            const int distance_squared = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
            return std::make_pair(distance_squared, hash(uint32_t(cell)));
        };
        std::partial_sort(cells.begin(), cells.begin() + needed, cells.end(),
            [&](int a, int b) { return rank(a) < rank(b); });
        cells.resize(needed);
        return true;
    }

public:
    bool queue_sand_pressure(V3 point, float radius, float pascals, int extent = 2) {
        if (!point.is_finite() || point.x < 0.f || point.y < 0.f || point.z < 0.f ||
                point.x >= nx * scale || point.y >= ny * scale || point.z >= nz * scale ||
                !std::isfinite(radius) || radius <= 0.f || radius > 1.f ||
                !std::isfinite(pascals) || pascals <= 0.f || (extent != 2 && extent != 4 && extent != 8) ||
                sand_pressure_pulses.size() >= 8) return false;
        sand_pressure_pulses.push_back({point, radius, pascals, extent});
        return true;
    }

private:
    bool pack_loaded_region(I3 p, int extent, MaterialLoad load) {
        if (!inside(p) || !inside(p + I3(extent - 1, extent - 1, extent - 1))) return false;
        const int cell = flat(p);
        const I3 corner(extent - 1, extent - 1, extent - 1);
        const int mass_goal = cube_cells(extent) * sand_packing_percent / 100;
        std::vector<int> blocks;
        std::vector<int> donors;
        bool full = true;
        int pores = 0;
        int mass_units = 0;
        int held_cells = 0;
        V3 held_anchor;
        std::optional<FormMotion> held_motion;
        const auto preserve_form = [&](int i, int cells) {
            if (!voxels[i].formed) return;
            const FormMotion &motion = form_motion.at(i);
            if (motion.expires != 0) return;
            if (!held_motion) held_motion = motion;
            held_cells += cells;
            held_anchor += motion.anchor * float(cells);
        };
        V3 momentum;
        each_cube(cell, extent, [&](int i) {
            if (!full) return;
            const Voxel &v = voxels[i];
            if (extent == 2 && v.kind == Empty && !body_at(i)) {
                full = ++pores <= 2;
                return;
            }
            if (v.kind == Sand) {
                ++mass_units;
                momentum += grains[v.occupant].velocity;
                preserve_form(i, 1);
                return;
            }
            if (v.kind != Gravel) { full = false; return; }
            const Grain &g = gravel[v.occupant];
            const I3 offset = decode(g.position) - p;
            full = g.extent < extent && offset.x >= 0 && offset.y >= 0 && offset.z >= 0 &&
                offset.x + g.extent <= extent && offset.y + g.extent <= extent && offset.z + g.extent <= extent;
            if (full && i == g.position) {
                blocks.push_back(i);
                const int units = gravel_mass_units(g);
                mass_units += units;
                momentum += g.velocity * units;
                preserve_form(i, cube_cells(g.extent));
            }
        });
        if (!full || !gather_gravel_shell(p, extent, mass_goal - mass_units, donors,
                    [&](I3 q) { return kind_at(q) == Sand; })) return false;
        // A pressure source must do work against the material's
        // compaction resistance before the occupancy can change.
        const float work = sand_compaction_work(extent, int(donors.size()));
        if (work > load.work) return false;

        for (int i : donors) {
            momentum += grains[voxels[i].occupant].velocity;
            replace_element_cell(i, Empty);
        }
        for (int origin : blocks) erase_gravel(voxels[origin].occupant);
        mass_units += int(donors.size());
        const int id = int(gravel.size());
        gravel.push_back({cell, cell, tick, extent, momentum / mass_units});
        gravel.back().sand_units = uint16_t(mass_units);
        mark_gravel_tile(p);
        each_cube(cell, extent, [&](int i) {
            if (voxels[i].kind == Sand) erase_sand(voxels[i].occupant);
            voxels[i] = {Gravel, voxel_stamp, id};
            set_mobile(i, false);
        });
        if (held_motion && held_cells * 2 >= cube_cells(extent)) {
            held_motion->anchor = held_anchor / float(held_cells);
            attach_form_motion(cell, *held_motion);
        }
        wake_box(p, p + corner, awake);
        ++sand_revision;
        return true;
    }

    MaterialLoad apply_impact_impulse(Grain &grain, Kind kind, V3 point, float radius, float force = 1.f,
            float fracture_fraction = 0.f) {
        const I3 p = decode(grain.position);
        const V3 relative = particle_center(p, grain.extent) - point;
        const float distance_squared = relative.length_squared();
        if (distance_squared > radius * radius) return {};
        const float distance = std::sqrt(distance_squared);
        const float strength = force * (2.8f + 3.6f * (1.f - distance / radius));
        V3 direction = distance > 0.0001f ? relative / distance : V3();
        direction.y += 1.15f;
        const float effective_width = scale * grain.extent;
        const float impulse = strength * sand_density * scale * effective_width * effective_width;
        const float mass = particle_mass(kind, grain);
        const V3 kick = direction * (impulse / mass);
        const float fracture_work = 0.5f * impulse * strength * fracture_fraction;
        grain.velocity += kick * std::sqrt(1.f - fracture_fraction);
        for (int axis = 0; axis < 3; ++axis)
            grain.velocity[axis] = std::clamp<float>(grain.velocity[axis], -maximum_swept_speed, maximum_swept_speed);
        if (kind == Sand) settled_sand_chunks[chunk_at(p)] = false;
        wake_near(p, awake);
        return {fracture_work, {}};
    }

public:
    int impact(V3 point, float radius, float strength = 1.f,
            V3 pressure_source = V3(std::numeric_limits<float>::infinity(), 0, 0)) {
        if (!point.is_finite() || !std::isfinite(radius) || radius <= 0 || radius > 1 ||
                !std::isfinite(strength) || strength <= 0.f) return 0;
        strength = std::min(strength, max_cast_strength);
        const I3 lo = grid(point - V3(radius, radius, radius));
        const I3 hi = grid(point + V3(radius, radius, radius));
        std::vector<int> targets;
        std::vector<std::pair<int, float>> surface_contacts;
        float surface_weight = 0.f;
        const bool from_surface = pressure_source.is_finite();
        for (int y = std::max(1, lo.y); y <= std::min(ny - 1, hi.y); ++y)
            for (int z = std::max(0, lo.z); z <= std::min(nz - 1, hi.z); ++z)
                for (int x = std::max(0, lo.x); x <= std::min(nx - 1, hi.x); ++x) {
                    const I3 p(x, y, z);
                    const float distance_squared = center(p).distance_squared_to(point);
                    if (distance_squared > radius * radius) continue;
                    const int index = flat(p);
                    const Voxel &v = voxels[index];
                    const bool mobile = mobile_at(index);
                    const bool movable = v.kind == Sand || v.kind == Rock || v.kind == Gravel || (mobile && element_particle(v.kind));
                    const bool receives_work = from_surface && !mobile && terrain_toughness(v.kind) > 0.f;
                    if (!movable && !receives_work) continue;
                    if (from_surface) {
                        const V3 ray = center(p) - pressure_source;
                        const V3 hit = ray_hit(pressure_source, ray, ray.length(), false);
                        if (!hit.is_finite() ? v.kind != Steam : grid(hit) != p) continue;
                    }
                    if (v.kind == Sand)
                        apply_impact_impulse(grains[v.occupant], Sand, point, radius, strength);
                    else if (v.kind == Rock)
                        apply_impact_impulse(rock_fragments[v.occupant], Rock, point, radius, strength);
                    else if (v.kind == Gravel)
                        targets.push_back(gravel[v.occupant].position);
                    else if (receives_work) {
                        const float taper = std::max(0.f, 1.f - std::sqrt(distance_squared) / radius);
                        const float weight = taper * taper * taper * taper * taper * taper;
                        surface_contacts.emplace_back(index, weight);
                        surface_weight += weight;
                    } else {
                        apply_impact_impulse(element_grains[element_grain_slot(v.kind)][v.occupant], v.kind, point, radius, strength);
                    }
                }
        std::sort(targets.begin(), targets.end());
        targets.erase(std::unique(targets.begin(), targets.end()), targets.end());
        for (int origin : targets) {
            const int id = voxels[origin].occupant;
            Grain &grain = gravel[id];
            record_packed_load(id, apply_impact_impulse(grain, Gravel, point, radius, strength, 0.62f));
        }
        const int packed_breaks = finish_packed_fractures();
        return packed_breaks + apply_surface_pulse(surface_contacts, surface_weight, pressure_source, strength);
    }

    int grain_count() const { return int(grains.size() + gravel.size()); }

    int carve(V3 point, float radius) {
        if (!point.is_finite() || !std::isfinite(radius) || radius <= 0 || radius > 1.0f) return 0;
        const size_t first_fragment = rock_fragments.size();
        const I3 lo = grid(point - V3(radius, radius, radius));
        const I3 hi = grid(point + V3(radius, radius, radius));
        std::vector<int> fractured;
        for (int y = std::max(1, lo.y); y <= std::min(ny - 1, hi.y); ++y)
            for (int z = std::max(0, lo.z); z <= std::min(nz - 1, hi.z); ++z)
                for (int x = std::max(0, lo.x); x <= std::min(nx - 1, hi.x); ++x) {
                    const I3 p(x, y, z);
                    if (kind_at(p) != Stone || center(p).distance_squared_to(point) > radius * radius) continue;
                    const int i = flat(p);
                    set_rock(i);
                    fractured.push_back(i);
                }
        if (!fractured.empty()) {
            const std::array<I3, 6> dirs = {{{0,1,0}, {0,-1,0}, {1,0,0}, {-1,0,0}, {0,0,1}, {0,0,-1}}};
            std::vector<int> seeds;
            for (int i : fractured) {
                const I3 p = decode(i);
                for (I3 d : dirs)
                    if (kind_at(p + d) == Stone && inside(p + d)) seeds.push_back(flat(p + d));
            }
            release_unsupported(seeds);
            element_support_dirty = true;
            for (size_t i = first_fragment; i < rock_fragments.size(); ++i)
                apply_impact_impulse(rock_fragments[i], Rock, point, radius + 0.24f);
            // Transfer freed voxels from the stone mesh to the fragment batch.
            ++stone_version;
            ++rock_revision;
        }
        return int(fractured.size());
    }

    int range_count(int material, I3 lo, I3 hi) const {
        int count = 0;
        for (int y = std::max(0, lo.y); y < std::min(ny, hi.y); ++y)
            for (int z = std::max(0, lo.z); z < std::min(nz, hi.z); ++z)
                for (int x = std::max(0, lo.x); x < std::min(nx, hi.x); ++x)
                    count += kind_at({x, y, z}) == material;
        return count;
    }

private:
    bool surface_water_at(I3 p) const {
        if (!inside(p)) return false;
        const Voxel &voxel = voxels[flat(p)];
        return voxel.kind == Water && !voxel.formed;
    }

public:
    WorldStats status() const {
        WorldStats data;
        data.refinement = refinement;
        data.cell_size = scale;
        data.step_seconds = dt;
        data.bulk_water_hz = bulk_water_hz;
        data.voxel_bytes = sizeof(Voxel);
        data.grain_bytes = sizeof(Grain);
        data.dense_grid_bytes = voxels.size() * sizeof(Voxel) + mobile_rows.size() + mobile_row_groups.size() + bulk_water_rows.size() +
            (water_chunk_counts.size() + sand_chunk_counts.size()) * sizeof(uint16_t) + settled_sand_chunks.size() + 2 * ChunkActivity::bytes +
            liquid_head.bytes() + previous_water_heads.capacity() * sizeof(LiquidHead) +
            body_mask.bytes() + support_seen.bytes();
#ifdef SOMEVOXEL_PARALLEL_WATER_WORKERS
        if (water_spans) data.dense_grid_bytes += water_spans->scratch_bytes();
#endif
        data.sand = grains.size();
        data.gravel = gravel.size();
        const int loose_and_mud = int(grains.size() + element_cells(Mud).size());
        int solid_volume = loose_and_mud;
        int solid_mass = loose_and_mud;
        int packed_blocks = 0;
        int coarse_9 = 0, coarse_18 = 0, coarse_36 = 0;
        for (const Grain &g : gravel) {
            solid_volume += cube_cells(g.extent);
            solid_mass += gravel_mass_units(g);
            packed_blocks += int(g.sand_units != 0);
            if (g.extent == 2) ++coarse_9;
            if (g.extent == 4) ++coarse_18;
            if (g.extent == 8) ++coarse_36;
        }
        data.sand_volume = solid_volume;
        data.sand_mass = solid_mass;
        data.packed_sand_blocks = packed_blocks;
        data.blocks_9 = coarse_9;
        data.blocks_18 = coarse_18;
        data.blocks_36 = coarse_36;
        data.displaced_water = water_displaced;
        data.water = water_positions.size();
        data.active_water = liquid_motion.size();
        data.water_volume = water_positions.size() + element_cells(Ice).size() +
            element_cells(Steam).size() + element_cells(Mud).size();
        data.ice = element_cells(Ice).size();
        data.wood = element_cells(Wood).size();
        data.grass = element_cells(Grass).size();
        data.ash = element_cells(Ash).size();
        data.steam = element_cells(Steam).size();
        data.air_water = air_water;
        data.air_collected = air_collected;
        data.steam_collected = steam_collected;
        data.air_recharged = air_recharged;
        data.lava = element_cells(Lava).size();
        data.vine = element_cells(Vine).size();
        data.mud = element_cells(Mud).size();
        data.char_cells = element_cells(Char).size();
        data.burning = burning_count;
        data.moved = moved_grains;
        data.moved_gravel = moved_gravel;
        data.chunks = active_count;
        data.chunk_capacity = chunk_count;
        data.stone_version = stone_version;
        data.water_version = water_version;
        data.elements_version = elements_version;
        data.contact_stone_fractures = terrain_contact_breaks[Stone];
        data.contact_wood_fractures = terrain_contact_breaks[Wood];
        data.contact_ice_fractures = terrain_contact_breaks[Ice];
        data.contact_char_fractures = terrain_contact_breaks[Char];
        return data;
    }

#include "detail/voxel_reactions.hpp"
#include "detail/voxel_liquid_pressure.hpp"
#include "detail/voxel_liquid_motion.hpp"
#include "detail/voxel_bulk_liquid.hpp"
#include "detail/voxel_gravel_liquid.hpp"
#include "detail/voxel_humidity.hpp"
#include "detail/voxel_forms.hpp"
#include "detail/voxel_earth_wave.hpp"
#include "detail/voxel_fire_jet.hpp"
#include "detail/voxel_terrain_fracture.hpp"
#include "detail/voxel_sand_tiles.hpp"
};

} // namespace somevoxel
