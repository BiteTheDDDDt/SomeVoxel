#pragma once

#include "../core/world.hpp"
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/packed_byte_array.hpp>
#include <godot_cpp/variant/packed_color_array.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/packed_vector3_array.hpp>
#include <godot_cpp/variant/variant.hpp>
#include <godot_cpp/variant/vector3.hpp>
#include <godot_cpp/variant/vector3i.hpp>

namespace somevoxel {

inline godot::Vector3 to_godot(Vec3 v) { return {v.x, v.y, v.z}; }
inline Vec3 from_godot(godot::Vector3 v) { return {v.x, v.y, v.z}; }
inline Cell from_godot(godot::Vector3i v) { return {v.x, v.y, v.z}; }

class WorldPresentation final : public World {
    using V3 = Vec3;
    using I3 = Cell;

#ifdef SOMEVOXEL_PARALLEL_RENDER_WORKERS
    mutable std::unique_ptr<HeadWorkers> render_workers;
    std::array<std::vector<int>, SOMEVOXEL_PARALLEL_RENDER_WORKERS> worker_visibility;
#endif

public:
    godot::PackedFloat32Array sand_buffer(float alpha) const {
        return build_particle_buffer(Sand, alpha, grain_count(), nullptr);
    }

    V3 grain_render_position(const Grain &grain, float blend) const {
        const V3 destination = center(decode(grain.position));
        return grain.moved == tick && grain.previous != grain.position
            ? center(decode(grain.previous)).lerp(destination, blend) : destination;
    }

    godot::PackedFloat32Array rock_buffer(float alpha) const {
        return build_particle_buffer(Rock, alpha, int(rock_fragments.size()), nullptr);
    }

    godot::Array visible_sand_buffer(float alpha, int previous_capacity) {
        return particle_batch(Sand, alpha, previous_capacity, true);
    }

    godot::Array particle_batch(int material, float alpha, int previous_capacity, bool cull) {
        const bool debris = structure_kind(material);
        if (material != Sand && material != Rock && material != Water && !debris) return {};
        const bool filter = material == Sand && cull;
        if (filter) update_sand_visibility();
        int count = material == Water ? formed_water_count : material == Rock ? int(rock_fragments.size())
            : filter ? int(visible_sand.size() + gravel.size()) : grain_count();
        if (debris) {
            const auto &particles = element_grains[element_grain_slot(Kind(material))];
            count = int(std::count_if(particles.begin(), particles.end(), [&](const Grain &grain) {
                return mobile_at(grain.position);
            }));
        }
        int capacity = count;
        if (count > 0 && (material != Sand || filter)) {
            const int padded = (count + 511) & ~511;
            capacity = previous_capacity < count || previous_capacity > padded + std::max(padded / 2, 512)
                ? padded : previous_capacity;
        }
        godot::Array result;
        result.push_back(count);
        result.push_back(capacity);
        result.push_back(build_particle_buffer(material, alpha, capacity, filter ? &visible_sand : nullptr));
        return result;
    }

    void update_sand_visibility() {
        if (visibility_revision != sand_revision) {
            visible_sand.clear();
            const auto select = [&](int start, int end, std::vector<int> &visible) {
                for (int i = start; i < end; ++i) {
                    const Grain &g = grains[i];
                    if (g.moved == tick && g.previous != g.position) {
                        visible.push_back(i);
                        continue;
                    }
                    const I3 p = decode(g.position);
                    // Cull only sand fully inside other sand. Stone, gravel, and
                    // water are deliberately ignored to keep contact surfaces visible.
                    const int n = g.position;
                    if (p.x == 0 || p.x == nx - 1 || p.y == 0 || p.y == ny - 1 ||
                            p.z == 0 || p.z == nz - 1 ||
                            voxels[n - 1].kind != Sand || voxels[n + 1].kind != Sand ||
                            voxels[n - nx].kind != Sand || voxels[n + nx].kind != Sand ||
                            voxels[n - slice].kind != Sand || voxels[n + slice].kind != Sand)
                        visible.push_back(i);
                }
            };
#ifdef SOMEVOXEL_PARALLEL_RENDER_WORKERS
            if (grains.size() >= 65536) {
                constexpr int count = SOMEVOXEL_PARALLEL_RENDER_WORKERS;
                if (!render_workers) render_workers = std::make_unique<HeadWorkers>(count);
                render_workers->run([&](int worker) {
                    auto &visible = worker_visibility[worker];
                    visible.clear();
                    select(int(grains.size() * worker / count), int(grains.size() * (worker + 1) / count), visible);
                });
                for (const auto &visible : worker_visibility)
                    visible_sand.insert(visible_sand.end(), visible.begin(), visible.end());
            } else
#endif
                select(0, int(grains.size()), visible_sand);
            visibility_revision = sand_revision;
        }
    }

    static V3 sand_color(float tint) {
        return V3(0.56f + tint * 0.19f, 0.29f + tint * 0.14f, 0.115f + tint * 0.093f);
    }

    static V3 gravel_color(int extent, bool compacted, float tint) {
        const V3 base = extent == 8 ? V3(0.24f, 0.104f, 0.065f) :
            extent == 4 ? V3(0.33f, 0.153f, 0.075f) : V3(0.41f, 0.218f, 0.11f);
        const V3 color = base + V3(tint * 0.09f, tint * 0.06f, tint * 0.04f);
        return color * (compacted ? 0.74f : 1.f);
    }

    godot::PackedByteArray sand_palette() const {
        godot::PackedFloat32Array values;
        values.resize(256 * 8 * 4);
        float *out = values.ptrw();
        for (int style = 0; style < 8; ++style) {
            for (int tint = 0; tint < 256; ++tint) {
                const int extent = 1 << (style & 3);
                const V3 color = extent == 1 ? sand_color(tint / 255.f) :
                    gravel_color(extent, (style & 4) != 0, tint / 255.f);
                *out++ = color.x;
                *out++ = color.y;
                *out++ = color.z;
                *out++ = 1.0f;
            }
        }
        return values.to_byte_array();
    }

    static constexpr int coordinate_bits(int axis_size) {
        int bits = 0;
        for (int last = axis_size - 1; last > 0; last >>= 1) ++bits;
        return bits;
    }

    godot::Array sand_snapshot(bool cull, int previous_rows) {
        // RGBA32F stores exact integers through 2^24. The pair of coordinates
        // per channel therefore fits for dimensions up to 4096 cells.
        if constexpr (nx > 4096 || ny > 4096 || nz > 4096) return {};
        if (cull) update_sand_visibility();
        const int sand_count = int(cull ? visible_sand.size() : grains.size());
        const int count = sand_count + int(gravel.size());
        constexpr int width = 1024;
        const int needed = (count + width - 1) / width;
        if (needed > 16384) return {};
        const int padded = (needed + 3) & ~3;
        int rows = padded;
        if (count != 0 && previous_rows >= needed && previous_rows <= 16384 &&
                previous_rows <= padded + std::max(padded / 2, 4))
            rows = previous_rows;
        constexpr I3 bits(coordinate_bits(nx), coordinate_bits(ny), coordinate_bits(nz));
        godot::PackedFloat32Array data;
        data.resize(rows * width * 4);
        float *cells = data.ptrw();
        const auto write_grain = [&](float *out, const Grain &grain, int index, int style) {
            const I3 current = decode(grain.position);
            const I3 previous = grain.moved == tick && grain.previous != grain.position ? decode(grain.previous) : current;
            out[0] = float(current.x | (previous.x << bits.x));
            out[1] = float(current.y | (previous.y << bits.y));
            out[2] = float(current.z | (previous.z << bits.z));
            out[3] = float((hash(uint32_t(index)) & 255u) | (uint32_t(style) << 8));
        };
        const auto pack_sand = [&](int start, int end) {
            for (int i = start; i < end; ++i) {
                const int index = cull ? visible_sand[i] : i;
                write_grain(cells + size_t(i) * 4, grains[index], index, 0);
            }
        };
#ifdef SOMEVOXEL_PARALLEL_RENDER_WORKERS
        if (sand_count >= 65536) {
            constexpr int workers = SOMEVOXEL_PARALLEL_RENDER_WORKERS;
            if (!render_workers) render_workers = std::make_unique<HeadWorkers>(workers);
            render_workers->run([&](int id) {
                pack_sand(int(int64_t(sand_count) * id / workers), int(int64_t(sand_count) * (id + 1) / workers));
            });
        } else
#endif
            pack_sand(0, sand_count);
        for (int i = 0; i < int(gravel.size()); ++i) {
            const Grain &grain = gravel[i];
            const int style = (grain.extent == 8 ? 3 : grain.extent == 4 ? 2 : 1) | (grain.sand_units ? 4 : 0);
            write_grain(cells + size_t(sand_count + i) * 4, grain, i, style);
        }
        godot::Array result;
        result.push_back(count);
        result.push_back(width);
        result.push_back(rows);
        result.push_back(data.to_byte_array());
        result.push_back(godot::Vector3i(bits.x, bits.y, bits.z));
        return result;
    }

    void append_particle_transform(float *&data, const Grain &grain, float blend, float size_factor) const {
        const float half_cell_offset = (grain.extent - 1) * 0.5f * scale;
        const V3 p = grain_render_position(grain, blend) + V3(half_cell_offset, half_cell_offset, half_cell_offset);
        const float size = size_factor * scale * grain.extent;
        const std::array<float, 12> transform = {size, 0, 0, p.x, 0, size, 0, p.y, 0, 0, size, p.z};
        std::copy(transform.begin(), transform.end(), data);
        data += transform.size();
    }

    template <bool Colored, typename Palette>
    void append_particle_instances(float *&data, const std::vector<Grain> &particles,
            const std::vector<int> *selection, float blend, float size_factor, Palette color,
            int start = 0, int end = -1) const {
        if (end < 0) end = selection == nullptr ? int(particles.size()) : int(selection->size());
        for (int entry = start; entry < end; ++entry) {
            const int i = selection == nullptr ? entry : (*selection)[entry];
            const Grain &grain = particles[i];
            append_particle_transform(data, grain, blend, size_factor);
            if constexpr (Colored) {
                const V3 rgb = color(grain, (hash(uint32_t(i)) & 255u) / 255.f);
                *data++ = rgb.x;
                *data++ = rgb.y;
                *data++ = rgb.z;
                *data++ = 1.0f;
            }
        }
    }

    godot::PackedFloat32Array build_particle_buffer(int material, float alpha, int capacity,
            const std::vector<int> *selection) const {
        godot::PackedFloat32Array buffer;
        const float blend = std::isfinite(alpha) ? std::clamp(alpha, 0.f, 1.f) : 1.f;
        if (material == Water) {
            buffer.resize(capacity * 12);
            float *data = buffer.ptrw();
            for (const auto &entry : form_motion)
                if (voxels[entry.first].kind == Water) append_particle_transform(data, liquid_particle(entry.first), blend, 0.98f);
        } else if (material == Rock) {
            buffer.resize(capacity * 12);
            float *data = buffer.ptrw();
            append_particle_instances<false>(data, rock_fragments, nullptr, blend, 0.99f, nullptr);
        } else if (structure_kind(material)) {
            buffer.resize(capacity * 12);
            float *data = buffer.ptrw();
            for (const Grain &grain : element_grains[element_grain_slot(Kind(material))])
                if (mobile_at(grain.position)) append_particle_transform(data, grain, blend, 0.985f);
        } else {
            buffer.resize(capacity * 16);
            float *data = buffer.ptrw();
            const auto sand_color = [](const Grain &, float tint) {
                return WorldPresentation::sand_color(tint);
            };
#ifdef SOMEVOXEL_PARALLEL_RENDER_WORKERS
            const size_t grain_count = selection ? selection->size() : grains.size();
            if (grain_count >= 65536) {
                constexpr int count = SOMEVOXEL_PARALLEL_RENDER_WORKERS;
                if (!render_workers) render_workers = std::make_unique<HeadWorkers>(count);
                render_workers->run([&](int worker) {
                    const int start = int(grain_count * worker / count);
                    const int end = int(grain_count * (worker + 1) / count);
                    float *output = data + size_t(start) * 16;
                    append_particle_instances<true>(output, grains, selection, blend, 0.974f, sand_color, start, end);
                });
                data += grain_count * 16;
            } else
#endif
                append_particle_instances<true>(data, grains, selection, blend, 0.974f, sand_color);
            append_particle_instances<true>(data, gravel, nullptr, blend, 0.98f, [](const Grain &grain, float tint) {
                return WorldPresentation::gravel_color(grain.extent, grain.sand_units != 0, tint);
            });
        }
        return buffer;
    }

    godot::Array build_surface(int material, const std::vector<int> &water_selection, int stone_patch = -1) const {
        using namespace godot;
        struct Face { I3 normal; I3 origin; I3 u; I3 v; };
        struct VisibleFace { I3 voxel; uint8_t face; uint8_t length; };
        const std::array<Face, 6> faces = {{
            {{1,0,0}, {1,0,0}, {0,0,1}, {0,1,0}},
            {{-1,0,0},{0,0,0}, {0,1,0}, {0,0,1}},
            {{0,1,0}, {0,1,0}, {1,0,0}, {0,0,1}},
            {{0,-1,0},{0,0,0}, {0,0,1}, {1,0,0}},
            {{0,0,1}, {0,0,1}, {0,1,0}, {1,0,0}},
            {{0,0,-1},{0,0,0}, {1,0,0}, {0,1,0}}
        }};
        const bool liquid = material == Water;
        const bool static_structure = structure_kind(material);
        std::vector<VisibleFace> visible;
        if (liquid) visible.reserve(water_selection.size());
        const auto water_top_matches = [&](I3 p, bool deep) {
            if (!surface_water_at(p)) return false;
            const I3 above = p + I3(0, 1, 0);
            if (inside(above) && (surface_water_at(above) || rock_surface(kind_at(above)))) return false;
            return surface_water_at(p + I3(0, -1, 0)) == deep;
        };
        const auto append_faces = [&](int i) {
            const I3 p = decode(i);
            for (uint8_t face = 0; face < faces.size(); ++face) {
                const I3 other = p + faces[face].normal;
                const Kind neighbor = kind_at(other);
                const bool same_surface = liquid ? surface_water_at(other) : neighbor == material &&
                    (!static_structure || (inside(other) && !mobile_at(flat(other))));
                if (inside(other) && (same_surface || (liquid && rock_surface(neighbor)) ||
                        (material == Stone && intact_stone(neighbor)))) continue;
                uint8_t length = 1;
                if (liquid && face == 2) {
                    // Join coplanar water tops with the same depth shading.
                    // Bound strips to eight cells so shading still samples wave
                    // geometry and dirty patches own separate complete strips.
                    const bool deep = surface_water_at(p + I3(0, -1, 0));
                    if (p.x % block != 0 && water_top_matches(p - I3(1, 0, 0), deep)) continue;
                    const int limit = block - p.x % block;
                    while (length < limit && water_top_matches(p + I3(length, 0, 0), deep)) ++length;
                }
                visible.push_back({p, face, length});
            }
        };
        if (liquid) {
            for (int i : water_selection)
                if (!voxels[i].formed) append_faces(i);
        } else if (element_kind(material)) {
            for (int i : element_cells(Kind(material)))
                if (!static_structure || !mobile_at(i)) append_faces(i);
        } else if (stone_patch >= 0) {
            const int x0 = (stone_patch % stone_patch_x) * stone_patch_width;
            const int z0 = (stone_patch / stone_patch_x) * stone_patch_width;
            const int x1 = std::min(nx, x0 + stone_patch_width);
            const int z1 = std::min(nz, z0 + stone_patch_width);
            for (int y = 0; y < ny; ++y)
                for (int z = z0; z < z1; ++z)
                    for (int i = flat({x0, y, z}), end = i + x1 - x0; i < end; ++i)
                        if (intact_stone(voxels[i].kind)) append_faces(i);
        } else {
            for (int i = 0; i < int(voxels.size()); ++i)
                if (voxels[i].kind == material || (material == Stone && intact_stone(voxels[i].kind))) append_faces(i);
        }
        if (visible.empty()) return {};

        PackedVector3Array vertices, normals;
        PackedColorArray colors;
        PackedInt32Array indices;
        vertices.resize(visible.size() * 4);
        normals.resize(visible.size() * 4);
        colors.resize(visible.size() * 4);
        indices.resize(visible.size() * 6);
        godot::Vector3 *positions = vertices.ptrw();
        godot::Vector3 *directions = normals.ptrw();
        Color *tints = colors.ptrw();
        int32_t *triangles = indices.ptrw();
        int base = 0;
        for (const VisibleFace &entry : visible) {
            const I3 p = entry.voxel;
            const Face &f = faces[entry.face];
            const I3 root = p + f.origin;
            const float depth = liquid && surface_water_at(p + I3(0, -1, 0)) ? 0.46f : 0.15f;
            const Color tint = liquid ? Color(depth, 0.07f, 0, 1)
                                      : Color(0.04f, 0, 0, 1);
            const V3 direction(f.normal);
            const I3 width = f.u * entry.length;
            for (I3 point : {root, root + width, root + f.v, root + width + f.v}) {
                *positions++ = to_godot(V3(point) * scale);
                *directions++ = to_godot(direction);
                *tints++ = tint;
            }
            for (int index : {0, 1, 2, 1, 3, 2}) *triangles++ = base + index;
            base += 4;
        }
        Array data;
        data.resize(13);
        data[0] = vertices;
        data[1] = normals;
        data[3] = colors;
        data[12] = indices;
        return data;
    }

    godot::Array surface(int material) const { return build_surface(material, water_positions); }

    godot::Array take_stone_patches() {
        godot::Array result;
        for (int id = 0; id < int(stone_dirty.size()); ++id) {
            if (!stone_dirty[id]) continue;
            result.push_back(id);
            result.push_back(build_surface(Stone, water_positions, id));
        }
        stone_dirty.fill(false);
        return result;
    }

    void set_water_patch_tracking(bool enabled) {
        track_water_patches = enabled;
        water_dirty.fill(enabled);
    }

    godot::Array take_water_patches() {
        godot::Array result;
        if (std::none_of(water_dirty.begin(), water_dirty.end(), [](bool dirty) { return dirty; })) return result;
        std::array<std::vector<int>, patch_x * patch_z> bins;
        const auto is_surface_water = [&](int i) {
            const Voxel &cell = voxels[i];
            return cell.kind == Water && !cell.formed;
        };
        for (int index : water_positions) {
            if (voxels[index].formed) continue;
            const I3 p = decode(index);
            const int id = patch_id(p);
            if (!water_dirty[id]) continue;
            // Fully submerged cells would emit no faces. Leave them out of
            // the patch lists and keep the original order of the boundary.
            if (p.x > 0 && p.x < nx - 1 && p.y > 0 && p.y < ny - 1 &&
                    p.z > 0 && p.z < nz - 1 &&
                    is_surface_water(index - 1) && is_surface_water(index + 1) &&
                    is_surface_water(index - nx) && is_surface_water(index + nx) &&
                    is_surface_water(index - slice) && is_surface_water(index + slice))
                continue;
            bins[id].push_back(index);
        }
        for (int id = 0; id < int(water_dirty.size()); ++id) {
            if (!water_dirty[id]) continue;
            result.push_back(id);
            result.push_back(build_surface(Water, bins[id]));
        }
        water_dirty.fill(false);
        return result;
    }


    godot::PackedFloat32Array fire_buffer() const {
        godot::PackedFloat32Array result;
        if (burning_count <= 0) return result;
        result.resize(burning_count * 12);
        float *out = result.ptrw();
        for (const auto &[i, cell] : reaction_cells) {
            if (!cell.burning) continue;
            const float variation = float(hash(uint32_t(i)) & 0xffu) / 255.f;
            const float height = scale * (2.7f + 2.6f * variation);
            const float width = scale * (1.10f + 0.36f * variation);
            const V3 p = center(decode(i)) + V3(0, 0.2f * height + 0.25f * scale, 0);
            const std::array<float, 12> instance = {
                width, 0, 0, p.x, 0, height, 0, p.y, 0, 0, width, p.z
            };
            std::copy(instance.begin(), instance.end(), out);
            out += instance.size();
        }
        return result;
    }

    godot::PackedFloat32Array fire_jet_buffer(float alpha) const {
        godot::PackedFloat32Array result;
        if (fire_packets.empty()) return result;
        result.resize(fire_jet_capacity * 12);
        float *output = result.ptrw();
        for (const auto &packet : fire_packets) {
            const float age = packet.age / packet.lifetime;
            const float fade = 1.f - age * age;
            const float width = packet.width * (0.62f + age * 1.3f) * fade;
            const V3 direction = packet.velocity.normalized();
            const V3 reference = std::abs(direction.y) > 0.95f ? V3(1, 0, 0) : V3(0, 1, 0);
            const V3 right = direction.cross(reference).normalized() * width;
            const V3 up = direction * (width * 3.6f);
            const V3 forward = right.normalized().cross(direction) * width;
            const V3 center = packet.previous.lerp(packet.position, std::clamp(alpha, 0.f, 1.f));
            const std::array<float, 12> instance = {
                right.x, up.x, forward.x, center.x,
                right.y, up.y, forward.y, center.y,
                right.z, up.z, forward.z, center.z
            };
            std::copy(instance.begin(), instance.end(), output);
            output += instance.size();
        }
        return result;
    }
};

} // namespace somevoxel
