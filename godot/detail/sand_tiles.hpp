private:
    static constexpr int sand_texture_width = 256;

    struct SandTile {
        std::vector<float> texels;
        I3 low, high;
        bool moving = false;
    };

    uint64_t presented_sand_generation = 0;
    bool presented_sand_cull = false;
    std::vector<std::vector<float>> presented_sand;
    std::vector<int> presented_sand_rows;
    std::vector<int> moving_sand_tiles;
    std::vector<SandTile> sand_tile_updates;

    static void write_sand_texel(float *out, I3 current, I3 previous, I3 bits, int index, int style) {
        out[0] = float(current.x | (previous.x << bits.x));
        out[1] = float(current.y | (previous.y << bits.y));
        out[2] = float(current.z | (previous.z << bits.z));
        out[3] = float((hash(uint32_t(index)) & 255u) | (uint32_t(style) << 8));
    }

    void scan_sand_tile(int id, bool cull, I3 shifts, SandTile &tile) const {
        const I3 origin = sand_tile_origin(id);
        const I3 limit(std::min(origin.x + sand_tile_width, nx),
            std::min(origin.y + sand_tile_width, ny), std::min(origin.z + sand_tile_width, nz));
        tile.texels.clear();
        tile.moving = false;
        tile.low = limit;
        tile.high = origin;
        for (int y = origin.y; y < limit.y; ++y) {
            for (int z = origin.z; z < limit.z; ++z) {
                const int first = flat({0, y, z});
                for (int x = origin.x; x < limit.x; ++x) {
                    const int cell = first + x;
                    const Voxel &voxel = voxels[cell];
                    const bool fine = voxel.kind == Sand;
                    if (!fine && voxel.kind != Gravel) continue;
                    const int index = voxel.occupant;
                    const Grain &grain = fine ? grains[index] : gravel[index];
                    if (grain.position != cell) continue;
                    const bool moves = grain_interpolates(grain);
                    const I3 current(x, y, z);
                    if (fine && cull && !moves && !exposed_sand(current, cell)) continue;
                    const I3 previous = moves ? decode(grain.previous) : current;
                    const int extent = grain.extent;
                    const int style = fine ? 0 : (extent == 8 ? 3 : extent == 4 ? 2 : 1) | (grain.sand_units ? 4 : 0);
                    const size_t start = tile.texels.size();
                    tile.texels.resize(start + 4);
                    write_sand_texel(tile.texels.data() + start, current, previous, shifts, index, style);
                    tile.low.x = std::min({tile.low.x, x, previous.x});
                    tile.low.y = std::min({tile.low.y, y, previous.y});
                    tile.low.z = std::min({tile.low.z, z, previous.z});
                    tile.high.x = std::max(tile.high.x, std::max(x, previous.x) + extent);
                    tile.high.y = std::max(tile.high.y, std::max(y, previous.y) + extent);
                    tile.high.z = std::max(tile.high.z, std::max(z, previous.z) + extent);
                    tile.moving |= moves;
                }
            }
        }
    }

public:
    godot::Array take_sand_tiles(bool cull) {
        if constexpr (nx > 4096 || ny > 4096 || nz > 4096) return {};
        const bool reset = presented_sand.empty() || sand_tile_generation != presented_sand_generation ||
            cull != presented_sand_cull;
        if (reset) {
            presented_sand_generation = sand_tile_generation;
            presented_sand_cull = cull;
            presented_sand.clear();
            presented_sand.resize(sand_tile_count);
            presented_sand_rows.assign(sand_tile_count, 0);
            sand_tile_dirty.assign(sand_tile_count, 0);
            sand_tile_changes.clear();
            for (const Grain &grain : grains) mark_sand_tile(sand_tile_id(decode(grain.position)));
            for (const Grain &grain : gravel) mark_sand_tile(sand_tile_id(decode(grain.position)));
        } else {
            // Tiles that used the last tick's motion must collapse their endpoints,
            // even when none of their cells moved during the current tick.
            for (int id : moving_sand_tiles) mark_sand_tile(id);
        }
        moving_sand_tiles.clear();
        constexpr I3 shifts(coordinate_bits(nx), coordinate_bits(ny), coordinate_bits(nz));
        const int count = int(sand_tile_changes.size());
        std::sort(sand_tile_changes.begin(), sand_tile_changes.end());
        sand_tile_updates.resize(count);
        const auto scan = [&](int index) {
            scan_sand_tile(sand_tile_changes[index], cull, shifts, sand_tile_updates[index]);
        };
#ifdef SOMEVOXEL_PARALLEL_RENDER_WORKERS
        if (count >= SOMEVOXEL_PARALLEL_RENDER_WORKERS * 2) {
            constexpr int workers = SOMEVOXEL_PARALLEL_RENDER_WORKERS;
            if (!render_workers) render_workers = std::make_unique<HeadWorkers>(workers);
            render_workers->run([&](int worker) {
                for (int i = worker; i < count; i += workers) scan(i);
            });
        } else
#endif
            for (int i = 0; i < count; ++i) scan(i);

        godot::Array updates;
        for (int index = 0; index < count; ++index) {
            const int id = sand_tile_changes[index];
            sand_tile_dirty[id] = 0;
            SandTile &current = sand_tile_updates[index];
            if (current.moving) moving_sand_tiles.push_back(id);
            auto &previous = presented_sand[id];
            if (current.texels == previous) continue;
            previous.swap(current.texels);
            const int instances = int(previous.size() / 4);
            godot::Array update;
            update.push_back(id);
            update.push_back(instances);
            if (instances != 0) {
                godot::PackedByteArray data;
                constexpr int stride = 4 * sizeof(float);
                const int needed = (instances + sand_texture_width - 1) / sand_texture_width;
                int &rows = presented_sand_rows[id];
                if (rows < needed || rows > needed + std::max(4, needed / 2)) {
                    rows = (needed + 3) & ~3;
                }
                const int size = rows * sand_texture_width * stride;
                data.resize(size);
                uint8_t *out = data.ptrw();
                std::memcpy(out, previous.data(), instances * stride);
                std::memset(out + instances * stride, 0, size - instances * stride);
                update.push_back(data);
                update.push_back(godot::Vector3i(current.low.x, current.low.y, current.low.z));
                update.push_back(godot::Vector3i(current.high.x, current.high.y, current.high.z));
                update.push_back(current.moving);
            } else presented_sand_rows[id] = 0;
            updates.push_back(update);
        }
        sand_tile_changes.clear();
        godot::Array result;
        result.push_back(reset);
        result.push_back(godot::Vector3i(shifts.x, shifts.y, shifts.z));
        result.push_back(sand_texture_width);
        result.push_back(updates);
        return result;
    }
