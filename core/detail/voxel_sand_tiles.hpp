private:
    static constexpr int sand_tile_width = 64;
    static constexpr int sand_tile_columns = (nx + sand_tile_width - 1) / sand_tile_width;
    static constexpr int sand_tile_rows = (nz + sand_tile_width - 1) / sand_tile_width;
    static constexpr int sand_tile_plane = sand_tile_columns * sand_tile_rows;
    static constexpr int sand_tile_count = sand_tile_plane * ((ny + sand_tile_width - 1) / sand_tile_width);

    // Allocated on first presentation request. Physics-only worlds keep no tile state.
    std::vector<uint8_t> sand_tile_dirty;
    std::vector<int> sand_tile_changes;
    uint64_t sand_tile_generation = 0;

    static int sand_tile_id(I3 p) {
        return p.x / sand_tile_width + sand_tile_columns *
            (p.z / sand_tile_width + sand_tile_rows * (p.y / sand_tile_width));
    }

    static I3 sand_tile_origin(int id) {
        return I3(id % sand_tile_columns, id / sand_tile_plane,
            id / sand_tile_columns % sand_tile_rows) * sand_tile_width;
    }

    void mark_sand_tile(int id) {
        if (sand_tile_dirty[id]) return;
        sand_tile_dirty[id] = 1;
        sand_tile_changes.push_back(id);
    }

    void mark_gravel_tile(I3 p) {
        if (!sand_tile_dirty.empty()) mark_sand_tile(sand_tile_id(p));
    }

    void mark_sand_tiles(I3 p) {
        if (sand_tile_dirty.empty()) return;
        const int id = sand_tile_id(p);
        mark_sand_tile(id);
        // A changed voxel also exposes or buries its six face neighbors.
        if (p.x > 0 && p.x % sand_tile_width == 0) mark_sand_tile(id - 1);
        if (p.x + 1 < nx && (p.x + 1) % sand_tile_width == 0) mark_sand_tile(id + 1);
        if (p.y > 0 && p.y % sand_tile_width == 0) mark_sand_tile(id - sand_tile_plane);
        if (p.y + 1 < ny && (p.y + 1) % sand_tile_width == 0) mark_sand_tile(id + sand_tile_plane);
        if (p.z > 0 && p.z % sand_tile_width == 0) mark_sand_tile(id - sand_tile_columns);
        if (p.z + 1 < nz && (p.z + 1) % sand_tile_width == 0) mark_sand_tile(id + sand_tile_columns);
    }

    void reset_sand_tiles() {
        ++sand_tile_generation;
        sand_tile_changes.clear();
        std::fill(sand_tile_dirty.begin(), sand_tile_dirty.end(), 0);
    }
