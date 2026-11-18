private:
    static constexpr float ambient_temperature = 18.f;
    static constexpr int reaction_interval = simulation_hz / 10;
    static constexpr float wind_radius = 1.5f;
    static constexpr float water_pull_radius = 2.4f;
    static constexpr float vine_pull_radius = 1.2f;
    static constexpr uint8_t mud_soak_passes = 6;
    static constexpr float lava_temperature = 480.f;
    static constexpr float fire_temperature = 450.f;
    static constexpr float wet_fuel_intake = 0.20f;

    struct ReactionCell {
        float temperature = ambient_temperature;
        float heat_delta = 0.f;
        float wetness = 0.f;
        float fuel_used = 0.f;
        uint32_t heat_pass = 0;
        bool burning = false;
    };

    std::unordered_map<int, ReactionCell> reaction_cells;
    std::unordered_set<int> wetted_fuel;
    std::unordered_map<int, uint8_t> wet_sand;
    std::vector<int> mud_ready;
    std::vector<int> reaction_order;
    std::vector<uint8_t> heat_brush_visited;
    std::vector<int> heat_brush_frontier;
    uint32_t reaction_pass = 0;
    int burning_count = 0;
    V3 wind_point;
    V3 wind_direction;
    V3 water_field;
    V3 vine_field;
    bool winding = false;
    bool pulling_water = false;
    bool pulling_vines = false;
    bool track_fuel_wetting = false;

    float wind_pressure(V3 point) const {
        if (!winding) return 0.f;
        const float radius = wind_radius * cast_radius_scale;
        const float distance = point.distance_squared_to(wind_point) / (radius * radius);
        return cast_strength * std::max(0.f, 1.f - distance);
    }

    static std::array<I3, 6> reaction_faces() {
        return {{{0, 1, 0}, {1, 0, 0}, {0, 0, 1}, {-1, 0, 0}, {0, 0, -1}, {0, -1, 0}}};
    }

    static bool has_fuel(Kind kind) {
        return kind == Grass || kind == Wood || kind == Char || kind == Vine;
    }

    static bool carries_heat(Kind kind) {
        return kind == Water || kind == Ice || kind == Steam || kind == Lava || kind == Mud ||
            kind == Stone || kind == Rock || kind == Sand || kind == Ash || has_fuel(kind);
    }

    static float heat_capacity(Kind kind) {
        switch (kind) {
            case Water: return 1.9f;
            case Mud: return 1.55f;
            case Lava: return 1.40f;
            case Ice: return 1.30f;
            default: return 1.f;
        }
    }

    template <typename Visitor>
    void for_reaction_sphere(V3 point, float radius, Visitor visit) {
        const V3 dimensions(nx * scale, ny * scale, nz * scale);
        const V3 reach(radius, radius, radius);
        const auto low = grid((point - reach).clamp(V3(), dimensions));
        const auto high = grid((point + reach).clamp(V3(), dimensions));
        const float radius_squared = radius * radius;
        for (int y = std::max(1, low.y); y <= std::min(ny - 1, high.y); ++y) {
            for (int z = std::max(0, low.z); z <= std::min(nz - 1, high.z); ++z) {
                for (int x = std::max(0, low.x); x <= std::min(nx - 1, high.x); ++x) {
                    const I3 p(x, y, z);
                    if (center(p).distance_squared_to(point) <= radius_squared) visit(flat(p), p);
                }
            }
        }
    }

    static bool heat_brush_passable(Kind kind) {
        return kind == Empty || kind == Water || kind == Steam;
    }

    I3 heat_brush_origin(V3 point, V3 incident) const {
        const I3 cell = grid(point);
        if (!inside(cell) || heat_brush_passable(kind_at(cell))) return cell;
        // The aim ray ends inside the first occupied cell. Start on its incoming face.
        const V3 direction = incident.normalized();
        float distance = std::numeric_limits<float>::infinity();
        for (int axis = 0; axis < 3; ++axis) {
            if (std::abs(direction[axis]) < 0.000001f) continue;
            const float boundary = (cell[axis] + (direction[axis] < 0.f)) * scale;
            distance = std::min(distance, (point[axis] - boundary) / direction[axis]);
        }
        return grid(point - direction * (std::max(0.f, distance) + scale * 0.001f));
    }

    template <typename Visitor>
    void for_exposed_heat(V3 point, float radius, V3 incident, Visitor visit) {
        const I3 origin = heat_brush_origin(point, incident);
        if (!inside(origin) || !heat_brush_passable(kind_at(origin))) return;
        const V3 reach(radius + scale, radius + scale, radius + scale);
        const I3 lo = grid(point - reach).clamp(I3(), I3(nx - 1, ny - 1, nz - 1));
        const I3 hi = grid(point + reach).clamp(I3(), I3(nx - 1, ny - 1, nz - 1));
        const I3 size = hi - lo + I3(1, 1, 1);
        heat_brush_visited.assign(size.x * size.y * size.z, 0);
        heat_brush_frontier.clear();
        const float radius_squared = radius * radius;
        const auto discover = [&](I3 p) {
            if (p.x < lo.x || p.x > hi.x || p.y < lo.y || p.y > hi.y || p.z < lo.z || p.z > hi.z) return;
            const I3 local = p - lo;
            uint8_t &visited = heat_brush_visited[local.x + size.x * (local.z + size.z * local.y)];
            if (visited) return;
            visited = 1;
            const float distance_squared = center(p).distance_squared_to(point);
            if (p != origin && distance_squared > radius_squared) return;
            const int i = flat(p);
            if (distance_squared <= radius_squared) visit(i, p);
            // Solid faces absorb heat. Their interiors use the regular conduction pass.
            if (heat_brush_passable(voxels[i].kind)) heat_brush_frontier.push_back(i);
        };
        discover(origin);
        for (size_t cursor = 0; cursor < heat_brush_frontier.size(); ++cursor) {
            const I3 p = decode(heat_brush_frontier[cursor]);
            for (I3 face : reaction_faces()) discover(p + face);
        }
    }

    void wake_reaction_area(V3 point, float radius) {
        const V3 reach(radius, radius, radius);
        const V3 dimensions(nx * scale, ny * scale, nz * scale);
        wake_box(grid((point - reach).clamp(V3(), dimensions)), grid((point + reach).clamp(V3(), dimensions)), awake);
    }

    void change_element_field(V3 point, bool active, V3 &position, bool &enabled, float reach) {
        const bool next_enabled = active && point.is_finite();
        if (enabled && (!next_enabled || point != position)) wake_reaction_area(position, reach);
        if (next_enabled && (!enabled || point != position)) wake_reaction_area(point, reach);
        enabled = next_enabled;
        if (next_enabled) position = point;
    }

    int water_contact(I3 p) const {
        for (I3 side : reaction_faces()) {
            const I3 q = p + side;
            if (inside(q)) {
                const int i = flat(q);
                if (voxels[i].kind == Water) return i;
            }
        }
        return -1;
    }

    bool has_water_contact(I3 p) const { return water_contact(p) >= 0; }

    bool water_in_contact_chunks(I3 p) const {
        const int id = chunk_at(p);
        if (water_chunk_counts[id]) return true;
        constexpr int edge = block - 1;
        return ((p.x & edge) == 0 && p.x > 0 && water_chunk_counts[id - 1]) ||
            ((p.x & edge) == edge && p.x + 1 < nx && water_chunk_counts[id + 1]) ||
            ((p.z & edge) == 0 && p.z > 0 && water_chunk_counts[id - bx]) ||
            ((p.z & edge) == edge && p.z + 1 < nz && water_chunk_counts[id + bx]) ||
            ((p.y & edge) == 0 && p.y > 0 && water_chunk_counts[id - bx * bz]) ||
            ((p.y & edge) == edge && p.y + 1 < ny && water_chunk_counts[id + bx * bz]);
    }

    void queue_water_contact(int i) {
        if (voxels[i].kind != Water || (!track_fuel_wetting && grains.empty())) return;
        const I3 p = decode(i);
        for (I3 side : reaction_faces()) {
            const I3 q = p + side;
            if (!inside(q)) continue;
            const int neighbor = flat(q);
            const Kind kind = voxels[neighbor].kind;
            if (kind == Sand) wet_sand.try_emplace(neighbor, 0);
            else if (track_fuel_wetting && has_fuel(kind)) wetted_fuel.insert(neighbor);
        }
    }

    void queue_sand_contact(int i) {
        if (water_positions.empty()) return;
        const I3 p = decode(i);
        if (water_in_contact_chunks(p) && has_water_contact(p)) wet_sand.try_emplace(i, 0);
    }

    void move_element_state(int from, int to) {
        if (!reaction_cells.empty()) {
            auto source = reaction_cells.extract(from);
            auto destination = reaction_cells.extract(to);
            if (source) {
                source.key() = to;
                reaction_cells.insert(std::move(source));
            }
            if (destination) {
                destination.key() = from;
                reaction_cells.insert(std::move(destination));
            }
        }
        if (!wetted_fuel.empty()) {
            const bool source_wetted = wetted_fuel.erase(from) != 0;
            const bool destination_wetted = wetted_fuel.erase(to) != 0;
            if (source_wetted && has_fuel(voxels[to].kind)) wetted_fuel.insert(to);
            if (destination_wetted && has_fuel(voxels[from].kind)) wetted_fuel.insert(from);
        }
        const bool sand_from = voxels[from].kind == Sand;
        const bool sand_to = voxels[to].kind == Sand;
        if (!wet_sand.empty()) {
            // Occupancy has already swapped. Carry the sand's exposure with it.
            auto source = sand_to ? wet_sand.extract(from) : decltype(wet_sand)::node_type{};
            auto destination = sand_from ? wet_sand.extract(to) : decltype(wet_sand)::node_type{};
            if (source) {
                source.key() = to;
                wet_sand.insert(std::move(source));
            }
            if (destination) {
                destination.key() = from;
                wet_sand.insert(std::move(destination));
            }
        }
        if (sand_from) queue_sand_contact(from);
        if (sand_to) queue_sand_contact(to);
        queue_water_contact(from);
        queue_water_contact(to);
        // Wetness follows fuel already exposed to moving water.
        if (track_fuel_wetting && has_fuel(voxels[to].kind) && has_water_contact(decode(to))) wetted_fuel.insert(to);
    }

    void reset_elements() {
        reaction_cells.clear();
        wetted_fuel.clear();
        wet_sand.clear();
        mud_ready.clear();
        reaction_order.clear();
        reaction_pass = 0;
        burning_count = 0;
        winding = false;
        pulling_water = false;
        pulling_vines = false;
        vine_support_dirty = !element_cells(Vine).empty();
        track_fuel_wetting = !element_cells(Grass).empty() || !element_cells(Wood).empty() ||
            !element_cells(Vine).empty() || !element_cells(Char).empty();
        for (int i : element_cells(Lava)) reaction_cells[i].temperature = lava_temperature;
        for (int i : element_cells(Steam)) reaction_cells[i].temperature = 118.f;
        for (int i : element_cells(Ice)) reaction_cells[i].temperature = -20.f;
        for (Kind fuel : {Grass, Wood, Vine, Char}) {
            for (int i : element_cells(fuel)) {
                if (has_water_contact(decode(i))) reaction_cells[i].wetness = 1.f;
            }
        }
        if (!grains.empty()) {
            for (int i : water_positions) queue_water_contact(i);
        }
    }

    void soak_sand() {
        mud_ready.clear();
        for (auto it = wet_sand.begin(); it != wet_sand.end();) {
            const int sand = it->first;
            if (voxels[sand].kind != Sand || !has_water_contact(decode(sand))) {
                it = wet_sand.erase(it);
                continue;
            }
            if (it->second < mud_soak_passes) ++it->second;
            if (it->second == mud_soak_passes) mud_ready.push_back(sand);
            ++it;
        }
        // Several sand cells can touch the same water. Resolve claims in a
        // stable order and test each donor again before consuming it.
        std::sort(mud_ready.begin(), mud_ready.end());
        for (int sand : mud_ready) {
            const Kind below = kind_at(decode(sand) - I3(0, 1, 0));
            // The current mud material represents a supported saturated bed.
            if (!rock_surface(below) && below != Sand && below != Gravel) continue;
            if (below == Lava || below == Vine || voxels[sand].formed) continue;
            const int water = water_contact(decode(sand));
            if (water < 0 || body_at(sand)) continue;
            const auto existing = reaction_cells.find(water);
            const float water_temperature = existing == reaction_cells.end()
                ? ambient_temperature : existing->second.temperature;
            Grain mixed = grains[voxels[sand].occupant];
            const V3 water_velocity = voxels[water].liquid_moving ? liquid_particle(water).velocity : V3();
            mixed.velocity = (mixed.velocity * particle_mass(Sand, 1) + water_velocity * particle_mass(Water, 1)) /
                particle_mass(Mud, 1);
            if (!replace_element_cell(sand, Mud)) continue;
            replace_element_cell(water, Empty);
            reaction_cells.erase(water);
            element_grains[element_grain_slot(Mud)][voxels[sand].occupant] = mixed;
            auto &mud = reaction_cells[sand];
            mud.temperature = (mud.temperature + water_temperature) * 0.5f;
        }
    }

    void dry_mud(int i, ReactionCell &cell) {
        const I3 p = decode(i);
        for (I3 side : reaction_faces()) {
            const I3 q = p + side;
            if (!inside(q)) continue;
            const int release = flat(q);
            if (voxels[release].kind != Empty || body_at(release)) continue;
            const Grain sediment = element_grains[element_grain_slot(Mud)][voxels[i].occupant];
            if (!replace_element_cell(release, Steam)) continue;
            if (!replace_element_cell(i, Sand)) {
                replace_element_cell(release, Empty);
                return;
            }
            grains[voxels[i].occupant] = sediment;
            cell.temperature = std::max(27.f, cell.temperature - 26.f);
            reaction_cells[release].temperature = 105.f;
            return;
        }
    }

    bool solidify_lava(int i) {
        const Grain motion = element_grains[element_grain_slot(Lava)][voxels[i].occupant];
        if (!replace_element_cell(i, Stone)) return false;
        if (motion.velocity.length_squared() <= 0.0009f) return true;
        const I3 p = decode(i);
        for (I3 side : structural_neighbors()) {
            if (inside(p + side) && intact_stone(kind_at(p + side))) return true;
        }
        // Cooling retains momentum when the new solid has no mineral support.
        set_rock(i);
        rock_fragments.back() = motion;
        rock_fragments.back().velocity *= particle_mass(Lava, 1) / particle_mass(Rock, 1);
        ++rock_revision;
        return true;
    }

    void update_combustion(int i, Kind kind, ReactionCell &cell) {
        if (has_water_contact(decode(i))) {
            cell.wetness = std::min(1.f, cell.wetness + wet_fuel_intake);
        }
        if (cell.wetness > 0.f) {
            // Contact replenishes moisture, while sustained heat dries it.
            // Water exchanges sensible heat in the common face conduction pass.
            const float evaporation = std::min(cell.wetness, 0.045f + 0.0018f * std::max(0.f, cell.temperature - 60.f));
            if (cell.temperature > 60.f) cell.temperature -= 22.f * evaporation;
            cell.wetness -= evaporation;
        }

        const float ignition = kind == Grass ? 90.f : kind == Vine ? 103.f : kind == Char ? 100.f : 135.f;
        if (cell.wetness > 0.28f || cell.temperature < 72.f) cell.burning = false;
        if (!cell.burning && cell.wetness <= 0.28f && cell.temperature >= ignition) cell.burning = true;
        if (!cell.burning) return;
        cell.temperature = std::max(cell.temperature, fire_temperature);
        const float burn_steps = kind == Grass ? 6.f : kind == Vine ? 10.f : kind == Char ? 9.f : 21.f;
        cell.fuel_used += 1.f;
        if (cell.fuel_used < burn_steps) return;
        const Kind next = kind == Wood ? Char : Ash;
        if (replace_element_cell(i, next)) {
            cell.fuel_used = 0.f;
            if (next == Ash) {
                cell.burning = false;
                cell.temperature = 155.f;
            }
        }
    }

    void step_elements() {
        // Fields keep their chunks scheduled across all fixed-step batches.
        if (winding) wake_reaction_area(wind_point, wind_radius * cast_radius_scale);
        if (pulling_water) wake_reaction_area(water_field, water_pull_radius * cast_radius_scale);
        if (pulling_vines) wake_reaction_area(vine_field, vine_pull_radius * cast_radius_scale);
        if (tick % reaction_interval != 0) return;

        soak_sand();
        for (int i : wetted_fuel) {
            if (!has_fuel(voxels[i].kind)) continue;
            auto &cell = reaction_cells[i];
            if (!has_water_contact(decode(i))) cell.wetness = std::min(1.f, cell.wetness + wet_fuel_intake);
        }
        wetted_fuel.clear();
        if (reaction_cells.empty()) return;

        ++reaction_pass;
        reaction_order.clear();
        for (auto &[i, cell] : reaction_cells) {
            cell.heat_pass = reaction_pass;
            reaction_order.push_back(i);
        }
        std::sort(reaction_order.begin(), reaction_order.end());
        // Accumulate heat flow from a snapshot. Each shared face is visited
        // once, so heat does not spread by an in-place update cascade.
        for (int i : reaction_order) {
            const Kind kind = voxels[i].kind;
            if (!carries_heat(kind)) continue;
            auto &cell = reaction_cells.at(i);
            const float rate = kind == Steam ? 0.065f : kind == Lava ? 0.0008f : kind == Ice ? 0.018f : 0.033f;
            cell.heat_delta += (ambient_temperature - cell.temperature) * rate * heat_capacity(kind);
            const I3 p = decode(i);
            for (I3 side : reaction_faces()) {
                const I3 q = p + side;
                if (!inside(q)) continue;
                const int n = flat(q);
                const Kind neighbor_kind = voxels[n].kind;
                if (!carries_heat(neighbor_kind)) continue;
                const auto existing = reaction_cells.find(n);
                if (existing != reaction_cells.end() && existing->second.heat_pass == reaction_pass && n < i) continue;
                const float neighbor_temperature = existing != reaction_cells.end()
                    ? existing->second.temperature : ambient_temperature;
                const bool fire_to_neighbor = cell.burning && has_fuel(neighbor_kind);
                const bool fire_from_neighbor = existing != reaction_cells.end() && existing->second.burning && has_fuel(kind);
                float coupling = fire_to_neighbor || fire_from_neighbor ? 0.16f
                    : kind == Water || neighbor_kind == Water ? 0.05f : 0.065f;
                if (kind == Lava || neighbor_kind == Lava) {
                    const Kind other = kind == Lava ? neighbor_kind : kind;
                    // A dry crust insulates the basin. Liquid water couples
                    // much more strongly and can cool the exposed lava face.
                    coupling = other == Water ? 0.22f : other == Lava ? 0.065f : 0.0026f;
                }
                if (winding && (fire_to_neighbor || fire_from_neighbor)) {
                    const bool outgoing = fire_to_neighbor && (!fire_from_neighbor || cell.temperature >= neighbor_temperature);
                    const I3 source = outgoing ? p : q;
                    const float pressure = wind_pressure(center(source));
                    if (pressure > 0.f) {
                        const float bias = wind_direction.dot(V3(outgoing ? side : -side));
                        coupling *= std::clamp(1.f + 0.38f * bias * pressure, 0.20f, 2.5f);
                    }
                }
                const float transfer = (cell.temperature - neighbor_temperature) * coupling;
                if (std::abs(transfer) < 0.14f) continue;
                cell.heat_delta -= transfer;
                reaction_cells[n].heat_delta += transfer;
            }
        }
        for (auto &[i, cell] : reaction_cells) {
            cell.temperature = std::clamp(cell.temperature + cell.heat_delta / heat_capacity(voxels[i].kind), -140.f, 820.f);
            cell.heat_delta = 0.f;
        }

        burning_count = 0;
        for (int i : reaction_order) {
            auto &cell = reaction_cells.at(i);
            const Kind kind = voxels[i].kind;
            if (kind == Water && cell.temperature <= -2.f && replace_element_cell(i, Ice)) {
                cell.temperature = std::clamp(cell.temperature * 0.45f, -26.f, -4.f);
            } else if (kind == Ice && cell.temperature >= 4.f && replace_element_cell(i, Water)) {
                cell.temperature -= 12.f;
            } else if (kind == Water && cell.temperature >= 99.f && replace_element_cell(i, Steam)) {
                cell.temperature = std::max(104.f, cell.temperature - 28.f);
            } else if (kind == Steam && cell.temperature <= 51.f) {
                replace_element_cell(i, Water);
            } else if (kind == Lava) {
                if (has_water_contact(decode(i))) cell.temperature -= 42.f;
                if (cell.temperature <= 155.f && solidify_lava(i)) {
                    cell.temperature = std::min(cell.temperature, 150.f);
                }
            } else if (kind == Mud && cell.temperature >= 92.f) {
                dry_mud(i, cell);
            } else if (has_fuel(kind)) {
                update_combustion(i, kind, cell);
            }
            if (!has_fuel(voxels[i].kind)) cell.burning = false;
            if (cell.burning) ++burning_count;
        }
        for (auto it = reaction_cells.begin(); it != reaction_cells.end();) {
            const auto &cell = it->second;
            const Kind kind = voxels[it->first].kind;
            const bool settles = kind != Ice && kind != Steam && kind != Lava && !cell.burning && cell.wetness <= 0.f &&
                cell.fuel_used == 0.f && std::abs(cell.temperature - ambient_temperature) < 0.8f;
            if (!carries_heat(kind) || settles) it = reaction_cells.erase(it);
            else ++it;
        }
    }

public:
    int apply_heat(V3 point, float radius, float amount, V3 incident = V3()) {
        const float incident_squared = incident.length_squared();
        if (!point.is_finite() || !std::isfinite(radius) || !std::isfinite(amount) ||
            !std::isfinite(incident_squared) || radius <= 0.f || radius > 1.25f || amount == 0.f) return 0;
        int touched = 0;
        const auto deposit_heat = [&](int i, I3 p) {
            if (!carries_heat(voxels[i].kind)) return;
            const float falloff = 1.f - 0.42f * center(p).distance_squared_to(point) / (radius * radius);
            auto &cell = reaction_cells[i];
            const float deposited = std::clamp(amount, -260.f, 320.f) * falloff / heat_capacity(voxels[i].kind);
            cell.temperature = std::clamp(cell.temperature + deposited, -140.f, 820.f);
            ++touched;
        };
        if (incident_squared > 0.000001f) for_exposed_heat(point, radius, incident, deposit_heat);
        else for_reaction_sphere(point, radius, deposit_heat);
        return touched;
    }

    void set_wind(V3 point, V3 direction, bool active) {
        change_element_field(point, active && direction.is_finite() && direction.length_squared() > 0.0001f,
            wind_point, winding, wind_radius * cast_radius_scale);
        if (winding) wind_direction = direction.normalized();
    }

    void set_water_field(V3 point, bool active) {
        change_element_field(point, active, water_field, pulling_water, water_pull_radius * cast_radius_scale);
    }

    void set_vine_field(V3 point, bool active) {
        const bool was_pulling = pulling_vines;
        change_element_field(point, active, vine_field, pulling_vines, vine_pull_radius * cast_radius_scale);
        if (was_pulling && !pulling_vines) vine_support_dirty = true;
    }
