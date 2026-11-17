private:
    // Air carries a coarse moisture reserve in liquid-voxel equivalents. It
    // stays separate from visible steam and does not occupy material cells.
    static constexpr int humidity_width = 4 * refinement;
    static constexpr int humidity_nx = nx / humidity_width;
    static constexpr int humidity_ny = ny / humidity_width;
    static constexpr int humidity_nz = nz / humidity_width;
    static constexpr int humidity_size = humidity_nx * humidity_ny * humidity_nz;
    static constexpr float humidity_cell = humidity_width * scale;
    static constexpr float reference_volume = refinement * refinement * refinement / 64.f;
    static constexpr int humidity_second = 30 * refinement;
    static constexpr int humidity_recovery_delay = 6 * humidity_second;

    struct HumidityCell {
        int reserve = 0;
        int capacity = 0;
        uint32_t recharge_at = 0;
    };

    struct HumiditySource {
        float distance;
        int index;
        V3 point;
    };

    std::array<HumidityCell, humidity_size> humidity{};
    int depleted_humidity_cells = 0;
    int64_t air_water = 0;
    int64_t air_collected = 0;
    int64_t steam_collected = 0;
    int64_t air_recharged = 0;
    V3 condensation_point;
    float condensation_radius = 0.f;
    float condensation_strength = 0.f;
    float condensation_effort = 0.f;
    std::vector<std::pair<float, int>> condensation_candidates;
    std::vector<HumiditySource> humidity_sources;

    static int humidity_index(I3 p) { return p.x + humidity_nx * (p.z + humidity_nz * p.y); }
    static I3 humidity_coords(I3 p) { return {p.x / humidity_width, p.y / humidity_width, p.z / humidity_width}; }
    static bool humidity_inside(I3 p) {
        return p.x >= 0 && p.x < humidity_nx && p.y >= 0 && p.y < humidity_ny && p.z >= 0 && p.z < humidity_nz;
    }

    void reset_humidity() {
        std::array<int, humidity_size> sources{};
        const auto record = [&](int i, int moisture) {
            const I3 p = decode(i);
            if (kind_at(p + I3(0, 1, 0)) != Empty) return;
            int &source = sources[humidity_index(humidity_coords(p))];
            source = std::max(source, moisture);
        };
        for (int i : water_positions) record(i, 38);
        for (int i : element_cells(Mud)) record(i, 29);
        for (int i : element_cells(Grass)) record(i, 15);

        std::array<int, humidity_size> climate{};
        climate.fill(5);
        for (int y = 0; y < humidity_ny; ++y) {
            for (int z = 0; z < humidity_nz; ++z) {
                for (int x = 0; x < humidity_nx; ++x) {
                    const I3 p(x, y, z);
                    const int source = sources[humidity_index(p)];
                    if (source == 0) continue;
                    for (int dy = -1; dy <= 2; ++dy) {
                        for (int dz = -2; dz <= 2; ++dz) {
                            for (int dx = -2; dx <= 2; ++dx) {
                                const I3 q = p + I3(dx, dy, dz);
                                if (!humidity_inside(q)) continue;
                                const int moisture = source - 6 * (dx * dx + dy * dy + dz * dz);
                                int &capacity = climate[humidity_index(q)];
                                capacity = std::max(capacity, moisture);
                            }
                        }
                    }
                }
            }
        }

        air_water = 0;
        air_collected = 0;
        steam_collected = 0;
        air_recharged = 0;
        depleted_humidity_cells = 0;
        condensation_strength = 0.f;
        condensation_effort = 0.f;
        for (int i = 0; i < humidity_size; ++i) {
            const int capacity = std::max(1, int(std::round(climate[i] * reference_volume)));
            humidity[i] = {capacity, capacity, 0};
            air_water += capacity;
        }
    }

    void recharge_humidity() {
        if (depleted_humidity_cells == 0 || tick % humidity_second != 0) return;
        const int wet_capacity = std::max(2, int(std::round(12 * reference_volume)));
        const int recharge_amount = std::max(1, int(std::round(reference_volume)));
        for (auto &cell : humidity) {
            if (cell.reserve == cell.capacity || int32_t(tick - cell.recharge_at) < 0) continue;
            const int replenished = std::min(recharge_amount, cell.capacity - cell.reserve);
            cell.reserve += replenished;
            air_water += replenished;
            air_recharged += replenished;
            cell.recharge_at = tick + humidity_second * (cell.capacity >= wet_capacity ? 1 : 3);
            if (cell.reserve == cell.capacity) --depleted_humidity_cells;
        }
    }

    bool condensation_path_clear(V3 target) const {
        const V3 offset = target - condensation_point;
        const float distance = offset.length();
        return distance < scale * 0.5f || !ray_hit(condensation_point, offset / distance, distance, false).is_finite();
    }

    int condense_visible_steam(int budget) {
        const auto &steam = element_cells(Steam);
        if (steam.empty()) return 0;
        condensation_candidates.clear();
        const float limit = condensation_radius * condensation_radius;
        const auto add_steam = [&](int i, I3 p) {
            if (body_at(i)) return;
            const float distance = center(p).distance_squared_to(condensation_point);
            if (distance <= limit) condensation_candidates.emplace_back(distance, i);
        };
        const float width = 2.f * condensation_radius / scale;
        if (steam.size() <= width * width * width) {
            for (int i : steam) add_steam(i, decode(i));
        } else {
            for_reaction_sphere(condensation_point, condensation_radius, [&](int i, I3 p) {
                if (voxels[i].kind == Steam) add_steam(i, p);
            });
        }
        const int nearby = std::min(int(condensation_candidates.size()), budget * 4 + 12);
        std::partial_sort(condensation_candidates.begin(), condensation_candidates.begin() + nearby,
            condensation_candidates.end());
        int caught = 0;
        for (int candidate = 0; candidate < nearby && caught < budget; ++candidate) {
            const int i = condensation_candidates[candidate].second;
            if (!condensation_path_clear(center(decode(i))) || !replace_element_cell(i, Water)) continue;
            auto existing = reaction_cells.find(i);
            if (existing != reaction_cells.end()) existing->second.temperature = std::min(45.f, existing->second.temperature);
            queue_water_contact(i);
            ++caught;
        }
        steam_collected += caught;
        return caught;
    }

    V3 nearest_humidity_point(I3 p) const {
        const V3 edge(scale * 0.2f, scale * 0.2f, scale * 0.2f);
        const V3 low = V3(p) * humidity_cell + edge;
        return condensation_point.clamp(low, low + V3(humidity_cell, humidity_cell, humidity_cell) - edge * 2.f);
    }

    void condense_ambient_water(int budget) {
        const float minimum_drop_radius = 2.f * scale;
        const float drop_radius = std::clamp(condensation_radius * 0.33f, minimum_drop_radius,
            std::max(minimum_drop_radius, 0.22f));
        condensation_candidates.clear();
        for_reaction_sphere(condensation_point, drop_radius, [&](int i, I3 p) {
            if (voxels[i].kind != Empty || body_at(i)) return;
            const float distance = center(p).distance_squared_to(condensation_point);
            condensation_candidates.emplace_back(distance, i);
        });
        if (condensation_candidates.empty()) return;
        std::sort(condensation_candidates.begin(), condensation_candidates.end());

        humidity_sources.clear();
        const V3 reach(condensation_radius, condensation_radius, condensation_radius);
        const I3 low = humidity_coords(grid(condensation_point - reach));
        const I3 high = humidity_coords(grid(condensation_point + reach));
        const float radius_squared = condensation_radius * condensation_radius;
        for (int y = std::max(0, low.y); y <= std::min(humidity_ny - 1, high.y); ++y) {
            for (int z = std::max(0, low.z); z <= std::min(humidity_nz - 1, high.z); ++z) {
                for (int x = std::max(0, low.x); x <= std::min(humidity_nx - 1, high.x); ++x) {
                    const I3 p(x, y, z);
                    const int i = humidity_index(p);
                    if (humidity[i].reserve == 0) continue;
                    const V3 nearest = nearest_humidity_point(p);
                    const float distance = nearest.distance_squared_to(condensation_point);
                    if (distance <= radius_squared) humidity_sources.push_back({distance, i, nearest});
                }
            }
        }
        std::sort(humidity_sources.begin(), humidity_sources.end(), [](const auto &a, const auto &b) {
            return std::make_pair(a.distance, a.index) < std::make_pair(b.distance, b.index);
        });

        size_t drop = 0;
        for (const auto &source : humidity_sources) {
            if (drop == condensation_candidates.size() || budget == 0) break;
            if (!condensation_path_clear(source.point)) continue;
            auto &cell = humidity[source.index];
            const int before = cell.reserve;
            while (cell.reserve > 0 && budget > 0 && drop < condensation_candidates.size()) {
                const int i = condensation_candidates[drop++].second;
                if (!condensation_path_clear(center(decode(i))) || !replace_element_cell(i, Water)) continue;
                queue_water_contact(i);
                --cell.reserve;
                --budget;
            }
            const int collected = before - cell.reserve;
            if (collected == 0) continue;
            if (before == cell.capacity) ++depleted_humidity_cells;
            cell.recharge_at = tick + humidity_recovery_delay;
            air_water -= collected;
            air_collected += collected;
        }
    }

    void step_humidity() {
        recharge_humidity();
        if (condensation_strength == 0.f || tick % reaction_interval != 0) return;
        const Kind at_focus = kind_at(grid(condensation_point));
        if (at_focus != Empty && at_focus != Water && at_focus != Steam) return;
        condensation_effort += 50.f * reference_volume * condensation_strength * reaction_interval * dt;
        const int budget = std::min(48, int(condensation_effort));
        condensation_effort -= std::floor(condensation_effort);
        if (budget == 0) return;
        const int steam = condense_visible_steam(budget);
        if (steam < budget) condense_ambient_water(budget - steam);
    }

public:
    void set_condensation(V3 point, bool active, float radius, float strength) {
        if (!active || !point.is_finite() || !std::isfinite(radius) || !std::isfinite(strength) ||
                radius <= 0.f || strength <= 0.f || !inside(grid(point))) {
            condensation_strength = 0.f;
            condensation_effort = 0.f;
            return;
        }
        condensation_point = point;
        condensation_radius = std::clamp(radius, 0.18f, 6.f);
        condensation_strength = std::min(strength, 5.f);
    }

    float humidity_at(V3 point) const {
        if (!point.is_finite() || !inside(grid(point))) return 0.f;
        return std::min(1.f, humidity[humidity_index(humidity_coords(grid(point)))].reserve / std::max(1.f, 38.f * reference_volume));
    }
