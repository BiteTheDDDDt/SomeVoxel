private:
    struct PressureNode {
        int parent = -1;
        int depth = 0;
    };

    static constexpr uint32_t pressure_interval = simulation_hz / 15;
    static constexpr int pressure_transfer_limit = 48;
    static constexpr int pressure_face_budget = 2048;
    using LiquidHead = std::conditional_t<(ny <= 255), uint8_t, uint16_t>;
    std::array<std::vector<int>, ny> water_by_height;
    std::vector<LiquidHead> liquid_head;
    std::vector<LiquidHead> previous_water_heads;
#ifdef SOMEVOXEL_PARALLEL_WATER_WORKERS
    std::unique_ptr<WaterSpans<nx, ny, nz>> water_spans;
#endif
    std::vector<int> pressure_component;
    std::vector<PressureNode> pressure_nodes;
    std::vector<int> pressure_sources;
    std::vector<int> pressure_outlets;
    std::vector<int> pressure_outlet_route;
    std::vector<int> pressure_source_route;
    std::array<int64_t, 7> pressure_revision{};
    V3 pressure_body_feet;
    float pressure_body_radius = 0.f;
    float pressure_body_height = 0.f;

    bool passive_water(int i) const {
        return voxels[i].kind == Water && !voxels[i].formed && !voxels[i].liquid_moving;
    }

    bool supported_pressure_surface(I3 point) const {
        const I3 down(0, -1, 0);
        if (can_enter(point + down, Water)) return false;
        for (I3 face : cardinal()) {
            const I3 side = point + face;
            if (can_enter(side, Water) && can_enter(side + down, Water)) return false;
        }
        return true;
    }

    bool pressure_route(int support, int source, int &budget) {
        pressure_outlet_route.clear();
        pressure_source_route.clear();
        // Both ends belong to the same face-connected component. Walk their
        // breadth-first parents to a common cell so no water crosses a wall.
        while (budget > 0) {
            --budget;
            if (support < 0 || source < 0 || !passive_water(support) || !passive_water(source)) return false;
            if (support == source) {
                pressure_outlet_route.push_back(support);
                return true;
            }
            const PressureNode &low = pressure_nodes[voxels[support].occupant];
            const PressureNode &high = pressure_nodes[voxels[source].occupant];
            if (low.depth >= high.depth) {
                pressure_outlet_route.push_back(support);
                support = low.parent;
            } else {
                pressure_source_route.push_back(source);
                source = high.parent;
            }
        }
        return false;
    }

    void discharge_pressure(int outlet, LiquidHead head) {
        // Shift the vacancy back along the filled route. Using ordinary moves
        // advects each parcel's heat and wetting state by one face as well.
        PressureNode vacancy_node;
        const auto shift = [&](int from) {
            const int id = voxels[from].occupant;
            const PressureNode old_node = pressure_nodes[id];
            move(decode(from), decode(outlet));
            pressure_nodes[id] = vacancy_node;
            vacancy_node = old_node;
            outlet = from;
        };
        liquid_head[outlet] = head;
        for (int site : pressure_outlet_route) shift(site);
        for (auto site = pressure_source_route.rbegin(); site != pressure_source_route.rend(); ++site) shift(*site);
    }

    void balance_pressure(int top, int &transfers, int &faces) {
        pressure_sources.clear();
        pressure_outlets.clear();
        if (top < 2) return;
        int highest_surface = -1;
        for (int i : pressure_component) {
            if (!passive_water(i)) continue;
            const int layer = i / slice;
            if (layer + 1 >= ny || voxels[i + slice].kind != Empty || body_at(i + slice)) continue;
            // A component with a gravity outlet is still draining. Its local
            // face transfers take precedence over hydrostatic redistribution.
            if (!supported_pressure_surface(decode(i))) return;
            if (layer >= highest_surface) {
                if (layer > highest_surface) pressure_sources.clear();
                highest_surface = layer;
                pressure_sources.push_back(i);
            }
            if (layer + 1 < top) pressure_outlets.push_back(i + slice);
        }
        if (pressure_sources.empty() || pressure_outlets.empty()) return;
        std::sort(pressure_outlets.begin(), pressure_outlets.end());
        // Visit the lowest available surface first, using each exposed source
        // once. Every accepted move takes mass from a higher to a lower layer.
        for (int outlet : pressure_outlets) {
            if (transfers == 0 || faces == 0 || pressure_sources.empty()) break;
            if (outlet / slice >= highest_surface) break;
            if (voxels[outlet].kind != Empty || body_at(outlet)) continue;
            const int source = pressure_sources.back();
            if (!pressure_route(outlet - slice, source, faces)) continue;
            discharge_pressure(outlet, LiquidHead(top + 1));
            pressure_sources.pop_back();
            --transfers;
        }
    }

    void refresh_liquid_head() {
        if (water_positions.empty()) return;
        if (liquid_head.empty()) liquid_head.resize(voxels.size());
        const std::array<int64_t, 7> revisions{water_topology_version, stone_version, rock_revision, sand_revision,
            elements_version, water_form_revision, int64_t(liquid_motion.size())};
        const bool balance = tick % pressure_interval == 0 && (pressure_revision != revisions ||
            pressure_body_feet != body_feet || pressure_body_radius != body_radius || pressure_body_height != body_height);
        if (head_version == water_topology_version && !balance) return;
        head_version = water_topology_version;
        if (balance) {
            pressure_revision = revisions;
            pressure_body_feet = body_feet;
            pressure_body_radius = body_radius;
            pressure_body_height = body_height;
            pressure_nodes.resize(water_positions.size());
        }

#ifdef SOMEVOXEL_PARALLEL_WATER_WORKERS
        if (!balance && water_positions.size() >= 16384) {
            if (!water_spans) {
                const unsigned available = std::thread::hardware_concurrency();
                constexpr unsigned configured = SOMEVOXEL_PARALLEL_WATER_WORKERS;
                const unsigned workers = available == 0 ? configured : std::min(available, configured);
                water_spans = std::make_unique<WaterSpans<nx, ny, nz>>(int(workers));
            }
            const auto apply_head = [this](int i, int head) {
                if (head == liquid_head[i]) return;
                liquid_head[i] = LiquidHead(head);
                const I3 p = decode(i);
                awake[chunk_id({p.x / block, p.y / block, p.z / block})] = true;
            };
            // Sparse scenes still benefit from visiting just their parcel list.
            if (size_t(water_scan_end / block) <= 4 * water_positions.size())
                water_spans->rebuild_rows(bulk_water_rows, liquid_motion, water_scan_end / slice, apply_head);
            else
                water_spans->rebuild(water_positions, apply_head);
            return;
        }
#endif

        // Only occupied water cells participate in the wake comparison. Keep
        // the snapshot by parcel ID for the duration of this rebuild.
        previous_water_heads.resize(water_positions.size());
        for (auto &height : water_by_height) height.clear();
        for (size_t id = 0; id < water_positions.size(); ++id) {
            const int i = water_positions[id];
            previous_water_heads[id] = liquid_head[i];
            liquid_head[i] = 0;
            water_by_height[i / slice].push_back(i);
        }
        int transfers = pressure_transfer_limit;
        int faces = pressure_face_budget;
        for (int y = ny - 1; y >= 0; --y) {
            const LiquidHead head = y + 1;
            for (int seed : water_by_height[y]) {
                if (liquid_head[seed] || voxels[seed].kind != Water) continue;
                pressure_component.clear();
                pressure_component.push_back(seed);
                liquid_head[seed] = head;
                const bool record_routes = balance && transfers > 0 && faces > 0;
                if (record_routes) pressure_nodes[voxels[seed].occupant] = {};
                for (size_t cursor = 0; cursor < pressure_component.size(); ++cursor) {
                    const int source = pressure_component[cursor];
                    const I3 point = decode(source);
                    const int source_id = voxels[source].occupant;
                    const auto visit = [&](int i) {
                        if (voxels[i].kind != Water || liquid_head[i]) return;
                        liquid_head[i] = head;
                        pressure_component.push_back(i);
                        if (record_routes) {
                            pressure_nodes[voxels[i].occupant] = {source, pressure_nodes[source_id].depth + 1};
                        }
                    };
                    if (head != previous_water_heads[source_id])
                        awake[chunk_id({point.x / block, point.y / block, point.z / block})] = true;
                    if (point.x < nx - 1) visit(source + 1);
                    if (point.x > 0) visit(source - 1);
                    if (point.y < ny - 1) visit(source + slice);
                    if (point.y > 0) visit(source - slice);
                    if (point.z < nz - 1) visit(source + nx);
                    if (point.z > 0) visit(source - nx);
                }
                if (record_routes) balance_pressure(y, transfers, faces);
            }
        }
    }
