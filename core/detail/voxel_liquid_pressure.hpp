private:
    struct PressureNode {
        int parent = -1;
        int depth = 0;
    };

    static constexpr uint32_t pressure_interval = simulation_hz / 15;
    static constexpr int pressure_transfer_limit = 48;
    static constexpr int pressure_face_budget = 2048;
    using LiquidHead = std::conditional_t<(ny <= 255), uint8_t, uint16_t>;

    class LiquidHeadPages {
        using Page = std::array<LiquidHead, block * block * block>;
        std::vector<std::unique_ptr<Page>> pages;
        size_t allocated = 0;

        static int offset(I3 p) { return p.x % block + block * (p.z % block + block * (p.y % block)); }
        static int page_id(I3 p) { return chunk_id({p.x / block, p.y / block, p.z / block}); }

        void allocate(int id) {
            pages[id] = std::make_unique<Page>();
            ++allocated;
        }

    public:
        void prepare(const std::vector<uint16_t> &wet_chunks) {
            if (pages.empty()) pages.resize(chunk_count);
            for (int id = 0; id < chunk_count; ++id) {
                if (wet_chunks[id] && !pages[id]) allocate(id);
            }
        }

        void clear() {
            pages.clear();
            allocated = 0;
        }

        size_t bytes() const { return pages.capacity() * sizeof(std::unique_ptr<Page>) + allocated * sizeof(Page); }

        LiquidHead get(I3 p) const {
            if (pages.empty()) return 0;
            const auto &page = pages[page_id(p)];
            return page ? (*page)[offset(p)] : 0;
        }

        LiquidHead get(int index) const { return get(decode(index)); }

        void set(int index, LiquidHead head) {
            const I3 p = decode(index);
            const int id = page_id(p);
            if (!pages[id]) allocate(id);
            (*pages[id])[offset(p)] = head;
        }

        // Prepare all occupied pages before dispatch. Each worker owns whole
        // Z chunk bands and publishes each occupied cell at most once.
        bool publish(I3 p, int width, LiquidHead head) {
            const auto first = pages[page_id(p)]->begin() + offset(p);
            const auto last = first + width;
            if (std::all_of(first, last, [head](LiquidHead value) { return value == head; })) return false;
            std::fill(first, last, head);
            return true;
        }
    };

    std::array<std::vector<int>, ny> water_by_height;
    LiquidHeadPages liquid_head;
    std::vector<LiquidHead> previous_water_heads;
#ifdef SOMEVOXEL_PARALLEL_WATER_WORKERS
    using PackedWaterSpans = WaterSpans<nx, ny, nz>;
    mutable std::unique_ptr<PackedWaterSpans> water_spans;
    typename PackedWaterSpans::Surfaces pressure_surfaces;
    int64_t pressure_surface_version = -1;
    int requested_water_workers = -1;
    WaterHeadAccelerator water_head_accelerator;
    int64_t accelerated_head_rebuilds = 0;
    int64_t accelerator_attempts = 0;
    int64_t accelerator_time_us = 0;
    detail::WaterHeadSelection head_selection;

    PackedWaterSpans &water_solver() const {
        if (!water_spans) water_spans = std::make_unique<PackedWaterSpans>(water_worker_count());
        return *water_spans;
    }
#endif
    WaterHeadTimes head_times;

    int64_t record_head_time(bool gpu, std::chrono::steady_clock::time_point start) {
        const int64_t elapsed = std::max<int64_t>(1, std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - start).count());
        int64_t &first = gpu ? head_times.first_gpu_us : head_times.first_cpu_us;
        if (!first) first = elapsed;
        (gpu ? head_times.gpu_us : head_times.cpu_us) = elapsed;
        head_times.last_gpu = gpu;
        return elapsed;
    }
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

#ifdef SOMEVOXEL_PARALLEL_WATER_WORKERS
    bool pressure_has_outlet(const typename PackedWaterSpans::Surfaces &surfaces) const {
        struct Levels {
            int low = ny;
            int high = -1;
            bool draining = false;
        };
        std::unordered_map<int, Levels> levels;
        for (const auto &band : surfaces) {
            for (const auto &run : band) {
                auto &component = levels[run.component];
                if (component.draining) continue;
                const int layer = run.cell / slice;
                for (int bit = 0; bit < block; ++bit) {
                    if (!(run.mask & (1u << bit))) continue;
                    const int i = run.cell + bit;
                    if (!passive_water(i) || voxels[i + slice].kind != Empty || body_at(i + slice)) continue;
                    if (!supported_pressure_surface(decode(i))) {
                        component.draining = true;
                        break;
                    }
                    component.low = std::min(component.low, layer);
                    component.high = std::max(component.high, layer);
                }
            }
        }
        // The lowest valid outlet is one layer above its support, and the
        // original pressure solver only moves water downhill by a full cell.
        for (const auto &entry : levels) {
            const auto &component = entry.second;
            if (!component.draining && component.low + 1 < component.high) return true;
        }
        return false;
    }
#endif

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
        liquid_head.set(outlet, head);
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
        const std::array<int64_t, 7> revisions{water_topology_version, stone_version, rock_revision, sand_revision,
            elements_version, water_form_revision, int64_t(liquid_motion.size())};
        const bool balance = tick % pressure_interval == 0 && (pressure_revision != revisions ||
            pressure_body_feet != body_feet || pressure_body_radius != body_radius || pressure_body_height != body_height);
        const bool head_current = head_version == water_topology_version;
        if (head_current && !balance) return;
        auto head_start = std::chrono::steady_clock::time_point{};
        if (!balance) head_start = std::chrono::steady_clock::now();
        if (!head_current) liquid_head.prepare(water_chunk_counts);
        head_version = water_topology_version;
        if (balance) {
            pressure_revision = revisions;
            pressure_body_feet = body_feet;
            pressure_body_radius = body_radius;
            pressure_body_height = body_height;
        }

#ifdef SOMEVOXEL_PARALLEL_WATER_WORKERS
        const bool use_rows = size_t(water_scan_end / block) <= 4 * water_positions.size();
        if (water_positions.size() >= 16384 && (!balance || use_rows)) {
            auto &solver = water_solver();
            const auto apply_head = [this](int i, int width, int head) {
                const I3 p = decode(i);
                if (liquid_head.publish(p, width, LiquidHead(head))) {
                    awake[chunk_id({p.x / block, p.y / block, p.z / block})] = true;
                }
            };
            if (balance) {
                // If no component can make the first transfer, the snapshot
                // will stay valid throughout this pressure update. Otherwise
                // keep the existing traversal and parcel routing order.
                const auto can_publish = [&](const auto &surfaces) { return !pressure_has_outlet(surfaces); };
                if (head_current && pressure_surface_version == water_topology_version) {
                    if (can_publish(pressure_surfaces)) return;
                } else {
                    pressure_surface_version = water_topology_version;
                    if (solver.try_pressure_rows(bulk_water_rows, liquid_motion, water_scan_end / slice,
                            pressure_surfaces, apply_head, can_publish)) return;
                }
            } else {
                const bool eligible = water_head_accelerator && water_positions.size() >= 1000000;
                const bool compare = eligible && head_times.automatic;
                const bool gpu = eligible && (!compare || head_selection.choose_gpu(
                        int64_t(water_positions.size()), water_scan_end / slice, water_topology_version));
                if (gpu) {
                    const bool solved = solver.try_accelerated_rows(bulk_water_rows, liquid_motion,
                        water_scan_end / slice, water_head_accelerator, apply_head);
                    const auto elapsed = record_head_time(true, head_start);
                    ++accelerator_attempts;
                    accelerator_time_us += elapsed;
                    if (solved) {
                        ++accelerated_head_rebuilds;
                        if (compare) head_selection.measured(elapsed);
                        return;
                    }
                    if (compare) head_selection.gpu_failed();
                    head_start = std::chrono::steady_clock::now();
                }
                // Sparse scenes benefit from visiting just their parcel list.
                if (use_rows) solver.rebuild_rows(bulk_water_rows, liquid_motion, water_scan_end / slice, apply_head);
                else solver.rebuild(water_positions, apply_head);
                const auto elapsed = record_head_time(false, head_start);
                if (compare && !gpu) head_selection.measured(elapsed);
                return;
            }
        }
#endif

        if (balance) pressure_nodes.resize(water_positions.size());
        // Only occupied water cells participate in the wake comparison. Keep
        // the snapshot by parcel ID for the duration of this rebuild.
        previous_water_heads.resize(water_positions.size());
        for (auto &height : water_by_height) height.clear();
        for (size_t id = 0; id < water_positions.size(); ++id) {
            const int i = water_positions[id];
            previous_water_heads[id] = liquid_head.get(i);
            liquid_head.set(i, 0);
            water_by_height[i / slice].push_back(i);
        }
        int transfers = pressure_transfer_limit;
        int faces = pressure_face_budget;
        for (int y = ny - 1; y >= 0; --y) {
            const LiquidHead head = y + 1;
            for (int seed : water_by_height[y]) {
                if (liquid_head.get(seed) || voxels[seed].kind != Water) continue;
                pressure_component.clear();
                pressure_component.push_back(seed);
                liquid_head.set(seed, head);
                const bool record_routes = balance && transfers > 0 && faces > 0;
                if (record_routes) pressure_nodes[voxels[seed].occupant] = {};
                for (size_t cursor = 0; cursor < pressure_component.size(); ++cursor) {
                    const int source = pressure_component[cursor];
                    const I3 point = decode(source);
                    const int source_id = voxels[source].occupant;
                    const auto visit = [&](int i) {
                        if (voxels[i].kind != Water || liquid_head.get(i)) return;
                        liquid_head.set(i, head);
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
        if (!balance) record_head_time(false, head_start);
    }
