private:
    struct LoadHistory {
        float energy = 0.f;
        V3 impulse;
        uint32_t last_contact = 0;
        bool queued = false;
    };

    struct TerrainStrain : LoadHistory {
        Kind material = Empty;
    };

    struct TerrainImpactPeak {
        float normal_speed = 0.f;
        float face_impulse = 0.f;
    };

    static constexpr size_t terrain_strain_limit = 8192;
    static constexpr float sand_yield_pressure = 14000.f;
    std::unordered_map<int, TerrainStrain> terrain_strain;
    std::vector<int> terrain_fracture_jobs;
    std::unordered_map<int, LoadHistory> packed_strain;
    std::vector<int> packed_fracture_jobs;
    std::vector<std::pair<int, float>> compaction_jobs;
    struct CompressionHistory {
        float work = 0.f;
        float pressure = 0.f;
        uint32_t last_contact = 0;
        uint32_t retry_after = 0;
    };
    std::unordered_map<int, CompressionHistory> contact_compression;
    std::vector<int> sampled_compression_sites;
    std::array<int, Char + 1> terrain_contact_breaks{};
    std::array<TerrainImpactPeak, Char + 1> terrain_impact_peaks{};

    static float terrain_toughness(Kind kind) {
        // Fracture work scales with the newly exposed face area.
        constexpr float face_ratio = (scale / 0.045f) * (scale / 0.045f);
        switch (kind) {
            case Stone: return 23.f * face_ratio;
            case Wood: return 13.f * face_ratio;
            case Ice: return 6.f * face_ratio;
            case Char: return 3.f * face_ratio;
            default: return 0.f;
        }
    }

    void reset_terrain_fractures() {
        terrain_strain.clear();
        terrain_fracture_jobs.clear();
        packed_strain.clear();
        packed_fracture_jobs.clear();
        compaction_jobs.clear();
        contact_compression.clear();
        sampled_compression_sites.clear();
        terrain_contact_breaks.fill(0);
        terrain_impact_peaks.fill({});
    }

    void record_terrain_contact(int index, const Grain &projectile, Kind kind, int axis, int direction,
            int bearing_cells = 1) {
        const Kind material = voxels[index].kind;
        const float toughness = terrain_toughness(material);
        if (toughness <= 0.f || mobile_at(index)) return;

        const float normal_speed = projectile.velocity[axis] * direction;
        // Resolve the stopped momentum over the cells actually bearing load.
        // The empty part of a projectile face cannot absorb any impact.
        const float local_impulse = particle_mass(kind, projectile) * normal_speed / bearing_cells;
        const float absorption = kind == Water ? 0.32f : 0.62f;
        const float work = 0.5f * local_impulse * normal_speed * absorption;
        if (work < 0.018f * toughness) return;
        auto &peak = terrain_impact_peaks[material];
        if ((kind == Sand || kind == Gravel) && normal_speed > peak.normal_speed)
            peak = {normal_speed, local_impulse};

        V3 impulse;
        impulse[axis] = direction * local_impulse;
        add_terrain_strain(index, material, toughness, {work, impulse});
    }

    bool accumulate_load(LoadHistory &strain, MaterialLoad load, float resistance) {
        if (strain.last_contact != tick) {
            const float seconds = (tick - strain.last_contact) * dt;
            strain.energy *= std::exp(-1.10f * seconds);
            strain.impulse = {};
            strain.last_contact = tick;
        }
        strain.energy += load.work;
        strain.impulse += load.impulse;
        if (strain.queued || strain.energy < resistance) return false;
        strain.queued = true;
        return true;
    }

    static float packed_fracture_work(const Grain &grain) {
        // Breaking creates an internal surface. Use physical area and a
        // higher work per square metre for compressed sand.
        const float edge = grain.extent * scale;
        return edge * edge * (grain.sand_units ? 760.f : 330.f);
    }

    static constexpr float sand_compaction_work(int extent, int donors) {
        return scale * scale * scale * (1000.f * extent * extent * extent + 24000.f * donors);
    }

    // At this relative speed, a pair of loose grains can supply the minimum
    // compaction work from half of their dissipated normal kinetic energy.
    static constexpr float minimum_sand_compaction_speed = 16.73f;

    void record_compaction_load(I3 contact, MaterialLoad load) {
        if (load.work < sand_compaction_work(2, 2) || compaction_jobs.size() >= 64) return;
        const I3 anchor(contact.x & ~1, contact.y & ~1, contact.z & ~1);
        compaction_jobs.emplace_back(flat(anchor), load.work);
    }

    void record_contact_pressure(I3 p, float pressure) {
        const I3 anchor(p.x & ~1, p.y & ~1, p.z & ~1);
        const int id = flat(anchor);
        auto entry = contact_compression.find(id);
        if (entry == contact_compression.end()) {
            if (contact_compression.size() >= terrain_strain_limit) return;
            entry = contact_compression.emplace(id, CompressionHistory{}).first;
        }
        auto &history = entry->second;
        if (history.pressure == 0.f) sampled_compression_sites.push_back(id);
        history.pressure = std::max(history.pressure, pressure);
    }

    void integrate_contact_pressure() {
        // Visit sites in sweep order. Each contributes its strongest sample
        // once, even when several grains press on the same region.
        for (int id : sampled_compression_sites) {
            auto &history = contact_compression.at(id);
            const float pressure = std::exchange(history.pressure, 0.f);
            accumulate_site_pressure(id, history, pressure);
        }
        sampled_compression_sites.clear();
    }

    void accumulate_site_pressure(int id, CompressionHistory &history, float pressure) {
        const uint32_t sample_gap = tick - history.last_contact;
        if (sample_gap > 2) {
            const float gap = (sample_gap - 2) * dt;
            history.work *= std::max(0.f, 1.f - 6.f * gap);
        }
        history.last_contact = tick;

        // Work is pressure above yield times the volume that can close this
        // step. Use one sample per packing region and a bounded strain rate.
        constexpr float closure_volume = 2.f * scale * scale * scale;
        constexpr float max_closure_rate = 3.f;
        const float threshold = sand_compaction_work(2, 2);
        const float duration = (sample_gap == 2 && history.work > 0.f ? 2.f : 1.f) * dt;
        history.work = std::min(threshold,
            history.work + (pressure - sand_yield_pressure) * closure_volume * max_closure_rate * duration);
        if (history.work >= threshold && tick >= history.retry_after && compaction_jobs.size() < 64) {
            compaction_jobs.emplace_back(id, history.work);
            history.retry_after = tick + std::max(1, simulation_hz / 15);
        }
    }

    template <typename Fits>
    void sample_bearing_contact(const Grain &grain, I3 from, Kind kind, V3 acceleration, const Fits &fits) {
        if (kind != Sand && kind != Gravel && kind != Rock) return;
        // Rotate between opposite corners so one side of each packing cell
        // is not permanently omitted.
        const int corner = int(tick & 1u);
        if (kind == Sand && (((from.x ^ corner) | (from.y ^ corner) | (from.z ^ corner)) & 1)) return;
        // One physical bearing layer approximates the load carried by
        // neighboring grains. A rigid packet uses its mass and face area.
        const float edge = scale * grain.extent;
        const float mass_per_area = kind == Sand ? sand_density * 0.12f
            : particle_mass(kind, grain) / (edge * edge);
        const V3 internal = acceleration + V3(0, gravity, 0);
        float best = sand_yield_pressure;
        int bearing = -1;
        I3 normal;
        for (int axis = 0; axis < 3; ++axis) {
            // Adjacent sand shares gravity. Only a fixed support can turn
            // common downward acceleration into contact load.
            const float applied = mass_per_area * acceleration[axis];
            const float relative = mass_per_area * internal[axis];
            if (std::max(std::abs(applied), std::abs(relative)) <= best) continue;
            for (int direction : {-1, 1}) {
                if (std::max(applied * direction, relative * direction) <= best ||
                        grain.velocity[axis] * direction < -0.1f) continue;
                I3 neighbor = from;
                neighbor[axis] += direction;
                int obstacle = -1;
                if (fits(neighbor, &obstacle) || obstacle < 0) continue;
                const Kind surface = voxels[obstacle].kind;
                const bool fixed = kind == Sand && retains_grain_face(obstacle);
                if (surface != Sand && !(kind == Sand && (surface == Gravel || fixed))) continue;
                const float pressure = (fixed || kind != Sand ? applied : relative) * direction;
                if (pressure > best) {
                    best = pressure;
                    bearing = obstacle;
                    normal = {};
                    normal[axis] = direction;
                }
            }
        }
        if (bearing < 0) return;
        if (voxels[bearing].kind != Sand) {
            if (kind == Sand) record_contact_pressure(from, best);
            return;
        }
        const I3 contact = decode(bearing);
        record_contact_pressure(contact, best);
        // Follow a short, solid path into the load-bearing sand. A gap ends
        // the path. Updating only the far end bounds cache traffic per sample.
        constexpr int bearing_cells = int(0.18f / scale + 0.5f);
        const float transmitted = best * 0.9f;
        if (transmitted <= sand_yield_pressure) return;
        I3 inner = contact;
        for (int i = 1; i < bearing_cells; ++i) {
            if (kind_at(inner + normal) != Sand) break;
            inner += normal;
        }
        if (inner != contact) record_contact_pressure(inner, transmitted);
    }

    void finish_contact_compaction() {
        integrate_contact_pressure();
        if (!contact_compression.empty() && tick % uint32_t(simulation_hz / 4) == 0) {
            for (auto entry = contact_compression.begin(); entry != contact_compression.end();) {
                if (tick - entry->second.last_contact > uint32_t(simulation_hz / 4)) entry = contact_compression.erase(entry);
                else ++entry;
            }
        }
        if (compaction_jobs.empty()) return;
        std::stable_sort(compaction_jobs.begin(), compaction_jobs.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });
        for (size_t i = 0; i < compaction_jobs.size();) {
            const auto [anchor, initial_work] = compaction_jobs[i++];
            float work = initial_work;
            while (i < compaction_jobs.size() && compaction_jobs[i].first == anchor) work += compaction_jobs[i++].second;
            const I3 site = decode(anchor);
            if (compact_with_work(particle_center(site, 2), std::min(1.f, 1.7f * scale), 2, {work, {}}))
                contact_compression.erase(anchor);
        }
        compaction_jobs.clear();
    }

    void record_packed_load(int id, MaterialLoad load) {
        const float resistance = packed_fracture_work(gravel[id]);
        if (load.work < 0.018f * resistance) return;
        auto it = packed_strain.find(id);
        if (it == packed_strain.end()) {
            if (packed_strain.size() >= terrain_strain_limit) return;
            it = packed_strain.emplace(id, LoadHistory{}).first;
        }
        if (accumulate_load(it->second, load, resistance)) packed_fracture_jobs.push_back(id);
    }

    void add_terrain_strain(int index, Kind material, float toughness, MaterialLoad load) {
        auto entry = terrain_strain.find(index);
        if (entry == terrain_strain.end()) {
            if (terrain_strain.size() >= terrain_strain_limit) return;
            entry = terrain_strain.emplace(index, TerrainStrain{}).first;
        }
        TerrainStrain &strain = entry->second;
        if (strain.material != material) {
            strain = {};
            strain.material = material;
        }
        if (accumulate_load(strain, load, toughness)) terrain_fracture_jobs.push_back(index);
    }

    int apply_surface_pulse(const std::vector<std::pair<int, float>> &contacts, float total_weight,
            V3 source, float strength) {
        if (contacts.empty()) return 0;
        // One pressure pulse supplies a fixed work budget. Receivers share it
        // according to their distance from the center of the strike.
        const float work_scale = 1050.f * strength * strength / std::max(1.f, total_weight);
        for (const auto &[index, weight] : contacts) {
            const Kind kind = voxels[index].kind;
            const float toughness = terrain_toughness(kind);
            const float work = work_scale * weight;
            if (work < 0.018f * toughness) continue;
            const float mass = particle_mass(kind == Stone ? Rock : kind, 1);
            const V3 direction = (center(decode(index)) - source).normalized();
            add_terrain_strain(index, kind, toughness, {work, direction * std::sqrt(2.f * mass * work)});
        }
        return finish_terrain_fractures();
    }

    bool record_gravel_terrain_contacts(I3 next, const Grain &projectile, int axis, int direction) {
        const int edge = projectile.extent - 1;
        if (direction > 0) next[axis] += edge;
        const int u = (axis + 1) % 3;
        const int v = (axis + 2) % 3;
        // Packed bodies have at most eight cells on each edge.
        std::array<int, 8 * 8> bearing{};
        int bearing_count = 0;
        for (int a = 0; a <= edge; ++a) {
            for (int b = 0; b <= edge; ++b) {
                I3 contact = next;
                contact[u] += a;
                contact[v] += b;
                const int index = flat(contact);
                const Kind material = voxels[index].kind;
                if (mobile_at(index) || (material != Bedrock && terrain_toughness(material) <= 0.f)) continue;
                bearing[bearing_count++] = index;
            }
        }
        // Bedrock participates in the division, but its own share cannot
        // fracture it. record_terrain_contact rejects its zero toughness.
        for (int i = 0; i < bearing_count; ++i)
            record_terrain_contact(bearing[i], projectile, Gravel, axis, direction, bearing_count);
        return bearing_count != 0;
    }

    bool split_packed_body(int id, std::vector<int> &overflow) {
        const Grain grain = gravel[id];
        const int extent = grain.extent;
        const int origin = grain.position;
        const I3 start = decode(origin);
        const int mass_units = gravel_mass_units(grain);
        if (extent == 2 && !gather_gravel_shell(start, extent, mass_units - cube_cells(extent), overflow,
                [&](I3 p) { return voxels[flat(p)].kind == Empty && !body_at(flat(p)); })) return false;

        const auto motion = voxels[origin].formed ? std::optional<FormMotion>(form_motion.at(origin)) : std::nullopt;
        erase_gravel(id);
        if (extent == 2) {
            invalidate_support_above(start, extent);
            const auto release = [&](int i) {
                voxels[i] = {Sand, voxel_stamp, int(grains.size())};
                grains.push_back({i, i, tick, 1, grain.velocity});
                if (!sand_tile_dirty.empty()) mark_sand_tiles(decode(i));
                queue_sand_contact(i);
                set_mobile(i, true);
                if (motion) attach_form_motion(i, *motion);
                wake_near(decode(i), awake);
            };
            each_cube(origin, extent, release);
            for (int i : overflow) release(i);
        } else {
            const int half = extent / 2;
            for (int offset : gravel_offsets()) {
                const int child = origin + half * offset;
                const int child_id = int(gravel.size());
                gravel.push_back({child, child, tick, half, grain.velocity});
                if (grain.sand_units) gravel.back().sand_units = uint16_t(mass_units / 8);
                if (!sand_tile_dirty.empty()) mark_gravel_tile(decode(child));
                each_cube(child, half, [&](int i) { voxels[i] = {Gravel, voxel_stamp, child_id}; });
                if (motion) attach_form_motion(child, *motion);
            }
        }
        wake_box(start, start + I3(extent - 1, extent - 1, extent - 1), awake);
        return true;
    }

    int finish_packed_fractures() {
        int fractured = 0;
        if (!packed_fracture_jobs.empty()) {
            // Erasing a packed body swaps in the last ID. Descending order
            // keeps every unprocessed job pointed at its original body.
            std::sort(packed_fracture_jobs.rbegin(), packed_fracture_jobs.rend());
            std::vector<int> overflow;
            for (int id : packed_fracture_jobs) {
                if (split_packed_body(id, overflow)) ++fractured;
                else packed_strain.at(id).queued = false;
            }
            packed_fracture_jobs.clear();
            if (fractured) ++sand_revision;
        }
        if (!packed_strain.empty() && tick % uint32_t(simulation_hz) == 0) {
            for (auto entry = packed_strain.begin(); entry != packed_strain.end();) {
                if (tick - entry->second.last_contact >= 3 * simulation_hz) entry = packed_strain.erase(entry);
                else ++entry;
            }
        }
        return fractured;
    }

    int finish_terrain_fractures() {
        int fractured = 0;
        bool fractured_stone = false;
        for (int index : terrain_fracture_jobs) {
            const auto entry = terrain_strain.find(index);
            if (entry == terrain_strain.end()) continue;
            const TerrainStrain strain = entry->second;
            terrain_strain.erase(entry);
            const Kind material = voxels[index].kind;
            if (material != strain.material || mobile_at(index)) continue;

            const I3 p = decode(index);
            if (material == Stone) {
                set_rock(index);
                fractured_stone = true;
                for (I3 face : structural_neighbors()) {
                    const I3 neighbor = p + face;
                    if (inside(neighbor) && kind_at(neighbor) == Stone) stone_support_pending.push_back(flat(neighbor));
                }
            } else {
                detach_element(index);
            }
            element_support_dirty = true;
            ++terrain_contact_breaks[material];
            ++fractured;

            // Only unused fracture work can leave as fragment motion. Earlier
            // impulses were carried by the intact support, not stored as speed.
            Grain &fragment = *movable_particle_at(index);
            const float mass = particle_mass(voxels[index].kind, fragment);
            const float surplus = std::max(0.f, strain.energy - terrain_toughness(material));
            const float speed = std::min(maximum_swept_speed, std::sqrt(2.f * surplus / mass));
            fragment.velocity = (strain.impulse / mass).limit_length(speed);
        }
        terrain_fracture_jobs.clear();
        if (fractured_stone) {
            ++stone_version;
            ++rock_revision;
        }

        if (!terrain_strain.empty() && tick % uint32_t(1.f / dt) == 0) {
            for (auto entry = terrain_strain.begin(); entry != terrain_strain.end();) {
                const TerrainStrain &strain = entry->second;
                const int index = entry->first;
                if ((tick - strain.last_contact) * dt >= 3.f || voxels[index].kind != strain.material || mobile_at(index))
                    entry = terrain_strain.erase(entry);
                else
                    ++entry;
            }
        }
        return fractured;
    }

public:
    std::optional<ImpactInfo> sand_impact_peak(int material) const {
        if (material < 0 || material > Char) return std::nullopt;
        const auto &peak = terrain_impact_peaks[material];
        return ImpactInfo{peak.normal_speed, peak.face_impulse};
    }
