private:
    struct TerrainStrain {
        Kind material = Empty;
        float energy = 0.f;
        V3 impulse;
        uint32_t last_contact = 0;
        bool queued = false;
    };

    struct TerrainImpactPeak {
        float normal_speed = 0.f;
        float face_impulse = 0.f;
    };

    static constexpr size_t terrain_strain_limit = 8192;
    std::unordered_map<int, TerrainStrain> terrain_strain;
    std::vector<int> terrain_fracture_jobs;
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
        add_terrain_strain(index, material, toughness, work, impulse);
    }

    void add_terrain_strain(int index, Kind material, float toughness, float work, V3 impulse) {
        auto entry = terrain_strain.find(index);
        if (entry == terrain_strain.end()) {
            if (terrain_strain.size() >= terrain_strain_limit) return;
            entry = terrain_strain.emplace(index, TerrainStrain{}).first;
        }
        TerrainStrain &strain = entry->second;
        if (strain.material != material) strain = {material, 0.f, V3(), tick, false};
        if (strain.last_contact != tick) {
            const float seconds = (tick - strain.last_contact) * dt;
            strain.energy *= std::exp(-1.10f * seconds);
            strain.impulse = {};
            strain.last_contact = tick;
        }
        strain.energy += work;
        strain.impulse += impulse;
        if (strain.energy >= toughness && !strain.queued) {
            strain.queued = true;
            terrain_fracture_jobs.push_back(index);
        }
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
            add_terrain_strain(index, kind, toughness, work, direction * std::sqrt(2.f * mass * work));
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
