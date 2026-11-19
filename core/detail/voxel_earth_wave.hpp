// Included by World. Metres and seconds use the same scale and
// fixed step as the occupancy and particle solver.
private:
    struct EarthArrival {
        I3 cell;
        float time;
        float pressure;
    };

    struct EarthContact {
        float time;
        float pressure;
        bool arrived = false;
    };

    struct EarthMaterial {
        float speed;
        float loss;
        float impedance;
    };

    using EarthImpactPeaks = std::unordered_map<int, float>;

    struct EarthWave {
        bool active = false;
        int64_t serial = 0;
        int ended = 0; // 1: ejected material, 2: lost contact or absorbed.
        int ejected = 0;
        int disturbed = 0;
        float strength = 0.f;
        float impulse = 0.f;
        float radius = 0.f;
        float range = 0.f;
        float travelled = 0.f;
        float time = 0.f;
        float front_score = 0.f;
        V3 origin;
        V3 direction;
        V3 position;
        std::vector<V3> trace;
        std::vector<EarthArrival> frontier;
        std::unordered_map<int, EarthContact> contacts;
        std::shared_ptr<EarthImpactPeaks> impact_peaks;
        std::unordered_set<int> lifted_cells;
    } earth_wave;
    // An airborne fragment can outlive its wave front. Hold its cast's
    // impact peaks until the last fragment stops rising.
    std::unordered_map<int, std::shared_ptr<EarthImpactPeaks>> airborne_earth_ejecta;

    static constexpr float earth_wave_range = 5.5f;
    static constexpr float earth_wave_minimum_strength = 0.045f;
    static constexpr float earth_wave_maximum_radius = 1.30f;
    static constexpr float earth_wave_step_height = 0.20f;
    static constexpr float earth_wave_trace_spacing = 0.145f;
    static constexpr int earth_wave_crest_limit = 360;
    static constexpr int earth_wave_ejection_limit = 640;
    static constexpr size_t earth_wave_contact_limit = 8192;

    static EarthMaterial earth_material(Kind kind) {
        switch (kind) {
            case Stone: return {4.9f, 0.045f, 1.f};
            case Rock: return {4.4f, 0.075f, 0.83f};
            case Gravel: return {4.0f, 0.10f, 0.69f};
            case Mud: return {3.3f, 0.15f, 0.51f};
            case Sand: return {3.6f, 0.23f, 0.36f};
            default: return {};
        }
    }

    bool earth_wave_conducts(I3 p) const {
        return inside(p) && earth_material(voxels[flat(p)].kind).impedance > 0.f;
    }

    I3 earth_wave_surface(I3 from, int height) const {
        for (int i = 0; i < height; ++i) {
            const I3 above = from + I3(0, 1, 0);
            if (!earth_wave_conducts(above)) break;
            from = above;
        }
        return from;
    }

    bool follow_earth_wave(I3 from, I3 side, I3 &surface) const {
        if (!earth_wave_conducts(from)) return false;
        const int grade = std::max(1, int(earth_wave_step_height / scale));
        for (int depth = 0; depth <= grade && earth_wave_conducts(from); ++depth, --from.y) {
            if (!earth_wave_conducts(from + side)) continue;
            surface = earth_wave_surface(from + side, grade);
            return true;
        }
        return false;
    }

    static V3 earth_surface_point(I3 p) {
        return center(p) + V3(0, scale * 0.5f, 0);
    }

    void move_earth_ejecta(int from, int to) {
        if (airborne_earth_ejecta.empty() || from == to) return;
        auto source = airborne_earth_ejecta.extract(from);
        auto destination = airborne_earth_ejecta.extract(to);
        if (source) {
            source.key() = to;
            airborne_earth_ejecta.insert(std::move(source));
        }
        if (destination) {
            destination.key() = from;
            airborne_earth_ejecta.insert(std::move(destination));
        }
    }

    bool record_earth_contact(const Grain &particle, I3 from, I3 to, EarthImpactPeaks &impact_peaks) {
        const V3 lo = V3(from) * scale;
        const float width = particle.extent * scale;
        const V3 hi = lo + V3(width, width, width);
        const float crossed_top = (to.y + particle.extent) * scale;
        float nearest = crossed_top;
        FormTarget *hit = nullptr;
        int id = 0;
        for (auto &entry : form_targets) {
            FormTarget &target = entry.second;
            if (target.lo.x >= hi.x || target.hi.x <= lo.x ||
                    target.lo.z >= hi.z || target.hi.z <= lo.z) continue;
            if (target.lo.y < hi.y || target.lo.y > nearest) continue;
            if (hit && target.lo.y == nearest && entry.first >= id) continue;
            nearest = target.lo.y;
            hit = &target;
            id = entry.first;
        }
        if (!hit) return false;
        // Targets read the peak incoming normal speed, independent of poll frequency.
        float &delivered = impact_peaks[id];
        const float speed = particle.velocity.y;
        if (speed > delivered) hit->pending_earth_lift += speed - delivered;
        delivered = std::max(delivered, speed);
        return true;
    }

    Grain *lift_earth_particle(I3 p, std::unordered_set<int> &lifted, std::vector<int> &fragmented) {
        const int index = flat(p);
        const Voxel &cell = voxels[index];
        const I3 up(0, 1, 0);
        if (cell.kind == Gravel) {
            Grain &grain = gravel[cell.occupant];
            if (lifted.count(grain.position) || !fits_gravel(decode(grain.position) + up, cell.occupant)) return nullptr;
            lifted.insert(grain.position);
            return &grain;
        }
        if (cell.kind != Stone && cell.kind != Sand && cell.kind != Rock) return nullptr;
        if (lifted.count(index) || !can_enter(p + up, cell.kind == Sand ? Sand : Rock)) return nullptr;
        if (cell.kind == Stone) {
            set_rock(index);
            fragmented.push_back(index);
        }
        lifted.insert(index);
        return movable_particle_at(index);
    }

    void impulse_earth_particle(Grain &particle, V3 impulse) {
        particle.velocity += impulse / gravel_density_ratio(particle);
        particle.velocity.y = std::clamp<float>(particle.velocity.y, 0.f, maximum_particle_speed);
        const I3 start = decode(particle.position);
        settled_sand_chunks[chunk_at(start)] = false;
        const int edge = particle.extent - 1;
        if (edge > 0) wake_box(start, start + I3(edge, edge, edge), awake);
        else wake_near(start, awake);
    }

    void update_earth_fragments(const std::vector<int> &fragmented) {
        if (fragmented.empty()) return;
        ++stone_version;
        ++rock_revision;
        element_support_dirty = true;
        for (int i : fragmented) {
            const I3 p = decode(i);
            for (I3 face : structural_neighbors()) {
                const I3 q = p + face;
                if (inside(q) && kind_at(q) == Stone) stone_support_pending.push_back(flat(q));
            }
        }
    }

    float earth_guide_width() const {
        return std::clamp(0.105f + earth_wave.radius * 0.42f, 0.13f, 0.48f);
    }

    V3 earth_relative(I3 p) const {
        V3 relative = center(p) - earth_wave.origin;
        relative.y = 0.f;
        return relative;
    }

    bool in_earth_guide(V3 relative) const {
        const float along = relative.dot(earth_wave.direction);
        if (along < -scale) return false;
        const float lateral_squared = (relative - earth_wave.direction * along).length_squared();
        if (along <= earth_wave.range && lateral_squared <= earth_guide_width() * earth_guide_width()) return true;
        const V3 from_focus = relative - earth_wave.direction * earth_wave.range;
        return from_focus.length_squared() <= earth_wave.radius * earth_wave.radius;
    }

    static int earth_column(I3 p) { return p.x + nx * p.z; }

    static bool later_earth_arrival(const EarthArrival &a, const EarthArrival &b) {
        if (a.time != b.time) return a.time > b.time;
        return earth_column(a.cell) > earth_column(b.cell);
    }

    void schedule_earth_contact(I3 p, float time, float pressure) {
        auto found = earth_wave.contacts.find(earth_column(p));
        if (found == earth_wave.contacts.end()) {
            if (earth_wave.contacts.size() >= earth_wave_contact_limit) return;
            earth_wave.contacts.emplace(earth_column(p), EarthContact{time, pressure});
        } else {
            EarthContact &old = found->second;
            if (old.arrived || time > old.time + 0.00001f ||
                    (time >= old.time - 0.00001f && pressure <= old.pressure)) return;
            old = {time, pressure};
        }
        earth_wave.frontier.push_back({p, time, pressure});
        std::push_heap(earth_wave.frontier.begin(), earth_wave.frontier.end(), later_earth_arrival);
    }

    bool match_earth_contact(I3 p, I3 step, I3 &surface) const {
        if (step.x == 0 || step.z == 0) return follow_earth_wave(p, step, surface);
        // A diagonal can advance faster than two full axial crossings, but a
        // shared corner by itself does not conduct a pulse.
        const I3 x(step.x, 0, 0), z(0, 0, step.z);
        I3 middle;
        return (follow_earth_wave(p, x, middle) && follow_earth_wave(middle, z, surface)) ||
            (follow_earth_wave(p, z, middle) && follow_earth_wave(middle, x, surface));
    }

    I3 live_earth_contact(I3 p) const {
        const int depth = std::max(1, int(0.13f / scale));
        for (int i = 0; i < depth && !earth_wave_conducts(p) && passable_body(kind_at(p)); ++i) --p.y;
        return p;
    }

    void transmit_earth_contact(I3 p, const EarthArrival &arrival, EarthMaterial source) {
        static const std::array<I3, 8> sides = {{
            {1, 0, 0}, {-1, 0, 0}, {0, 0, 1}, {0, 0, -1},
            {1, 0, 1}, {1, 0, -1}, {-1, 0, 1}, {-1, 0, -1}
        }};
        for (I3 side : sides) {
            if (!in_earth_guide(earth_relative(p + side))) continue;
            I3 target;
            if (!match_earth_contact(p, side, target)) continue;
            const EarthMaterial other = earth_material(kind_at(target));
            const float interface = 2.f * std::sqrt(source.impedance * other.impedance) /
                (source.impedance + other.impedance);
            const float length = V3(target - p).length() * scale;
            const float pressure = arrival.pressure * interface - 0.5f * (source.loss + other.loss) * length;
            if (pressure < earth_wave_minimum_strength) continue;
            const float time = length * 0.5f * (1.f / source.speed + 1.f / other.speed);
            schedule_earth_contact(target, arrival.time + time, pressure);
        }
    }

    void mark_earth_front(I3 p, V3 relative, float pressure) {
        const float along = relative.dot(earth_wave.direction);
        const float off_axis = (relative - earth_wave.direction * along).length();
        const float score = along - 0.43f * off_axis - 2.f * std::max(0.f, along - earth_wave.range);
        if (score <= earth_wave.front_score) return;
        earth_wave.front_score = score;
        earth_wave.travelled = std::clamp(along, 0.f, earth_wave.range);
        earth_wave.position = earth_surface_point(p);
        earth_wave.strength = pressure;
    }

    void release_earth_pressure(I3 p, V3 relative, float pressure, std::vector<int> &fragmented) {
        const float along = relative.dot(earth_wave.direction);
        if (along < 0.12f) return;
        const float width = earth_guide_width();
        const V3 from_focus = relative - earth_wave.direction * earth_wave.range;
        const float focus_distance = from_focus.length();
        const bool exits = focus_distance <= earth_wave.radius;
        const float profile = exits ? 1.f - focus_distance / earth_wave.radius :
            std::clamp(1.f - (relative - earth_wave.direction * along).length() / width, 0.f, 1.f);
        const float footprint = exits ? 3.141593f * earth_wave.radius * earth_wave.radius :
            2.f * width * earth_wave.range;
        const int limit = exits ? earth_wave_ejection_limit : earth_wave_crest_limit;
        int &count = exits ? earth_wave.ejected : earth_wave.disturbed;
        if (count >= limit || (exits && pressure < earth_wave_minimum_strength +
                std::min(0.067f, earth_wave.radius * 0.08f))) return;
        // One active cell per column. Sample the stressed surface across the
        // whole guide so wider casts cannot exhaust their budget at its start.
        const float coverage = std::min(exits ? 1.f : 0.42f, limit * scale * scale / footprint);
        const float threshold = float(hash(uint32_t(earth_column(p)) ^ 0xc1ab3917u) & 0xffffu) / 65535.f;
        if (threshold >= coverage && (!exits || focus_distance >= 1.15f * scale)) return;
        Grain *particle = lift_earth_particle(p, earth_wave.lifted_cells, fragmented);
        if (!particle) return;
        const float speed = exits ? pressure * (2.2f + 3.1f * profile) :
            std::min(2.95f, pressure * (1.75f + 1.65f * profile));
        const V3 horizontal = exits ? from_focus * (0.28f * speed / earth_wave.radius) :
            earth_wave.direction * (0.26f * speed);
        impulse_earth_particle(*particle, horizontal + V3(0, speed, 0));
        ++count;
        if (!exits) return;
        if (!form_targets.empty()) {
            if (!earth_wave.impact_peaks) earth_wave.impact_peaks = std::make_shared<EarthImpactPeaks>();
            airborne_earth_ejecta.insert_or_assign(particle->position, earth_wave.impact_peaks);
        }
        earth_wave.impulse = std::max<float>(earth_wave.impulse, particle->velocity.y);
    }

    void step_earth_wave() {
        if (!earth_wave.active) return;
        earth_wave.time += dt;
        std::vector<int> fragmented;
        while (!earth_wave.frontier.empty() && earth_wave.frontier.front().time <= earth_wave.time) {
            std::pop_heap(earth_wave.frontier.begin(), earth_wave.frontier.end(), later_earth_arrival);
            const EarthArrival arrival = earth_wave.frontier.back();
            earth_wave.frontier.pop_back();
            EarthContact &contact = earth_wave.contacts.at(earth_column(arrival.cell));
            if (contact.arrived || contact.time != arrival.time || contact.pressure != arrival.pressure) continue;
            contact.arrived = true;
            const I3 p = live_earth_contact(arrival.cell);
            if (!earth_wave_conducts(p)) continue;
            const V3 relative = earth_relative(p);
            mark_earth_front(p, relative, arrival.pressure);
            transmit_earth_contact(p, arrival, earth_material(kind_at(p)));
            release_earth_pressure(p, relative, arrival.pressure, fragmented);
        }
        update_earth_fragments(fragmented);
        if (earth_wave.position.distance_squared_to(earth_wave.trace.back()) >=
                earth_wave_trace_spacing * earth_wave_trace_spacing) earth_wave.trace.push_back(earth_wave.position);
        if (!earth_wave.frontier.empty()) return;
        earth_wave.active = false;
        earth_wave.ended = earth_wave.ejected > 0 ? 1 : 2;
    }

public:
    bool start_earth_wave(V3 origin_foot, V3 target, float radius, float force) {
        if (earth_wave.active || !origin_foot.is_finite() || !target.is_finite() ||
                !std::isfinite(radius) || !std::isfinite(force) ||
                radius < scale || radius > earth_wave_maximum_radius ||
                force <= 0.f || force > max_cast_strength) return false;
        V3 direction = target - origin_foot;
        direction.y = 0.f;
        const float distance = std::hypot(direction.x, direction.z);
        if (!std::isfinite(distance) || distance < 0.35f) return false;
        I3 seed = grid(origin_foot + V3(0, scale * 0.45f, 0));
        if (!inside(seed)) return false;
        const int depth = std::max(1, int(0.26f / scale));
        for (int i = 0; i < depth; ++i) {
            if (!inside(seed) || !passable_body(kind_at(seed))) break;
            --seed.y;
        }
        if (!earth_wave_conducts(seed)) return false;

        const int64_t serial = earth_wave.serial + 1;
        earth_wave = {};
        earth_wave.active = true;
        earth_wave.serial = serial;
        earth_wave.origin = origin_foot;
        earth_wave.direction = direction / distance;
        earth_wave.position = V3(origin_foot.x, earth_surface_point(seed).y, origin_foot.z);
        earth_wave.strength = force;
        earth_wave.range = std::min(distance, earth_wave_range);
        earth_wave.radius = radius;
        earth_wave.trace.push_back(earth_wave.position);
        schedule_earth_contact(seed, 0.f, force);
        return true;
    }

    EarthWaveInfo earth_wave_state() const {
        EarthWaveInfo state;
        state.active = earth_wave.active;
        state.position = earth_wave.position;
        state.strength = std::clamp(earth_wave.strength, 0.f, 1.f);
        state.serial = earth_wave.serial;
        state.ended = earth_wave.ended;
        state.radius = earth_wave.radius;
        state.ejected = earth_wave.ejected;
        state.disturbed = earth_wave.disturbed;
        state.impulse = earth_wave.impulse;
        state.direction = earth_wave.direction;
        state.travelled = earth_wave.travelled;
        state.range = earth_wave.range;
        state.contact_count = earth_wave.contacts.size();
        state.trail.assign(earth_wave.trace.begin(), earth_wave.trace.end());
        if (!state.trail.empty() && state.trail.back() != earth_wave.position)
            state.trail.push_back(earth_wave.position);
        return state;
    }

    TargetContacts take_earth_contacts() {
        TargetContacts contacts;
        for (auto &entry : form_targets) {
            FormTarget &target = entry.second;
            if (target.pending_earth_lift <= 0.f) continue;
            contacts.emplace_back(entry.first, target.pending_earth_lift);
            target.pending_earth_lift = 0.f;
        }
        return contacts;
    }
