private:
    struct ParticleForm {
        V3 point;
        V3 forward;
        V3 right;
        V3 up;
        V3 half;
        V3 handle;
        Kind material = Empty;
        FormStyle style = NoForm;
        float radius = 0.f;
        float strength = 1.f;
    } particle_form;

    static constexpr int whip_node_count = 10;
    struct WhipPulse {
        std::array<V3, whip_node_count> offset{};
        std::array<V3, whip_node_count> velocity{};
        uint32_t ready = 0;
        bool moving = false;
    } whip_pulse;

    struct FormTarget {
        V3 lo;
        V3 hi;
        float pending_damage = 0.f;
        float pending_earth_lift = 0.f;
    };

    static constexpr size_t form_motion_limit = 1600;
    std::unordered_map<int, FormMotion> form_motion;
    std::unordered_map<int, FormTarget> form_targets;
    std::vector<std::pair<float, int>> form_candidates;
    uint32_t next_form_recruit = 0;
    uint32_t whip_mobile_until = 0;
    int held_form_count = 0;
    int formed_water_count = 0;
    int moved_water_form = 0;
    int64_t water_form_revision = 0;

    static V3 form_half_extents(FormStyle style, float radius) {
        radius = std::min(radius, 3.2f);
        if (style == BladeForm) return V3(std::max(0.17f, 0.58f * radius),
            std::max(0.062f, 0.105f * radius), std::max(0.047f, 0.052f * radius));
        const float width = std::max(0.11f, 0.21f * radius);
        return {width, width, width};
    }

    static float form_random(uint32_t seed) { return float(hash(seed) & 0xffffu) / 65535.f; }

    static V3 form_anchor(uint32_t key, FormStyle style) {
        const float a = form_random(key ^ 0x6c8e9cf5u) * 6.2831853f;
        if (style == WhipForm) {
            const float radius = std::sqrt(form_random(key));
            return V3(form_random(key ^ 0x97bb41d7u), std::cos(a) * radius, std::sin(a) * radius);
        }
        const float y = form_random(key ^ 0x97bb41d7u) * 2.f - 1.f;
        const float disk = std::sqrt(std::max(0.f, 1.f - y * y));
        return V3(std::cos(a) * disk, y, std::sin(a) * disk) * std::cbrt(form_random(key));
    }

    bool whip_is_lashing() const {
        return whip_pulse.moving;
    }

    void step_whip_pulse() {
        if (!whip_pulse.moving) return;
        constexpr int substeps = std::max(1, 4 / refinement);
        constexpr float timestep = dt / substeps;
        float largest_motion = 0.f;
        for (int step = 0; step < substeps; ++step) {
            for (int i = 1; i < whip_node_count; ++i) {
                const V3 node = whip_pulse.offset[i];
                const V3 previous = whip_pulse.offset[i - 1];
                const V3 next = whip_pulse.offset[std::min(i + 1, whip_node_count - 1)];
                // The handle is fixed relative to the held shape and the tip
                // is free. Neighbor tension transmits the impulse with damping.
                V3 &velocity = whip_pulse.velocity[i];
                velocity += ((previous + next - node * 2.f) * 1300.f - node * 70.f - velocity * 6.f) * timestep;
            }
            largest_motion = 0.f;
            for (int i = 1; i < whip_node_count; ++i) {
                V3 &node = whip_pulse.offset[i];
                const V3 step_offset = whip_pulse.velocity[i] * timestep;
                node = (node + step_offset).limit_length(1.65f);
                largest_motion = std::max(largest_motion, node.length_squared() + step_offset.length_squared());
            }
        }
        if (largest_motion < 0.000004f) {
            whip_pulse.offset.fill(V3());
            whip_pulse.velocity.fill(V3());
            whip_pulse.moving = false;
        }
    }

    V3 whip_destination(V3 anchor) const {
        const float u = anchor.x;
        const float width = 0.037f + 0.063f * (1.f - u);
        const float bend = std::sin(3.14159265f * u);
        V3 target = particle_form.handle.lerp(particle_form.point, u);
        target += particle_form.right * (anchor.y * width + bend * 0.09f);
        target += particle_form.up * (anchor.z * width - bend * 0.14f);
        if (whip_is_lashing()) {
            const float sample = u * (whip_node_count - 1);
            const int first = std::min(int(sample), whip_node_count - 2);
            target += whip_pulse.offset[first].lerp(whip_pulse.offset[first + 1], sample - first);
        }
        return target;
    }

    float form_charge_gather() const {
        return particle_form.style == ShotForm || particle_form.style == BladeForm ? charge_gather : 0.f;
    }

    V3 form_destination(const FormMotion &motion) const {
        if (motion.style == WhipForm) return whip_destination(motion.anchor);
        const V3 offset = motion.anchor * particle_form.half;
        return particle_form.point + particle_form.right * offset.x
            + particle_form.up * offset.y + particle_form.forward * offset.z;
    }

    int form_capacity() const {
        if (particle_form.style == WhipForm) {
            const float length = particle_form.handle.distance_to(particle_form.point);
            return std::clamp(int(0.011f * length / (scale * scale * scale)), 18, 420);
        }
        const float width = 4.18879f * (particle_form.half.x / scale) * (particle_form.half.y / scale)
            * (particle_form.half.z / scale);
        return std::clamp(int(width * 0.48f), 12, 620);
    }

    void reset_particle_forms() {
        particle_form = {};
        whip_pulse = {};
        form_motion.clear();
        form_targets.clear();
        form_candidates.clear();
        next_form_recruit = 0;
        whip_mobile_until = 0;
        held_form_count = 0;
        formed_water_count = 0;
        moved_water_form = 0;
        ++water_form_revision;
    }

    void set_form_voxel(int i, bool formed) {
        Voxel &cell = voxels[i];
        cell.formed = formed;
        if (cell.kind != Water) return;
        formed_water_count += formed ? 1 : -1;
        ++water_form_revision;
        ++water_version;
        dirty_water(decode(i));
    }

    void forget_particle_form(int i) {
        if (!voxels[i].formed) return;
        auto motion = form_motion.find(i);
        if (motion->second.expires == 0) --held_form_count;
        form_motion.erase(motion);
        set_form_voxel(i, false);
    }

    void move_form_state(int a, int b) {
        // Occupancy has already swapped. Rekey motion for each displaced cell.
        auto first = form_motion.extract(a);
        auto second = form_motion.extract(b);
        if (first) {
            first.key() = b;
            form_motion.insert(std::move(first));
        }
        if (second) {
            second.key() = a;
            form_motion.insert(std::move(second));
        }
    }

    void release_held_form() {
        if (particle_form.style == NoForm) return;
        for (auto item = form_motion.begin(); item != form_motion.end();) {
            if (item->second.expires != 0) {
                ++item;
                continue;
            }
            set_form_voxel(item->first, false);
            wake_near(decode(item->first), awake);
            item = form_motion.erase(item);
        }
        held_form_count = 0;
        particle_form = {};
        whip_pulse = {};
        whip_mobile_until = 0;
    }

    void recruit_form_particles() {
        const int limit = std::min({form_capacity() - held_form_count, int(form_motion_limit - form_motion.size()), 120});
        if (limit <= 0) return;

        form_candidates.clear();
        const float reach_squared = particle_form.radius * particle_form.radius;
        const bool whip = particle_form.style == WhipForm;
        const V3 line = particle_form.point - particle_form.handle;
        const float inverse_length = whip ? 1.f / std::max(0.000001f, line.length_squared()) : 0.f;
        const auto consider = [&](int i) {
            if (voxels[i].formed) return;
            const V3 sample = center(decode(i));
            const float nearest = whip ? std::clamp((sample - particle_form.handle).dot(line) * inverse_length, 0.f, 1.f) : 1.f;
            const float distance = sample.distance_squared_to(particle_form.handle + line * nearest);
            if (distance > reach_squared) return;
            form_candidates.emplace_back(distance, i);
        };
        if (particle_form.material == Water) {
            for (int i : water_positions) consider(i);
        } else {
            for (const Grain &grain : grains) consider(grain.position);
        }
        if (int(form_candidates.size()) > limit) {
            std::nth_element(form_candidates.begin(), form_candidates.begin() + limit, form_candidates.end());
            form_candidates.resize(limit);
        }
        for (const auto &candidate : form_candidates) {
            const int i = candidate.second;
            FormMotion motion;
            motion.anchor = form_anchor(uint32_t(i) + tick * 0x9e3779b9u, particle_form.style);
            motion.style = particle_form.style;
            if (particle_form.material == Water) ensure_liquid_motion(i);
            form_motion.emplace(i, motion);
            set_form_voxel(i, true);
            ++held_form_count;
        }
    }

    void trim_held_form() {
        const int capacity = form_capacity();
        if (held_form_count <= capacity) return;
        form_candidates.clear();
        for (const auto &entry : form_motion) {
            if (entry.second.expires == 0) {
                const float distance = center(decode(entry.first)).distance_squared_to(particle_form.point);
                form_candidates.emplace_back(distance, entry.first);
            }
        }
        std::nth_element(form_candidates.begin(), form_candidates.begin() + capacity, form_candidates.end());
        for (size_t n = capacity; n < form_candidates.size(); ++n) {
            const int i = form_candidates[n].second;
            forget_particle_form(i);
            wake_near(decode(i), awake);
        }
    }

    void prepare_particle_forms() {
        step_whip_pulse();
        if (particle_form.style != NoForm && tick >= next_form_recruit) {
            recruit_form_particles();
            next_form_recruit = tick + std::max(1u, uint32_t(refinement * 2));
        }
        for (auto item = form_motion.begin(); item != form_motion.end();) {
            wake_near(decode(item->first), awake);
            if (item->second.expires != 0 && tick >= item->second.expires) {
                set_form_voxel(item->first, false);
                item = form_motion.erase(item);
            } else {
                ++item;
            }
        }
    }

    bool contact_form_target(FormMotion &motion, Kind material, I3 from, I3 to, V3 velocity, int axis) {
        const bool launched = motion.expires != 0;
        const bool whip = motion.style == WhipForm;
        if (form_targets.empty() || (!launched && motion.style != BladeForm && !whip)) return false;
        const V3 a = center(from), b = center(to);
        const V3 padding = V3(0.48f, 0.48f, 0.48f) * scale;
        const int other_a = (axis + 1) % 3, other_b = (axis + 2) % 3;
        FormTarget *hit_target = nullptr;
        float nearest = std::numeric_limits<float>::infinity();
        int hit_id = 0;
        for (auto &entry : form_targets) {
            FormTarget &target = entry.second;
            const V3 lo = target.lo - padding, hi = target.hi + padding;
            if (a[other_a] < lo[other_a] || a[other_a] > hi[other_a] ||
                    a[other_b] < lo[other_b] || a[other_b] > hi[other_b]) continue;
            const bool positive = b[axis] > a[axis];
            const float start = a[axis], end = b[axis];
            if (positive ? start > lo[axis] || end < lo[axis] : start < hi[axis] || end > hi[axis]) continue;
            const float distance = positive ? lo[axis] - start : start - hi[axis];
            if (distance < nearest || (distance == nearest && entry.first < hit_id)) {
                nearest = distance;
                hit_target = &target;
                hit_id = entry.first;
            }
        }
        if (!hit_target) return false;

        const float speed = std::abs(velocity[axis]);
        const bool lashing = whip && whip_is_lashing();
        const bool striking = !whip || lashing || tick < whip_mobile_until;
        if (striking && speed > (launched ? 1.05f : 0.85f) && tick >= motion.last_contact) {
            const V3 facing = launched ? motion.forward : particle_form.forward;
            const float slice = std::abs(velocity.normalized().dot(facing));
            const float profile = motion.style == BladeForm ? 0.72f + 1.18f * slice : whip ? 0.5f + 0.65f * motion.anchor.x : 1.f;
            const float response = launched ? 0.055f : lashing ? 0.14f : 0.18f;
            hit_target->pending_damage += response * particle_mass(material, 1) * speed * speed * profile;
            motion.last_contact = tick + std::max(1u, uint32_t(refinement * 5));
        }
        return true;
    }

    int nearest_packed_sand(V3 point) const {
        if (!point.is_finite() || !inside(grid(point))) return -1;
        const float reach = std::clamp(0.76f * cast_radius_scale, 0.18f, 1.18f);
        float closest = reach * reach;
        int nearest = -1;
        for (int i = 0; i < int(gravel.size()); ++i) {
            const Grain &grain = gravel[i];
            if (grain.sand_units == 0) continue;
            const V3 location = particle_center(decode(grain.position), grain.extent);
            const float distance = point.distance_squared_to(location);
            if (distance >= closest) continue;
            closest = distance;
            nearest = i;
        }
        return nearest;
    }

    float packed_sand_speed(const Grain &grain, float impulse_scale) const {
        const float effort = std::sqrt(cast_strength * std::clamp(impulse_scale, 0.25f, 6.f));
        const float ideal = 17.5f * effort / (1.f + 0.065f * (grain.extent - 2));
        const float ratio = ideal / maximum_swept_speed;
        // Available impulse follows the footprint's uncompressed mass. A
        // denser body consumes more of it for the same velocity increment.
        return ideal / (std::sqrt(1.f + ratio * ratio) * gravel_density_ratio(grain));
    }

    struct ThrowSample {
        V3 position_sum;
        V3 momentum;
        float mass = 0.f;
        float impulse = 0.f;
        float fastest = 0.f;
        int count = 0;

        void add(Kind kind, const Grain &grain, float speed = 0.f) {
            const float weight = particle_mass(kind, grain);
            // Keep the estimated launch origin inside the occupied cell.
            constexpr float half = scale * 0.499f;
            const V3 bound(half, half, half);
            const V3 position = particle_center(decode(grain.position), grain.extent) + grain.travel.clamp(-bound, bound);
            position_sum += position * weight;
            momentum += grain.velocity * weight;
            mass += weight;
            impulse += weight * speed;
            fastest = std::max(fastest, speed);
            ++count;
        }

        float speed() const { return mass > 0.f ? impulse / mass : 0.f; }

        ThrowInfo result(float speed) const {
            if (count == 0) return {};
            return {position_sum / mass, momentum / mass, speed, mass, count};
        }
    };

    static float charge_speed_increment(float mass, float base_speed, float fastest, float charge) {
        if (mass <= 0.f || charge <= 0.f) return 0.f;
        constexpr float reference_mass = 170.f;
        constexpr float reference_speed = 23.f;
        // Work applied in the actuator frame: M*(u*dv + 0.5*dv*dv),
        // with u the mass-weighted base increment of the selected bodies.
        const float twice_work_per_kg = std::min(mass, reference_mass) / mass
            * reference_speed * reference_speed * std::min(charge, 1.f);
        const float increment = twice_work_per_kg / (base_speed + std::sqrt(base_speed * base_speed + twice_work_per_kg));
        return std::clamp(increment, 0.f, std::max(0.f, maximum_swept_speed - fastest));
    }

    float free_sand_focus_reach() const { return std::clamp(0.82f * cast_radius_scale, 0.18f, 1.18f); }

    static float free_sand_focus_distance_squared(V3 point, V3 position) {
        V3 offset = point - position;
        offset.y *= 1.4f;
        return offset.length_squared();
    }

    struct FreeSandCandidates {
        std::vector<std::pair<float, int>> fine;
        std::vector<std::pair<float, int>> blocks;
    };

    FreeSandCandidates free_sand_candidates(V3 point) const {
        FreeSandCandidates candidates;
        if (!pulling || !point.is_finite() || point.distance_squared_to(field) > 4.f * scale * scale)
            return candidates;
        const float reach = free_sand_focus_reach();
        const float reach_squared = reach * reach;
        const auto select = [&](const std::vector<Grain> &material, std::vector<std::pair<float, int>> &nearest, size_t limit) {
            nearest.reserve(std::min(limit, material.size()));
            for (int i = 0; i < int(material.size()); ++i) {
                const Grain &grain = material[i];
                if (voxels[grain.position].formed) continue;
                const V3 center = particle_center(decode(grain.position), grain.extent);
                const float distance = free_sand_focus_distance_squared(field, center);
                if (distance < reach_squared) nearest.emplace_back(distance, i);
            }
            if (nearest.size() <= limit) return;
            std::nth_element(nearest.begin(), nearest.begin() + limit, nearest.end());
            nearest.resize(limit);
        };
        select(grains, candidates.fine, 900);
        select(gravel, candidates.blocks, 32);
        return candidates;
    }

    float loose_sand_speed(float impulse_scale) const {
        const float effort = std::sqrt(cast_strength * std::clamp(impulse_scale, 0.25f, 6.f));
        const float ideal = 32.f * effort;
        const float ratio = ideal / maximum_swept_speed;
        return ideal * (particle_mass(Water, 1) / particle_mass(Sand, 1)) / std::sqrt(1.f + ratio * ratio);
    }

    float free_sand_block_speed(const Grain &grain, float impulse_scale, float fine_speed) const {
        return std::min(packed_sand_speed(grain, impulse_scale), fine_speed * 0.95f / gravel_density_ratio(grain));
    }

    ThrowSample sample_free_sand(const FreeSandCandidates &candidates, float impulse_scale, float fine_speed) const {
        ThrowSample sample;
        for (const auto &candidate : candidates.fine) sample.add(Sand, grains[candidate.second]);
        sample.impulse = sample.mass * fine_speed;
        if (!candidates.fine.empty()) sample.fastest = fine_speed;
        for (const auto &candidate : candidates.blocks) {
            const Grain &grain = gravel[candidate.second];
            sample.add(Gravel, grain, free_sand_block_speed(grain, impulse_scale, fine_speed));
        }
        return sample;
    }

    void apply_sand_throw(Grain &grain, V3 direction, float speed) {
        grain.velocity += direction * speed;
        const I3 p = decode(grain.position);
        const int edge = grain.extent - 1;
        wake_box(p, p + I3(edge, edge, edge), awake);
    }

    bool form_launchable(int index, const FormMotion &motion, V3 margin) const {
        if (motion.expires != 0) return false;
        const V3 offset = center(decode(index)) - particle_form.point;
        return std::abs(offset.dot(particle_form.right)) <= margin.x &&
            std::abs(offset.dot(particle_form.up)) <= margin.y &&
            std::abs(offset.dot(particle_form.forward)) <= margin.z;
    }

    ThrowSample sample_particle_form(V3 margin) const {
        ThrowSample sample;
        const Kind kind = particle_form.material;
        for (const auto &entry : form_motion) {
            const int i = entry.first;
            if (!form_launchable(i, entry.second, margin)) continue;
            const Grain &grain = kind == Water ? liquid_particle(i) : grains[voxels[i].occupant];
            sample.add(kind, grain);
        }
        return sample;
    }

public:
    ThrowInfo packed_sand_throw(V3 point, float impulse_scale = 1.f) const {
        if (!std::isfinite(impulse_scale)) return {};
        const int id = nearest_packed_sand(point);
        if (id < 0) return {};
        const Grain &grain = gravel[id];
        ThrowSample sample;
        sample.add(Gravel, grain);
        return sample.result(packed_sand_speed(grain, impulse_scale));
    }

    float launch_packed_sand(V3 point, V3 direction, float impulse_scale = 1.f) {
        const float length_squared = direction.length_squared();
        if (!std::isfinite(impulse_scale) || !std::isfinite(length_squared) || length_squared < 0.00001f)
            return 0.f;
        const int id = nearest_packed_sand(point);
        if (id < 0) return 0.f;

        Grain &grain = gravel[id];
        const float speed = packed_sand_speed(grain, impulse_scale);
        apply_sand_throw(grain, direction / std::sqrt(length_squared), speed);
        return speed;
    }

    ThrowInfo free_sand_throw(V3 point, float impulse_scale = 1.f, float charge = 0.f) const {
        if (!std::isfinite(impulse_scale) || !std::isfinite(charge)) return {};
        const FreeSandCandidates candidates = free_sand_candidates(point);
        if (candidates.fine.empty() && candidates.blocks.empty()) return {};
        const float fine_speed = loose_sand_speed(impulse_scale);
        const ThrowSample sample = sample_free_sand(candidates, impulse_scale, fine_speed);
        const float speed = sample.speed();
        ThrowInfo result = sample.result(speed + charge_speed_increment(sample.mass, speed, sample.fastest, charge));
        result.blocks = int(candidates.blocks.size());
        return result;
    }

    int launch_free_sand(V3 point, V3 direction, float impulse_scale = 1.f, float charge = 0.f) {
        const float length_squared = direction.length_squared();
        if (!std::isfinite(impulse_scale) || !std::isfinite(charge) || !std::isfinite(length_squared) || length_squared < 0.00001f)
            return 0;
        const FreeSandCandidates candidates = free_sand_candidates(point);
        direction /= std::sqrt(length_squared);
        const float fine_speed = loose_sand_speed(impulse_scale);
        float boost = 0.f;
        if (charge > 0.f) {
            const ThrowSample sample = sample_free_sand(candidates, impulse_scale, fine_speed);
            boost = charge_speed_increment(sample.mass, sample.speed(), sample.fastest, charge);
        }
        for (const auto &candidate : candidates.fine) apply_sand_throw(grains[candidate.second], direction, fine_speed + boost);
        for (const auto &candidate : candidates.blocks) {
            Grain &grain = gravel[candidate.second];
            apply_sand_throw(grain, direction, free_sand_block_speed(grain, impulse_scale, fine_speed) + boost);
        }
        return int(candidates.fine.size() + candidates.blocks.size());
    }

    void set_particle_form(V3 point, V3 direction, int material, int form, float radius, float strength, V3 origin = V3()) {
        const float norm = direction.length_squared();
        const bool whipping = form == WhipForm && material == Water;
        if ((!whipping && form != ShotForm && form != BladeForm) || (material != Sand && material != Water) ||
                !point.is_finite() || !inside(grid(point)) || !std::isfinite(norm) || norm < 0.00001f ||
                !std::isfinite(radius) || radius <= 0.f || !std::isfinite(strength) || strength <= 0.f) {
            release_held_form();
            return;
        }
        if (particle_form.style != form || particle_form.material != material) {
            release_held_form();
            next_form_recruit = tick;
        }
        const V3 forward = direction / std::sqrt(norm);
        radius = std::clamp(radius, 0.18f, 6.f);
        V3 handle = point;
        V3 axis = forward;
        if (whipping) {
            const float length = std::clamp(0.75f + radius * 0.79f, 0.85f, 4.8f);
            if (!origin.is_finite() || origin == V3() || !inside(grid(origin))) {
                handle = point - forward * std::min(length, std::max(0.65f, radius));
                handle = handle.clamp(V3(scale, scale, scale), V3(nx - 1, ny - 1, nz - 1) * scale);
            } else {
                handle = origin.move_toward(point, 0.32f);
            }
            if (handle.distance_squared_to(point) > 0.0001f) axis = (point - handle).normalized();
            point = handle + (point - handle).limit_length(length);
            if (particle_form.style == WhipForm && (point.distance_squared_to(particle_form.point) > 0.0001f ||
                    handle.distance_squared_to(particle_form.handle) > 0.0001f))
                whip_mobile_until = tick + uint32_t(0.12f / dt);
        }
        const V3 horizontal = V3(0.f, 1.f, 0.f).cross(axis);
        const V3 right = horizontal.length_squared() > 0.0025f ? horizontal.normalized() : V3(1.f, 0.f, 0.f);
        const bool shrinking = radius < particle_form.radius;
        particle_form = {point, axis, right, axis.cross(right).normalized(), form_half_extents(FormStyle(form), radius),
            handle, Kind(material), FormStyle(form), radius, std::clamp(strength, 0.1f, 5.f)};
        if (shrinking) trim_held_form();
    }

    int particle_form_count() const { return held_form_count; }

    V3 particle_form_tip() const {
        const float miss = std::numeric_limits<float>::infinity();
        return particle_form.style == NoForm ? V3(miss, miss, miss) : particle_form.point;
    }

    float particle_form_speed(float impulse_scale = 1.f) const {
        if (particle_form.style == NoForm || particle_form.style == WhipForm || !std::isfinite(impulse_scale)) return 0.f;
        const float base = particle_form.style == BladeForm ? 14.f : 17.f;
        const float desired = base * std::sqrt(particle_form.strength * std::clamp(impulse_scale, 0.25f, 6.f));
        // Equal-sized Water and Sand cells receive the same nominal impulse.
        const float mass_ratio = particle_mass(Water, 1) / particle_mass(particle_form.material, 1);
        const float ratio = desired / maximum_swept_speed;
        return desired * mass_ratio / std::sqrt(1.f + ratio * ratio);
    }

    ThrowInfo particle_form_throw(float impulse_scale = 1.f, float charge = 0.f) const {
        if (held_form_count == 0 || particle_form.style == WhipForm || !std::isfinite(impulse_scale) || !std::isfinite(charge))
            return {};
        const V3 margin = particle_form.half + V3(0.11f, 0.11f, 0.11f);
        const ThrowSample sample = sample_particle_form(margin);
        const float speed = particle_form_speed(impulse_scale);
        return sample.result(speed + charge_speed_increment(sample.mass, speed, speed, charge));
    }

    int launch_particle_form(V3 direction, float impulse_scale = 1.f, float charge = 0.f) {
        const float norm = direction.length_squared();
        if (held_form_count == 0 || !std::isfinite(norm) || norm < 0.00001f || !std::isfinite(impulse_scale) || !std::isfinite(charge))
            return 0;
        direction /= std::sqrt(norm);
        if (particle_form.style == WhipForm) {
            if (tick < whip_pulse.ready) return 0;
            const float kick = 55.f * std::sqrt(std::clamp(impulse_scale, 0.25f, 6.f) * particle_form.strength);
            const V3 stroke = (direction + particle_form.right * 0.16f).normalized() * std::clamp(kick, 9.f, 90.f);
            whip_pulse.velocity[1] += stroke;
            whip_pulse.velocity[2] += stroke * 0.6f;
            whip_pulse.velocity[3] += stroke * 0.25f;
            whip_pulse.moving = true;
            whip_pulse.ready = tick + uint32_t(0.48f / dt);
            return held_form_count;
        }
        const V3 launch_margin = particle_form.half + V3(0.11f, 0.11f, 0.11f);
        float speed = particle_form_speed(impulse_scale);
        if (charge > 0.f) {
            const ThrowSample sample = sample_particle_form(launch_margin);
            speed += charge_speed_increment(sample.mass, speed, speed, charge);
        }
        const V3 impulse = direction * (speed * particle_mass(particle_form.material, 1));
        int launched = 0;
        for (auto &entry : form_motion) {
            FormMotion &motion = entry.second;
            const int i = entry.first;
            if (!form_launchable(i, motion, launch_margin)) continue;
            Grain &grain = particle_form.material == Water ? liquid_particle(i) : grains[voxels[i].occupant];
            grain.velocity += impulse / particle_mass(particle_form.material, grain);
            motion.forward = direction;
            motion.expires = tick + std::max(1u, uint32_t(refinement * 24));
            ++launched;
        }
        if (launched > 0) release_held_form();
        return launched;
    }

    void set_form_target(int id, V3 location, V3 half_extents) {
        if (!location.is_finite() || !half_extents.is_finite() || half_extents.x <= 0.f || half_extents.y <= 0.f ||
                half_extents.z <= 0.f || half_extents.x > 4.f || half_extents.y > 4.f || half_extents.z > 4.f) {
            form_targets.erase(id);
            return;
        }
        if (!inside(grid(location)) || (form_targets.size() >= 32 && form_targets.count(id) == 0)) return;
        FormTarget &target = form_targets[id];
        target.lo = location - half_extents;
        target.hi = location + half_extents;
    }

    TargetContacts take_form_hits() {
        TargetContacts hits;
        for (auto &entry : form_targets) {
            FormTarget &target = entry.second;
            if (target.pending_damage <= 0.f) continue;
            hits.emplace_back(entry.first, target.pending_damage);
            target.pending_damage = 0.f;
        }
        return hits;
    }
