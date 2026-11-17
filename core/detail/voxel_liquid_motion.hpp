private:
    // Only controlled or still-moving liquid needs velocity. The bulk solver
    // continues to use compact occupancy and connected water head.
    static constexpr size_t liquid_motion_budget = 1600;
    std::unordered_map<int, Grain> liquid_motion;

    bool liquid_motion_has_room() const { return liquid_motion.size() < liquid_motion_budget; }

    Grain &ensure_liquid_motion(int index) {
        voxels[index].liquid_moving = true;
        sync_bulk_water(index);
        return liquid_motion.try_emplace(index, Grain{index, index, tick}).first->second;
    }

    Grain &liquid_particle(int index) { return liquid_motion.at(index); }
    const Grain &liquid_particle(int index) const { return liquid_motion.at(index); }

    void remove_liquid_motion(int index) {
        voxels[index].liquid_moving = false;
        sync_bulk_water(index);
        liquid_motion.erase(index);
    }

    void move_liquid_motion(int a, int b) {
        // Occupancy has already swapped. Extract both before changing keys, so
        // even a two-way exchange keeps the records with their own water cells.
        auto first = liquid_motion.extract(a);
        auto second = liquid_motion.extract(b);
        const auto place = [&](auto &entry, int target) {
            if (!entry) return;
            int ordinary_move = 0;
            record_grain_move(entry.mapped(), target, voxels[target].formed ? moved_water_form : ordinary_move);
            entry.key() = target;
            liquid_motion.insert(std::move(entry));
        };
        place(first, b);
        place(second, a);
    }

    bool step_liquid_motion(I3 p, int index) {
        const bool tracked = voxels[index].liquid_moving;
        if (!tracked && ((!pulling_water && !winding) || !liquid_motion_has_room())) return false;
        const V3 position = center(p);
        const V3 offset = pulling_water ? water_field - position : V3();
        const float reach = water_pull_radius * cast_radius_scale;
        const bool attracted = pulling_water && offset.length_squared() < reach * reach;
        const float wind = winding && kind_at(p + I3(0, 1, 0)) == Empty ? wind_pressure(position) : 0.f;
        const bool driven = attracted || wind > 0.02f;
        // The pressure solver owns the bulk. Only give a parcel its own kinetic
        // state where an exposed face allows it to respond to the field.
        const auto can_follow_field = [&] {
            const V3 direction = (attracted ? offset : V3()) + wind_direction * wind;
            for (I3 face : structural_neighbors()) {
                if (direction.dot(V3(face)) > scale * 0.10f && can_enter(p + face, Water)) return true;
            }
            return false;
        };
        if (!tracked) {
            if (!driven || !can_follow_field()) return false;
            ensure_liquid_motion(index);
        }

        Grain &grain = liquid_particle(index);
        const bool supported = !can_enter(p + I3(0, -1, 0), Water);
        if (supported && grain.velocity.length_squared() < 0.0225f && tick - grain.moved >= 8u &&
                (!driven || !can_follow_field())) {
            remove_liquid_motion(index);
            return false;
        }

        const V3 acceleration = attracted ? pull_acceleration(offset - grain.travel, grain.velocity, reach, 1.f, 74.f, 7.f)
            : grain.velocity * (supported ? -6.5f : -1.5f);
        const ParticlePath path = advance_particle(grain, p, Water, [&](I3 candidate, int *contact = nullptr) {
            return can_enter(candidate, Water, contact);
        }, nullptr, acceleration);
        commit_particle_path(p, path);
        next_awake[chunk_id({path.target.x / block, path.target.y / block, path.target.z / block})] = true;
        return true;
    }
