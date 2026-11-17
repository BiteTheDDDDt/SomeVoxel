private:
    static constexpr int gravel_water_search_budget = 4096;

    struct GravelWaterGroups {
        std::array<int, 8 * 8> parents, demand, capacity{};
        int unfilled;

        explicit GravelWaterGroups(int count) : unfilled(count) {
            for (int i = 0; i < count; ++i) {
                parents[i] = i;
                demand[i] = 1;
            }
        }

        int root(int group) {
            while (parents[group] != group) {
                parents[group] = parents[parents[group]];
                group = parents[group];
            }
            return group;
        }

        void unite(int a, int b) {
            a = root(a);
            b = root(b);
            if (a == b) return;
            unfilled -= (capacity[a] < demand[a]) + (capacity[b] < demand[b]);
            parents[b] = a;
            capacity[a] += capacity[b];
            demand[a] += demand[b];
            unfilled += capacity[a] < demand[a];
        }

        void add_capacity(int group) {
            group = root(group);
            if (++capacity[group] == demand[group]) --unfilled;
        }
    };

    static bool in_gravel_box(I3 p, I3 anchor, int extent) {
        const I3 delta = p - anchor;
        return delta.x >= 0 && delta.x < extent && delta.y >= 0 && delta.y < extent &&
            delta.z >= 0 && delta.z < extent;
    }

    bool gravel_can_enter_water(const Grain &grain, int obstacle) const {
        return obstacle >= 0 && voxels[obstacle].kind == Water &&
            sand_density * gravel_density_ratio(grain) > granular_material(Water).density;
    }

    bool plan_gravel_liquid_move(const Grain &grain, I3 from, I3 target, int axis, int direction,
            int &obstacle, GravelLiquidMove &plan) const {
        const int extent = grain.extent;
        const I3 occupied = decode(grain.position);
        I3 face = target;
        if (direction > 0) face[axis] += extent - 1;
        const int u = (axis + 1) % 3, v = (axis + 2) % 3;

        plan.count = 0;
        plan.axis = axis;
        plan.direction = direction;
        std::array<int, 8 * 8> inlet;
        inlet.fill(-1);
        for (int a = 0; a < extent; ++a) {
            for (int b = 0; b < extent; ++b) {
                I3 cell = face;
                cell[u] += a;
                cell[v] += b;
                const int i = flat(cell);
                if (body_at(i)) {
                    obstacle = -1;
                    return false;
                }
                const Voxel &voxel = voxels[i];
                if (voxel.kind == Empty || in_gravel_box(cell, occupied, extent)) continue;
                if (voxel.kind != Water || voxel.stamp == voxel_stamp) {
                    obstacle = i;
                    return false;
                }
                inlet[a * extent + b] = plan.count;
                plan.sources[plan.count++] = i;
            }
        }
        if (plan.count == 0) return false;
        GravelWaterGroups groups(plan.count);
        // Water touching along the compressed face can already flow between
        // those cells. Dry gaps on that face cannot connect separate pockets.
        for (int a = 0; a < extent; ++a) {
            for (int b = 0; b < extent; ++b) {
                const int i = a * extent + b;
                if (inlet[i] < 0) continue;
                if (a > 0 && inlet[i - extent] >= 0) groups.unite(inlet[i], inlet[i - extent]);
                if (b > 0 && inlet[i - 1] >= 0) groups.unite(inlet[i], inlet[i - 1]);
            }
        }

        // Capacity must be connected around the body. Its trailing volume
        // becomes available after moving, but a sealing piston cannot push
        // water through itself into that volume.
        const int halo = std::max(4, extent / 2);
        const I3 lo(std::max(0, std::min(from.x, target.x) - halo), 0,
            std::max(0, std::min(from.z, target.z) - halo));
        const I3 hi(std::min(nx, std::max(from.x, target.x) + extent + halo), ny,
            std::min(nz, std::max(from.z, target.z) + extent + halo));
        const I3 span = hi - lo;
        plan.seen.assign(span.x * span.y * span.z, 0);
        plan.frontier.clear();
        plan.outlets.clear();
        const auto available = [&](I3 cell, int i) {
            return voxels[i].kind == Empty || in_gravel_box(cell, occupied, extent);
        };
        const auto label_at = [&](I3 cell) -> uint8_t & {
            const I3 local = cell - lo;
            return plan.seen[local.x + span.x * (local.z + span.z * local.y)];
        };
        const auto offer = [&](I3 cell, int group) {
            if (cell.x < lo.x || cell.x >= hi.x || cell.y < lo.y || cell.y >= hi.y ||
                    cell.z < lo.z || cell.z >= hi.z || in_gravel_box(cell, target, extent)) return;
            uint8_t &label = label_at(cell);
            if (label != 0) {
                groups.unite(group, label - 1);
                return;
            }
            if (plan.frontier.size() >= gravel_water_search_budget) return;
            const int i = flat(cell);
            if (body_at(i)) return;
            const bool empty = available(cell, i);
            if (!empty && voxels[i].kind != Water) return;
            label = uint8_t(group + 1);
            plan.frontier.push_back(i);
            if (empty) {
                plan.outlets.push_back(i);
                groups.add_capacity(group);
            }
        };
        const auto options = structural_neighbors();
        for (int n = 0; n < plan.count; ++n)
            for (I3 offset : options) offer(decode(plan.sources[n]) + offset, n);
        std::array<bool, 8 * 8> has_portal{};
        for (int entry : plan.frontier) has_portal[groups.root(label_at(decode(entry)) - 1)] = true;
        for (int n = 0; n < plan.count; ++n)
            if (!has_portal[groups.root(n)]) return false;
        for (size_t cursor = 0; cursor < plan.frontier.size() && groups.unfilled > 0; ++cursor) {
            const I3 cell = decode(plan.frontier[cursor]);
            const int group = label_at(cell) - 1;
            for (I3 offset : options) offer(cell + offset, group);
        }
        if (groups.unfilled > 0) return false;

        std::array<int, 8 * 8> pending, next;
        pending.fill(-1);
        for (int n = plan.count - 1; n >= 0; --n) {
            const int group = groups.root(n);
            next[n] = pending[group];
            pending[group] = n;
        }
        int placed = 0;
        for (int exit : plan.outlets) {
            int &n = pending[groups.root(label_at(decode(exit)) - 1)];
            if (n < 0) continue;
            plan.exits[n] = exit;
            n = next[n];
            if (++placed == plan.count) break;
        }
        return placed == plan.count;
    }

    void commit_gravel_liquid_move(Grain &grain, const GravelLiquidMove &plan) {
        const float mass = particle_mass(Gravel, grain);
        const float parcel_mass = particle_mass(Water, 1);
        const int axis = plan.axis;
        float momentum = mass * grain.velocity[axis];
        for (int n = 0; n < plan.count; ++n) {
            const int i = plan.sources[n];
            if (voxels[i].liquid_moving) momentum += parcel_mass * liquid_particle(i).velocity[axis];
        }
        const float speed = momentum / (mass + parcel_mass * plan.count);
        grain.velocity[axis] = speed;
        for (int n = 0; n < plan.count; ++n) {
            const int i = plan.sources[n];
            const I3 from = decode(i), to = decode(plan.exits[n]);
            I3 difference = to - from;
            const bool follows_impact = difference[axis] == plan.direction && speed * plan.direction > 0.f;
            difference[axis] = 0;
            // Pressure can turn at a wall and vent sideways or upward.
            // Only an adjacent outlet on the contact axis inherits its speed.
            if (follows_impact && difference == I3() &&
                    (voxels[i].liquid_moving || (std::abs(speed) > 1.7f && liquid_motion_has_room())))
                ensure_liquid_motion(i).velocity[axis] = speed;
            move(from, to);
        }
    }
