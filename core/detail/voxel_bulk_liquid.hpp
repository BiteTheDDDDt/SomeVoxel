private:
    enum class LiquidHeadModel { Column, Connected };
    enum class LiquidMotionStorage { Sparse, Resident };

    struct BulkLiquidMaterial {
        int bulk_hz;
        LiquidHeadModel head;
        LiquidMotionStorage motion;
        int downhill_faces;
        float drag;
        float rest_speed_squared;
    };

    static constexpr BulkLiquidMaterial water_flow{
        bulk_water_hz, LiquidHeadModel::Connected, LiquidMotionStorage::Sparse, 2, 0.f, 0.f};
    static constexpr BulkLiquidMaterial lava_flow{
        10, LiquidHeadModel::Column, LiquidMotionStorage::Resident, 1, 8.f, 0.0025f};

    static constexpr const BulkLiquidMaterial *bulk_liquid_material(Kind kind) {
        switch (kind) {
            case Water: return &water_flow;
            case Lava: return &lava_flow;
            default: return nullptr;
        }
    }

    bool step_resident_liquid(I3 p, const Voxel &voxel, const BulkLiquidMaterial &material) {
        const Kind kind = voxel.kind;
        Grain &grain = element_grains[element_grain_slot(kind)][voxel.occupant];
        const float reach = granular_material(kind).attraction_range * cast_radius_scale;
        const bool driven = pulling && (field - center(p) - grain.travel).length_squared() < reach * reach;
        // Even a parcel at rest leaves this path only while it has support.
        if (!driven && grain.velocity.length_squared() < material.rest_speed_squared &&
                !can_enter(p + I3(0, -1, 0), kind)) {
            grain.velocity = {};
            grain.travel = {};
            grain.blocked_faces = 0;
            return false;
        }
        const ParticlePath path = advance_particle(grain, p, kind, [&](I3 candidate, int *contact = nullptr) {
            return can_enter(candidate, kind, contact);
        }, nullptr, -grain.velocity * material.drag);
        commit_particle_path(p, path);
        return true;
    }

    void step_bulk_liquid(I3 p, int index, Kind kind, const BulkLiquidMaterial &material) {
        const bool scheduled = material.bulk_hz == bulk_water_hz ? water_tick
            : (uint64_t(tick) * material.bulk_hz) % simulation_hz < material.bulk_hz;
        if (!scheduled) {
            next_awake.set_for_cell(p);
            return;
        }

        const I3 down(0, -1, 0);
        if (can_enter(p + down, kind)) {
            move(p, p + down);
            return;
        }
        const auto dirs = cardinal();
        const int start = hash(uint32_t(index) ^ (tick * 0x9e3779b9u)) & 3u;
        I3 level = p;
        for (int i = 0; i < 4; ++i) {
            const I3 side = p + dirs[(start + i) & 3];
            if (!can_enter(side, kind)) continue;
            if (can_enter(side + down, kind)) {
                if (material.downhill_faces == 2) move_across_two_faces(p, side, side + down);
                else move(p, side);
                return;
            }
            if (level == p || kind_at(side + down) == Empty) level = side;
        }
        if (level == p) return;

        // A level surface needs a pressure difference to keep moving.
        const bool pressurized = material.head == LiquidHeadModel::Connected
            ? liquid_head.get(p) > p.y + 1 : kind_at(p - down) == kind;
        if (pressurized) move(p, level);
    }

    void step_liquid(I3 p, int index, const Voxel &voxel, const BulkLiquidMaterial &material) {
        const bool integrated = material.motion == LiquidMotionStorage::Sparse
            ? step_liquid_motion(p, index) : step_resident_liquid(p, voxel, material);
        if (!integrated) step_bulk_liquid(p, index, voxel.kind, material);
    }
