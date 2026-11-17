private:
    struct FirePacket {
        V3 position;
        V3 previous;
        V3 velocity;
        float age;
        float lifetime;
        float width;
        float temperature;
    };

    static constexpr int fire_jet_capacity = 128;
    std::vector<FirePacket> fire_packets;
    std::vector<std::pair<int, float>> fire_heat_contacts;
    V3 fire_jet_origin;
    V3 fire_jet_direction;
    float fire_jet_radius = 0.f;
    float fire_jet_strength = 0.f;
    float fire_jet_emission = 0.f;
    uint32_t fire_jet_serial = 0;

    void reset_fire_jet() {
        fire_packets.clear();
        fire_jet_strength = 0.f;
        fire_jet_emission = 0.f;
        fire_jet_serial = 0;
    }

    void emit_fire_packet() {
        if (int(fire_packets.size()) >= fire_jet_capacity) return;
        const auto random = [](uint32_t seed) { return float(hash(seed) & 0xffffu) / 65535.f; };
        const uint32_t serial = ++fire_jet_serial;
        const float spin = 6.2831853f * random(serial);
        const float spread = std::sqrt(random(serial ^ 0x4d1e77f3u)) * fire_jet_radius;
        const V3 up = std::abs(fire_jet_direction.y) > 0.95f ? V3(1, 0, 0) : V3(0, 1, 0);
        const V3 right = fire_jet_direction.cross(up).normalized();
        const V3 radial = right * std::cos(spin) + right.cross(fire_jet_direction) * std::sin(spin);
        const V3 direction = (fire_jet_direction + radial * (spread * 0.28f)).normalized();
        const float variation = random(serial ^ 0xab3cf1d7u);
        const float speed = (5.5f + 1.6f * variation) * std::pow(fire_jet_strength, 0.23f);
        const float lifetime = 0.37f + 0.17f * variation;
        const float width = 0.042f + fire_jet_radius * (0.10f + 0.12f * variation);
        const float temperature = 230.f * std::sqrt(fire_jet_strength);
        fire_packets.push_back({fire_jet_origin, fire_jet_origin, direction * speed, 0.f, lifetime, width, temperature});
    }

    void deposit_fire_packet(const FirePacket &packet, V3 contact) {
        const float radius = std::clamp(0.075f + 0.63f * packet.width, scale, 0.22f);
        // Budget in voxel heat units. A wider packet carries more heat, but
        // additional exposed faces share it instead of duplicating it.
        const float reference_faces = 3.14159265f * radius * radius / (scale * scale);
        const float energy = packet.temperature * (1.f - 0.54f * packet.age / packet.lifetime) * reference_faces;
        float total_weight = 0.f;
        fire_heat_contacts.clear();
        for_reaction_sphere(contact, radius, [&](int i, I3 p) {
            const Kind kind = voxels[i].kind;
            if (!carries_heat(kind)) return;
            const V3 offset = center(p) - packet.position;
            const float length = offset.length();
            if (length > 0.00001f) {
                const V3 hit = ray_hit(packet.position, offset / length, length, true);
                if (hit.is_finite() && grid(hit) != p) return;
            }
            const float weight = 1.f - 0.55f * center(p).distance_squared_to(contact) / (radius * radius);
            total_weight += weight;
            fire_heat_contacts.emplace_back(i, weight);
        });
        const float available = energy / std::max(reference_faces, total_weight);
        for (const auto &[i, weight] : fire_heat_contacts) {
            auto &cell = reaction_cells[i];
            cell.temperature = std::min(820.f, cell.temperature + available * weight / heat_capacity(voxels[i].kind));
        }
    }

    void step_fire_jet() {
        if (fire_jet_strength > 0.f) {
            fire_jet_emission += (34.f + 43.f * std::sqrt(fire_jet_strength)) * dt;
            while (fire_jet_emission >= 1.f) {
                emit_fire_packet();
                fire_jet_emission -= 1.f;
            }
        }
        for (size_t i = 0; i < fire_packets.size();) {
            auto &packet = fire_packets[i];
            packet.age += dt;
            packet.velocity += (V3(0, 4.2f, 0) + wind_direction * (10.f * wind_pressure(packet.position))) * dt;
            const V3 movement = packet.velocity * dt;
            const float length = movement.length();
            const Kind phase = kind_at(grid(packet.position));
            const V3 contact = phase == Water ? packet.position : ray_hit(packet.position, movement, length, true);
            if (contact.is_finite() || packet.age >= packet.lifetime || !inside(grid(packet.position + movement))) {
                if (contact.is_finite()) deposit_fire_packet(packet, contact);
                packet = fire_packets.back();
                fire_packets.pop_back();
                continue;
            }
            packet.previous = packet.position;
            packet.position += movement;
            ++i;
        }
    }

public:
    void set_fire_jet(V3 origin, V3 direction, float radius, float strength, bool active) {
        const float length_squared = direction.length_squared();
        if (!active || !origin.is_finite() || !std::isfinite(length_squared) || length_squared < 0.00001f ||
                !std::isfinite(radius) || radius <= 0.f || !std::isfinite(strength) || strength <= 0.f ||
                !inside(grid(origin))) {
            fire_jet_strength = 0.f;
            fire_jet_emission = 0.f;
            return;
        }
        fire_jet_origin = origin;
        fire_jet_direction = direction / std::sqrt(length_squared);
        fire_jet_radius = std::clamp(radius, 0.045f, 1.25f);
        fire_jet_strength = std::clamp(strength, 0.1f, 5.f);
    }

    int fire_jet_count() const { return int(fire_packets.size()); }
    int fire_jet_buffer_capacity() const { return fire_jet_capacity; }
