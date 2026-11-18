#[compute]
#version 450
layout(local_size_x = 128) in;

layout(set = 0, binding = 0, std430) restrict buffer Directory { int directory[]; };
layout(set = 0, binding = 1, std430) restrict buffer Materials { int atlas[]; };
layout(set = 0, binding = 2, std430) restrict buffer Origins { ivec4 pages[]; };
layout(set = 0, binding = 3, std430) restrict buffer Residency { int writable[]; };
layout(set = 0, binding = 4, std430) restrict buffer Support { uint flags[]; };
layout(set = 0, binding = 5, std430) restrict buffer Counters {
    int resident;
    int pending_active;
    int active_count;
    int capacity_misses;
    int edit_failed;
    int surface_count;
    int blocked;
    uint completed_tick;
    uint sweep_pending;
    uint sweep_count;
    uint sweep_pairs;
};
layout(set = 0, binding = 7, std430) restrict buffer Commands { ivec4 cells[]; };
layout(set = 0, binding = 8, std430) restrict buffer Travel { uint travel[]; };
layout(set = 0, binding = 9, std430) restrict buffer SweepPages { uint sweep_tick[]; };
layout(set = 0, binding = 10, std430) readonly buffer PressureRows { uint pressure_rows[]; };
layout(set = 0, binding = 11, std430) restrict buffer SweepList { int sweep_pages[]; };
layout(push_constant, std430) uniform Params {
    ivec4 shape_capacity;
    ivec4 bounds_mode;
    ivec4 axis_parity_count;
    vec4 focus;
    vec4 field;
    ivec4 region_origin;
    ivec4 region_size_limit;
    ivec4 body;
};

bool inside(ivec3 p) {
    return all(greaterThanEqual(p, ivec3(0))) && all(lessThan(p, bounds_mode.xyz));
}

bool inside_body(ivec3 p) {
    int half_width = body.w & 65535;
    int height = body.w >> 16;
    return p.y >= body.y && p.y < body.y + height
        && all(lessThan(abs(p.xz - body.xz), ivec2(half_width)));
}

vec3 body_acceleration(ivec3 p) {
    ivec2 offset = p.xz - body.xz;
    if (abs(offset.x) >= abs(offset.y)) return vec3(offset.x >= 0 ? 0.8 : -0.8, 0.0, 0.0);
    return vec3(0.0, 0.0, offset.y >= 0 ? 0.8 : -0.8);
}

int tile_key(ivec3 tile) {
    return tile.x + shape_capacity.x * (tile.z + shape_capacity.z * tile.y);
}

int address(ivec3 p) {
    if (!inside(p)) return -2;
    int slot = directory[tile_key(p / 8)];
    if (slot < 0) return slot;
    ivec3 local = p & 7;
    return slot * 512 + local.x + 8 * (local.z + 8 * local.y);
}

int material(int location) {
    return location < 0 ? -1 - location : atlas[location];
}

void record_support(uint invocation) {
    if (invocation >= uint(active_count) * 512u) return;
    ivec4 page = pages[invocation >> 9];
    int local = int(invocation & 511u);
    ivec3 p = page.xyz + ivec3(local & 7, local >> 6, (local >> 3) & 7);
    int location = page.w * 512 + local;
    int below, above;
    if (local >= 64 && local < 448 && p.y + 1 < bounds_mode.y && p.x < bounds_mode.x && p.z < bounds_mode.z) {
        below = atlas[location - 64];
        above = atlas[location + 64];
    } else {
        below = material(address(p - ivec3(0, 1, 0)));
        above = material(address(p + ivec3(0, 1, 0)));
    }
    bool pressurized = above == 3;
    if (!pressurized && axis_parity_count.z != 0 && atlas[location] == 3)
        pressurized = ((pressure_rows[location >> 3] >> (8 + (location & 7))) & 1u) != 0u;
    flags[location] = (flags[location] & ~3u) | uint(below == 0) | (uint(pressurized) << 1);
}

int speed(uint state, int axis) {
    return bitfieldExtract(int(state), 2 + axis * 10, 10);
}

int budget(uint state, int axis) {
    return int(bitfieldExtract(state, axis * 10 + 5, 5));
}

bool has_whole_travel(uint state) {
    const uint whole_bits = 31u << 5 | 31u << 15 | 31u << 25;
    return (state & whole_bits) != 0u;
}

void mark_sweep(int slot) {
    atomicMax(sweep_tick[slot], uint(region_origin.z));
}

ivec3 local_cell(int index) {
    return ivec3(index & 7, index >> 6, (index >> 3) & 7);
}

bool can_enter(int kind, int other) {
    return other == 0 || (kind == 2 && other == 3);
}

// Velocity and residual travel use thirty-seconds of a cell. The force runs
// once per fixed tick; each committed face crossing consumes one budget unit.
void integrate_field(uint invocation) {
    if (invocation >= uint(active_count) * 512u) return;
    ivec4 page = pages[invocation >> 9];
    int local = int(invocation & 511u);
    int location = page.w * 512 + local;
    int kind = atlas[location];
    if (kind != 2 && kind != 3) return;
    uint previous = flags[location];
    uint previous_travel = travel[location];
    ivec3 p = page.xyz + local_cell(local);
    vec3 delta = focus.xyz - (vec3(p) + vec3(0.5));
    float distance_sq = dot(delta, delta);
    bool affected = kind == int(field.z) && field.x > 0.0 && distance_sq < focus.w;
    bool body_contact = body.w != 0 && inside_body(p);
    if (!affected && !body_contact && (previous >> 2) == 0u && kind == 3) return;
    int below = local >= 64 && inside(p) ? atlas[location - 64] : material(address(p - ivec3(0, 1, 0)));
    bool supported = !can_enter(kind, below);
    vec3 velocity = vec3(speed(previous, 0), speed(previous, 1), speed(previous, 2)) / 32.0;
    float dt = intBitsToFloat(region_origin.y);
    vec3 acceleration = vec3(0.0, -intBitsToFloat(region_origin.x), 0.0);
    float damping = kind == 3 ? (supported ? 6.5 : 1.5) : 0.0;
    if (body_contact) acceleration += body_acceleration(p);
    if (affected) {
        float response = clamp(2.8 * (1.0 - distance_sq / focus.w), 0.0, 1.0) * field.x;
        float stiffness = kind == 3 ? 74.0 : 94.0;
        acceleration += delta * (stiffness * response * dt * dt);
        damping = (kind == 3 ? 7.0 : 13.0) * sqrt(response);
    }
    if (supported && velocity.y <= 0.0 && acceleration.y <= 0.0) {
        if (-velocity.y < intBitsToFloat(region_size_limit.y)) velocity.y = 0.0;
        acceleration.y = 0.0;
        if (kind == 2 && !affected) {
            ivec3 slope = ivec3(0);
            if (region_origin.w != 0) {
                const ivec3 sides[4] = ivec3[4](ivec3(1, 0, 0), ivec3(0, 0, 1),
                    ivec3(-1, 0, 0), ivec3(0, 0, -1));
                uint key = uint(p.x + bounds_mode.x * (p.z + bounds_mode.z * p.y));
                key = (key ^ (key >> 16)) * 0x7feb352du;
                key = (key ^ (key >> 15)) * 0x846ca68bu;
                int first = int((key ^ (key >> 16)) & 3u);
                for (int i = 0; i < 4; i++) {
                    ivec3 side = sides[(first + i) & 3];
                    if (can_enter(kind, material(address(p + side))) &&
                            can_enter(kind, material(address(p + side - ivec3(0, 1, 0))))) {
                        slope = side;
                        break;
                    }
                }
            }
            if (slope == ivec3(0)) velocity.xz *= intBitsToFloat(region_size_limit.x);
            else acceleration += vec3(slope) * (1.55 * intBitsToFloat(region_origin.x));
        }
    }
    if ((previous >> 2) == 0u && all(equal(acceleration, vec3(0.0)))) return;
    velocity += acceleration - velocity * (damping * dt);
    uint state = previous & 3u;
    uint displacement = 0u;
    for (int axis = 0; axis < 3; axis++) {
        int last = speed(previous, axis);
        int next = clamp(int(round(velocity[axis] * 32.0)), -511, 511);
        if (next == 0) continue;
        ivec3 neighbor = p;
        neighbor[axis] += sign(next);
        int other = material(address(neighbor));
        if (!can_enter(kind, other) && !(kind == 2 && other == 2)) continue;
        int fraction = sign(next) == sign(last) ? int(bitfieldExtract(previous_travel, axis * 10, 5)) : 0;
        fraction += abs(next);
        state = bitfieldInsert(state, uint(next), 2 + axis * 10, 10);
        displacement = bitfieldInsert(displacement, uint(fraction), axis * 10, 10);
    }
    flags[location] = state;
    travel[location] = displacement;
    if (budget(displacement, 0) >= 2 || budget(displacement, 1) >= 2 || budget(displacement, 2) >= 2)
        mark_sweep(page.w);
}

void add_impulse(uint invocation) {
    if (invocation >= uint(active_count) * 512u) return;
    ivec4 page = pages[invocation >> 9];
    int index = int(invocation & 511u);
    int location = page.w * 512 + index;
    if (atlas[location] != axis_parity_count.z) return;
    vec3 delta = vec3(page.xyz + local_cell(index)) + vec3(0.5) - focus.xyz;
    float distance_sq = dot(delta, delta);
    if (distance_sq >= focus.w) return;
    float weight = 1.0 - sqrt(distance_sq) * field.w;
    uint state = flags[location];
    for (int axis = 0; axis < 3; axis++) {
        int last = speed(state, axis);
        int velocity = clamp(last + int(round(field[axis] * weight * 32.0)), -511, 511);
        if (sign(velocity) != sign(last)) travel[location] = bitfieldInsert(travel[location], 0u, axis * 10, 10);
        state = bitfieldInsert(state, uint(velocity), 2 + axis * 10, 10);
    }
    flags[location] = state;
}

void extract_surface(uint invocation) {
    ivec3 size = region_size_limit.xyz;
    if (invocation >= uint(size.x * size.y * size.z)) return;
    int index = int(invocation);
    ivec3 p = region_origin.xyz + ivec3(index % size.x, index / (size.x * size.z), (index / size.x) % size.z);
    int kind = material(address(p));
    if (kind == 0) return;
    int faces = 0;
    for (int face = 0; face < 6; face++) {
        ivec3 neighbor = p;
        neighbor[face / 2] += (face & 1) == 0 ? -1 : 1;
        if (!inside(neighbor) || material(address(neighbor)) == 0) faces |= 1 << face;
    }
    if (faces == 0) return;
    int output_index = atomicAdd(surface_count, 1);
    if (output_index < region_size_limit.w) cells[output_index] = ivec4(p, kind | (faces << 8));
}

// Equal-mass grains share only their closing normal momentum. The parcels
// retain their tangential velocity and stay on their own side of the face.
void close_sand_contact(int a, int b, int axis) {
    int left = speed(flags[a], axis), right = speed(flags[b], axis);
    if (left <= right || !((left > 0 && budget(travel[a], axis) != 0)
            || (right < 0 && budget(travel[b], axis) != 0))) return;
    int sum = left + right;
    int normal_velocity = sum >> 1;
    int shift = 2 + axis * 10;
    flags[a] = bitfieldInsert(flags[a], uint(normal_velocity), shift, 10);
    flags[b] = bitfieldInsert(flags[b], uint(sum - normal_velocity), shift, 10);
    travel[a] = bitfieldInsert(travel[a], 0u, axis * 10, 10);
    travel[b] = bitfieldInsert(travel[b], 0u, axis * 10, 10);
}

void exchange_pairs(uint invocation) {
    if (invocation >= uint(active_count) * 256u) return;
    ivec4 page = pages[invocation >> 8];
    int cell = int(invocation & 255u);
    int axis = axis_parity_count.x;
    int parity = axis_parity_count.y;
    ivec3 local = ivec3(cell & 7, cell >> 6, (cell >> 3) & 7);
    if (axis == 0) local = ivec3((cell & 3) * 2 + parity, cell >> 5, (cell >> 2) & 7);
    else if (axis == 1) local.y = local.y * 2 + parity;
    else local = ivec3(cell & 7, cell >> 5, ((cell >> 3) & 3) * 2 + parity);
    ivec3 p = page.xyz + local;
    if (!inside(p)) return;
    ivec3 q = p;
    q[axis] += 1;
    int a = page.w * 512 + local.x + 8 * (local.z + 8 * local.y);
    if (q[axis] >= bounds_mode[axis]) return;
    int b;
    if (local[axis] < 7) {
        b = a + (axis == 0 ? 1 : (axis == 1 ? 64 : 8));
    } else {
        b = address(q);
        if (b < 0 || writable[b / 512] == 0) return;
    }
    int left = atlas[a], right = atlas[b];
    if (left == right) {
        if (left == 2 && axis_parity_count.w != 0) close_sand_contact(a, b, axis);
        return;
    }
    if (body.w != 0) {
        bool a_inside = inside_body(p), b_inside = inside_body(q);
        // Existing grains can move out, but moving grains cannot enter.
        if (a_inside != b_inside && (a_inside ? right != 0 : left != 0)) return;
    }
    uint af = flags[a], bf = flags[b];
    uint at = travel[a], bt = travel[b];
    bool swap = false;
    if (axis_parity_count.w != 0 && left >= 2 && (right == 0 || (left == 2 && right == 3)) && (af >> 2) != 0u) {
        swap = budget(at, axis) > 0 && speed(af, axis) > 0;
    } else if (axis_parity_count.w != 0 && right >= 2 && (left == 0 || (right == 2 && left == 3)) && (bf >> 2) != 0u) {
        swap = budget(bt, axis) > 0 && speed(bf, axis) < 0;
    } else if (axis == 1) {
        swap = left == 0 && right == 3;
        if (axis_parity_count.w < 2) swap = swap || (right == 2 && (left == 0 || left == 3));
    }
    else if (axis_parity_count.w == 0 && region_origin.w != 0 && left == 2 && right == 0)
        swap = (af & 1u) == 0u && (bf & 1u) != 0u;
    else if (axis_parity_count.w == 0 && region_origin.w != 0 && right == 2 && left == 0)
        swap = (bf & 1u) == 0u && (af & 1u) != 0u;
    else if (left == 3 && right == 0) {
        swap = (af & 1u) == 0u && ((bf & 1u) != 0u || ((af & 2u) != 0u && (bf & 2u) == 0u));
    } else if (right == 3 && left == 0) {
        swap = (bf & 1u) == 0u && ((af & 1u) != 0u || ((bf & 2u) != 0u && (af & 2u) == 0u));
    }
    if (swap) {
        atlas[a] = right;
        atlas[b] = left;
        if (axis_parity_count.w != 0) {
            uint bit = 1u << (axis * 10 + 5);
            if (speed(bf, axis) < 0 && budget(bt, axis) > 0) bt -= bit;
            if (speed(af, axis) > 0 && budget(at, axis) > 0) at -= bit;
            flags[a] = (af & 3u) | (bf & ~3u);
            flags[b] = (bf & 3u) | (af & ~3u);
            travel[a] = bt;
            travel[b] = at;
            if (a / 512 != b / 512) {
                if (has_whole_travel(bt)) mark_sweep(a / 512);
                if (has_whole_travel(at)) mark_sweep(b / 512);
            }
        }
    }
}

// Each invocation owns one full line across a disjoint pair of neighboring
// pages. A second checkerboard phase covers the intervening page boundaries.
int line_address(int offset, int first, int second, int stride) {
    return offset < 8 ? first + offset * stride : second + (offset - 8) * stride;
}

// Gather once per checkerboard phase. Its pairs have disjoint footprints.
void collect_sweep_pages(uint invocation) {
    if (invocation >= uint(active_count)) return;
    ivec4 page = pages[invocation];
    int axis = axis_parity_count.x;
    if (((page[axis] >> 3) & 1) != axis_parity_count.y) return;
    uint stamp = uint(region_origin.z);
    if (sweep_tick[page.w] != stamp) {
        ivec3 neighbor = page.xyz;
        neighbor[axis] += 8;
        if (!inside(neighbor)) return;
        int slot = directory[tile_key(neighbor / 8)];
        if (slot < 0 || writable[slot] == 0 || sweep_tick[slot] != stamp) return;
    }
    sweep_pages[atomicAdd(sweep_pending, 1u)] = int(invocation);
}

void sweep_lines(uint invocation) {
    bool compact = axis_parity_count.z != 0;
    uint size = compact ? sweep_count : uint(active_count);
    if (invocation >= size * 64u) return;
    uint index = invocation >> 6;
    ivec4 page = pages[compact ? sweep_pages[index] : int(index)];
    int axis = axis_parity_count.x;
    if (((page[axis] >> 3) & 1) != axis_parity_count.y) return;
    int lane = int(invocation & 63u);
    ivec3 p = page.xyz;
    p[(axis + 1) % 3] += lane & 7;
    p[(axis + 2) % 3] += lane >> 3;
    if (!inside(p)) return;
    ivec3 q = p;
    q[axis] += 8;
    int first = address(p);
    int second = address(q);
    bool paired = second >= 0 && writable[second / 512] != 0;
    uint stamp = uint(region_origin.z);
    if (!compact && atomicAdd(sweep_tick[page.w], 0u) != stamp
            && (!paired || atomicAdd(sweep_tick[second / 512], 0u) != stamp)) return;
    int width = min(bounds_mode[axis] - p[axis], paired ? 16 : 8);
    int strides[3] = int[3](1, 64, 8);
    int stride = strides[axis];
    for (int sign_ = -1; sign_ <= 1; sign_ += 2) {
        for (int index = 0; index < width; index++) {
            int offset = sign_ > 0 ? width - 1 - index : index;
            int from = line_address(offset, first, second, stride);
            uint motion = travel[from];
            int remaining = budget(motion, axis);
            if (remaining == 0 || speed(flags[from], axis) * sign_ <= 0) continue;
            int kind = atlas[from];
            if (kind < 2) continue;
            int distance = 0, to = from, contact = -1;
            ivec3 point = p;
            point[axis] += offset;
            bool in_body = body.w != 0 && inside_body(point);
            for (int j = 1; j <= remaining; j++) {
                int destination = offset + j * sign_;
                if (destination < 0 || destination >= width) break;
                int next = line_address(destination, first, second, stride);
                if (atlas[next] != 0) {
                    if (kind == 2 && atlas[next] == 2) contact = next;
                    break;
                }
                point[axis] += sign_;
                if (body.w != 0) {
                    bool next_in_body = inside_body(point);
                    if (!in_body && next_in_body) break;
                    in_body = next_in_body;
                }
                distance = j;
                to = next;
            }
            if (distance != 0) {
                atlas[from] = 0;
                atlas[to] = kind;
                flags[to] = (flags[to] & 3u) | (flags[from] & ~3u);
                flags[from] &= 3u;
                travel[to] = motion - (uint(distance) << (axis * 10 + 5));
                travel[from] = 0u;
                if (to / 512 != from / 512 && has_whole_travel(travel[to])) mark_sweep(to / 512);
            }
            if (contact >= 0) {
                if (sign_ > 0) close_sand_contact(to, contact, axis);
                else close_sand_contact(contact, to, axis);
            }
        }
    }
}

// Record the highest contiguous runs without filling the gaps between them.
// Uniform air and stone pages need one directory lookup per eight cells.
void column_map(uint invocation) {
    ivec2 dimensions = region_size_limit.xz;
    uint count = uint(dimensions.x * dimensions.y);
    if (invocation >= count) return;
    int stride = region_size_limit.w;
    int layers = region_origin.w;
    ivec2 column = ivec2(int(invocation) % dimensions.x, int(invocation) / dimensions.x);
    ivec2 xz = min(column * stride + stride / 2, bounds_mode.xz - 1);
    ivec4 spans[2] = ivec4[2](ivec4(0), ivec4(0));
    ivec2 completed = ivec2(0);
    for (int y = bounds_mode.y - 1; y >= 0 && any(lessThan(completed, ivec2(layers)));) {
        int slot = directory[tile_key(ivec3(xz.x, y, xz.y) / 8)];
        int bottom = y & ~7;
        int local = (xz.x & 7) + 8 * (xz.y & 7);
        do {
            int kind = slot < 0 ? 0 : atlas[slot * 512 + local + 64 * (y & 7)];
            for (int material_index = 0; material_index < 2; material_index++) {
                int layer = completed[material_index];
                if (layer == layers) continue;
                int channel = material_index * 2;
                if (kind == material_index + 2) {
                    if (spans[layer][channel + 1] == 0) spans[layer][channel + 1] = y + 1;
                    spans[layer][channel] = y;
                } else if (spans[layer][channel + 1] != 0) {
                    completed[material_index]++;
                }
            }
            y--;
        } while (slot >= 0 && y >= bottom && any(lessThan(completed, ivec2(layers))));
        y = bottom - 1;
    }
    for (int layer = 0; layer < layers; layer++)
        cells[invocation + uint(layer) * count] = floatBitsToInt(vec4(spans[layer]));
}

void trace_ray() {
    vec3 direction = field.xyz;
    float entry = 0.0, exit = focus.w;
    cells[0] = ivec4(0);
    for (int axis = 0; axis < 3; axis++) {
        if (abs(direction[axis]) < 1e-8) {
            if (focus[axis] < 0.0 || focus[axis] >= float(bounds_mode[axis])) return;
            continue;
        }
        float a = -focus[axis] / direction[axis];
        float b = (float(bounds_mode[axis]) - focus[axis]) / direction[axis];
        entry = max(entry, min(a, b));
        exit = min(exit, max(a, b));
    }
    if (entry > exit) return;
    ivec3 p = ivec3(floor(focus.xyz + direction * min(entry + 0.0001, exit)));
    p = clamp(p, ivec3(0), bounds_mode.xyz - 1);
    ivec3 advance = ivec3(sign(direction));
    vec3 next = vec3(1e30), spacing = vec3(1e30);
    for (int axis = 0; axis < 3; axis++) {
        if (abs(direction[axis]) < 1e-8) continue;
        spacing[axis] = abs(1.0 / direction[axis]);
        next[axis] = (float(p[axis] + (advance[axis] > 0 ? 1 : 0)) - focus[axis]) / direction[axis];
    }
    while (inside(p) && entry <= exit) {
        int kind = material(address(p));
        if ((axis_parity_count.z & (1 << kind)) != 0) {
            cells[0] = ivec4(floatBitsToInt(focus.xyz + direction * entry), kind);
            return;
        }
        entry = min(next.x, min(next.y, next.z));
        for (int axis = 0; axis < 3; axis++) {
            if (next[axis] <= entry) {
                p[axis] += advance[axis];
                next[axis] += spacing[axis];
            }
        }
    }
}

// Allocation and material updates use separate dispatches. Only the winner of
// the directory claim initializes an atlas slot or publishes an active page.
void ensure_page(ivec3 origin, bool allow_stone) {
    int key = tile_key(origin / 8);
    int slot = atomicAdd(directory[key], 0);
    if (slot == -3 || (!allow_stone && slot == -2)) return;
    if (slot < 0) {
        if (atomicCompSwap(directory[key], slot, -3) != slot) return;
        int next = atomicAdd(resident, 0);
        while (next < shape_capacity.w) {
            int observed = atomicCompSwap(resident, next, next + 1);
            if (observed == next) break;
            next = observed;
        }
        if (next == shape_capacity.w) {
            atomicAdd(capacity_misses, 1);
            atomicExchange(directory[key], slot);
            return;
        }
        int kind = -1 - slot;
        for (int i = 0; i < 512; i++) {
            atlas[next * 512 + i] = kind;
            flags[next * 512 + i] = 0u;
            travel[next * 512 + i] = 0u;
        }
        atomicExchange(writable[next], 1);
        sweep_tick[next] = 0u;
        pages[atomicAdd(pending_active, 1)] = ivec4(origin, next);
        atomicExchange(directory[key], next);
    } else if (atomicCompSwap(writable[slot], 0, 1) == 0) {
        pages[atomicAdd(pending_active, 1)] = ivec4(origin, slot);
    }
}

void expand_frontier(uint invocation) {
    if (invocation >= uint(active_count) * 6u) return;
    ivec4 page = pages[invocation / 6u];
    int face = int(invocation % 6u);
    int axis = face / 2;
    int side = face & 1;
    ivec3 neighbor = page.xyz;
    neighbor[axis] += side == 0 ? -8 : 8;
    if (!inside(neighbor)) return;
    int slot = atomicAdd(directory[tile_key(neighbor / 8)], 0);
    if (slot <= -2 || (slot >= 0 && atomicAdd(writable[slot], 0) != 0)) return;
    int u_axis = (axis + 1) % 3;
    int v_axis = (axis + 2) % 3;
    ivec3 local = ivec3(0);
    int scan_depth = axis_parity_count.w == 0 ? 2 : 8;
    for (int depth = 0; depth < scan_depth; depth++) {
        local[axis] = side == 0 ? depth : 7 - depth;
        for (int v = 0; v < 8; v++) {
            local[v_axis] = v;
            for (int u = 0; u < 8; u++) {
                local[u_axis] = u;
                int kind = atlas[page.w * 512 + local.x + 8 * (local.z + 8 * local.y)];
                if (kind >= 2) {
                    ensure_page(neighbor, false);
                    return;
                }
            }
        }
    }
}

void main() {
    if (blocked != 0) return;
    uint invocation = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * 16384u * gl_WorkGroupSize.x;
    int mode = bounds_mode.w;
    if (mode == 0) record_support(invocation);
    else if (mode == 1) exchange_pairs(invocation);
    else if (mode == 2) expand_frontier(invocation);
    else if (mode == 8) integrate_field(invocation);
    else if (mode == 9) add_impulse(invocation);
    else if (mode == 10) extract_surface(invocation);
    else if (mode == 11) column_map(invocation);
    else if (mode == 12 && invocation == 0u) trace_ray();
    else if (mode == 13) sweep_lines(invocation);
    else if (mode == 14) collect_sweep_pages(invocation);
    else if (mode >= 4 && invocation < uint(axis_parity_count.z)) {
        ivec3 p = cells[invocation].xyz;
        if (mode == 4) ensure_page(p & ~7, true);
        else if (mode == 5 && address(p) < 0) atomicOr(edit_failed, 1);
        else if (mode == 6 && edit_failed == 0) {
            int location = address(p);
            atlas[location] = cells[invocation].w;
            flags[location] = 0u;
            travel[location] = 0u;
        }
        else if (mode == 7) {
            cells[invocation].w = inside(p) ? material(address(p)) : -1;
        }
    }
}
