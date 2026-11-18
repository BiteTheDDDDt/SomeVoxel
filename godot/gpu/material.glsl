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
};
layout(set = 0, binding = 6, std430) restrict buffer Indirect { uint groups[]; };
layout(set = 0, binding = 7, std430) restrict buffer Commands { ivec4 cells[]; };
layout(push_constant, std430) uniform Params {
    ivec4 shape_capacity;
    ivec4 bounds_mode;
    ivec4 axis_parity_count;
    vec4 focus;
    vec4 field;
    ivec4 region_origin;
    ivec4 region_size_limit;
};

const uint GROUP_WIDTH = 128u;
const uint GROUP_LIMIT = 16384u;

bool inside(ivec3 p) {
    return all(greaterThanEqual(p, ivec3(0))) && all(lessThan(p, bounds_mode.xyz));
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
    int below = material(address(p - ivec3(0, 1, 0)));
    int above = material(address(p + ivec3(0, 1, 0)));
    int location = page.w * 512 + local;
    flags[location] = (flags[location] & ~3u) | uint(below == 0) | (uint(above == 3) << 1);
}

int speed(uint state, int axis) {
    return int(bitfieldExtract(state, 2 + axis * 9, 5)) - (int(bitfieldExtract(state, 6 + axis * 9, 1)) << 5);
}

uint ready_bit(int axis) {
    return 1u << (29 + axis);
}

ivec3 local_cell(int index) {
    return ivec3(index & 7, index >> 6, (index >> 3) & 7);
}

// Velocity and subcell travel are stored in sixteenths of a cell per tick.
void integrate_field(uint invocation) {
    if (invocation >= uint(active_count) * 512u) return;
    ivec4 page = pages[invocation >> 9];
    int local = int(invocation & 511u);
    int location = page.w * 512 + local;
    if (atlas[location] != 2) return;
    uint previous = flags[location];
    ivec3 p = page.xyz + local_cell(local);
    vec3 delta = focus.xyz - vec3(p);
    float distance_sq = dot(delta, delta);
    bool affected = field.x > 0.0 && distance_sq < focus.w;
    if (!affected && (previous >> 2) == 0u) return;
    vec3 acceleration = vec3(0.0, -0.125, 0.0);
    if (affected && distance_sq > 0.0) {
        float distance = sqrt(distance_sq);
        // Linear falloff at the field boundary and a soft center.
        float weight = (1.0 - distance * field.y) / max(distance, 2.0);
        acceleration += delta * (field.x * weight);
    }
    uint state = previous & 3u;
    for (int axis = 0; axis < 3; axis++) {
        int last = speed(previous, axis);
        int velocity = clamp(last - sign(last) + int(round(acceleration[axis] * 16.0)), -15, 15);
        if (velocity == 0) continue;
        ivec3 neighbor = p;
        neighbor[axis] += sign(velocity);
        int obstacle = material(address(neighbor));
        if (obstacle != 0 && obstacle != 3) continue;
        int shift = 2 + axis * 9;
        int fraction = sign(velocity) == sign(last) ? int(bitfieldExtract(previous, shift + 5, 4)) : 0;
        fraction += abs(velocity);
        if (fraction >= 16) state |= ready_bit(axis);
        state |= (uint(velocity) & 31u) << shift;
        state |= (uint(fraction) & 15u) << (shift + 5);
    }
    flags[location] = state;
}

void add_impulse(uint invocation) {
    if (invocation >= uint(active_count) * 512u) return;
    ivec4 page = pages[invocation >> 9];
    int index = int(invocation & 511u);
    int location = page.w * 512 + index;
    if (atlas[location] != 2) return;
    vec3 delta = vec3(page.xyz + local_cell(index)) - focus.xyz;
    float distance_sq = dot(delta, delta);
    if (distance_sq >= focus.w) return;
    float weight = 1.0 - sqrt(distance_sq) * field.w;
    uint state = flags[location];
    for (int axis = 0; axis < 3; axis++) {
        int last = speed(state, axis);
        int velocity = clamp(last + int(round(field[axis] * weight * 16.0)), -15, 15);
        int shift = 2 + axis * 9;
        if (sign(velocity) != sign(last)) state = bitfieldInsert(state, 0u, shift + 5, 4);
        state = bitfieldInsert(state, uint(velocity), shift, 5);
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
    int b = address(q);
    if (b < 0 || writable[b / 512] == 0) return;
    int left = atlas[a], right = atlas[b];
    if (left == right) return;
    uint af = flags[a], bf = flags[b];
    uint ready = ready_bit(axis);
    bool swap = false;
    if (axis_parity_count.w != 0 && left == 2 && (right == 0 || right == 3) && (af >> 2) != 0u) {
        swap = (af & ready) != 0u && speed(af, axis) > 0;
    } else if (axis_parity_count.w != 0 && right == 2 && (left == 0 || left == 3) && (bf >> 2) != 0u) {
        swap = (bf & ready) != 0u && speed(bf, axis) < 0;
    } else if (axis == 1) swap = (left == 0 && (right == 2 || right == 3)) || (left == 3 && right == 2);
    else if (left == 3 && right == 0) {
        swap = (af & 1u) == 0u && ((bf & 1u) != 0u || ((af & 2u) != 0u && (bf & 2u) == 0u));
    } else if (right == 3 && left == 0) {
        swap = (bf & 1u) == 0u && ((af & 1u) != 0u || ((bf & 2u) != 0u && (af & 2u) == 0u));
    }
    if (swap) {
        atlas[a] = right;
        atlas[b] = left;
        if (axis_parity_count.w != 0) {
            flags[a] = (af & 3u) | (bf & ~(3u | ready));
            flags[b] = (bf & 3u) | (af & ~(3u | ready));
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
        }
        atomicExchange(writable[next], 1);
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
    // Two exchanges per axis can move material by two cells in one update.
    for (int depth = 0; depth < 2; depth++) {
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

void dispatch_size(uint offset, uint invocations) {
    uint total = max(1u, (invocations + GROUP_WIDTH - 1u) / GROUP_WIDTH);
    groups[offset] = min(total, GROUP_LIMIT);
    groups[offset + 1u] = (total + GROUP_LIMIT - 1u) / GROUP_LIMIT;
    groups[offset + 2u] = 1u;
}

void publish_dispatch_sizes() {
    active_count = pending_active;
    dispatch_size(0u, uint(active_count) * 512u);
    dispatch_size(3u, uint(active_count) * 256u);
    dispatch_size(6u, uint(active_count) * 6u);
}

void main() {
    uint invocation = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * GROUP_LIMIT * GROUP_WIDTH;
    int mode = bounds_mode.w;
    if (mode == 0) record_support(invocation);
    else if (mode == 1) exchange_pairs(invocation);
    else if (mode == 2) expand_frontier(invocation);
    else if (mode == 3 && invocation == 0u) publish_dispatch_sizes();
    else if (mode == 8) integrate_field(invocation);
    else if (mode == 9) add_impulse(invocation);
    else if (mode == 10) extract_surface(invocation);
    else if (mode >= 4 && invocation < uint(axis_parity_count.z)) {
        ivec3 p = cells[invocation].xyz;
        if (mode == 4) ensure_page(p & ~7, true);
        else if (mode == 5 && address(p) < 0) atomicOr(edit_failed, 1);
        else if (mode == 6 && edit_failed == 0) {
            int location = address(p);
            atlas[location] = cells[invocation].w;
            flags[location] = 0u;
        }
        else if (mode == 7) {
            cells[invocation].w = inside(p) ? material(address(p)) : -1;
        }
    }
}
