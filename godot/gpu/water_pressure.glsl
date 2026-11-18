#[compute]
#version 450
layout(local_size_x = 128) in;

layout(set = 0, binding = 0, std430) readonly buffer Directory { int directory[]; };
layout(set = 0, binding = 1, std430) restrict buffer Materials { int atlas[]; };
layout(set = 0, binding = 2, std430) readonly buffer Origins { ivec4 pages[]; };
layout(set = 0, binding = 3, std430) readonly buffer Residency { int writable[]; };
layout(set = 0, binding = 4, std430) restrict buffer Motion { uint flags[]; };
layout(set = 0, binding = 5, std430) readonly buffer Counters {
    int resident;
    int pending_active;
    int active_count;
    int capacity_misses;
    int edit_failed;
    int surface_count;
    int blocked;
};
layout(set = 0, binding = 6, std430) restrict buffer Rows { uint masks[]; };
struct Run {
    uint parent;
    uint source;
    uint outlet;
};
layout(set = 0, binding = 7, std430) restrict buffer Runs { Run runs[]; };
layout(set = 0, binding = 8, std430) restrict buffer Transfers {
    uint transfer_count;
    uint total_transfers;
    uvec2 transfers[48];
};
layout(set = 0, binding = 9, std430) restrict buffer Travel { uint travel[]; };

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

const uint NO_ROOT = 0xffffffffu;
const uint POSITION_MASK = 0x7fffffffu;
const uint MAX_TRANSFERS = 48u;
const int SCAN_RADIUS = 8;
const ivec3 UP = ivec3(0, 1, 0);
const ivec3 SIDES[4] = ivec3[4](ivec3(-1, 0, 0), ivec3(1, 0, 0), ivec3(0, 0, -1), ivec3(0, 0, 1));

bool inside(ivec3 p) {
    return all(greaterThanEqual(p, ivec3(0))) && all(lessThan(p, bounds_mode.xyz));
}

bool inside_body(ivec3 p) {
    int half_width = body.w & 65535;
    int height = body.w >> 16;
    return p.y >= body.y && p.y < body.y + height
        && all(lessThan(abs(p.xz - body.xz), ivec2(half_width)));
}

int address(ivec3 p) {
    if (!inside(p)) return -2;
    ivec3 tile = p / 8;
    int slot = directory[tile.x + shape_capacity.x * (tile.z + shape_capacity.z * tile.y)];
    if (slot < 0) return slot;
    ivec3 local = p & 7;
    return slot * 512 + local.x + 8 * (local.z + 8 * local.y);
}

bool water_enter(ivec3 p) {
    if (body.w != 0 && inside_body(p)) return false;
    int site = address(p);
    return site == -1 || (site >= 0 && atlas[site] == 0);
}

uint grid_index(ivec3 p) {
    return uint(p.x + bounds_mode.x * (p.z + bounds_mode.z * p.y));
}

ivec3 unflat(uint i) {
    uint width = uint(bounds_mode.x);
    uint depth = uint(bounds_mode.z);
    return ivec3(i % width, i / (width * depth), (i / width) % depth);
}

uint starts(uint mask) {
    return mask & ~(mask << 1);
}

uint run_at(uint row, uint start_mask, int column) {
    return row * 4u + uint(bitCount(start_mask & ((2u << column) - 1u))) - 1u;
}

// Every parent link decreases. A failed CAS retries the edge with the
// updated root rather than overwriting a competing union.
uint find_root(uint node) {
    for (;;) {
        uint parent = atomicAdd(runs[node].parent, 0u);
        if (parent == node) return node;
        uint ancestor = atomicAdd(runs[parent].parent, 0u);
        if (ancestor < parent) atomicMin(runs[node].parent, ancestor);
        node = ancestor;
    }
}

void join(uint first, uint second) {
    for (;;) {
        first = find_root(first);
        second = find_root(second);
        if (first == second) return;
        uint high = max(first, second);
        if (atomicCompSwap(runs[high].parent, high, min(first, second)) == high) return;
    }
}

uint read_root(uint node) {
    while (runs[node].parent != node) node = runs[node].parent;
    return node;
}

void build_runs(uint row, ivec3 p) {
    uint mask = 0u;
    if (!inside(p)) {
        masks[row] = mask;
        return;
    }
    for (int x = 0; x < min(8, bounds_mode.x - p.x); ++x) {
        int site = int(row * 8u) + x;
        if (atlas[site] == 3 && (flags[site] >> 2) == 0u
                && (body.w == 0 || !inside_body(p + ivec3(x, 0, 0)))) mask |= 1u << x;
    }
    masks[row] = mask;
    uint count = uint(bitCount(starts(mask)));
    for (uint i = 0u; i < count; ++i) {
        uint id = row * 4u + i;
        runs[id] = Run(id, 0u, 0u);
    }
}

void join_rows(uint row, uint mask, int adjacent) {
    if (adjacent < 0 || writable[adjacent / 512] == 0) return;
    uint other = uint(adjacent) >> 3;
    uint other_mask = masks[other] & 255u;
    uint overlap = mask & other_mask;
    if (overlap == 0u) return;
    uint first_runs = starts(mask);
    uint second_runs = starts(other_mask);
    for (uint edges = starts(overlap); edges != 0u; edges &= edges - 1u) {
        int column = findLSB(edges);
        join(run_at(row, first_runs, column), run_at(other, second_runs, column));
    }
}

void link_runs(uint row, uint mask, ivec3 p) {
    if ((mask & 128u) != 0u) {
        int adjacent = address(p + ivec3(8, 0, 0));
        if (adjacent >= 0 && writable[adjacent / 512] != 0) {
            uint other = uint(adjacent) >> 3;
            if ((masks[other] & 1u) != 0u) join(run_at(row, starts(mask), 7), other * 4u);
        }
    }
    join_rows(row, mask, address(p + ivec3(0, 1, 0)));
    join_rows(row, mask, address(p + ivec3(0, 0, 1)));
}

uint row_mask(ivec3 p) {
    int site = address(p);
    return site >= 0 && writable[site / 512] != 0 ? masks[site >> 3] & 255u : 0u;
}

uint rank_outlet(ivec3 p) {
    int site = address(p);
    if (site < 0 || writable[site / 512] == 0 || atlas[site] != 0 || (body.w != 0 && inside_body(p))) return 0u;
    if (water_enter(p - UP)) return 0u;
    return POSITION_MASK - grid_index(p);
}

void rank_surfaces(uint row, uint mask, ivec3 p) {
    uint start_mask = starts(mask);
    uint previous_root = NO_ROOT, best_source = 0u, best_outlet = 0u;
    uint above = row_mask(p + UP);
    uint sides[4] = uint[4](
        (mask << 1) | (row_mask(p + ivec3(-8, 0, 0)) >> 7),
        (mask >> 1) | (row_mask(p + ivec3(8, 0, 0)) << 7),
        row_mask(p + SIDES[2]), row_mask(p + SIDES[3]));
    uint candidates = mask & ~(above & sides[0] & sides[1] & sides[2] & sides[3]);
    for (uint remaining = candidates; remaining != 0u; remaining &= remaining - 1u) {
        int x = findLSB(remaining);
        ivec3 source = p + ivec3(x, 0, 0);
        uint up = 0u, destination = 0u;
        uint bit = 1u << x;
        if ((above & bit) == 0u && water_enter(source + UP)) {
            up = grid_index(source) + 1u;
            destination = rank_outlet(source + UP);
        }
        for (int side = 0; side < 4; ++side) {
            if ((sides[side] & bit) == 0u) destination = max(destination, rank_outlet(source + SIDES[side]));
        }
        if (up == 0u && destination == 0u) continue;
        uint root = read_root(run_at(row, start_mask, x));
        if (root != previous_root && previous_root != NO_ROOT) {
            atomicMax(runs[previous_root].source, best_source);
            atomicMax(runs[previous_root].outlet, best_outlet);
            best_source = 0u;
            best_outlet = 0u;
        }
        previous_root = root;
        best_source = max(best_source, up);
        best_outlet = max(best_outlet, destination);
    }
    if (previous_root != NO_ROOT) {
        atomicMax(runs[previous_root].source, best_source);
        atomicMax(runs[previous_root].outlet, best_outlet);
    }
}

bool belongs(ivec3 p, uint root) {
    int site = address(p);
    if (site < 0 || writable[site / 512] == 0) return false;
    uint row = uint(site) >> 3;
    uint mask = masks[row] & 255u;
    int relative_x = site & 7;
    return (mask & (1u << relative_x)) != 0u
        && read_root(run_at(row, starts(mask), relative_x)) == root;
}

// Scan nearby cells at the same high and low levels. The fixed global
// transfer budget matches the CPU solver and bounds the worst-case search.
ivec2 next_endpoint(ivec3 first, int start, uint root, bool take) {
    const int diameter = SCAN_RADIUS * 2 + 1;
    for (int i = start; i < diameter * diameter; ++i) {
        ivec3 p = first + ivec3(i % diameter - SCAN_RADIUS, 0, i / diameter - SCAN_RADIUS);
        int site = address(p);
        if (site < 0 || writable[site / 512] == 0) continue;
        if (take) {
            if (water_enter(p + UP) && belongs(p, root)) return ivec2(site, i + 1);
        } else if (rank_outlet(p) != 0u) {
            if (belongs(p - UP, root)) return ivec2(site, i + 1);
            for (int side = 0; side < 4; ++side)
                if (belongs(p + SIDES[side], root)) return ivec2(site, i + 1);
        }
    }
    return ivec2(-1);
}

void plan_transfers(uint root) {
    uint high = runs[root].source;
    uint low = runs[root].outlet;
    if (high == 0u || low == 0u) return;
    ivec3 source = unflat(high - 1u), outlet = unflat(POSITION_MASK - low);
    if (outlet.y >= source.y) return;
    ivec2 a = ivec2(0), b = ivec2(0);
    for (uint i = 0u; i < MAX_TRANSFERS; ++i) {
        a = next_endpoint(source, a.y, root, true);
        b = next_endpoint(outlet, b.y, root, false);
        if (a.x < 0 || b.x < 0) break;
        uint ticket = atomicAdd(transfer_count, 1u);
        if (ticket >= MAX_TRANSFERS) return;
        transfers[ticket] = uvec2(a.x, b.x);
    }
}

void publish_heads(uint row, uint mask, int y) {
    uint top_mask = 0u;
    uint start_mask = starts(mask);
    for (uint remaining = start_mask; remaining != 0u; remaining &= remaining - 1u) {
        int x = findLSB(remaining);
        uint root = read_root(run_at(row, start_mask, x));
        uint high = runs[root].source;
        if (high == 0u || unflat(high - 1u).y <= y) continue;
        int width = findLSB(~(mask >> x));
        top_mask |= ((1u << width) - 1u) << x;
    }
    masks[row] = mask | (top_mask << 8);
}

void main() {
    if (blocked != 0) return;
    uint invocation = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * 16384u * gl_WorkGroupSize.x;
    int mode = bounds_mode.w;
    if (mode == 0 && invocation == 0u) transfer_count = 0u;
    if (mode == 4) {
        uint count = min(transfer_count, MAX_TRANSFERS);
        if (invocation < count) {
            // All joined parcels here are passive water. Moving the vacancy to
            // the source leaves the same occupancy as shifting those parcels
            // along a face-connected route.
            uvec2 sites = transfers[invocation];
            if (atomicCompSwap(atlas[sites.y], 0, 3) != 0) return;
            atlas[sites.x] = 0;
            flags[sites.x] = flags[sites.y] = 0u;
            travel[sites.x] = travel[sites.y] = 0u;
            atomicAdd(total_transfers, 1u);
        }
        return;
    }
    if (invocation >= uint(active_count) * 64u) return;
    ivec4 page = pages[invocation >> 6];
    int local = int(invocation & 63u);
    ivec3 p = page.xyz + ivec3(0, local >> 3, local & 7);
    uint row = uint(page.w) * 64u + uint(local);
    if (mode == 0) {
        build_runs(row, p);
        return;
    }
    uint mask = masks[row] & 255u;
    if (mask == 0u) return;
    if (mode == 1) link_runs(row, mask, p);
    else if (mode == 2) rank_surfaces(row, mask, p);
    else if (mode == 3) {
        uint count = uint(bitCount(starts(mask)));
        for (uint i = 0u; i < count; ++i) {
            uint root = row * 4u + i;
            if (runs[root].parent == root) plan_transfers(root);
        }
    } else if (mode == 5) publish_heads(row, mask, p.y);
}
