#[compute]
#version 450
#ifndef HEAD_WORKGROUP_SIZE
#define HEAD_WORKGROUP_SIZE 128
#endif
layout(local_size_x = HEAD_WORKGROUP_SIZE) in;
layout(set=0,binding=0,std430) readonly buffer Rows { uint rows[]; };
layout(set=0,binding=1,std430) restrict buffer Runs { uvec2 runs[]; };
layout(set=0,binding=2,std430) restrict buffer Head { uvec2 heads[]; };
layout(push_constant,std430) uniform Parameters { ivec4 shape; };

uint row_mask(uint row) {
    return (rows[row >> 2u] >> ((row & 3u) * 8u)) & 255u;
}

uint start_mask(uint mask) { return mask & ~(mask << 1); }
uint run_at(uint row, uint starts, int x) {
    return row * 4u + uint(bitCount(starts & ((2u << x) - 1u))) - 1u;
}
uint root(uint id) {
    for (;;) {
        uint parent = atomicAdd(runs[id].x, 0u);
        if (parent == id) return id;
        uint ancestor = atomicAdd(runs[parent].x, 0u);
        if (ancestor < parent) atomicMin(runs[id].x, ancestor);
        id = ancestor;
    }
}

// Join is complete before rank and publish. Parent links stay immutable in
// those passes, so readers can use loads without contending on atomic units.
uint resolved_root(uint id) {
    uint parent = runs[id].x;
    while (parent != id) {
        id = parent;
        parent = runs[id].x;
    }
    return id;
}

void join(uint a, uint b) {
    for (;;) {
        a = root(a);
        b = root(b);
        if (a == b) return;
        uint high = max(a, b);
        if (atomicCompSwap(runs[high].x, high, min(a,b)) == high) return;
    }
}
void join_rows(uint row, uint other, uint mask, uint a) {
    uint other_mask = row_mask(other);
    uint b = start_mask(other_mask);
    for (uint edges = start_mask(mask & other_mask); edges != 0u; edges &= edges - 1u) {
        int x = findLSB(edges);
        join(run_at(row, a, x), run_at(other, b, x));
    }
}
void main() {
    uint row = gl_GlobalInvocationID.x;
    uint plane = uint(shape.x * shape.z);
    if (row >= plane * uint(shape.y)) return;
    uint mask = row_mask(row);
    if (mask == 0u) {
        if (shape.w == 3) heads[row] = uvec2(0);
        return;
    }
    uint starts = start_mask(mask);
    uint count = uint(bitCount(starts));
    uint y = row / plane;
    uint z = row / uint(shape.x) % uint(shape.z);
    uint x = row % uint(shape.x);
    if (shape.w == 0) {
        for (uint i = 0u; i < count; ++i) runs[row * 4u + i] = uvec2(row * 4u + i, 0);
    } else if (shape.w == 1) {
        if (x + 1u < uint(shape.x) && (mask & 128u) != 0u && (row_mask(row + 1u) & 1u) != 0u)
            join(row * 4u + count - 1u, (row + 1u) * 4u);
        if (y + 1u < uint(shape.y)) join_rows(row, row + plane, mask, starts);
        if (z + 1u < uint(shape.z)) join_rows(row, row + uint(shape.x), mask, starts);
    } else if (shape.w == 2) {
        uint upper = y + 1u < uint(shape.y) ? row_mask(row + plane) : 0u;
        // Only terminal upward faces may set their component's maximum Y.
        uint last_root = 0xffffffffu;
        for (uint exposed = mask & ~upper; exposed != 0u; exposed &= exposed - 1u) {
            uint id = resolved_root(run_at(row, starts, findLSB(exposed)));
            if (id == last_root) continue;
            atomicMax(runs[id].y, y + 1u);
            last_root = id;
        }
    } else {
        uvec2 packed = uvec2(0u);
        for (uint i = 0u; i < count; ++i)
            packed[i >> 1] |= runs[resolved_root(row * 4u + i)].y << ((i & 1u) * 16u);
        heads[row] = packed;
    }
}
