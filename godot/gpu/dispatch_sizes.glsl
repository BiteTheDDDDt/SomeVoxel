#[compute]
#version 450
layout(local_size_x = 1) in;

layout(set = 0, binding = 0, std430) restrict buffer Counters {
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
layout(set = 0, binding = 1, std430) writeonly buffer Indirect { uint groups[]; };

layout(push_constant, std430) uniform Params {
    uint mode;
    uint tick;
    uint guard_pool;
    uint reset_sweep_pairs;
};

void dispatch_size(uint offset, uint invocations) {
    const uint width = 128u;
    const uint limit = 16384u;
    uint total = max(1u, (invocations + width - 1u) / width);
    groups[offset] = min(total, limit);
    groups[offset + 1u] = (total + limit - 1u) / limit;
    groups[offset + 2u] = 1u;
}

void main() {
    if (blocked != 0) return;
    if (mode == 1u) {
        if (reset_sweep_pairs != 0u) sweep_pairs = 0u;
        sweep_count = sweep_pending;
        sweep_pending = 0u;
        sweep_pairs += sweep_count;
        dispatch_size(15u, sweep_count * 64u);
        return;
    }
    // Capacity exhaustion is discovered before any material pass for the
    // tick. Keep the old frontier list length for replay after growth.
    if (guard_pool != 0u && capacity_misses != 0) {
        blocked = 1;
        return;
    }
    completed_tick = tick;
    active_count = pending_active;
    dispatch_size(0u, uint(active_count) * 512u);
    dispatch_size(3u, uint(active_count) * 256u);
    dispatch_size(6u, uint(active_count) * 6u);
    dispatch_size(9u, uint(active_count) * 64u);
    dispatch_size(12u, uint(active_count));
}
