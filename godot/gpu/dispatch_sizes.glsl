#[compute]
#version 450
layout(local_size_x = 1) in;

layout(set = 0, binding = 0, std430) restrict buffer Counters {
    int resident;
    int pending_active;
    int active_count;
};
layout(set = 0, binding = 1, std430) writeonly buffer Indirect { uint groups[]; };

void dispatch_size(uint offset, uint invocations) {
    const uint width = 128u;
    const uint limit = 16384u;
    uint total = max(1u, (invocations + width - 1u) / width);
    groups[offset] = min(total, limit);
    groups[offset + 1u] = (total + limit - 1u) / limit;
    groups[offset + 2u] = 1u;
}

void main() {
    active_count = pending_active;
    dispatch_size(0u, uint(active_count) * 512u);
    dispatch_size(3u, uint(active_count) * 256u);
    dispatch_size(6u, uint(active_count) * 6u);
    dispatch_size(9u, uint(active_count) * 64u);
}
