"""Exercise shared sand and water occupancy on a full, paged CUDA grid."""

import argparse
import hashlib
import json
import time

import numpy as np
import warp as wp

from paged_material_grid import build_pages


@wp.func
def page_address(p: wp.vec3i, directory: wp.array(dtype=int), shape: wp.vec3i) -> int:
    if p[0] < 0 or p[1] < 0 or p[2] < 0 or p[0] >= shape[0] * 8 or p[1] >= shape[1] * 8 or p[2] >= shape[2] * 8:
        return -2
    key = (p[0] >> 3) + shape[0] * ((p[2] >> 3) + shape[2] * (p[1] >> 3))
    slot = directory[key]
    if slot < 0:
        return slot
    return slot * 512 + (p[0] & 7) + 8 * ((p[2] & 7) + 8 * (p[1] & 7))


@wp.func
def material(address: int, atlas: wp.array(dtype=int)) -> int:
    if address < 0:
        return -1 - address
    return atlas[address]


@wp.kernel
def record_water_faces(
    origins: wp.array(dtype=wp.vec3i), slots: wp.array(dtype=int),
    directory: wp.array(dtype=int), atlas: wp.array(dtype=int),
    flags: wp.array(dtype=wp.uint8), shape: wp.vec3i,
):
    i = wp.tid()
    row = i // 512
    local = i & 511
    point = origins[row] + wp.vec3i(local & 7, local >> 6, (local >> 3) & 7)
    index = slots[row] * 512 + local
    below = material(page_address(point - wp.vec3i(0, 1, 0), directory, shape), atlas)
    above = material(page_address(point + wp.vec3i(0, 1, 0), directory, shape), atlas)
    flags[index] = wp.uint8(int(below == 0) | (int(above == 3) << 1))


@wp.func
def preference(point: wp.vec3i, axis: int, sign: int, focus: wp.vec3, radius_sq: float) -> bool:
    delta = focus - wp.vec3(float(point[0]), float(point[1]), float(point[2]))
    return delta[axis] * float(sign) > 0.5 and wp.dot(delta, delta) < radius_sq


@wp.kernel
def exchange_pairs(
    origins: wp.array(dtype=wp.vec3i), slots: wp.array(dtype=int),
    directory: wp.array(dtype=int), atlas: wp.array(dtype=int), flags: wp.array(dtype=wp.uint8),
    writable: wp.array(dtype=wp.uint8), counts: wp.array2d(dtype=int), shape: wp.vec3i,
    axis: int, parity: int, focus: wp.vec3, radius_sq: float,
):
    thread = wp.tid()
    page = thread // 256
    local = thread & 255
    x, z, y = local & 7, (local >> 3) & 7, local >> 6
    if axis == 0:
        x = (local & 3) * 2 + parity
        z = (local >> 2) & 7
        y = local >> 5
    elif axis == 1:
        y = y * 2 + parity
    else:
        z = ((local >> 3) & 3) * 2 + parity
        y = local >> 5
    p = origins[page] + wp.vec3i(x, y, z)
    q = wp.vec3i(p)
    q[axis] += 1
    a = slots[page] * 512 + x + 8 * (z + 8 * y)
    b = page_address(q, directory, shape)
    left = material(a, atlas)
    right = material(b, atlas)
    if left == right:
        return
    swap = False
    # Pairing by coordinate parity grants a single writer for each cell.
    if left == 2 and (right == 0 or right == 3) and preference(p, axis, 1, focus, radius_sq):
        swap = True
    elif right == 2 and (left == 0 or left == 3) and preference(q, axis, -1, focus, radius_sq):
        swap = True
    elif axis == 1:
        swap = (left == 0 and (right == 2 or right == 3)) or (left == 3 and right == 2)
    elif b >= 0 and writable[b // 512] != wp.uint8(0):
        # These support flags come from a separate snapshot, so an adjacent
        # pair cannot change the water gradient while it is being read.
        af, bf = int(flags[a]), int(flags[b])
        if left == 3 and right == 0:
            swap = (af & 1) == 0 and ((bf & 1) != 0 or ((af & 2) != 0 and (bf & 2) == 0))
        elif right == 3 and left == 0:
            swap = (bf & 1) == 0 and ((af & 1) != 0 or ((bf & 2) != 0 and (af & 2) == 0))
    if not swap:
        return
    if b < 0 or writable[b // 512] == wp.uint8(0):
        wp.atomic_add(counts, page, 3, 1)
        return
    atlas[a] = right
    atlas[b] = left
    if a // 512 != b // 512:
        if left == 2 or right == 2:
            wp.atomic_add(counts, page, 0, 1)
        if left == 3 or right == 3:
            wp.atomic_add(counts, page, 1, 1)
    if left + right == 5:
        wp.atomic_add(counts, page, 2, 1)


def allocate_halo(boxes, bounds, halo):
    if any(n % 8 for n in bounds):
        raise ValueError("The motion probe requires whole eight-cell pages.")
    directory, packed, totals = build_pages(boxes, bounds, 8, [2, 3])
    pages = packed.reshape(-1, 512)
    shape = tuple(n // 8 for n in bounds)
    table = directory.reshape(shape[1], shape[2], shape[0])
    active = np.zeros(table.shape, dtype=bool)
    occupied_slots = table >= 0
    mobile = ((pages == 2) | (pages == 3)).any(axis=1)
    active[occupied_slots] = mobile[table[occupied_slots]]
    for _ in range(halo):
        expanded = active.copy()
        for axis in range(3):
            view = np.moveaxis(active, axis, 0)
            out = np.moveaxis(expanded, axis, 0)
            out[1:] |= view[:-1]
            out[:-1] |= view[1:]
        active = expanded
    # Uniform solid pages remain encoded in the directory. Empty pages in
    # the halo receive real storage, including pages across tile boundaries.
    active &= table != -2
    new = active & (table == -1)
    table[new] = np.arange(len(pages), len(pages) + np.count_nonzero(new), dtype=np.int32)
    pages = np.concatenate((pages, np.zeros((np.count_nonzero(new), 512), np.int32)))
    y, z, x = np.nonzero(active)
    origins = np.column_stack((x, y, z)).astype(np.int32) * 8
    return directory, pages, origins, table[active].copy(), shape, totals


def run(layout, bounds, halo, frames, use_graphs, trace):
    boxes = np.asarray(layout["boxes"], dtype=np.int32).reshape((-1, 7))
    if (np.any(boxes[:, :3] < 0) or np.any(boxes[:, 3:6] > bounds)
            or np.any(boxes[:, :3] >= boxes[:, 3:6]) or np.any(boxes[:, 6] < 0) or np.any(boxes[:, 6] > 3)):
        raise ValueError("Expected in-bounds boxes with material IDs 0..3.")
    directory, pages, origins, slots, shape, totals = allocate_halo(boxes, bounds, halo)
    wp.set_device("cuda:0")
    atlas = wp.array(pages.ravel(), dtype=int)
    device_directory = wp.array(directory, dtype=int)
    device_origins = wp.array(origins, dtype=wp.vec3i)
    device_slots = wp.array(slots, dtype=int)
    writable = np.zeros(len(pages), dtype=np.uint8)
    writable[slots] = 1
    device_writable = wp.array(writable, dtype=wp.uint8)
    flags = wp.zeros(atlas.shape, dtype=wp.uint8)
    counts = wp.zeros((len(slots), 4), dtype=int)
    grid = wp.vec3i(*shape)
    sand_boxes = boxes[boxes[:, 6] == 2]
    if len(sand_boxes) == 0:
        raise ValueError("The motion fixture needs a sand source.")
    peak = sand_boxes[np.argmax(sand_boxes[:, 4])]
    anchor = np.asarray(layout.get("focus", (peak[:3] + peak[3:6]) / 2), dtype=np.float32)
    def launch_update(tick):
        target = anchor + np.array([0.75 * min(tick, frames // 2), 0.0, 0.25 * tick], dtype=np.float32)
        focus = wp.vec3(*target)
        reach = 85.0 ** 2 if tick < frames * 2 // 3 else 0.0
        for axis in (1, 0, 2):
            if axis != 1:
                wp.launch(record_water_faces, len(slots) * 512,
                          inputs=[device_origins, device_slots, device_directory, atlas, flags, grid])
            for color in range(2):
                wp.launch(exchange_pairs, len(slots) * 256,
                          inputs=[device_origins, device_slots, device_directory, atlas, flags,
                                  device_writable, counts, grid, axis, color ^ (tick & 1), focus, reach])

    graphs = []
    if use_graphs:
        wp.load_module(device="cuda:0")
        for tick in range(frames):
            # Capturing records commands without advancing material state.
            with wp.ScopedCapture() as capture:
                launch_update(tick)
            graphs.append(capture.graph)
    samples = []
    begin, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
    for tick in range(frames):
        wp.record_event(begin)
        start = time.perf_counter()
        if use_graphs:
            wp.capture_launch(graphs[tick])
        else:
            launch_update(tick)
        wp.record_event(end)
        wp.synchronize_event(end)
        if tick >= 4:
            samples.append([wp.get_event_elapsed_time(begin, end), (time.perf_counter() - start) * 1000])
    final = atlas.numpy().reshape(pages.shape)
    final_counts = np.bincount(final.ravel(), minlength=4)
    before_counts = np.bincount(pages.ravel(), minlength=4)
    sand, water, exchanges, edge = counts.numpy().sum(axis=0, dtype=np.int64)
    conserved = np.array_equal(before_counts, final_counts)
    static_errors = int(np.count_nonzero((pages == 1) != (final == 1)))
    # Hash mobile material by logical cell coordinates, independent of the
    # assigned page slots and of how many empty pages the halo allocated.
    by_page = final[slots]
    mobile_page, mobile_cell = np.nonzero((by_page == 2) | (by_page == 3))
    mobile_key = origins[mobile_page].astype(np.int64)
    mobile_key[:, 0] += mobile_cell & 7
    mobile_key[:, 1] += mobile_cell >> 6
    mobile_key[:, 2] += (mobile_cell >> 3) & 7
    mobile_key[:, 0] += bounds[0] * (mobile_key[:, 2] + bounds[2] * mobile_key[:, 1])
    material_ids = by_page[mobile_page, mobile_cell]
    hashes = hashlib.sha256((4 * mobile_key[:, 0] + material_ids).tobytes())
    result = {
        "layout": layout["layout"], "halo_tiles": halo, "graphs": use_graphs, "updates": frames,
        "resident_pages": len(pages), "dispatched_pages": len(slots),
        "sand": totals.get(2, 0), "water": totals.get(3, 0),
        "tracked_mib": sum(a.size * a.dtype.itemsize for a in (pages, directory, slots, origins)) / 2**20
        + (flags.size + counts.size * 4 + writable.nbytes) / 2**20,
        "cuda_p50_ms": float(np.percentile(samples, 50, axis=0)[0]),
        "cuda_p95_ms": float(np.percentile(samples, 95, axis=0)[0]),
        "fenced_p95_ms": float(np.percentile(samples, 95, axis=0)[1]),
        "sand_crossings": int(sand), "water_crossings": int(water), "sand_water_swaps": int(exchanges),
        "unbacked_move_attempts": int(edge), "conserved": bool(conserved), "static_errors": static_errors,
        "material_count_delta": (final_counts - before_counts).tolist(),
        "hash": hashes.hexdigest(),
    }
    if trace:
        result["cuda_ms_after_warmup"] = [round(sample[0], 3) for sample in samples]
    print(json.dumps(result), flush=True)
    return bool(conserved and not static_errors and sand > 0 and water > 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("layouts", type=argparse.FileType("r", encoding="utf-8"))
    parser.add_argument("--bounds", type=int, nargs=3, required=True)
    parser.add_argument("--halo", type=int, default=3)
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--graph", action="store_true", help="Replay one pre-recorded CUDA graph per update.")
    parser.add_argument("--trace", action="store_true", help="Include individual CUDA timings.")
    args = parser.parse_args()
    if min(args.bounds) <= 0 or args.halo < 1 or args.frames < 6:
        parser.error("Use positive bounds, a halo of at least one page and at least six updates.")
    errors = 0
    for line in args.layouts:
        errors += not run(json.loads(line), args.bounds, args.halo, args.frames, args.graph, args.trace)
    return int(errors > 0)


if __name__ == "__main__":
    raise SystemExit(main())
