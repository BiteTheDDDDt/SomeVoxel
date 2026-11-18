"""Build and check an immutable paged material grid for a box-authored layout."""

import argparse
import json

import numpy as np
import warp as wp


@wp.kernel
def sample_material(directory: wp.array(dtype=int), pages: wp.array(dtype=int),
                    points: wp.array(dtype=wp.vec3i), output: wp.array(dtype=int),
                    px: int, pz: int, tile: int):
    i = wp.tid()
    p = points[i]
    slot = p[0] // tile + px * (p[2] // tile + pz * (p[1] // tile))
    page = directory[slot]
    if page < 0:
        output[i] = -1 - page
    else:
        local = p[0] % tile + tile * (p[2] % tile + tile * (p[1] % tile))
        output[i] = pages[page * tile * tile * tile + local]


def build_pages(boxes, bounds, tile, dynamic):
    nx, ny, nz = bounds
    px, py, pz = ((n + tile - 1) // tile for n in bounds)
    directory = np.empty((py, pz, px), np.int32)
    slabs = []
    page_count = 0
    materials = {}
    for slab in range(py):
        y0 = slab * tile
        data = np.zeros((tile, pz * tile, px * tile), dtype=np.uint8)
        for x0, low_y, z0, x1, high_y, z1, kind in boxes:
            start = max(0, low_y - y0)
            stop = min(tile, high_y - y0)
            if start < stop:
                data[start:stop, z0:z1, x0:x1] = kind
        counts = np.bincount(data[:min(tile, ny - y0), :nz, :nx].ravel())
        for kind, count in enumerate(counts):
            materials[kind] = materials.get(kind, 0) + int(count)
        tiled = data.reshape(tile, pz, tile, px, tile).transpose(1, 3, 0, 2, 4)
        minimum = tiled.min(axis=(2, 3, 4))
        maximum = tiled.max(axis=(2, 3, 4))
        needs_page = (minimum != maximum) | np.isin(maximum, dynamic)
        selected = tiled[needs_page]
        directory[slab] = -1 - maximum.astype(np.int32)
        directory[slab][needs_page] = np.arange(page_count, page_count + len(selected), dtype=np.int32)
        page_count += len(selected)
        if len(selected):
            slabs.append(selected.astype(np.int32).ravel())
    pages = np.concatenate(slabs) if slabs else np.zeros(1, dtype=np.int32)
    return directory.ravel(), pages, materials


def verify(boxes, bounds, tile, dynamic, samples):
    directory, pages, materials = build_pages(boxes, bounds, tile, dynamic)
    rng = np.random.default_rng(127)
    coordinates = rng.integers(0, bounds, size=(samples, 3), dtype=np.int32)
    # Also query just inside and outside every authored box face.
    edges = np.concatenate([boxes[:, :3], boxes[:, 3:6] - 1])
    edges = np.concatenate([edges + delta for delta in np.vstack((np.zeros(3), np.eye(3), -np.eye(3)))])
    points = np.vstack((coordinates, np.clip(edges, 0, np.array(bounds) - 1))).astype(np.int32)
    expected = np.zeros(len(points), dtype=np.int32)
    for box in boxes:
        expected[np.all((points >= box[:3]) & (points < box[3:6]), axis=1)] = box[6]
    device = "cuda:0"
    output = wp.empty(len(points), dtype=int, device=device)
    inputs = [wp.array(directory, dtype=int, device=device), wp.array(pages, dtype=int, device=device),
              wp.array(points, dtype=wp.vec3i, device=device), output,
              (bounds[0] + tile - 1) // tile, (bounds[2] + tile - 1) // tile, tile]
    wp.launch(sample_material, len(points), inputs=inputs, device=device)
    wp.synchronize_device(device)
    return {
        "materials": materials,
        "initial_grid_MiB": (directory.nbytes + pages.nbytes) / 2**20,
        "dense_grid_MiB": int(np.prod(bounds)) * 4 / 2**20,
        "queried_cells": len(points),
        "material_errors": int(np.count_nonzero(output.numpy() != expected)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("layouts", type=argparse.FileType("r", encoding="utf-8"))
    parser.add_argument("--bounds", type=int, nargs=3, required=True)
    parser.add_argument("--tile", type=int, choices=(8, 16, 32), default=8)
    parser.add_argument("--dynamic", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--samples", type=int, default=500_000)
    args = parser.parse_args()
    if min(args.bounds) <= 0 or args.samples < 1:
        parser.error("Bounds and sample count must be positive.")
    errors = 0
    for line in args.layouts:
        layout = json.loads(line)
        boxes = np.asarray(layout["boxes"], dtype=np.int32).reshape(-1, 7)
        if (np.any(boxes[:, :3] < 0) or np.any(boxes[:, 3:6] > args.bounds)
                or np.any(boxes[:, :3] >= boxes[:, 3:6]) or np.any(boxes[:, 6] < 0)
                or np.any(boxes[:, 6] > 255)):
            parser.error("Invalid box bounds or byte-sized material.")
        result = verify(boxes, args.bounds, args.tile, args.dynamic, args.samples)
        errors += result["material_errors"]
        print(json.dumps({"layout": layout["layout"], "tile": args.tile, **result}), flush=True)
    return int(errors > 0)


if __name__ == "__main__":
    raise SystemExit(main())
