"""Prepare Vulkan fixtures and compare their final material state with CUDA."""

import argparse
import json
from pathlib import Path

import numpy as np

from paged_material_motion import allocate_halo, logical_hash


def prepare(args):
    for index, entry in enumerate(args.layouts):
        layout = json.loads(entry)
        boxes = np.asarray(layout["boxes"], dtype=np.int32).reshape(-1, 7)
        if (np.any(boxes[:, :3] < 0) or np.any(boxes[:, 3:6] > args.bounds)
                or np.any(boxes[:, :3] >= boxes[:, 3:6]) or np.any(boxes[:, 6] < 0)
                or np.any(boxes[:, 6] > 3)):
            raise ValueError("Expected in-bounds boxes with material IDs 0..3.")
        directory, atlas, origins, slots, shape, _ = allocate_halo(boxes, args.bounds, args.halo)
        folder = args.output / str(index)
        folder.mkdir(parents=True, exist_ok=True)
        active = np.zeros(len(atlas), dtype=np.int32)
        active[slots] = 1
        for name, array in (("directory", directory), ("initial", atlas),
                            ("origins", np.column_stack((origins, slots))), ("writable", active)):
            array.astype("<i4", copy=False).tofile(folder / (name + ".i32"))
        sand = boxes[boxes[:, 6] == 2]
        peak = sand[np.argmax(sand[:, 4])]
        focus = layout.get("focus", ((peak[:3] + peak[3:6]) / 2).tolist())
        meta = {"bounds": args.bounds, "shape": shape, "focus": focus, "page_count": len(slots),
                "updates": args.frames, "layout": layout["layout"], "halo": args.halo}
        (folder / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        print(folder, flush=True)
    return 0


def verify(args):
    expected = {}
    for line in args.cuda_log:
        if line.startswith("{"):
            row = json.loads(line)
            expected[row["layout"]] = row
    failures = 0
    for folder in args.fixtures:
        meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        initial = np.fromfile(folder / "initial.i32", dtype="<i4").reshape(-1, 512)
        final = np.fromfile(folder / "vulkan.i32", dtype="<i4").reshape(initial.shape)
        origins = np.fromfile(folder / "origins.i32", dtype="<i4").reshape(-1, 4)
        digest = logical_hash(final, origins[:, :3], origins[:, 3], meta["bounds"])
        counts = np.bincount(final.ravel(), minlength=4) - np.bincount(initial.ravel(), minlength=4)
        walls = np.count_nonzero((initial == 1) != (final == 1))
        reference = expected[meta["layout"]]
        matched = (reference["halo_tiles"] == meta["halo"] and reference["updates"] == meta["updates"]
                   and digest == reference["hash"])
        failures += bool(np.any(counts) or walls or not matched)
        print(json.dumps({"layout": meta["layout"], "count_delta": counts.tolist(),
                          "static_errors": int(walls), "matches_cuda": matched, "hash": digest}))
    return int(failures > 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("prepare")
    setup.add_argument("layouts", type=argparse.FileType("r", encoding="utf-8"))
    setup.add_argument("output", type=Path)
    setup.add_argument("--bounds", type=int, nargs=3, required=True)
    setup.add_argument("--halo", type=int, default=3)
    setup.add_argument("--frames", type=int, default=48)
    check = commands.add_parser("verify")
    check.add_argument("cuda_log", type=argparse.FileType("r", encoding="utf-8"))
    check.add_argument("fixtures", type=Path, nargs="+")
    args = parser.parse_args()
    return prepare(args) if args.command == "prepare" else verify(args)


if __name__ == "__main__":
    raise SystemExit(main())
