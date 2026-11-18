"""Measure GPU memory, time and physical error for compressed lattice state.

The variants keep the same force, contact, slip and render packing schedule.
Arithmetic and draw output remain FP32. Travel is measured in cell widths so
that the face-crossing threshold is exactly representable in every format.
"""

import argparse
import hashlib
import json
import time
from typing import Any

import numpy as np
import warp as wp

from matched_cuda import acceleration, fixture, index_of, percentile, side_step, valid


Q14Vec = wp.types.vector(3, wp.int16)
Q6Vec = wp.types.vector(3, wp.int8)
FORMATS = (
    "fp32", "packed-cells", "fp16-v", "fp16", "q14", "q6",
    "packed-fp16-v", "packed-fp16", "packed-q6",
    "packed-cells-gather", "packed-fp16-v-gather", "packed-q13-v-gather",
)
wp.set_module_options({"enable_backward": False})


def storage_types(layout):
    layout = layout.removesuffix("-gather")
    cells = wp.int32 if layout.startswith("packed-") else wp.vec3i
    velocity = wp.vec3 if layout in ("fp32", "packed-cells") else wp.vec3h
    if layout.endswith("q13-v"):
        velocity = Q14Vec
    travel = wp.vec3
    if layout.endswith("fp16"):
        travel = wp.vec3h
    elif layout.endswith("q14"):
        travel = Q14Vec
    elif layout.endswith("q6"):
        travel = Q6Vec
    return cells, velocity, travel


@wp.func
def cell_load(value: wp.vec3i) -> wp.vec3i:
    return value


@wp.func
def cell_load(value: wp.int32) -> wp.vec3i:
    return wp.vec3i(value & 1023, (value >> 10) & 511, (value >> 19) & 1023)


@wp.func
def cell_store(cells: wp.array(dtype=wp.vec3i), i: int, value: wp.vec3i):
    cells[i] = value


@wp.func
def cell_store(cells: wp.array(dtype=wp.int32), i: int, value: wp.vec3i):
    cells[i] = value[0] | (value[1] << 10) | (value[2] << 19)


@wp.func
def motion_load(value: wp.vec3) -> wp.vec3:
    return value


@wp.func
def motion_load(value: wp.vec3h) -> wp.vec3:
    return wp.vec3(float(value[0]), float(value[1]), float(value[2]))


@wp.func
def motion_load(value: Q14Vec) -> wp.vec3:
    return wp.vec3(float(value[0]), float(value[1]), float(value[2])) * (1.0 / 16384.0)


@wp.func
def motion_load(value: Q6Vec) -> wp.vec3:
    return wp.vec3(float(value[0]), float(value[1]), float(value[2])) * (1.0 / 64.0)


@wp.func
def motion_store(data: wp.array(dtype=wp.vec3), i: int, value: wp.vec3):
    data[i] = value


@wp.func
def motion_store(data: wp.array(dtype=wp.vec3h), i: int, value: wp.vec3):
    data[i] = wp.vec3h(wp.float16(value[0]), wp.float16(value[1]), wp.float16(value[2]))


@wp.func
def motion_store(data: wp.array(dtype=Q14Vec), i: int, value: wp.vec3):
    packed = wp.vec3i()
    for axis in range(3):
        packed[axis] = int(wp.clamp(wp.round(value[axis] * 16384.0), -32768.0, 32767.0))
    data[i] = Q14Vec(wp.int16(packed[0]), wp.int16(packed[1]), wp.int16(packed[2]))


@wp.func
def motion_store(data: wp.array(dtype=Q6Vec), i: int, value: wp.vec3):
    packed = wp.vec3i()
    for axis in range(3):
        packed[axis] = int(wp.clamp(wp.round(value[axis] * 64.0), -128.0, 127.0))
    data[i] = Q6Vec(wp.int8(packed[0]), wp.int8(packed[1]), wp.int8(packed[2]))


@wp.func
def velocity_load(value: Any) -> wp.vec3:
    return motion_load(value)


@wp.func
def velocity_load(value: Q14Vec) -> wp.vec3:
    return motion_load(value) * 2.0


@wp.func
def velocity_store(data: wp.array[Any], i: int, value: wp.vec3):
    motion_store(data, i, value)


@wp.func
def velocity_store(data: wp.array[Q14Vec], i: int, value: wp.vec3):
    motion_store(data, i, value * 0.5)


@wp.kernel
def integrate(
    cells: wp.array[Any], velocity: wp.array[Any], travel: wp.array[Any],
    dx: float, dt: float, axis: int, phase: int,
):
    i = wp.tid()
    p = cell_load(cells[i])
    pos = (wp.vec3(float(p[0]), float(p[1]), float(p[2])) + wp.vec3(0.5)) * dx
    v = velocity_load(velocity[i])
    r = motion_load(travel[i])
    v[axis] = wp.clamp(v[axis] + dt * acceleration(pos, phase)[axis], -dx / dt, dx / dt)
    r[axis] += (dt / dx) * v[axis]
    velocity_store(velocity, i, v)
    motion_store(travel, i, r)


@wp.kernel
def paired_contact(
    cells: wp.array[Any], velocity: wp.array[Any], travel: wp.array[Any],
    occupancy: wp.array[int], bounds: wp.vec3i, axis: int, parity: int, friction: float,
):
    i = wp.tid()
    p = cell_load(cells[i])
    direction = 1 if (p[axis] & 1) == parity else -1
    q = p
    q[axis] += direction
    if q[axis] < 0 or q[axis] >= bounds[axis]:
        return
    dest = index_of(q, bounds)
    neighbor = occupancy[dest]
    if neighbor < 0:
        return
    if neighbor == 0:
        r = motion_load(travel[i])
        if float(direction) * r[axis] >= 1.0:
            occupancy[index_of(p, bounds)] = 0
            occupancy[dest] = i + 1
            cell_store(cells, i, q)
            r[axis] -= float(direction)
            motion_store(travel, i, r)
        return
    # A filled pair belongs to its lower endpoint. The higher endpoint
    # checks occupancy only and cannot read or write the shared motion state.
    if direction < 0:
        return
    other = neighbor - 1
    a = motion_load(travel[i])
    b = motion_load(travel[other])
    if a[axis] < 1.0 and b[axis] > -1.0:
        return
    va = velocity_load(velocity[i])
    vb = velocity_load(velocity[other])
    if va[axis] <= vb[axis]:
        return
    relative = va - vb
    relative[axis] = 0.0
    tangent_speed = wp.length(relative)
    if tangent_speed > 0.0:
        impulse = relative * wp.min(0.5, friction * (va[axis] - vb[axis]) * 0.5 / tangent_speed)
        va -= impulse
        vb += impulse
    normal_speed = (va[axis] + vb[axis]) * 0.5
    va[axis] = normal_speed
    vb[axis] = normal_speed
    velocity_store(velocity, i, va)
    velocity_store(velocity, other, vb)
    if a[axis] >= 1.0:
        a[axis] = 0.0
        motion_store(travel, i, a)
    if b[axis] <= -1.0:
        b[axis] = 0.0
        motion_store(travel, other, b)


@wp.kernel
def propose_slip(
    cells: wp.array[Any], travel: wp.array[Any], occupancy: wp.array[int],
    claims: wp.array[int], choice: wp.array[int], bounds: wp.vec3i, dense_claims: bool,
):
    i = wp.tid()
    p = cell_load(cells[i])
    choice[i] = -1
    if motion_load(travel[i])[1] > -1.0 or p[1] < 2:
        return
    down = wp.vec3i(0, -1, 0)
    if occupancy[index_of(p + down, bounds)] <= 0:
        return
    first = int(wp.rand_init(i, index_of(p, bounds)) & wp.uint32(3))
    for k in range(4):
        direction = (first + k) & 3
        side = p + side_step(direction)
        dest = side + down
        if valid(dest, bounds) and occupancy[index_of(side, bounds)] == 0 and occupancy[index_of(dest, bounds)] == 0:
            choice[i] = direction
            if dense_claims:
                wp.atomic_max(claims, index_of(dest, bounds), i + 1)
            return


@wp.kernel
def confirm_slip(
    cells: wp.array[Any], occupancy: wp.array[int], choice: wp.array[int],
    winners: wp.array[wp.int8], bounds: wp.vec3i,
):
    i = wp.tid()
    direction = choice[i]
    if direction < 0:
        return
    down = wp.vec3i(0, -1, 0)
    dest = cell_load(cells[i]) + side_step(direction) + down
    # Compare proposals while occupancy remains read-only. A target has
    # four possible sources. Highest grain ID wins as with atomic_max.
    winners[i] = wp.int8(0)
    for k in range(4):
        source = dest - side_step(k) - down
        if not valid(source, bounds):
            continue
        other = occupancy[index_of(source, bounds)] - 1
        if other > i and choice[other] == k:
            return
    winners[i] = wp.int8(1)


@wp.kernel
def apply_slip(
    cells: wp.array[Any], velocity: wp.array[Any], travel: wp.array[Any],
    occupancy: wp.array[int], claims: wp.array[int], choice: wp.array[int],
    winners: wp.array[wp.int8], bounds: wp.vec3i, dense_claims: bool,
):
    i = wp.tid()
    if choice[i] < 0:
        return
    p = cell_load(cells[i])
    side = side_step(choice[i])
    dest = p + side + wp.vec3i(0, -1, 0)
    offset = index_of(dest, bounds)
    if dense_claims:
        if wp.atomic_cas(claims, offset, i + 1, 0) != i + 1:
            return
    elif winners[i] == wp.int8(0):
        return
    v = velocity_load(velocity[i])
    r = motion_load(travel[i])
    tangent = wp.vec3(float(side[0]), -1.0, float(side[2]))
    axis = 0 if side[0] != 0 else 2
    speed = wp.max(0.0, 0.5 * wp.dot(v, tangent))
    v[axis] = speed * tangent[axis]
    v[1] = -speed
    r[axis] = 0.0
    r[1] += 1.0
    occupancy[index_of(p, bounds)] = 0
    occupancy[offset] = i + 1
    cell_store(cells, i, dest)
    velocity_store(velocity, i, v)
    motion_store(travel, i, r)


@wp.kernel
def resolve(
    cells: wp.array[Any], velocity: wp.array[Any], travel: wp.array[Any],
    occupancy: wp.array[int], bounds: wp.vec3i, contacts: wp.array[wp.vec3i], friction: float,
):
    i = wp.tid()
    v = velocity_load(velocity[i])
    r = motion_load(travel[i])
    hits = contacts[i]
    for axis in range(3):
        if wp.abs(r[axis]) < 1.0:
            continue
        dest = cell_load(cells[i])
        dest[axis] += 1 if r[axis] > 0.0 else -1
        inside = valid(dest, bounds)
        if not inside or occupancy[index_of(dest, bounds)] < 0:
            if inside:
                hits[axis] += 1
            tangent = wp.vec3(v)
            tangent[axis] = 0.0
            speed = wp.length(tangent)
            if speed > 0.0:
                v -= tangent * wp.min(1.0, friction * 1.25 * wp.abs(v[axis]) / speed)
            v[axis] *= -0.25
        r[axis] = 0.0
    velocity_store(velocity, i, v)
    motion_store(travel, i, r)
    contacts[i] = hits


@wp.kernel
def position_output(
    cells: wp.array[Any], travel: wp.array[Any], output: wp.array[wp.vec3],
    dx: float, lo: wp.vec3, hi: wp.vec3,
):
    i = wp.tid()
    p = cell_load(cells[i])
    x = (wp.vec3(float(p[0]), float(p[1]), float(p[2])) + wp.vec3(0.5) + motion_load(travel[i])) * dx
    output[i] = wp.vec3(wp.clamp(x[0], lo[0], hi[0]), wp.clamp(x[1], lo[1], hi[1]), wp.clamp(x[2], lo[2], hi[2]))


def register_variants(formats):
    # Register every signature up front. This leaves JIT work outside all
    # timed intervals.
    for layout in formats:
        cells, velocity, travel = storage_types(layout)
        grid_state = {"cells": wp.array[cells], "travel": wp.array[travel]}
        grain_state = {**grid_state, "velocity": wp.array[velocity]}
        for kernel in (integrate, paired_contact, apply_slip, resolve):
            wp.overload(kernel, grain_state)
        for kernel in (propose_slip, position_output):
            wp.overload(kernel, grid_state)
        wp.overload(confirm_slip, {"cells": wp.array[cells]})


class Lattice:
    def __init__(self, initial, bounds, wall, dx, substeps, layout, speed=None):
        self.n = len(initial)
        self.dx, self.dt = dx, 1.0 / (60 * substeps)
        self.bounds = wp.vec3i(*bounds)
        self.lo = wp.vec3(0.5 * dx, 1.5 * dx, 0.5 * dx)
        self.hi = wp.vec3((wall - 0.5) * dx, (bounds[1] - 0.5) * dx, (bounds[2] - 0.5) * dx)
        cell_dtype, velocity_dtype, travel_dtype = storage_types(layout)
        self.packed = cell_dtype == wp.int32
        self.dense_claims = not layout.endswith("-gather")
        self.fixed_velocity = velocity_dtype == Q14Vec
        if self.packed:
            x, y, z = initial.T
            self.cells = wp.array(x | (y << 10) | (z << 19), dtype=wp.int32)
        else:
            self.cells = wp.array(initial, dtype=wp.vec3i)
        if speed is not None and self.fixed_velocity:
            speed = np.clip(np.rint(speed * 8192.0), -32768, 32767).astype(np.int16)
        self.velocity = wp.zeros(self.n, dtype=velocity_dtype) if speed is None else wp.array(speed, dtype=velocity_dtype)
        self.travel = wp.zeros(self.n, dtype=travel_dtype)
        self.output = wp.zeros(self.n, dtype=wp.vec3)
        self.contacts = wp.zeros(self.n, dtype=wp.vec3i)
        occupancy = np.zeros((bounds[1], bounds[2], bounds[0]), np.int32)
        occupancy[0, :, :] = -1
        occupancy[:, :, wall] = -1
        occupancy[initial[:, 1], initial[:, 2], initial[:, 0]] = np.arange(1, self.n + 1, dtype=np.int32)
        self.occupancy = wp.array(occupancy.ravel(), dtype=int)
        self.claims = wp.zeros(self.occupancy.shape if self.dense_claims else 1, dtype=int)
        self.choice = wp.empty(self.n, dtype=int)
        self.winners = wp.empty(1 if self.dense_claims else self.n, dtype=wp.int8)

    def step(self, phase):
        state = [self.cells, self.velocity, self.travel]
        for axis in (1, 0, 2):
            wp.launch(integrate, self.n, inputs=[*state, self.dx, self.dt, axis, phase])
            for parity in (0, 1):
                wp.launch(paired_contact, self.n, inputs=[*state, self.occupancy, self.bounds, axis, parity, 0.45])
            if axis == 1:
                wp.launch(propose_slip, self.n, inputs=[
                    self.cells, self.travel, self.occupancy, self.claims, self.choice, self.bounds, self.dense_claims,
                ])
                if not self.dense_claims:
                    wp.launch(confirm_slip, self.n, inputs=[
                        self.cells, self.occupancy, self.choice, self.winners, self.bounds,
                    ])
                wp.launch(apply_slip, self.n, inputs=[
                    *state, self.occupancy, self.claims, self.choice, self.winners, self.bounds, self.dense_claims,
                ])
        wp.launch(resolve, self.n, inputs=[*state, self.occupancy, self.bounds, self.contacts, 0.45])

    def pack(self):
        wp.launch(position_output, self.n, inputs=[self.cells, self.travel, self.output, self.dx, self.lo, self.hi])

    def state_bytes(self):
        arrays = [self.cells, self.velocity, self.travel, self.output, self.contacts,
                  self.occupancy, self.claims, self.choice, self.winners]
        return {key: val.capacity for key, val in zip(
            ("positions", "velocity", "travel", "output", "contacts", "occupancy", "claims", "choice", "winners"), arrays
        )}

    def read_velocity(self):
        result = self.velocity.numpy().astype(np.float32)
        if self.fixed_velocity:
            result *= 1.0 / 8192.0
        return result

    def validate(self, wall):
        positions = self.cells.numpy()
        if self.packed:
            x, y, z = positions & 1023, (positions >> 10) & 511, (positions >> 19) & 1023
        else:
            x, y, z = positions.T
        state = self.occupancy.numpy().reshape(self.bounds[1], self.bounds[2], self.bounds[0])
        ids = state[y, z, x]
        digest = hashlib.sha256()
        for array in (np.column_stack((x, y, z)), self.velocity.numpy(), self.travel.numpy()):
            digest.update(array.tobytes())
        return {
            "id_errors": int(np.count_nonzero(ids != np.arange(1, self.n + 1))),
            "occupancy_error": int(np.count_nonzero(state > 0)) - self.n,
            "through_wall": int(np.count_nonzero(x >= wall)),
            "uncleared_claims": int(np.count_nonzero(self.claims.numpy())),
            "changed_solids": int(np.count_nonzero(state[0] != -1) + np.count_nonzero(state[:, :, wall] != -1)),
            "hash": digest.hexdigest(),
        }


def isolated_checks(layout, substeps):
    # Compare tangent momentum and restitution without contact ordering noise.
    initial = np.array([[20, 25, 20], [21, 25, 20]], np.int32)
    speed = np.array([[2.4, 0, 1.2], [0, 0, 0]], np.float32)
    collision = Lattice(initial, (48, 48, 48), 44, 0.03, substeps, layout, speed)
    for _ in range(int(np.ceil(0.03 / (2.4 * collision.dt))) + 1):
        collision.step(2)
    velocities = collision.read_velocity()[:, [0, 2]]
    contact_error = float(np.max(np.abs(velocities - [[1.2, .66], [1.2, .54]])))
    momentum_error = float(np.max(np.abs(velocities.sum(axis=0) - [2.4, 1.2])))

    # Use 1 mm/s over 0.25 seconds, before floor contact. Coarse travel
    # quantization can zero every displacement at this speed.
    initial = np.array([[12, 40, 12]], np.int32)
    speed = np.array([[0.001, 0, 0]], np.float32)
    drift = Lattice(initial, (48, 48, 48), 44, 0.03, substeps, layout, speed)
    for _ in range(15 * substeps):
        drift.step(2)
    drift.pack()
    motion_mm = float((drift.output.numpy()[0, 0] - 12.5 * 0.03) * 1000)

    initial = np.array([[36, 40, 12]], np.int32)
    speed = np.array([[2.4, 0, 0]], np.float32)
    bounce = Lattice(initial, (48, 48, 48), 44, 0.03, substeps, layout, speed)
    for _ in range(12 * substeps):
        bounce.step(2)
    return {
        "pair_max_component_error_m_s": contact_error,
        "pair_total_momentum_error_per_kg_m_s": momentum_error,
        "slow_drift_mm": motion_mm,
        "slow_drift_expected_mm": 0.25,
        "wall_rebound_m_s": float(bounce.read_velocity()[0, 0]),
        "wall_hits": int(bounce.contacts.numpy()[0, 0]),
    }


def run_scene(initial, bounds, wall, dx, substeps, layout, frames, profile):
    sim = Lattice(initial, bounds, wall, dx, substeps, layout)
    graphs = []
    for phase in range(3):
        # The isolated checks warm the kernels. Graph capture is untimed.
        with wp.ScopedCapture(force_module_load=True) as capture:
            for _ in range(substeps):
                sim.step(phase)
            sim.pack()
        graphs.append(capture.graph)
    times = []
    start_event = wp.Event(enable_timing=True)
    end_event = wp.Event(enable_timing=True)
    for frame in range(frames):
        phase = min(frame // 12, 2)
        wp.record_event(start_event)
        wp.capture_launch(graphs[phase])
        wp.record_event(end_event)
        wp.synchronize_event(end_event)
        if frame >= 8:
            times.append(wp.get_event_elapsed_time(start_event, end_event))
    result = sim.validate(wall)
    position = sim.output.numpy()
    speed = sim.read_velocity()
    solid_contacts = sim.contacts.numpy().sum(axis=0, dtype=np.int64).tolist()
    bucket = np.floor(position / 0.12).astype(np.int32)
    bucket_shape = tuple(int(np.ceil(size * dx / 0.12)) + 1 for size in bounds)
    histogram = np.bincount(np.ravel_multi_index(bucket.T, bucket_shape), minlength=int(np.prod(bucket_shape)))
    stage_ms = {}
    if profile:
        wp.timing_begin(wp.TIMING_KERNEL)
        for _ in range(substeps):
            sim.step(2)
        sim.pack()
        for event in wp.timing_end():
            name = event.name.removeprefix("forward kernel ").rsplit("_", 1)[0]
            stage_ms[name] = stage_ms.get(name, 0.0) + event.elapsed
    result.update({
        "format": layout, "grains": sim.n, "dx_m": dx, "substeps": substeps, "ms_per_60hz": percentile(times),
        "bytes": sim.state_bytes(), "finite": bool(np.isfinite(position).all() and np.isfinite(speed).all()),
        "com_m": position.mean(axis=0, dtype=np.float64).tolist(),
        "rms_speed_m_s": float(np.sqrt(np.mean(np.sum(speed * speed, axis=1), dtype=np.float64))),
        "solid_contacts": solid_contacts,
        "stages_ms": stage_ms,
    })
    return result, histogram


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", choices=("sparse", "dense", "local"))
    parser.add_argument("--scale", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument("--formats", nargs="+", default=FORMATS, choices=FORMATS)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or args.frames < 25 or (args.scene == "local" and args.scale != 3) or args.formats[0] != "fp32":
        parser.error("List fp32 first, run at least 25 frames, and use scale 3 for the local scene.")
    wp.set_device("cuda:0")
    register_variants(args.formats)
    initial, bounds, wall, dx = fixture(args.scene, args.scale)
    substeps = 2 * args.scale
    checks = {layout: isolated_checks(layout, substeps) for layout in args.formats}
    controls = None
    hashes = {}
    passed = True
    for repeat in range(args.repeats):
        # Alternate order to lessen GPU clock and temperature bias.
        order = args.formats if repeat % 2 == 0 else list(reversed(args.formats))
        for layout in order:
            result, histogram = run_scene(initial, bounds, wall, dx, substeps, layout, args.frames, args.profile)
            if controls is None:
                controls = result, histogram
            control, control_hist = controls
            repeat_hash_equal = hashes.setdefault(layout, result["hash"]) == result["hash"]
            result.update({
                "repeat": repeat + 1, "scene": args.scene, "isolation": checks[layout],
                "com_delta_cm": float(np.linalg.norm(np.asarray(result["com_m"]) - control["com_m"]) * 100),
                "density_tv_12cm": float(np.abs(histogram - control_hist).sum() / (2 * len(initial))),
                "repeat_hash_equal": repeat_hash_equal,
                "exact_fp32_hash": result["hash"] == control["hash"],
            })
            print(json.dumps(result), flush=True)
            passed = passed and repeat_hash_equal and result["finite"] and not any(
                result[key] for key in ("id_errors", "occupancy_error", "through_wall", "uncleared_claims", "changed_solids")
            )
            if layout in ("packed-cells", "packed-cells-gather"):
                passed = passed and result["exact_fp32_hash"]
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
