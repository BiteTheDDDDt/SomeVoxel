"""Matched sand fixtures for CUDA lattice and APIC/MPM experiments."""
import argparse
import hashlib
import json
import time

import numpy as np
import warp as wp

from gpu_mpm import base_node, p2g, shape_weights


@wp.func
def acceleration(p: wp.vec3, phase: int) -> wp.vec3:
    gravity = wp.vec3(0.0, -9.81, 0.0)
    center = wp.vec3(6.5, 3.8, 2.25)
    if phase == 1:
        center[2] = 4.65
    delta = center - p
    if phase < 2 and wp.dot(delta, delta) <= 30.25:
        return gravity + 9.0 * delta
    return gravity


@wp.func
def index_of(p: wp.vec3i, bounds: wp.vec3i) -> int:
    return p[0] + bounds[0] * (p[2] + bounds[2] * p[1])


@wp.func
def valid(p: wp.vec3i, bounds: wp.vec3i) -> bool:
    return p[0] >= 0 and p[0] < bounds[0] and p[1] >= 0 and p[1] < bounds[1] and p[2] >= 0 and p[2] < bounds[2]


@wp.kernel
def integrate(
    cells: wp.array(dtype=wp.vec3i), velocity: wp.array(dtype=wp.vec3),
    travel: wp.array(dtype=wp.vec3), dx: float, dt: float, axis: int, phase: int,
):
    i = wp.tid()
    p = cells[i]
    x = (wp.vec3(float(p[0]), float(p[1]), float(p[2])) + wp.vec3(0.5)) * dx
    v = velocity[i]
    r = travel[i]
    v[axis] = wp.clamp(v[axis] + dt * acceleration(x, phase)[axis], -dx / dt, dx / dt)
    r[axis] += dt * v[axis]
    velocity[i] = v
    travel[i] = r


@wp.kernel
def pairs(
    cells: wp.array(dtype=wp.vec3i), travel: wp.array(dtype=wp.vec3),
    occupancy: wp.array(dtype=int), bounds: wp.vec3i, dx: float, axis: int, parity: int,
):
    i = wp.tid()
    r = travel[i]
    if wp.abs(r[axis]) < dx:
        return
    source = cells[i]
    direction = 1
    if r[axis] < 0.0:
        direction = -1
    dest = source
    dest[axis] += direction
    if (wp.min(source[axis], dest[axis]) & 1) != parity or not valid(dest, bounds):
        return
    offset = index_of(dest, bounds)
    # Each phase owns disjoint two-cell pairs. An occupied pair cannot write
    # either cell. A pair with one grain has exactly one writer.
    if occupancy[offset] == 0:
        occupancy[index_of(source, bounds)] = 0
        occupancy[offset] = i + 1
        cells[i] = dest
        r[axis] -= float(direction) * dx
        travel[i] = r


@wp.func
def transfer_contact(
    i: int, other: int, axis: int, dx: float, friction: float,
    velocity: wp.array(dtype=wp.vec3), travel: wp.array(dtype=wp.vec3),
):
    a, b = travel[i], travel[other]
    if a[axis] < dx and b[axis] > -dx:
        return
    va = wp.vec3(velocity[i])
    vb = wp.vec3(velocity[other])
    if va[axis] <= vb[axis]:
        return
    # Equal masses, normal restitution zero. Each phase owns its pairs.
    relative = va - vb
    relative[axis] = 0.0
    tangent_speed = wp.length(relative)
    if tangent_speed > 0.0:
        tangent_impulse = relative * wp.min(0.5, friction * (va[axis] - vb[axis]) * 0.5 / tangent_speed)
        va -= tangent_impulse
        vb += tangent_impulse
    speed = 0.5 * (va[axis] + vb[axis])
    va[axis] = speed
    vb[axis] = speed
    velocity[i] = va
    velocity[other] = vb
    if a[axis] >= dx:
        a[axis] = 0.0
        travel[i] = a
    if b[axis] <= -dx:
        b[axis] = 0.0
        travel[other] = b


@wp.kernel
def pair_impulses(
    cells: wp.array(dtype=wp.vec3i), velocity: wp.array(dtype=wp.vec3),
    travel: wp.array(dtype=wp.vec3), occupancy: wp.array(dtype=int),
    bounds: wp.vec3i, dx: float, axis: int, parity: int, friction: float,
):
    i = wp.tid()
    low = cells[i]
    if (low[axis] & 1) != parity or low[axis] + 1 >= bounds[axis]:
        return
    high = low
    high[axis] += 1
    other = occupancy[index_of(high, bounds)] - 1
    if other < 0:
        return
    transfer_contact(i, other, axis, dx, friction, velocity, travel)


@wp.kernel
def contact_pairs(
    cells: wp.array(dtype=wp.vec3i), velocity: wp.array(dtype=wp.vec3),
    travel: wp.array(dtype=wp.vec3), occupancy: wp.array(dtype=int),
    bounds: wp.vec3i, dx: float, axis: int, parity: int, friction: float,
):
    i = wp.tid()
    p = cells[i]
    direction = 1 if (p[axis] & 1) == parity else -1
    q = p
    q[axis] += direction
    if q[axis] < 0 or q[axis] >= bounds[axis]:
        return
    destination = index_of(q, bounds)
    neighbor = occupancy[destination]
    if neighbor < 0:
        return
    if neighbor == 0:
        r = travel[i]
        if float(direction) * r[axis] >= dx:
            occupancy[index_of(p, bounds)] = 0
            occupancy[destination] = i + 1
            cells[i] = q
            r[axis] -= float(direction) * dx
            travel[i] = r
        return
    # If both cells are occupied, only the lower endpoint reads or writes
    # velocity and travel. Neither endpoint can change occupancy.
    if direction < 0:
        return
    other = neighbor - 1
    transfer_contact(i, other, axis, dx, friction, velocity, travel)


@wp.func
def side_step(direction: int) -> wp.vec3i:
    if direction == 0:
        return wp.vec3i(1, 0, 0)
    if direction == 1:
        return wp.vec3i(-1, 0, 0)
    if direction == 2:
        return wp.vec3i(0, 0, 1)
    return wp.vec3i(0, 0, -1)


@wp.kernel
def propose_slip(
    cells: wp.array(dtype=wp.vec3i), travel: wp.array(dtype=wp.vec3),
    occupancy: wp.array(dtype=int), claims: wp.array(dtype=int),
    choice: wp.array(dtype=int), bounds: wp.vec3i, dx: float,
):
    i = wp.tid()
    p = cells[i]
    choice[i] = -1
    if travel[i][1] > -dx or p[1] < 2:
        return
    down = wp.vec3i(0, -1, 0)
    if occupancy[index_of(p + down, bounds)] <= 0:
        return
    # Vary equal-cost choices spatially to avoid a preferred direction.
    first = wp.rand_init(i, index_of(p, bounds)) & wp.uint32(3)
    for k in range(4):
        direction = (int(first) + k) & 3
        side = p + side_step(direction)
        dest = side + down
        if valid(dest, bounds) and occupancy[index_of(side, bounds)] == 0 and occupancy[index_of(dest, bounds)] == 0:
            choice[i] = direction
            wp.atomic_max(claims, index_of(dest, bounds), i + 1)
            return


@wp.kernel
def apply_slip(
    cells: wp.array(dtype=wp.vec3i), velocity: wp.array(dtype=wp.vec3),
    travel: wp.array(dtype=wp.vec3), occupancy: wp.array(dtype=int),
    claims: wp.array(dtype=int), choice: wp.array(dtype=int), bounds: wp.vec3i, dx: float,
):
    i = wp.tid()
    if choice[i] < 0:
        return
    p = cells[i]
    side = side_step(choice[i])
    dest = p + side + wp.vec3i(0, -1, 0)
    offset = index_of(dest, bounds)
    if wp.atomic_cas(claims, offset, i + 1, 0) != i + 1:
        return
    v = velocity[i]
    r = travel[i]
    tangent = wp.vec3(float(side[0]), -1.0, float(side[2]))
    # Project the downward velocity onto the open downhill tangent.
    # Preserve velocity along the third axis.
    axis = 0 if side[0] != 0 else 2
    speed = wp.max(0.0, 0.5 * wp.dot(v, tangent))
    v[axis] = speed * tangent[axis]
    v[1] = -speed
    r[axis] = 0.0
    r[1] += dx
    occupancy[index_of(p, bounds)] = 0
    occupancy[offset] = i + 1
    cells[i] = dest
    velocity[i] = v
    travel[i] = r


@wp.kernel
def resolve(
    cells: wp.array(dtype=wp.vec3i), velocity: wp.array(dtype=wp.vec3),
    travel: wp.array(dtype=wp.vec3), occupancy: wp.array(dtype=int),
    bounds: wp.vec3i, dx: float, contacts: wp.array(dtype=wp.vec3i), keep_grain_velocity: bool,
    friction: float,
):
    i = wp.tid()
    v = wp.vec3(velocity[i])
    r = wp.vec3(travel[i])
    hits = contacts[i]
    for axis in range(3):
        if wp.abs(r[axis]) >= dx:
            dest = cells[i]
            direction = 1
            if r[axis] < 0.0:
                direction = -1
            dest[axis] += direction
            bounce = 0.0
            if not valid(dest, bounds):
                bounce = 0.25
            else:
                value = occupancy[index_of(dest, bounds)]
                if value < 0:
                    bounce = 0.25
                    hits[axis] += 1
                elif keep_grain_velocity:
                    # Paired impulses already handled grain contacts. A request
                    # that missed its parity phase can be retried next step.
                    bounce = -1.0
            if bounce > 0.0 and friction > 0.0:
                tangent = wp.vec3(v)
                tangent[axis] = 0.0
                speed = wp.length(tangent)
                if speed > 0.0:
                    v -= tangent * wp.min(1.0, friction * (1.0 + bounce) * wp.abs(v[axis]) / speed)
            v[axis] *= -bounce
            r[axis] = 0.0
    velocity[i] = v
    travel[i] = r
    contacts[i] = hits


@wp.kernel
def position_output(
    cells: wp.array(dtype=wp.vec3i), travel: wp.array(dtype=wp.vec3),
    output: wp.array(dtype=wp.vec3), dx: float, lo: wp.vec3, hi: wp.vec3,
):
    i = wp.tid()
    p = cells[i]
    x = (wp.vec3(float(p[0]), float(p[1]), float(p[2])) + wp.vec3(0.5)) * dx + travel[i]
    output[i] = wp.vec3(wp.clamp(x[0], lo[0], hi[0]), wp.clamp(x[1], lo[1], hi[1]), wp.clamp(x[2], lo[2], hi[2]))


@wp.kernel
def mpm_grid(
    mass: wp.array3d(dtype=float), grid: wp.array3d(dtype=wp.vec3), dx: float, dt: float, phase: int,
):
    u, q, w = wp.tid()
    value = mass[u, q, w]
    if value > 0.0:
        p = (wp.vec3(float(u), float(q), float(w)) - wp.vec3(2.0)) * dx
        grid[u, q, w] = grid[u, q, w] / value + dt * acceleration(p, phase)


@wp.kernel
def mpm_g2p(
    x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), C: wp.array(dtype=wp.mat33),
    grid: wp.array3d(dtype=wp.vec3), dx: float, dt: float, lo: wp.vec3, hi: wp.vec3,
    contacts: wp.array(dtype=wp.vec3i),
):
    i = wp.tid()
    xp = x[i] / dx
    base = base_node(xp)
    f = xp - wp.vec3(float(base[0]), float(base[1]), float(base[2]))
    weights = shape_weights(f)
    speed = wp.vec3(0.0)
    affine = wp.mat33(0.0)
    for u in range(3):
        for q in range(3):
            for w in range(3):
                delta = wp.vec3(float(u), float(q), float(w)) - f
                gv = grid[base[0] + u, base[1] + q, base[2] + w]
                weight = weights[u, 0] * weights[q, 1] * weights[w, 2]
                speed += gv * weight
                affine += wp.outer(gv, delta) * (weight * 4.0 / dx)
    position = x[i] + dt * speed
    hits = contacts[i]
    for axis in range(3):
        if position[axis] < lo[axis] or position[axis] > hi[axis]:
            hits[axis] += 1
            position[axis] = wp.clamp(position[axis], lo[axis], hi[axis])
            speed[axis] *= -0.25
            # Do not transfer affine normal velocity through the collider.
            affine[axis, 0] = 0.0
            affine[axis, 1] = 0.0
            affine[axis, 2] = 0.0
    x[i] = position
    v[i] = speed
    C[i] = affine
    contacts[i] = hits


def fixture(size, scale):
    if size == "local":
        # Approximately 0.84 M grains in a 1.0 x 0.6 x 1.4 m volume
        # at 1 cm pitch, in the same world as the other fixtures.
        y, z, x = np.mgrid[25:85, 130:270, 205:305]
        cells = np.column_stack((x.ravel(), y.ravel(), z.ravel())).astype(np.int32)
        return cells, (864, 432, 672), 471, 0.01
    dense = size == "dense"
    y, z, x = np.mgrid[
        5 * scale:(63 if dense else 35) * scale,
        28 * scale:(176 if dense else 98) * scale,
        30 * scale:(153 if dense else 82) * scale,
    ]
    cells = np.column_stack((x.ravel(), y.ravel(), z.ravel())).astype(np.int32)
    return cells, (288 * scale, 144 * scale, 224 * scale), 157 * scale, 0.03 / scale


class Lattice:
    def __init__(self, initial, bounds, wall, dx, substeps, contact, slip, fused, friction, speed=None):
        self.n = len(initial)
        self.contact, self.slip, self.fused, self.friction = contact, slip, fused, friction
        self.lo = wp.vec3(0.5 * dx, 1.5 * dx, 0.5 * dx)
        self.hi = wp.vec3((wall - 0.5) * dx, (bounds[1] - 0.5) * dx, (bounds[2] - 0.5) * dx)
        self.dx, self.dt = dx, 1 / (60 * substeps)
        self.bounds = wp.vec3i(*bounds)
        self.cells = wp.array(initial, dtype=wp.vec3i)
        self.velocity = wp.zeros(self.n, dtype=wp.vec3) if speed is None else wp.array(speed, dtype=wp.vec3)
        self.travel = wp.zeros(self.n, dtype=wp.vec3)
        self.output = wp.zeros(self.n, dtype=wp.vec3)
        self.contacts = wp.zeros(self.n, dtype=wp.vec3i)
        occupancy = np.zeros((bounds[1], bounds[2], bounds[0]), np.int32)
        occupancy[0, :, :] = -1
        occupancy[:, :, wall] = -1
        occupancy[initial[:, 1], initial[:, 2], initial[:, 0]] = np.arange(1, self.n + 1, dtype=np.int32)
        self.occupancy = wp.array(occupancy.ravel(), dtype=int)
        if slip:
            self.claims = wp.zeros(self.occupancy.shape, dtype=int)
            self.choice = wp.empty(self.n, dtype=int)

    def step(self, phase):
        for axis in (1, 0, 2):
            wp.launch(integrate, self.n, inputs=[self.cells, self.velocity, self.travel, self.dx, self.dt, axis, phase])
            for parity in (0, 1):
                if self.fused:
                    wp.launch(contact_pairs, self.n, inputs=[
                        self.cells, self.velocity, self.travel, self.occupancy, self.bounds, self.dx, axis, parity, self.friction,
                    ])
                else:
                    wp.launch(pairs, self.n, inputs=[self.cells, self.travel, self.occupancy, self.bounds, self.dx, axis, parity])
            if self.slip and axis == 1:
                wp.launch(propose_slip, self.n, inputs=[
                    self.cells, self.travel, self.occupancy, self.claims, self.choice, self.bounds, self.dx,
                ])
                wp.launch(apply_slip, self.n, inputs=[
                    self.cells, self.velocity, self.travel, self.occupancy,
                    self.claims, self.choice, self.bounds, self.dx,
                ])
            if self.contact and not self.fused:
                for parity in (0, 1):
                    wp.launch(pair_impulses, self.n, inputs=[
                        self.cells, self.velocity, self.travel, self.occupancy, self.bounds, self.dx, axis, parity, self.friction,
                    ])
        wp.launch(resolve, self.n, inputs=[self.cells, self.velocity, self.travel, self.occupancy, self.bounds, self.dx, self.contacts, self.contact, self.friction])

    def pack(self):
        wp.launch(position_output, self.n, inputs=[self.cells, self.travel, self.output, self.dx, self.lo, self.hi])

    def validate(self, wall):
        cells = self.cells.numpy()
        state = self.occupancy.numpy().reshape(self.bounds[1], self.bounds[2], self.bounds[0])
        ids = state[cells[:, 1], cells[:, 2], cells[:, 0]]
        digest = hashlib.sha256()
        for array in (cells, self.velocity.numpy(), self.travel.numpy()):
            digest.update(array.tobytes())
        return {
            "missing_or_duplicate": int(np.count_nonzero(ids != np.arange(1, self.n + 1))),
            "extra_occupancy": int(np.count_nonzero(state > 0)) - self.n,
            "through_wall": int(np.count_nonzero(cells[:, 0] >= wall)),
            "uncleared_claims": int(np.count_nonzero(self.claims.numpy())) if self.slip else 0,
            "changed_solids": int(np.count_nonzero(state[0] != -1) + np.count_nonzero(state[:, :, wall] != -1)),
            "hash": digest.hexdigest(),
        }


class Mpm:
    def __init__(self, initial, bounds, wall, dx, substeps, grid_factor, speed=None):
        self.n, self.dt = len(initial), 1 / (60 * substeps)
        self.dx = dx * grid_factor
        self.offset = 2 * self.dx
        self.volume = dx ** 3
        self.particle_mass = 1600.0 * self.volume
        self.shape = tuple(int(np.ceil(length * dx / self.dx)) + 5 for length in (wall, bounds[1], bounds[2]))
        self.output = wp.array((initial + 0.5) * dx + self.offset, dtype=wp.vec3)
        self.velocity = wp.zeros(self.n, dtype=wp.vec3) if speed is None else wp.array(speed, dtype=wp.vec3)
        self.contacts = wp.zeros(self.n, dtype=wp.vec3i)
        self.affine = wp.zeros(self.n, dtype=wp.mat33)
        self.compaction = wp.ones(self.n, dtype=float)
        self.mass = wp.zeros(self.shape, dtype=float)
        self.grid = wp.zeros(self.shape, dtype=wp.vec3)
        self.lo = wp.vec3(0.5 * dx + self.offset, 1.5 * dx + self.offset, 0.5 * dx + self.offset)
        self.hi = wp.vec3((wall - 0.5) * dx + self.offset, (bounds[1] - 0.5) * dx + self.offset, (bounds[2] - 0.5) * dx + self.offset)

    def step(self, phase):
        self.mass.zero_()
        self.grid.zero_()
        wp.launch(p2g, self.n, inputs=[
            self.output, self.velocity, self.affine, self.compaction,
            self.mass, self.grid, self.dx, self.dt, self.particle_mass, self.volume,
        ])
        wp.launch(mpm_grid, self.shape, inputs=[self.mass, self.grid, self.dx, self.dt, phase])
        wp.launch(mpm_g2p, self.n, inputs=[
            self.output, self.velocity, self.affine, self.grid, self.dx, self.dt, self.lo, self.hi, self.contacts,
        ])

    def pack(self):
        pass

    def validate(self, wall):
        return {
            "mass_error": abs(self.mass.numpy().sum(dtype=np.float64) / (self.n * self.particle_mass) - 1),
            "mean_volume_ratio": float(self.compaction.numpy().mean()),
            "through_wall": int(np.count_nonzero(self.output.numpy()[:, 0] > self.hi[0])),
            "grid": self.shape,
        }


def snapshot(sim, wall, dx):
    x = sim.output.numpy() - getattr(sim, "offset", 0.0)
    v = sim.velocity.numpy()
    wall_layer = x[:, 0] >= (wall - 1) * dx
    floor_layer = x[:, 1] < 2 * dx
    return {
        "position": x.mean(axis=0, dtype=np.float64).tolist(),
        "velocity": v.mean(axis=0, dtype=np.float64).tolist(),
        "wall_layer": int(wall_layer.sum()), "floor_layer": int(floor_layer.sum()),
        "rebounding_at_wall": int(np.count_nonzero(wall_layer & (v[:, 0] < -0.05))),
        "finite": bool(np.isfinite(x).all() and np.isfinite(v).all()),
    }


def ballistic(factory, substeps, scale=1, speed=2.4):
    dx = 0.03 / scale
    wall = 157 * scale
    # Isolated grain with no field. Start farther from the wall for a
    # fast impact so the first checkpoint is in free flight.
    cells_from_wall = (8 if speed < 3.0 else 48) * scale
    start_cell = wall - cells_from_wall
    initial = np.array([[start_cell, 50 * scale, 50 * scale]], np.int32)
    sim = factory(initial, (wall + 2, 100 * scale, 100 * scale), wall, dx, substeps,
                  speed=np.array([[speed, 0.0, 0.0]], np.float32))
    checkpoints = []
    for frame in range(15):
        for _ in range(substeps):
            sim.step(2)
        if frame in (2, 11, 14):
            sim.pack()
            checkpoints.append(snapshot(sim, wall, dx))
    x, y, _ = checkpoints[0]["position"]
    vx = checkpoints[1]["velocity"][0]
    expected_x = (start_cell + 0.5) * dx + speed * 0.05
    expected_y = (50 * scale + 0.5) * dx - 0.5 * 9.81 * 0.05 ** 2
    error = max(abs(x - expected_x), abs(y - expected_y))
    valid_run = error < 0.005 and abs(vx + 0.25 * speed) < 0.005 and sim.contacts.numpy()[0, 0] == 1
    return {"initial_vx_m_s": speed, "pre_contact_error_m": error, "rebound_vx_m_s": vx, "passed": bool(valid_run)}


def pair_test(factory, substeps, friction):
    sim = factory(np.array([[20, 25, 20], [21, 25, 20]], np.int32), (48, 48, 48), 44, 0.03, substeps,
                  speed=np.array([[2.4, 0, 1.2], [0, 0, 0]], np.float32))
    steps = int(np.ceil(0.03 / (2.4 * sim.dt))) + 1
    for _ in range(steps):
        sim.step(2)
    v = sim.velocity.numpy()
    dv = min(0.6, friction * 1.2)
    return {
        "normal_velocities": v[:, 0].tolist(),
        "tangent_velocities": v[:, 2].tolist(),
        "passed": bool(np.allclose(v[:, (0, 2)], [[1.2, 1.2 - dv], [1.2, dv]], atol=1e-5)),
    }


def pile_test(factory, substeps):
    dx = 0.03
    y, z, x = np.mgrid[8:52, 58:70, 58:70]
    initial = np.column_stack((x.ravel(), y.ravel(), z.ravel())).astype(np.int32)
    sim = factory(initial, (128, 64, 128), 127, dx, substeps)
    with wp.ScopedCapture() as capture:
        for _ in range(substeps):
            sim.step(2)
    for _ in range(120):
        wp.capture_launch(capture.graph)
    cells = sim.cells.numpy()
    spread = np.ptp(cells, axis=0) + 1
    # Percentiles describe the heap without letting a few ejecta determine
    # its diameter. Report the full bounds separately.
    core = np.percentile(cells, 95, axis=0) - np.percentile(cells, 5, axis=0) + 1
    result = sim.validate(127)
    errors = sum(abs(result[key]) for key in ("missing_or_duplicate", "extra_occupancy", "through_wall", "changed_solids", "uncleared_claims"))
    return {
        "initial_cells_xyz": [12, 44, 12],
        "final_cells_xyz": spread.tolist(),
        "core_cells_xyz": core.tolist(),
        "passed": bool(errors == 0 and spread[1] < 44 and spread[0] > 12 and spread[2] > 12),
    }


def percentile(values):
    p50, p95 = np.percentile(values, [50, 95])
    return {"p50": float(p50), "p95": float(p95)}


def run(args):
    wp.set_device("cuda:0")
    def factory(*values, **options):
        if args.solver == "lattice":
            return Lattice(*values, contact=args.contact, slip=args.slip, fused=args.fused, friction=args.friction, **options)
        return Mpm(*values, grid_factor=args.grid_factor, **options)
    ballistic_result = ballistic(factory, args.substeps, args.scale)
    high_speed = ballistic(factory, args.substeps, args.scale, speed=12.0) if args.substeps >= 7 * args.scale else None
    pair_result = pair_test(factory, args.substeps, args.friction) if args.contact and args.scale == 1 else None
    pile_result = pile_test(factory, args.substeps) if args.slip and args.scale == 1 else None
    initial, bounds, wall, dx = fixture(args.size, args.scale)
    sim = factory(initial, bounds, wall, dx, args.substeps)
    del initial
    # The isolated impact test has already compiled the kernels. Capturing
    # the graphs before the timed interval preserves the initial state.
    graphs = []
    for phase in range(3):
        with wp.ScopedCapture(force_module_load=True) as capture:
            for _ in range(args.substeps):
                sim.step(phase)
            sim.pack()
        graphs.append(capture.graph)
    sim.pack()
    snapshots = [snapshot(sim, wall, dx)]
    device_times = []
    fence_times = []
    copy_times = []
    combined_times = []
    start_event = wp.Event(enable_timing=True)
    end_event = wp.Event(enable_timing=True)
    for frame in range(72):
        phase = 2 if args.passive else min(frame // 12, 2)
        start = time.perf_counter()
        wp.record_event(start_event)
        wp.capture_launch(graphs[phase])
        wp.record_event(end_event)
        wp.synchronize_event(end_event)
        done = time.perf_counter()
        if args.readback:
            sim.output.numpy()
        end = time.perf_counter()
        if frame >= 8:
            device_times.append(wp.get_event_elapsed_time(start_event, end_event))
            fence_times.append((done - start) * 1000)
            if args.readback:
                copy_times.append((end - done) * 1000)
                combined_times.append((end - start) * 1000)
        if frame + 1 in (12, 18, 24, 36, 54, 72):
            snapshots.append(snapshot(sim, wall, dx))
    result = sim.validate(wall)
    hits = sim.contacts.numpy()
    stage_ms = {}
    if args.profile:
        # An independent launch after the measurement interval keeps detailed
        # timing instrumentation out of the graph and its P95.
        wp.timing_begin(wp.TIMING_KERNEL | wp.TIMING_KERNEL_BUILTIN | wp.TIMING_MEMSET)
        for _ in range(args.substeps):
            sim.step(2)
        sim.pack()
        for item in wp.timing_end():
            stage_ms[item.name] = stage_ms.get(item.name, 0.0) + item.elapsed
    result.update({
        "solver": args.solver, "fixture": args.size, "passive": args.passive, "particles": sim.n, "cell_m": dx,
        "substeps": args.substeps, "device_ms": percentile(device_times), "fenced_ms": percentile(fence_times),
        "readback_ms": percentile(copy_times) if args.readback else None,
        "with_readback_ms": percentile(combined_times) if args.readback else None,
        "collision_events_xyz": hits.sum(axis=0, dtype=np.int64).tolist(),
        "ballistic": ballistic_result, "fast_impact": high_speed, "checkpoints": snapshots,
        "contact": args.contact, "slip": args.slip, "fused": args.fused, "friction": args.friction, "stage_ms": stage_ms, "pair": pair_result, "pile": pile_result,
    })
    print(json.dumps(result), flush=True)
    checks = (
        ballistic_result["passed"] and (high_speed is None or high_speed["passed"])
        and all(s["finite"] for s in snapshots)
        and not any(result.get(key, 0) for key in ("missing_or_duplicate", "extra_occupancy", "through_wall", "changed_solids", "uncleared_claims"))
        and (pair_result is None or pair_result["passed"])
        and (pile_result is None or pile_result["passed"])
        and result.get("mass_error", 0.0) < 1e-5 and hits[:, 1].sum() > 0
        and (args.passive or hits[:, 0].sum() > 0)
    )
    return bool(checks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("solver", choices=("lattice", "mpm"))
    parser.add_argument("size", choices=("sparse", "dense", "local"))
    parser.add_argument("--scale", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--substeps", type=int, default=2)
    parser.add_argument("--grid-factor", type=float, default=2.0)
    parser.add_argument("--readback", action="store_true")
    parser.add_argument("--contact", action="store_true")
    parser.add_argument("--slip", action="store_true")
    parser.add_argument("--friction", type=float, default=0.0)
    parser.add_argument("--fused", action="store_true", help="Resolve paired impulse and transport in one dispatch.")
    parser.add_argument("--profile", action="store_true", help="Time individual kernels for one extra frame after the run.")
    parser.add_argument("--passive", action="store_true")
    args = parser.parse_args()
    if args.size == "local" and (args.scale != 3 or args.solver != "lattice"):
        parser.error("The local scene uses the lattice solver at scale three.")
    if args.substeps < 2 * args.scale or not np.isfinite(args.grid_factor) or args.grid_factor < 1.0:
        parser.error("Use at least two substeps per scale and a grid factor of at least one.")
    if not np.isfinite(args.friction) or args.friction < 0 or (args.friction and not args.contact):
        parser.error("A nonnegative friction coefficient requires contact handling when nonzero.")
    if args.fused and not args.contact:
        parser.error("Fused mode requires contact handling.")
    if args.solver != "lattice" and (args.slip or args.contact):
        parser.error("Contact and slip switches apply to the lattice solver.")
    raise SystemExit(0 if run(args) else 1)
