"""CUDA APIC/MPM benchmark with a simple compressible plastic material."""
import argparse
import json
import time

import numpy as np
import warp as wp


@wp.func
def base_node(xp: wp.vec3) -> wp.vec3i:
    return wp.vec3i(int(wp.floor(xp[0] - 0.5)), int(wp.floor(xp[1] - 0.5)), int(wp.floor(xp[2] - 0.5)))


@wp.func
def shape_weights(f: wp.vec3) -> wp.mat33:
    a = 0.5 * wp.cw_mul(wp.vec3(1.5) - f, wp.vec3(1.5) - f)
    b = wp.vec3(0.75) - wp.cw_mul(f - wp.vec3(1.0), f - wp.vec3(1.0))
    c = 0.5 * wp.cw_mul(f - wp.vec3(0.5), f - wp.vec3(0.5))
    return wp.mat33(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2])


@wp.kernel
def p2g(
    x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
    C: wp.array(dtype=wp.mat33), J: wp.array(dtype=float),
    mass: wp.array3d(dtype=float), momentum: wp.array3d(dtype=wp.vec3),
    dx: float, dt: float, particle_mass: float, volume: float,
):
    i = wp.tid()
    xp = x[i] / dx
    base = base_node(xp)
    f = xp - wp.vec3(float(base[0]), float(base[1]), float(base[2]))
    weights = shape_weights(f)
    # Compression-only pressure and irreversible volumetric compaction.
    j = wp.clamp(J[i] * (1.0 + dt * wp.trace(C[i])), 0.65, 1.0)
    J[i] = j
    stress = dt * volume * 4.0 / (dx * dx) * 3500.0 * (1.0 - j)
    affine = C[i] * particle_mass + wp.mat33(stress, 0.0, 0.0, 0.0, stress, 0.0, 0.0, 0.0, stress)
    for u in range(3):
        for q in range(3):
            for w in range(3):
                d = wp.vec3(float(u), float(q), float(w)) - f
                weight = weights[u, 0] * weights[q, 1] * weights[w, 2]
                gx = base[0] + u
                gy = base[1] + q
                gz = base[2] + w
                wp.atomic_add(mass, gx, gy, gz, weight * particle_mass)
                wp.atomic_add(momentum, gx, gy, gz, weight * (v[i] * particle_mass + affine * (d * dx)))


@wp.kernel
def grid_step(
    mass: wp.array3d(dtype=float), velocity: wp.array3d(dtype=wp.vec3),
    stepsize: float, width: int, forced: int,
):
    u, q, w = wp.tid()
    value = mass[u, q, w]
    if value > 0.0:
        p = velocity[u, q, w] / value + wp.vec3(float(forced) * 7.0, -9.81, 0.0) * stepsize
        if q < 3 and p[1] < 0.0:
            p = wp.vec3(p[0] * 0.67, 0.0, p[2] * 0.67)
        if (u < 3 and p[0] < 0.0) or (u >= width - 3 and p[0] > 0.0):
            p[0] = 0.0
        if q >= width - 3 and p[1] > 0.0:
            p[1] = 0.0
        if (w < 3 and p[2] < 0.0) or (w >= width - 3 and p[2] > 0.0):
            p[2] = 0.0
        velocity[u, q, w] = p


@wp.kernel
def g2p(
    x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), C: wp.array(dtype=wp.mat33),
    grid: wp.array3d(dtype=wp.vec3), dx: float, dt: float, width: int,
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
                d = wp.vec3(float(u), float(q), float(w)) - f
                weight = weights[u, 0] * weights[q, 1] * weights[w, 2]
                gv = grid[base[0] + u, base[1] + q, base[2] + w]
                speed += gv * weight
                affine += wp.outer(gv, d) * (weight * 4.0 / dx)
    position = x[i] + dt * speed
    lo = 2.0 * dx
    hi = float(width - 3) * dx
    x[i] = wp.vec3(wp.clamp(position[0], lo, hi), wp.clamp(position[1], lo, hi), wp.clamp(position[2], lo, hi))
    v[i] = speed
    C[i] = affine


def run(particles, width, substeps):
    wp.init()
    device = "cuda:0"
    dx = 1.0 / width
    dt = 1.0 / (60 * substeps)
    volume = 0.32 * 0.35 * 0.37 / particles
    particle_mass = 1600.0 * volume
    random = np.random.default_rng(74)
    initial = random.uniform([0.16, 0.25, 0.28], [0.48, 0.60, 0.65], (particles, 3)).astype(np.float32)
    x = wp.array(initial, dtype=wp.vec3, device=device)
    v = wp.zeros(particles, dtype=wp.vec3, device=device)
    C = wp.zeros(particles, dtype=wp.mat33, device=device)
    J = wp.ones(particles, dtype=float, device=device)
    shape = (width, width, width)
    mass = wp.zeros(shape, dtype=float, device=device)
    velocity = wp.zeros(shape, dtype=wp.vec3, device=device)

    def step(forced):
        mass.zero_()
        velocity.zero_()
        wp.launch(p2g, particles, inputs=[x, v, C, J, mass, velocity, dx, dt, particle_mass, volume], device=device)
        wp.launch(grid_step, shape, inputs=[mass, velocity, dt, width, forced], device=device)
        wp.launch(g2p, particles, inputs=[x, v, C, velocity, dx, dt, width], device=device)

    step(0)
    wp.synchronize_device(device)
    samples = []
    for frame in range(60):
        start = time.perf_counter()
        for _ in range(substeps):
            step(int(frame < 30))
        wp.synchronize_device(device)
        if frame >= 8:
            samples.append((time.perf_counter() - start) * 1000)
    delta = x.numpy() - initial
    particle_total = particle_mass * particles
    mass_error = abs(np.sum(mass.numpy(), dtype=np.float64) / particle_total - 1.0)
    compaction = J.numpy()
    moved = int(np.count_nonzero(np.linalg.norm(delta, axis=1) > dx))
    p50, p95 = np.percentile(samples, [50, 95])
    print(json.dumps({
        "particles": particles, "grid": width, "substeps_per_60hz": substeps,
        "p50_ms": p50, "p95_ms": p95, "mass_relative_error": mass_error,
        "moved": moved, "vertical_delta": float(delta[:, 1].mean()),
        "mean_volume_ratio": float(compaction.mean()), "finite": bool(np.isfinite(delta).all()),
    }))
    return mass_error < 1e-5 and moved > particles // 2 and np.isfinite(delta).all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=109000)
    parser.add_argument("--grid", type=int, default=48)
    parser.add_argument("--substeps", type=int, default=8)
    args = parser.parse_args()
    if args.particles < 1 or args.grid < 16 or args.substeps < 1:
        parser.error("Use a positive particle/step count and a grid of at least 16 cells.")
    raise SystemExit(0 if run(args.particles, args.grid, args.substeps) else 1)
