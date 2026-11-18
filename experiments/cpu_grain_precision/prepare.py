"""Prepare an isolated copy of the core for grain precision and SIMD probes.
The source core is read only. A new destination directory is required.
"""
import argparse
import os
import shutil
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("destination", type=Path)
parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2] / "core")
args = parser.parse_args()
destination = args.destination.resolve()
source = args.source.resolve()
if not (source / "world.hpp").is_file():
    parser.error("The source must be a SomeVoxel core directory.")
destination.mkdir(parents=True, exist_ok=False)
shutil.copytree(source, destination / "core")
shutil.copyfile(Path(__file__).with_name("bench.cpp"), destination / "bench.cpp")
os.chdir(destination)

p = Path("core/world.hpp")
s = p.read_text()
s = s.replace("    friend class WorldPresentation;", "    friend class WorldPresentation;\n    friend struct GrainPrecisionProbe;")
s = s.replace("    struct Grain {", """#if defined(SOMEVOXEL_HALF_TRAVEL) || defined(SOMEVOXEL_HALF_VELOCITY)
    struct HalfVector {
        _Float16 x = 0, y = 0, z = 0;
        HalfVector() = default;
        HalfVector(V3 v) : x(_Float16(v.x)), y(_Float16(v.y)), z(_Float16(v.z)) {}
        operator V3() const { return {float(x), float(y), float(z)}; }
        _Float16 &operator[](int axis) { return axis == 0 ? x : axis == 1 ? y : z; }
        float operator[](int axis) const { return axis == 0 ? float(x) : axis == 1 ? float(y) : float(z); }
        bool operator==(V3 v) const { return V3(*this) == v; }
        bool operator!=(V3 v) const { return V3(*this) != v; }
        V3 operator-() const { return -V3(*this); }
        V3 operator*(float s) const { return V3(*this) * s; }
        HalfVector &operator*=(float s) { return *this = V3(*this) * s; }
        HalfVector &operator+=(V3 v) { return *this = V3(*this) + v; }
        float length_squared() const { return V3(*this).length_squared(); }
        V3 clamp(V3 lo, V3 hi) const { return V3(*this).clamp(lo, hi); }
    };
#endif
#ifdef SOMEVOXEL_HALF_VELOCITY
    using VelocityState = HalfVector;
#else
    using VelocityState = V3;
#endif
#ifdef SOMEVOXEL_HALF_TRAVEL
    using TravelState = HalfVector;
#else
    using TravelState = V3;
#endif
    struct Grain {""")
s = s.replace("        V3 velocity;\n        V3 travel;", "        VelocityState velocity;\n        TravelState travel;")
s = s.replace("static_assert(sizeof(Grain) == 16 + 2 * sizeof(V3));", "static_assert(sizeof(Grain) == ((19 + sizeof(VelocityState) + sizeof(TravelState)) & ~size_t(3)));")
a = s.index("    ParticlePath sweep_particle(")
b = s.index("    // Integrate forces", a)
scope = s[a:b]
scope = scope.replace("const V3 previous_travel = grain.travel;", "const V3 previous_travel = grain.travel;\n        V3 travel = previous_travel;")
scope = scope.replace("grain.travel[", "travel[")
scope = scope.replace("if (!crosses_face) return {origin, origin};", "if (!crosses_face) {\n            grain.travel = travel;\n            return {origin, origin};\n        }")
scope = scope.replace("        return {position, origin};", "        grain.travel = travel;\n        return {position, origin};")
s = s[:a] + scope + s[b:]
s = s.replace("std::abs(grain.velocity[axis])", "std::abs(float(grain.velocity[axis]))")
p.write_text(s, newline="\n")
for path in Path('core/detail').glob('*.hpp'):
    s = path.read_text()
    s = s.replace("? liquid_particle(water).velocity : V3()", "? V3(liquid_particle(water).velocity) : V3()")
    path.write_text(s, newline="\n")

p = Path("core/world.hpp")
s = p.read_text(encoding="utf-8-sig")
s = s.replace("#pragma once", "#pragma once\n\n#ifdef SOMEVOXEL_SIMD_SWEEP\n#include <immintrin.h>\n#endif")
needle = """        bool crosses_face = false;
        for (int axis = 0; axis < 3; ++axis) {
            float speed"""
i = s.index(needle)
insertion = """        bool crosses_face = false;
#ifdef SOMEVOXEL_SIMD_SWEEP
        const auto xyz = [](V3 v) { return _mm_setr_ps(v.x, v.y, v.z, 0.f); };
        const auto unpack = [](__m128 v) {
            return V3(_mm_cvtss_f32(v), _mm_cvtss_f32(_mm_shuffle_ps(v, v, 0x55)),
                _mm_cvtss_f32(_mm_shuffle_ps(v, v, 0xaa)));
        };
        const __m128 a = xyz(acceleration);
        const __m128 delta = _mm_set1_ps(dt);
        const __m128 sign = _mm_set1_ps(-0.f);
        const __m128 zero = _mm_setzero_ps();
        const __m128 speed = _mm_min_ps(
            _mm_max_ps(_mm_add_ps(xyz(previous_velocity), _mm_mul_ps(a, delta)), _mm_set1_ps(-maximum_swept_speed)),
            _mm_set1_ps(maximum_swept_speed));
        const __m128 below = _mm_cmplt_ps(_mm_andnot_ps(sign, speed), _mm_set1_ps(0.03f));
        const __m128 stopped = _mm_and_ps(below, _mm_cmpeq_ps(a, zero));
        const __m128 next_speed = _mm_andnot_ps(stopped, speed);
        const __m128 displacement = _mm_mul_ps(next_speed, delta);
        const __m128 next_travel = _mm_add_ps(xyz(previous_travel), displacement);
        grain.velocity = unpack(next_speed);
        travel = unpack(next_travel);
        const int crosses = _mm_movemask_ps(_mm_cmpge_ps(_mm_andnot_ps(sign, next_travel), _mm_set1_ps(scale)));
        if (!crosses && previous_faces == 0) {
            grain.travel = travel;
            return {origin, origin};
        }
        const V3 distances = unpack(displacement);
        for (int axis = 0; axis < 3; ++axis) {
            const float distance = distances[axis];
            const float previous = previous_travel[axis];
#else
        for (int axis = 0; axis < 3; ++axis) {
            float speed"""
s = s[:i] + s[i:].replace(needle, insertion, 1)
s = s.replace("            travel[axis] += distance;\n            if (distance", "            travel[axis] += distance;\n#endif\n            if (distance", 1)
p.write_text(s, newline="\n")
