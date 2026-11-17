#pragma once

#include <algorithm>
#include <cmath>

namespace somevoxel {

template <typename T>
struct Vector3 {
    T x = 0, y = 0, z = 0;

    constexpr Vector3() = default;
    constexpr Vector3(T x, T y, T z) : x(x), y(y), z(z) {}
    template <typename U>
    explicit constexpr Vector3(Vector3<U> v) : x(T(v.x)), y(T(v.y)), z(T(v.z)) {}

    constexpr T &operator[](int axis) { return axis == 0 ? x : axis == 1 ? y : z; }
    constexpr T operator[](int axis) const { return axis == 0 ? x : axis == 1 ? y : z; }
    constexpr bool operator==(Vector3 v) const { return x == v.x && y == v.y && z == v.z; }
    constexpr bool operator!=(Vector3 v) const { return !(*this == v); }
    constexpr Vector3 operator-() const { return {-x, -y, -z}; }
    constexpr Vector3 operator+(Vector3 v) const { return {x + v.x, y + v.y, z + v.z}; }
    constexpr Vector3 operator-(Vector3 v) const { return {x - v.x, y - v.y, z - v.z}; }
    constexpr Vector3 operator*(Vector3 v) const { return {x * v.x, y * v.y, z * v.z}; }
    constexpr Vector3 operator*(T s) const { return {x * s, y * s, z * s}; }
    constexpr Vector3 operator/(T s) const { return {x / s, y / s, z / s}; }
    friend constexpr Vector3 operator*(T s, Vector3 v) { return v * s; }
    Vector3 &operator+=(Vector3 v) { return *this = *this + v; }
    Vector3 &operator-=(Vector3 v) { return *this = *this - v; }
    Vector3 &operator*=(T s) { return *this = *this * s; }
    Vector3 &operator/=(T s) { return *this = *this / s; }

    T dot(Vector3 v) const { return x * v.x + y * v.y + z * v.z; }
    Vector3 cross(Vector3 v) const {
        return {y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x};
    }
    T length_squared() const { return dot(*this); }
    T length() const { return std::sqrt(length_squared()); }
    T distance_squared_to(Vector3 v) const { return (v - *this).length_squared(); }
    T distance_to(Vector3 v) const { return (v - *this).length(); }
    Vector3 normalized() const {
        const T squared = length_squared();
        return squared == 0 ? Vector3{} : *this / std::sqrt(squared);
    }
    Vector3 limit_length(T maximum) const {
        const T n = length();
        return n > 0 && maximum < n ? (*this / n) * maximum : *this;
    }
    Vector3 move_toward(Vector3 to, T delta) const {
        const Vector3 offset = to - *this;
        const T distance = offset.length();
        return distance <= delta || distance < T(0.00001) ? to : *this + offset / distance * delta;
    }
    Vector3 lerp(Vector3 to, T weight) const { return *this + (to - *this) * weight; }
    Vector3 clamp(Vector3 low, Vector3 high) const {
        return {std::clamp(x, low.x, high.x), std::clamp(y, low.y, high.y), std::clamp(z, low.z, high.z)};
    }
    bool is_finite() const { return std::isfinite(x) && std::isfinite(y) && std::isfinite(z); }
};

using Vec3 = Vector3<float>;
using Cell = Vector3<int>;

} // namespace somevoxel
