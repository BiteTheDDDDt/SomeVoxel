#pragma once

#include "world_presentation.hpp"
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/dictionary.hpp>

namespace somevoxel {
class VoxelWorld : public godot::RefCounted {
    GDCLASS(VoxelWorld, godot::RefCounted)

    somevoxel::WorldPresentation world;
    using V3 = godot::Vector3;

    static godot::Dictionary throw_summary(const somevoxel::ThrowInfo &state) {
        godot::Dictionary result;
        if (state.count == 0) return result;
        result["position"] = somevoxel::to_godot(state.position);
        result["velocity"] = somevoxel::to_godot(state.velocity);
        result["speed"] = state.speed;
        result["mass"] = state.mass;
        result["count"] = state.count;
        return result;
    }

    static godot::Dictionary contact_map(const somevoxel::TargetContacts &contacts) {
        godot::Dictionary result;
        for (const auto &[id, effect] : contacts) result[id] = effect;
        return result;
    }

protected:
    static void _bind_methods() {
        using godot::ClassDB;
        using godot::D_METHOD;
        ClassDB::bind_method(D_METHOD("load_boxes", "layout"), &VoxelWorld::load_boxes);
        ClassDB::bind_method(D_METHOD("step"), &VoxelWorld::step);
        ClassDB::bind_method(D_METHOD("packed_particle_mass", "extent"), &VoxelWorld::packed_particle_mass);
        ClassDB::bind_method(D_METHOD("spawn_packed_particle", "position", "extent", "impulse"),
            &VoxelWorld::spawn_packed_particle, DEFVAL(V3()));
        ClassDB::bind_method(D_METHOD("packed_particle_state", "handle"), &VoxelWorld::packed_particle_state);
        ClassDB::bind_method(D_METHOD("gravity_acceleration"), &VoxelWorld::gravity_acceleration);
        ClassDB::bind_method(D_METHOD("set_field", "point", "active"), &VoxelWorld::set_field);
        ClassDB::bind_method(D_METHOD("set_cast_profile", "radius_scale", "strength"), &VoxelWorld::set_cast_profile);
        ClassDB::bind_method(D_METHOD("set_charge_gather", "progress"), &VoxelWorld::set_charge_gather);
        ClassDB::bind_method(D_METHOD("wind_acceleration_at", "point"), &VoxelWorld::wind_acceleration_at);
        ClassDB::bind_method(D_METHOD("set_water_field", "point", "active"), &VoxelWorld::set_water_field);
        ClassDB::bind_method(D_METHOD("set_vine_field", "point", "active"), &VoxelWorld::set_vine_field);
        ClassDB::bind_method(D_METHOD("set_wind", "point", "direction", "active"), &VoxelWorld::set_wind);
        ClassDB::bind_method(D_METHOD("apply_heat", "point", "radius", "amount", "incident"), &VoxelWorld::apply_heat,
            DEFVAL(V3()));
        ClassDB::bind_method(D_METHOD("set_fire_jet", "origin", "direction", "radius", "strength", "active"), &VoxelWorld::set_fire_jet);
        ClassDB::bind_method(D_METHOD("fire_jet_count"), &VoxelWorld::fire_jet_count);
        ClassDB::bind_method(D_METHOD("fire_jet_capacity"), &VoxelWorld::fire_jet_buffer_capacity);
        ClassDB::bind_method(D_METHOD("fire_jet_buffer", "alpha"), &VoxelWorld::fire_jet_buffer);
        ClassDB::bind_method(D_METHOD("set_condensation", "point", "active", "radius", "strength"), &VoxelWorld::set_condensation);
        ClassDB::bind_method(D_METHOD("humidity_at", "point"), &VoxelWorld::humidity_at);
        ClassDB::bind_method(D_METHOD("fire_buffer"), &VoxelWorld::fire_buffer);
        ClassDB::bind_method(D_METHOD("element_version", "kind"), &VoxelWorld::element_version);
        ClassDB::bind_method(D_METHOD("carve", "point", "radius"), &VoxelWorld::carve);
        ClassDB::bind_method(D_METHOD("start_earth_wave", "origin_foot", "target", "radius", "force"),
            &VoxelWorld::start_earth_wave);
        ClassDB::bind_method(D_METHOD("earth_wave_state"), &VoxelWorld::earth_wave_state);
        ClassDB::bind_method(D_METHOD("take_earth_contacts"), &VoxelWorld::take_earth_contacts);
        ClassDB::bind_method(D_METHOD("compact", "point", "radius", "extent"), &VoxelWorld::compact, DEFVAL(2));
        ClassDB::bind_method(D_METHOD("impact", "point", "radius", "strength", "pressure_source"),
            &VoxelWorld::impact, DEFVAL(1.f), DEFVAL(V3(std::numeric_limits<float>::infinity(), 0, 0)));
        ClassDB::bind_method(D_METHOD("grain_count"), &VoxelWorld::grain_count);
        ClassDB::bind_method(D_METHOD("sample", "point"), &VoxelWorld::sample);
        ClassDB::bind_method(D_METHOD("ray_hit", "origin", "direction", "distance", "include_water", "ignore_formed"),
            &VoxelWorld::ray_hit, DEFVAL(false));
        ClassDB::bind_method(D_METHOD("body_clear", "feet", "radius", "height"), &VoxelWorld::body_clear);
        ClassDB::bind_method(D_METHOD("push_body_up", "feet", "radius", "height", "speed", "body_mass"), &VoxelWorld::push_body_up);
        ClassDB::bind_method(D_METHOD("place_body", "feet", "radius", "height"), &VoxelWorld::place_body);
        ClassDB::bind_method(D_METHOD("remove_body"), &VoxelWorld::remove_body);
        ClassDB::bind_method(D_METHOD("surface", "material"), &VoxelWorld::surface);
        ClassDB::bind_method(D_METHOD("take_stone_patches"), &VoxelWorld::take_stone_patches);
        ClassDB::bind_method(D_METHOD("take_water_patches"), &VoxelWorld::take_water_patches);
        ClassDB::bind_method(D_METHOD("set_water_patch_tracking", "enabled"), &VoxelWorld::set_water_patch_tracking);
        ClassDB::bind_method(D_METHOD("sand_buffer", "interpolation"), &VoxelWorld::sand_buffer);
        ClassDB::bind_method(D_METHOD("visible_sand_buffer", "interpolation", "previous_capacity"),
            &VoxelWorld::visible_sand_buffer);
        ClassDB::bind_method(D_METHOD("sand_motion"), &VoxelWorld::sand_motion);
        ClassDB::bind_method(D_METHOD("rock_motion"), &VoxelWorld::rock_motion);
        ClassDB::bind_method(D_METHOD("rock_buffer", "interpolation"), &VoxelWorld::rock_buffer);
        ClassDB::bind_method(D_METHOD("particle_motion", "material"), &VoxelWorld::particle_motion);
        ClassDB::bind_method(D_METHOD("particle_batch", "material", "interpolation", "previous_capacity", "cull"),
            &VoxelWorld::particle_batch);
        ClassDB::bind_method(D_METHOD("sand_snapshot", "cull", "previous_rows"), &VoxelWorld::sand_snapshot);
        ClassDB::bind_method(D_METHOD("sand_palette"), &VoxelWorld::sand_palette);
        ClassDB::bind_method(D_METHOD("range_count", "material", "minimum", "maximum"), &VoxelWorld::range_count);
        ClassDB::bind_method(D_METHOD("status"), &VoxelWorld::status);
        ClassDB::bind_method(D_METHOD("sand_impact_peak", "material"), &VoxelWorld::sand_impact_peak);
        ClassDB::bind_method(D_METHOD("set_particle_form", "point", "direction", "material", "form", "radius", "strength", "origin"),
            &VoxelWorld::set_particle_form, DEFVAL(V3()));
        ClassDB::bind_method(D_METHOD("launch_particle_form", "direction", "impulse_scale", "charge"),
            &VoxelWorld::launch_particle_form, DEFVAL(1.f), DEFVAL(0.f));
        ClassDB::bind_method(D_METHOD("packed_sand_throw", "point", "impulse_scale"),
            &VoxelWorld::packed_sand_throw, DEFVAL(1.f));
        ClassDB::bind_method(D_METHOD("launch_packed_sand", "point", "direction", "impulse_scale"),
            &VoxelWorld::launch_packed_sand, DEFVAL(1.f));
        ClassDB::bind_method(D_METHOD("free_sand_throw", "point", "impulse_scale", "charge"),
            &VoxelWorld::free_sand_throw, DEFVAL(1.f), DEFVAL(0.f));
        ClassDB::bind_method(D_METHOD("launch_free_sand", "point", "direction", "impulse_scale", "charge"),
            &VoxelWorld::launch_free_sand, DEFVAL(1.f), DEFVAL(0.f));
        ClassDB::bind_method(D_METHOD("particle_form_count"), &VoxelWorld::particle_form_count);
        ClassDB::bind_method(D_METHOD("particle_form_tip"), &VoxelWorld::particle_form_tip);
        ClassDB::bind_method(D_METHOD("particle_form_speed", "impulse_scale"), &VoxelWorld::particle_form_speed, DEFVAL(1.f));
        ClassDB::bind_method(D_METHOD("particle_form_throw", "impulse_scale", "charge"),
            &VoxelWorld::particle_form_throw, DEFVAL(1.f), DEFVAL(0.f));
        ClassDB::bind_method(D_METHOD("set_form_target", "id", "center", "half_extents"), &VoxelWorld::set_form_target);
        ClassDB::bind_method(D_METHOD("take_form_hits"), &VoxelWorld::take_form_hits);
    }

public:
    bool load_boxes(const godot::PackedInt32Array &layout) {
        return world.load_boxes(layout.ptr(), layout.size());
    }

    float packed_particle_mass(int extent) const {
        return World::packed_particle_mass(extent);
    }

    int64_t spawn_packed_particle(V3 position, int extent, V3 impulse) {
        return world.spawn_packed_particle(from_godot(position), extent, from_godot(impulse));
    }

    godot::Dictionary packed_particle_state(int64_t handle) const {
        const auto state = world.packed_particle_state(handle);
        godot::Dictionary result;
        if (state) {
            result["position"] = to_godot(state->position);
            result["previous_position"] = to_godot(state->previous_position);
            result["velocity"] = to_godot(state->velocity);
            result["extent"] = state->extent;
        }
        return result;
    }

    godot::Dictionary status() const {
        const auto state = world.status();
        godot::Dictionary result;
        result["refinement"] = state.refinement;
        result["cell_size"] = state.cell_size;
        result["step_seconds"] = state.step_seconds;
        result["bulk_water_hz"] = state.bulk_water_hz;
        result["voxel_bytes"] = state.voxel_bytes;
        result["grain_bytes"] = state.grain_bytes;
        result["dense_grid_bytes"] = state.dense_grid_bytes;
        result["sand"] = state.sand;
        result["gravel"] = state.gravel;
        result["sand_volume"] = state.sand_volume;
        result["sand_mass"] = state.sand_mass;
        result["packed_sand_blocks"] = state.packed_sand_blocks;
        result["blocks_9"] = state.blocks_9;
        result["blocks_18"] = state.blocks_18;
        result["blocks_36"] = state.blocks_36;
        result["displaced_water"] = state.displaced_water;
        result["water"] = state.water;
        result["active_water"] = state.active_water;
        result["water_volume"] = state.water_volume;
        result["ice"] = state.ice;
        result["wood"] = state.wood;
        result["grass"] = state.grass;
        result["ash"] = state.ash;
        result["steam"] = state.steam;
        result["air_water"] = state.air_water;
        result["air_collected"] = state.air_collected;
        result["steam_collected"] = state.steam_collected;
        result["air_recharged"] = state.air_recharged;
        result["lava"] = state.lava;
        result["vine"] = state.vine;
        result["mud"] = state.mud;
        result["char"] = state.char_cells;
        result["burning"] = state.burning;
        result["moved"] = state.moved;
        result["moved_gravel"] = state.moved_gravel;
        result["chunks"] = state.chunks;
        result["chunk_capacity"] = state.chunk_capacity;
        result["stone_version"] = state.stone_version;
        result["water_version"] = state.water_version;
        result["elements_version"] = state.elements_version;
        result["contact_stone_fractures"] = state.contact_stone_fractures;
        result["contact_wood_fractures"] = state.contact_wood_fractures;
        result["contact_ice_fractures"] = state.contact_ice_fractures;
        result["contact_char_fractures"] = state.contact_char_fractures;
        return result;
    }

    godot::Dictionary earth_wave_state() const {
        const auto state = world.earth_wave_state();
        godot::Dictionary result;
        result["active"] = state.active;
        result["position"] = somevoxel::to_godot(state.position);
        result["strength"] = state.strength;
        result["serial"] = state.serial;
        result["ended"] = state.ended;
        result["radius"] = state.radius;
        result["ejected"] = state.ejected;
        result["disturbed"] = state.disturbed;
        result["impulse"] = state.impulse;
        result["direction"] = somevoxel::to_godot(state.direction);
        result["travelled"] = state.travelled;
        result["range"] = state.range;
        result["contact_count"] = state.contact_count;
        godot::PackedVector3Array trail;
        trail.resize(state.trail.size());
        auto *points = trail.ptrw();
        for (auto sample : state.trail) *points++ = somevoxel::to_godot(sample);
        result["trail"] = trail;
        return result;
    }

    godot::Dictionary sand_impact_peak(int material) const {
        const auto state = world.sand_impact_peak(material);
        godot::Dictionary result;
        if (state) {
            result["speed"] = state->speed;
            result["impulse"] = state->impulse;
        }
        return result;
    }

    godot::Dictionary free_sand_throw(V3 point, float impulse_scale, float charge) const {
        const auto state = world.free_sand_throw(somevoxel::from_godot(point), impulse_scale, charge);
        auto result = throw_summary(state);
        if (state.count != 0) result["blocks"] = state.blocks;
        return result;
    }

    void step() {
        world.step();
    }

    float gravity_acceleration() const {
        return world.gravity_acceleration();
    }

    void set_field(V3 point, bool active) {
        world.set_field(somevoxel::from_godot(point), active);
    }

    void set_cast_profile(float radius_scale, float strength) {
        world.set_cast_profile(radius_scale, strength);
    }

    void set_charge_gather(float progress) {
        world.set_charge_gather(progress);
    }

    V3 wind_acceleration_at(V3 point) const {
        return somevoxel::to_godot(world.wind_acceleration_at(somevoxel::from_godot(point)));
    }

    void set_water_field(V3 point, bool active) {
        world.set_water_field(somevoxel::from_godot(point), active);
    }

    void set_vine_field(V3 point, bool active) {
        world.set_vine_field(somevoxel::from_godot(point), active);
    }

    void set_wind(V3 point, V3 direction, bool active) {
        world.set_wind(somevoxel::from_godot(point), somevoxel::from_godot(direction), active);
    }

    int apply_heat(V3 point, float radius, float amount, V3 incident) {
        return world.apply_heat(somevoxel::from_godot(point), radius, amount, somevoxel::from_godot(incident));
    }

    void set_fire_jet(V3 origin, V3 direction, float radius, float strength, bool active) {
        world.set_fire_jet(somevoxel::from_godot(origin), somevoxel::from_godot(direction), radius, strength, active);
    }

    int fire_jet_count() const {
        return world.fire_jet_count();
    }

    int fire_jet_buffer_capacity() const {
        return world.fire_jet_buffer_capacity();
    }

    godot::PackedFloat32Array fire_jet_buffer(float alpha) const {
        return world.fire_jet_buffer(alpha);
    }

    void set_condensation(V3 point, bool active, float radius, float strength) {
        world.set_condensation(somevoxel::from_godot(point), active, radius, strength);
    }

    float humidity_at(V3 point) const {
        return world.humidity_at(somevoxel::from_godot(point));
    }

    godot::PackedFloat32Array fire_buffer() const {
        return world.fire_buffer();
    }

    int64_t element_version(int kind) const {
        return world.element_version(kind);
    }

    int carve(V3 point, float radius) {
        return world.carve(somevoxel::from_godot(point), radius);
    }

    bool start_earth_wave(V3 origin_foot, V3 target, float radius, float force) {
        return world.start_earth_wave(somevoxel::from_godot(origin_foot), somevoxel::from_godot(target), radius, force);
    }

    godot::Dictionary take_earth_contacts() {
        return contact_map(world.take_earth_contacts());
    }

    int compact(V3 point, float radius, int extent) {
        return world.compact(somevoxel::from_godot(point), radius, extent);
    }

    int impact(V3 point, float radius, float strength, V3 pressure_source) {
        return world.impact(somevoxel::from_godot(point), radius, strength, somevoxel::from_godot(pressure_source));
    }

    int grain_count() const {
        return world.grain_count();
    }

    int sample(V3 point) const {
        return world.sample(somevoxel::from_godot(point));
    }

    V3 ray_hit(V3 origin, V3 direction, float distance, bool include_water, bool ignore_formed) const {
        return somevoxel::to_godot(world.ray_hit(somevoxel::from_godot(origin), somevoxel::from_godot(direction), distance, include_water, ignore_formed));
    }

    bool body_clear(V3 feet, float radius, float height) const {
        return world.body_clear(somevoxel::from_godot(feet), radius, height);
    }

    float push_body_up(V3 feet, float radius, float height, float speed, float body_mass) {
        return world.push_body_up(somevoxel::from_godot(feet), radius, height, speed, body_mass);
    }

    bool place_body(V3 feet, float radius, float height) {
        return world.place_body(somevoxel::from_godot(feet), radius, height);
    }

    void remove_body() {
        world.remove_body();
    }

    godot::Array surface(int material) const {
        return world.surface(material);
    }

    godot::Array take_stone_patches() {
        return world.take_stone_patches();
    }

    godot::Array take_water_patches() {
        return world.take_water_patches();
    }

    void set_water_patch_tracking(bool enabled) {
        world.set_water_patch_tracking(enabled);
    }

    godot::PackedFloat32Array sand_buffer(float alpha) const {
        return world.sand_buffer(alpha);
    }

    godot::Array visible_sand_buffer(float alpha, int previous_capacity) {
        return world.visible_sand_buffer(alpha, previous_capacity);
    }

    int64_t sand_motion() const {
        return world.sand_motion();
    }

    int64_t rock_motion() const {
        return world.rock_motion();
    }

    godot::PackedFloat32Array rock_buffer(float alpha) const {
        return world.rock_buffer(alpha);
    }

    int64_t particle_motion(int material) const {
        return world.particle_motion(material);
    }

    godot::Array particle_batch(int material, float alpha, int previous_capacity, bool cull) {
        return world.particle_batch(material, alpha, previous_capacity, cull);
    }

    godot::Array sand_snapshot(bool cull, int previous_rows) { return world.sand_snapshot(cull, previous_rows); }
    godot::PackedByteArray sand_palette() const { return world.sand_palette(); }

    int range_count(int material, godot::Vector3i lo, godot::Vector3i hi) const {
        return world.range_count(material, somevoxel::from_godot(lo), somevoxel::from_godot(hi));
    }

    void set_particle_form(V3 point, V3 direction, int material, int form, float radius, float strength, V3 origin) {
        world.set_particle_form(somevoxel::from_godot(point), somevoxel::from_godot(direction), material, form, radius, strength, somevoxel::from_godot(origin));
    }

    int launch_particle_form(V3 direction, float impulse_scale, float charge) {
        return world.launch_particle_form(somevoxel::from_godot(direction), impulse_scale, charge);
    }

    godot::Dictionary packed_sand_throw(V3 point, float impulse_scale) const {
        return throw_summary(world.packed_sand_throw(somevoxel::from_godot(point), impulse_scale));
    }

    float launch_packed_sand(V3 point, V3 direction, float impulse_scale) {
        return world.launch_packed_sand(somevoxel::from_godot(point), somevoxel::from_godot(direction), impulse_scale);
    }

    int launch_free_sand(V3 point, V3 direction, float impulse_scale, float charge) {
        return world.launch_free_sand(somevoxel::from_godot(point), somevoxel::from_godot(direction), impulse_scale, charge);
    }

    int particle_form_count() const {
        return world.particle_form_count();
    }

    V3 particle_form_tip() const {
        return somevoxel::to_godot(world.particle_form_tip());
    }

    float particle_form_speed(float impulse_scale) const {
        return world.particle_form_speed(impulse_scale);
    }

    godot::Dictionary particle_form_throw(float impulse_scale, float charge) const {
        return throw_summary(world.particle_form_throw(impulse_scale, charge));
    }

    void set_form_target(int id, V3 location, V3 half_extents) {
        world.set_form_target(id, somevoxel::from_godot(location), somevoxel::from_godot(half_extents));
    }

    godot::Dictionary take_form_hits() {
        return contact_map(world.take_form_hits());
    }
};

} // namespace somevoxel
