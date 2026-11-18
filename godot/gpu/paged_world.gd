extends RefCounted
## Sparse GPU material world. Coordinates and the attraction radius are in cells.

const PageLayout = preload("page_layout.gd")
const MaterialShader = preload("material.glsl")
const DispatchSizesShader = preload("dispatch_sizes.glsl")
const GROUP_WIDTH := 128
const MAX_BATCH := 65536
const MAX_SURFACE_VOLUME := 2_000_000

enum Kind { AIR, STONE, SAND, WATER }
enum Buffer { DIRECTORY, ATLAS, ORIGINS, WRITABLE, SUPPORT, COUNTERS, INDIRECT, COMMANDS, TRAVEL, SWEEP_PAGES }
enum Pass {
	SUPPORT, EXCHANGE, FRONTIER, RESERVE = 4, VALIDATE, WRITE, SAMPLE,
	INTEGRATE, IMPULSE, SURFACE, COLUMNS, RAY, SWEEP,
}

var _rd: RenderingDevice
var _shader := RID()
var _pipeline := RID()
var _uniforms := RID()
var _dispatch_shader := RID()
var _dispatch_pipeline := RID()
var _dispatch_uniforms := RID()
var _buffers: Array[RID] = []
var _push := PackedByteArray()
var _bounds := Vector3i.ZERO
var _capacity := 0
var _bytes := 0
var _tick := 0
var _motion_started := false
var _pending := false
var granular_sliding := false
var ballistic_sand := true
var _cell_metres := 0.01
var _seconds_per_step := 1.0 / 120.0

func load_boxes(bounds: Vector3i, boxes: PackedInt32Array, max_resident_pages := 0) -> bool:
	var layout := PageLayout.new()
	return layout.build(bounds, boxes, max_resident_pages) and _load_layout(layout, max_resident_pages)

func _load_layout(layout: RefCounted, max_resident_pages: int) -> bool:
	var slots: int = layout.writable.size()
	var tiles: int = layout.directory.size()
	var capacity := max_resident_pages
	if capacity == 0:
		capacity = mini(tiles, slots + maxi(32, slots >> 2))
	if capacity < maxi(1, slots) or capacity > tiles:
		return false
	close()
	_rd = RenderingServer.create_local_rendering_device()
	if _rd == null:
		return false
	var spirv: RDShaderSPIRV = MaterialShader.get_spirv()
	var shader_error := spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_COMPUTE)
	if not shader_error.is_empty():
		push_error(shader_error)
		close()
		return false
	_shader = _rd.shader_create_from_spirv(spirv)
	if not _shader.is_valid():
		close()
		return false
	_pipeline = _rd.compute_pipeline_create(_shader)
	if not _pipeline.is_valid():
		close()
		return false
	_dispatch_shader = _rd.shader_create_from_spirv(DispatchSizesShader.get_spirv())
	if not _dispatch_shader.is_valid():
		close()
		return false
	_dispatch_pipeline = _rd.compute_pipeline_create(_dispatch_shader)
	if not _dispatch_pipeline.is_valid():
		close()
		return false
	var active_count: int = layout.origins.size() >> 2
	var initial := [
		layout.directory.to_byte_array(), layout.atlas.to_byte_array(),
		layout.origins.to_byte_array(), layout.writable.to_byte_array(),
		PackedByteArray(),
		PackedInt32Array([slots, active_count, active_count, 0, 0, 0]).to_byte_array(),
		PackedByteArray(), PackedByteArray(), PackedByteArray(), PackedByteArray(),
	]
	var sizes := [
		tiles * 4, capacity * PageLayout.PAGE_CELLS * 4, capacity * 16, capacity * 4,
		capacity * PageLayout.PAGE_CELLS * 4, 24, 48, MAX_BATCH * 32, capacity * PageLayout.PAGE_CELLS * 4, capacity * 4,
	]
	var bindings: Array[RDUniform] = []
	for binding in sizes.size():
		var contents: PackedByteArray = initial[binding]
		if not contents.is_empty() and contents.resize(sizes[binding]) != OK:
			close()
			return false
		var usage := RenderingDevice.STORAGE_BUFFER_USAGE_DISPATCH_INDIRECT if binding == Buffer.INDIRECT else 0
		var buffer := _rd.storage_buffer_create(sizes[binding], contents, usage)
		if not buffer.is_valid():
			close()
			return false
		_buffers.append(buffer)
		_bytes += sizes[binding]
		if binding != Buffer.INDIRECT:
			bindings.append(_storage_uniform(binding, buffer))
	_uniforms = _rd.uniform_set_create(bindings, _shader, 0)
	var dispatch_bindings: Array[RDUniform] = [
		_storage_uniform(0, _buffers[Buffer.COUNTERS]),
		_storage_uniform(1, _buffers[Buffer.INDIRECT]),
	]
	_dispatch_uniforms = _rd.uniform_set_create(dispatch_bindings, _dispatch_shader, 0)
	if not _uniforms.is_valid() or not _dispatch_uniforms.is_valid():
		close()
		return false
	_bounds = layout.bounds
	_capacity = capacity
	_rd.buffer_clear(_buffers[Buffer.SUPPORT], 0, sizes[Buffer.SUPPORT])
	_rd.buffer_clear(_buffers[Buffer.TRAVEL], 0, sizes[Buffer.TRAVEL])
	_rd.buffer_clear(_buffers[Buffer.SWEEP_PAGES], 0, sizes[Buffer.SWEEP_PAGES])
	_push.resize(128)
	_push.encode_s32(0, layout.shape.x)
	_push.encode_s32(4, layout.shape.y)
	_push.encode_s32(8, layout.shape.z)
	_push.encode_s32(12, capacity)
	_push.encode_s32(16, _bounds.x)
	_push.encode_s32(20, _bounds.y)
	_push.encode_s32(24, _bounds.z)
	_publish_dispatch_sizes()
	_rd.submit()
	_rd.sync()
	return true

## Set the size of a cell and the duration represented by one simulation step.
func configure_motion(cell_metres: float, seconds_per_step: float) -> bool:
	if not is_finite(cell_metres) or cell_metres < 0.0001 or cell_metres > 100.0:
		return false
	if not is_finite(seconds_per_step) or seconds_per_step < 0.0001 or seconds_per_step > 0.1:
		return false
	_cell_metres = cell_metres
	_seconds_per_step = seconds_per_step
	return true

## Adds velocity in cells per step to one material inside a sphere, with linear falloff.
func apply_impulse(center: Vector3, radius: float, velocity: Vector3, kind := Kind.SAND) -> bool:
	if _rd == null or not center.is_finite() or not velocity.is_finite() or (kind != Kind.SAND and kind != Kind.WATER):
		return false
	if not is_finite(radius) or radius <= 0.0 or radius > 1e8:
		return false
	var magnitude := velocity.abs()
	if magnitude[magnitude.max_axis_index()] > 16.0:
		return false
	_push.encode_float(48, center.x)
	_push.encode_float(52, center.y)
	_push.encode_float(56, center.z)
	_push.encode_float(60, radius * radius)
	_push.encode_float(64, velocity.x)
	_push.encode_float(68, velocity.y)
	_push.encode_float(72, velocity.z)
	_push.encode_float(76, 1.0 / maxf(radius, 1e-8))
	var commands := _begin()
	_dispatch(commands, Pass.IMPULSE, 0, 0, kind, 0)
	_finish()
	_motion_started = true
	return true

## Returns exposed cells as (x, y, z, material | face_mask << 8), in GPU order.
func surface_cells(origin: Vector3i, size: Vector3i, limit := MAX_BATCH) -> Dictionary:
	if _rd == null or size.x <= 0 or size.y <= 0 or size.z <= 0 or limit < 1 or limit > MAX_BATCH:
		return {}
	if origin != origin.clamp(Vector3i.ZERO, _bounds - Vector3i.ONE) or size != size.min(_bounds - origin):
		return {}
	var volume := size.x * size.y * size.z
	if volume > MAX_SURFACE_VOLUME:
		return {}
	for axis in 3:
		_push.encode_s32(80 + 4 * axis, origin[axis])
		_push.encode_s32(96 + 4 * axis, size[axis])
	_push.encode_s32(108, limit)
	wait_for_steps()
	_rd.buffer_clear(_buffers[Buffer.COUNTERS], 20, 4)
	var commands := _begin()
	_dispatch(commands, Pass.SURFACE, 0, 0, volume)
	_finish()
	var total := _rd.buffer_get_data(_buffers[Buffer.COUNTERS], 20, 4).decode_s32(0)
	var cells := PackedInt32Array()
	if total > 0:
		cells = _rd.buffer_get_data(_buffers[Buffer.COMMANDS], 0, mini(total, limit) * 16).to_int32_array()
	return {"total": total, "cells": cells}

## Up to two RGBA32F layers stacked vertically in X/Z order. Channels are
## the inclusive base and exclusive top of the sand and water runs.
func column_map(stride: int, layers := 1) -> Image:
	if _rd == null or stride <= 0 or layers < 1 or layers > 2:
		return null
	var width := ceili(float(_bounds.x) / stride)
	var depth := ceili(float(_bounds.z) / stride)
	if width * depth > MAX_BATCH or width > 16384 or depth * layers > 16384:
		return null
	_push.encode_s32(96, width)
	_push.encode_s32(104, depth)
	_push.encode_s32(108, stride)
	_push.encode_s32(92, layers)
	var commands := _begin()
	_dispatch(commands, Pass.COLUMNS, 0, 0, width * depth)
	_finish()
	var data := _rd.buffer_get_data(_buffers[Buffer.COMMANDS], 0, width * depth * layers * 16)
	return Image.create_from_data(width, depth * layers, false, Image.FORMAT_RGBAF, data)

## Traces the authoritative GPU grid in cell coordinates. Bit 1 << Kind
## enables a material. The returned position is INF when the ray misses.
func ray_hit(origin: Vector3, direction: Vector3, distance: float, material_mask := 14) -> Vector3:
	if _rd == null or not origin.is_finite() or not direction.is_finite() or direction.is_zero_approx():
		return Vector3.INF
	if not is_finite(distance) or distance <= 0.0 or material_mask <= 0 or material_mask > 14 or material_mask & 1:
		return Vector3.INF
	direction = direction.normalized()
	for axis in 3:
		_push.encode_float(48 + 4 * axis, origin[axis])
		_push.encode_float(64 + 4 * axis, direction[axis])
	_push.encode_float(60, distance)
	var commands := _begin()
	_dispatch(commands, Pass.RAY, 0, 0, material_mask)
	_finish()
	var data := _rd.buffer_get_data(_buffers[Buffer.COMMANDS], 0, 16)
	if data.decode_s32(12) == 0:
		return Vector3.INF
	return Vector3(data.decode_float(0), data.decode_float(4), data.decode_float(8))

## Defines a kinematic obstacle for particles. It blocks entry and pushes
## existing particles toward the nearest open side over subsequent steps.
func set_kinematic_box(base: Vector3i, half_width: int, height: int) -> bool:
	if _rd == null or half_width <= 0 or half_width > 65535 or height <= 0 or height > 32767:
		return false
	if base != base.clamp(-Vector3i(65535, 32767, 65535), _bounds + Vector3i(65535, 32767, 65535)):
		return false
	for axis in 3:
		_push.encode_s32(112 + 4 * axis, base[axis])
	_push.encode_s32(124, half_width | (height << 16))
	_motion_started = true
	return true

func clear_kinematic_box() -> void:
	if _rd != null:
		_push.encode_s32(124, 0)

func step(count := 1, focus := Vector3.ZERO, radius := 0.0, expand := true,
		strength := 0.8, kind := Kind.SAND) -> bool:
	if not submit_steps(count, focus, radius, expand, strength, kind):
		return false
	wait_for_steps()
	return true

## Runs the GPU while the caller does other work. A second submission waits for the first.
func submit_steps(count := 1, focus := Vector3.ZERO, radius := 0.0, expand := true,
		strength := 0.8, kind := Kind.SAND) -> bool:
	if _rd == null or count < 1 or count > 1024 or not focus.is_finite():
		return false
	if not is_finite(radius) or radius < 0.0 or radius > 1e8 or not is_finite(strength) or strength < 0.0 or strength > 16.0:
		return false
	if kind != Kind.SAND and kind != Kind.WATER:
		return false
	_motion_started = _motion_started or (radius > 0.0 and strength > 0.0)
	var motion_mode := 2 if ballistic_sand else int(_motion_started)
	_push.encode_s32(44, motion_mode)
	_push.encode_float(48, focus.x)
	_push.encode_float(52, focus.y)
	_push.encode_float(56, focus.z)
	_push.encode_float(60, radius * radius if strength > 0.0 else 0.0)
	_push.encode_float(64, strength)
	_push.encode_float(72, float(kind))
	_push.encode_float(80, 13.0 * _seconds_per_step * _seconds_per_step / _cell_metres)
	_push.encode_float(84, _seconds_per_step)
	_push.encode_s32(92, int(granular_sliding))
	var commands := _begin()
	for _i in count:
		_push.encode_u32(88, _tick + 1)
		if expand:
			_dispatch(commands, Pass.FRONTIER, 0, 0, 0, 24)
			_rd.compute_list_end()
			_publish_dispatch_sizes()
			commands = _begin()
		if motion_mode != 0:
			_dispatch(commands, Pass.INTEGRATE, 0, 0, 0, 0)
		var axes := [1, 2, 0] if granular_sliding and (_tick & 1) != 0 else [1, 0, 2]
		for axis in axes:
			if axis != 1:
				_dispatch(commands, Pass.SUPPORT, 0, 0, 0, 0)
			_dispatch(commands, Pass.EXCHANGE, axis, _tick & 1, 0, 12)
			_dispatch(commands, Pass.EXCHANGE, axis, (_tick & 1) ^ 1, 0, 12)
			if motion_mode != 0:
				_dispatch(commands, Pass.SWEEP, axis, _tick & 1, 0, 36)
				_dispatch(commands, Pass.SWEEP, axis, (_tick & 1) ^ 1, 0, 36)
		_tick += 1
	_rd.compute_list_end()
	_rd.submit()
	_pending = true
	return true

func wait_for_steps() -> void:
	if _pending:
		_rd.sync()
		_pending = false

## A flat array of unique (x, y, z, material) edits. Failure leaves materials unchanged.
func set_cells(edits: PackedInt32Array) -> bool:
	var count := edits.size() >> 2
	if _rd == null or edits.size() % 4 != 0 or count > MAX_BATCH:
		return false
	if count == 0:
		return true
	var seen := {}
	for i in range(0, edits.size(), 4):
		var p := Vector3i(edits[i], edits[i + 1], edits[i + 2])
		if p != p.clamp(Vector3i.ZERO, _bounds - Vector3i.ONE) or edits[i + 3] < 0 or edits[i + 3] > Kind.WATER or seen.has(p):
			return false
		seen[p] = true
	var bytes := edits.to_byte_array()
	wait_for_steps()
	_rd.buffer_update(_buffers[Buffer.COMMANDS], 0, bytes.size(), bytes)
	_rd.buffer_clear(_buffers[Buffer.COUNTERS], 16, 4)
	var commands := _begin()
	_dispatch(commands, Pass.RESERVE, 0, 0, count)
	_dispatch(commands, Pass.VALIDATE, 0, 0, count)
	_dispatch(commands, Pass.WRITE, 0, 0, count)
	_rd.compute_list_end()
	_publish_dispatch_sizes()
	_rd.submit()
	_rd.sync()
	return _rd.buffer_get_data(_buffers[Buffer.COUNTERS], 16, 4).decode_s32(0) == 0

## Returns one material per (x, y, z) triplet, or -1 outside the world.
func sample_cells(points: PackedInt32Array) -> PackedInt32Array:
	@warning_ignore("integer_division")
	var count := points.size() / 3
	var result := PackedInt32Array()
	if _rd == null or points.size() % 3 != 0 or count == 0 or count > MAX_BATCH:
		return result
	var payload := PackedInt32Array()
	payload.resize(count * 4)
	for i in count:
		payload[i * 4] = points[i * 3]
		payload[i * 4 + 1] = points[i * 3 + 1]
		payload[i * 4 + 2] = points[i * 3 + 2]
	var bytes := payload.to_byte_array()
	wait_for_steps()
	_rd.buffer_update(_buffers[Buffer.COMMANDS], 0, bytes.size(), bytes)
	var commands := _begin()
	_dispatch(commands, Pass.SAMPLE, 0, 0, count)
	_finish()
	var response := _rd.buffer_get_data(_buffers[Buffer.COMMANDS], 0, bytes.size()).to_int32_array()
	result.resize(count)
	for i in count:
		result[i] = response[i * 4 + 3]
	return result

func status() -> Dictionary:
	if _rd == null:
		return {}
	wait_for_steps()
	var counters := _rd.buffer_get_data(_buffers[Buffer.COUNTERS]).to_int32_array()
	return {
		"bounds": _bounds, "tick": _tick, "resident_pages": counters[0],
		"active_pages": counters[2], "capacity_misses": counters[3],
		"page_capacity": _capacity, "buffer_bytes": _bytes,
	}

func close() -> void:
	if _rd == null:
		return
	wait_for_steps()
	for resource: RID in [_uniforms, _dispatch_uniforms, _pipeline, _dispatch_pipeline, _shader, _dispatch_shader]:
		if resource.is_valid():
			_rd.free_rid(resource)
	for buffer in _buffers:
		_rd.free_rid(buffer)
	_rd.free()
	_rd = null
	_uniforms = RID()
	_pipeline = RID()
	_shader = RID()
	_dispatch_uniforms = RID()
	_dispatch_pipeline = RID()
	_dispatch_shader = RID()
	_buffers.clear()
	_push.clear()
	_bounds = Vector3i.ZERO
	_capacity = 0
	_bytes = 0
	_tick = 0
	_motion_started = false

func _notification(what: int) -> void:
	if what == NOTIFICATION_PREDELETE and _rd != null:
		if _pending:
			_rd.sync()
		for resource: RID in [_uniforms, _dispatch_uniforms, _pipeline, _dispatch_pipeline, _shader, _dispatch_shader]:
			if resource.is_valid():
				_rd.free_rid(resource)
		for buffer in _buffers:
			_rd.free_rid(buffer)
		_rd.free()

func _storage_uniform(binding: int, buffer: RID) -> RDUniform:
	var uniform := RDUniform.new()
	uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	uniform.binding = binding
	uniform.add_id(buffer)
	return uniform

func _publish_dispatch_sizes() -> void:
	# The publisher owns the storage write. The solver only consumes the
	# buffer as dispatch arguments, so the render graph orders the accesses.
	var commands := _rd.compute_list_begin()
	_rd.compute_list_bind_compute_pipeline(commands, _dispatch_pipeline)
	_rd.compute_list_bind_uniform_set(commands, _dispatch_uniforms, 0)
	_rd.compute_list_dispatch(commands, 1, 1, 1)
	_rd.compute_list_end()

func _begin() -> int:
	wait_for_steps()
	var commands := _rd.compute_list_begin()
	_rd.compute_list_bind_compute_pipeline(commands, _pipeline)
	_rd.compute_list_bind_uniform_set(commands, _uniforms, 0)
	return commands

func _dispatch(commands: int, mode: int, axis := 0, parity := 0, count := 0, indirect_offset := -1) -> void:
	_push.encode_s32(28, mode)
	_push.encode_s32(32, axis)
	_push.encode_s32(36, parity)
	_push.encode_s32(40, count)
	_rd.compute_list_set_push_constant(commands, _push, _push.size())
	if indirect_offset >= 0:
		_rd.compute_list_dispatch_indirect(commands, _buffers[Buffer.INDIRECT], indirect_offset)
	else:
		_rd.compute_list_dispatch(commands, maxi(1, ceili(float(count) / GROUP_WIDTH)), 1, 1)
	_rd.compute_list_add_barrier(commands)

func _finish() -> void:
	_rd.compute_list_end()
	_rd.submit()
	_rd.sync()
