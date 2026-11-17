extends SceneTree

# Sand-only probe. Keep occupancy on the GPU between dispatches.
const PASSES := 4
const DIRECTIONAL_PHASES := [0, 7, 2, 5, 4, 3, 6, 1]
var size := Vector3i(144, 72, 144)
var barrier_x := 80
var rd: RenderingDevice
var timestamps: Array[float] = []

func _initialize() -> void:
	call_deferred("run_test")

func index_of(x: int, y: int, z: int) -> int:
	return x + size.x * (z + size.z * y)

func run_test() -> void:
	var args := OS.get_cmdline_user_args()
	var resolution := args[0] if not args.is_empty() else "small"
	if resolution == "large":
		size = Vector3i(288, 144, 288)
		barrier_x = 160
	elif resolution != "small":
		push_error("Expected small or large")
		quit(1)
		return
	var factor := 2 if resolution == "large" else 1
	rd = RenderingServer.create_local_rendering_device()
	if rd == null:
		push_error("Run without --headless, using a RenderingDevice backend.")
		quit(1)
		return
	var source := RDShaderSource.new()
	source.source_compute = FileAccess.get_file_as_string("res://block.glsl.txt")
	var shader_code: RDShaderSPIRV = rd.shader_compile_spirv_from_source(source)
	var compile_error := shader_code.get_stage_compile_error(RenderingDevice.SHADER_STAGE_COMPUTE)
	if not compile_error.is_empty():
		push_error(compile_error)
		rd.free()
		quit(1)
		return
	var shader := rd.shader_create_from_spirv(shader_code)
	var pipeline := rd.compute_pipeline_create(shader)

	var cells := PackedInt32Array()
	cells.resize(size.x * size.y * size.z)
	for z in size.z:
		for x in size.x:
			cells[index_of(x, 0, z)] = 1
	for y in size.y:
		for z in size.z:
			cells[index_of(barrier_x, y, z)] = 1
	var next_id := 2
	var origins := PackedInt32Array([0, 0])
	for y in range(12 * factor, 49 * factor):
		for z in range(18 * factor, 110 * factor):
			for x in range(17 * factor, 65 * factor):
				var cell_index := index_of(x, y, z)
				cells[cell_index] = next_id
				origins.append(cell_index)
				next_id += 1
	var initial := cells.to_byte_array()
	var storage := rd.storage_buffer_create(initial.size(), initial)
	var binding := RDUniform.new()
	binding.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	binding.binding = 0
	binding.add_id(storage)
	var uniform_set := rd.uniform_set_create([binding], shader, 0)
	var push := PackedInt32Array([size.x, size.y, size.z, 0]).to_byte_array()
	push.append_array(PackedFloat32Array([104.0 * factor, 32.0 * factor, 64.0 * factor, 74.0 * factor]).to_byte_array())

	var count := 0
	var first_tick_max_cells := 0
	var group_count := Vector3i(ceili((size.x + 1) / 8.0), ceili((size.y + 1) / 8.0), ceili((size.z + 1) / 8.0))
	for step in 56:
		var start := Time.get_ticks_usec()
		var commands := rd.compute_list_begin()
		rd.compute_list_bind_compute_pipeline(commands, pipeline)
		rd.compute_list_bind_uniform_set(commands, uniform_set, 0)
		for pass_number in PASSES:
			push.encode_s32(12, DIRECTIONAL_PHASES[count % DIRECTIONAL_PHASES.size()])
			count += 1
			rd.compute_list_set_push_constant(commands, push, push.size())
			rd.compute_list_dispatch(commands, group_count.x, group_count.y, group_count.z)
			if pass_number != PASSES - 1:
				rd.compute_list_add_barrier(commands)
		rd.compute_list_end()
		rd.submit()
		rd.sync()
		if step == 0:
			var first := rd.buffer_get_data(storage).to_int32_array()
			for i in first.size():
				if first[i] <= 1 or first[i] >= next_id:
					continue
				var from := decode(origins[first[i]])
				var to := decode(i)
				var travel := (to - from).abs()
				first_tick_max_cells = maxi(first_tick_max_cells, maxi(travel.x, maxi(travel.y, travel.z)))
		if step >= 8:
			timestamps.append((Time.get_ticks_usec() - start) / 1000.0)
	var readback := rd.buffer_get_data(storage).to_int32_array()
	var seen := PackedByteArray()
	seen.resize(next_id)
	var duplicates := 0
	var through := 0
	var solid_changes := 0
	var moved := 0
	for i in readback.size():
		var value := readback[i]
		if (cells[i] == 1) != (value == 1):
			solid_changes += 1
		if value <= 1:
			continue
		if i % size.x >= barrier_x:
			through += 1
		if value >= next_id or seen[value] != 0:
			duplicates += 1
			continue
		seen[value] = 1
		if value != cells[i]:
			moved += 1
	var missing := 0
	for id in range(2, next_id):
		missing += int(seen[id] == 0)
	timestamps.sort()
	print(JSON.stringify({
		"grid": resolution, "dimensions": str(size), "voxels": cells.size(), "grains": next_id - 2,
		"moves_per_tick": PASSES, "roundtrip_p95_ms": timestamps[ceili(timestamps.size() * 0.95) - 1],
		"roundtrip_p50_ms": timestamps[timestamps.size() / 2], "moved": moved, "duplicate": duplicates,
		"missing": missing, "through_wall": through, "changed_stone": solid_changes, "first_tick_cells": first_tick_max_cells,
	}))
	rd.free_rid(uniform_set)
	rd.free_rid(storage)
	rd.free_rid(pipeline)
	rd.free_rid(shader)
	rd.free()
	quit(0 if duplicates == 0 and missing == 0 and through == 0 and solid_changes == 0 and moved > 0
		and first_tick_max_cells > 1 and first_tick_max_cells <= PASSES else 1)

func decode(i: int) -> Vector3i:
	return Vector3i(i % size.x, i / (size.x * size.z), (i / size.x) % size.z)
