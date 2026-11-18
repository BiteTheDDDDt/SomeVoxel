extends SceneTree

const WIDTH := 40
const HEIGHT := 112
const DEPTH := 256
const ROW_COUNT := WIDTH * HEIGHT * DEPTH
const WORKGROUP_COUNT := ROW_COUNT >> 7
const WARMUP := 4
const SAMPLES := 12

func _initialize() -> void:
	run.call_deferred()

func run() -> void:
	var device := RenderingServer.create_local_rendering_device()
	if device == null:
		push_error("This benchmark needs a Vulkan rendering device.")
		quit(1)
		return
	var rows := FileAccess.get_file_as_bytes("res://rows.u8")
	var expected := FileAccess.get_file_as_bytes("res://expected.u16")
	var spirv: RDShaderSPIRV = preload("head.glsl").get_spirv()
	var error := spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_COMPUTE)
	if not error.is_empty() or rows.size() != ROW_COUNT or expected.size() != ROW_COUNT * 8:
		push_error("Invalid shader or fixture. " + error)
		device.free()
		quit(1)
		return
	var shader := device.shader_create_from_spirv(spirv)
	var pipeline := device.compute_pipeline_create(shader)
	var masks := device.storage_buffer_create(rows.size(), rows)
	var runs := device.storage_buffer_create(ROW_COUNT * 32)
	var heads := device.storage_buffer_create(ROW_COUNT * 8)
	var bindings: Array[RDUniform] = []
	for target: RID in [masks, runs, heads]:
		var uniform := RDUniform.new()
		uniform.binding = bindings.size()
		uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
		uniform.add_id(target)
		bindings.append(uniform)
	var set_id := device.uniform_set_create(bindings, shader, 0)
	var fenced: Array[float] = []
	var roundtrip: Array[float] = []
	var mismatch := 0
	for cycle in 2:
		for sample in WARMUP + SAMPLES:
			var begin := Time.get_ticks_usec()
			if cycle != 0:
				device.buffer_update(masks, 0, rows.size(), rows)
			var list := device.compute_list_begin()
			device.compute_list_bind_compute_pipeline(list, pipeline)
			device.compute_list_bind_uniform_set(list, set_id, 0)
			for mode in 4:
				var push := PackedInt32Array([WIDTH, HEIGHT, DEPTH, mode]).to_byte_array()
				device.compute_list_set_push_constant(list, push, push.size())
				device.compute_list_dispatch(list, WORKGROUP_COUNT, 1, 1)
				if mode < 3:
					device.compute_list_add_barrier(list)
			device.compute_list_end()
			device.submit()
			device.sync()
			var output := device.buffer_get_data(heads) if cycle != 0 else PackedByteArray()
			var cost := (Time.get_ticks_usec() - begin) / 1000.0
			if sample >= WARMUP:
				if cycle == 0:
					fenced.append(cost)
				else:
					roundtrip.append(cost)
			if cycle != 0 and output != expected:
				mismatch += 1
	fenced.sort()
	roundtrip.sort()
	print(JSON.stringify({"gpu_p50_ms": fenced[5], "gpu_p95_ms": fenced.back(),
		"roundtrip_p50_ms": roundtrip[5], "roundtrip_p95_ms": roundtrip.back(), "mismatch_trials": mismatch,
		"input_bytes": rows.size(), "result_bytes": expected.size()}))
	for item: RID in [set_id, pipeline, shader, masks, runs, heads]:
		device.free_rid(item)
	device.free()
	quit(0 if mismatch == 0 else 1)
