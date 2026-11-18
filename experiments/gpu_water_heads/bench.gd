extends SceneTree

const WIDTH := 40
const HEIGHT := 112
const DEPTH := 256
const ROW_COUNT := WIDTH * HEIGHT * DEPTH
const GROUP_SIZES := [64, 128, 256]
const STAGES := ["init", "union", "rank", "publish"]
const WARMUP := 6
const SAMPLES := 40
const TIMESTAMP_PREFIX := "head-pass-"

func _initialize() -> void:
	run.call_deferred()

func run() -> void:
	var rows := FileAccess.get_file_as_bytes("res://rows.u8")
	var expected := FileAccess.get_file_as_bytes("res://expected.u16")
	if rows.size() != ROW_COUNT or expected.size() != ROW_COUNT * 8:
		push_error("Invalid fixture.")
		quit(1)
		return
	var sizes := GROUP_SIZES
	var args := OS.get_cmdline_user_args()
	if not args.is_empty():
		if args.size() != 2 or args[0] != "--group-size" or not args[1].is_valid_int() or not GROUP_SIZES.has(args[1].to_int()):
			push_error("Expected --group-size 64, 128, or 256.")
			quit(1)
			return
		sizes = [args[1].to_int()]
	var device := RenderingServer.create_local_rendering_device()
	if device == null:
		push_error("This benchmark needs a Vulkan rendering device.")
		quit(1)
		return
	var mismatches := 0
	for group_size: int in sizes:
		mismatches += benchmark(device, rows, expected, group_size)
	device.free()
	quit(0 if mismatches == 0 else 1)

func benchmark(device: RenderingDevice, rows: PackedByteArray, expected: PackedByteArray, group_size: int) -> int:
	var source := RDShaderSource.new()
	var text := FileAccess.get_file_as_string("res://head.glsl")
	source.source_compute = text.substr(text.find("\n") + 1).replace(
		"#version 450", "#version 450\n#define HEAD_WORKGROUP_SIZE %d" % group_size)
	var spirv := device.shader_compile_spirv_from_source(source)
	var error := spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_COMPUTE)
	if not error.is_empty():
		push_error(error)
		return 1
	var shader := device.shader_create_from_spirv(spirv)
	var pipeline := device.compute_pipeline_create(shader)
	var masks := device.storage_buffer_create(ROW_COUNT, rows)
	var runs := device.storage_buffer_create(ROW_COUNT * 32)
	var heads := device.storage_buffer_create(ROW_COUNT * 8)
	var upload_probe := device.storage_buffer_create(4)
	var bindings: Array[RDUniform] = []
	for target: RID in [masks, runs, heads]:
		var uniform := RDUniform.new()
		uniform.binding = bindings.size()
		uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
		uniform.add_id(target)
		bindings.append(uniform)
	var set_id := device.uniform_set_create(bindings, shader, 0)
	var groups := ceili(float(ROW_COUNT) / group_size)
	var constants: Array[PackedByteArray] = []
	for mode in STAGES.size():
		constants.append(PackedInt32Array([WIDTH, HEIGHT, DEPTH, mode]).to_byte_array())
	var times := {"fenced": [], "roundtrip": [], "upload_enqueue": [], "roundtrip_submit_wait": [],
		"readback": [], "upload_fenced": []}
	var phase_times: Dictionary = {}
	for stage: String in STAGES:
		phase_times[stage] = []
	var mismatches := 0
	for sample in WARMUP + SAMPLES:
		var begin := Time.get_ticks_usec()
		dispatch(device, pipeline, set_id, groups, constants)
		device.submit()
		device.sync()
		if sample >= WARMUP:
			times.fenced.append(milliseconds(begin))

		begin = Time.get_ticks_usec()
		device.buffer_update(masks, 0, rows.size(), rows)
		var after_upload := Time.get_ticks_usec()
		dispatch(device, pipeline, set_id, groups, constants)
		device.submit()
		device.sync()
		var after_compute := Time.get_ticks_usec()
		var output := device.buffer_get_data(heads)
		var end := Time.get_ticks_usec()
		if output != expected:
			mismatches += 1
		if sample >= WARMUP:
			times.roundtrip.append((end - begin) / 1000.0)
			times.upload_enqueue.append((after_upload - begin) / 1000.0)
			times.roundtrip_submit_wait.append((after_compute - after_upload) / 1000.0)
			times.readback.append((end - after_compute) / 1000.0)

	for sample in WARMUP + SAMPLES:
		var begin := Time.get_ticks_usec()
		device.buffer_update(masks, 0, rows.size(), rows)
		# Force the submitted queue to consume the uploaded mask.
		device.buffer_copy(masks, upload_probe, 0, 0, 4)
		device.submit()
		device.sync()
		if sample >= WARMUP:
			times.upload_fenced.append(milliseconds(begin))

	device.buffer_clear(heads, 0, expected.size())
	device.submit()
	device.sync()
	for sample in WARMUP + SAMPLES:
		var count := STAGES.size()
		for mode in count:
			device.capture_timestamp(TIMESTAMP_PREFIX + str(mode))
			var list := begin_compute(device, pipeline, set_id)
			device.compute_list_set_push_constant(list, constants[mode], constants[mode].size())
			device.compute_list_dispatch(list, groups, 1, 1)
			device.compute_list_end()
		device.capture_timestamp(TIMESTAMP_PREFIX + str(count))
		device.submit()
		device.sync()
		if sample >= WARMUP:
			var marks := gpu_timestamps(device)
			for mode in count:
				if marks.has(mode) and marks.has(mode + 1):
					phase_times[STAGES[mode]].append((marks[mode + 1] - marks[mode]) / 1_000_000.0)
	if device.buffer_get_data(heads) != expected:
		mismatches += 1
	var results := {
		"workgroup": group_size,
		"groups": groups,
		"input_bytes": rows.size(),
		"result_bytes": expected.size(),
		"mismatch_trials": mismatches,
		"host_ms": summarize(times),
		"gpu_stage_ms": summarize(phase_times),
	}
	print(JSON.stringify(results))
	for item: RID in [set_id, pipeline, shader, masks, runs, heads, upload_probe]:
		device.free_rid(item)
	return mismatches

func dispatch(device: RenderingDevice, pipeline: RID, set_id: RID, groups: int, constants: Array[PackedByteArray]) -> void:
	var list := begin_compute(device, pipeline, set_id)
	for mode in STAGES.size():
		var push := constants[mode]
		device.compute_list_set_push_constant(list, push, push.size())
		device.compute_list_dispatch(list, groups, 1, 1)
		if mode + 1 < STAGES.size():
			device.compute_list_add_barrier(list)
	device.compute_list_end()

func begin_compute(device: RenderingDevice, pipeline: RID, set_id: RID) -> int:
	var list := device.compute_list_begin()
	device.compute_list_bind_compute_pipeline(list, pipeline)
	device.compute_list_bind_uniform_set(list, set_id, 0)
	return list

func gpu_timestamps(device: RenderingDevice) -> Dictionary:
	var marks := {}
	for i in device.get_captured_timestamps_count():
		var name_ := device.get_captured_timestamp_name(i)
		if name_.begins_with(TIMESTAMP_PREFIX):
			marks[name_.trim_prefix(TIMESTAMP_PREFIX).to_int()] = device.get_captured_timestamp_gpu_time(i)
	return marks

func milliseconds(start: int) -> float:
	return (Time.get_ticks_usec() - start) / 1000.0

func summarize(series: Dictionary) -> Dictionary:
	var output := {}
	for stage: String in series:
		var data: Array = series[stage]
		if data.is_empty():
			continue
		data.sort()
		output[stage] = {"p50": data[ceili(data.size() * 0.50) - 1], "p95": data[ceili(data.size() * 0.95) - 1]}
	return output
