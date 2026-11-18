extends SceneTree

const GROUP_WIDTH := 128
const GROUP_LIMIT_X := 16384

var rd: RenderingDevice
var push := PackedByteArray()
var page_count := 0

func _initialize() -> void:
	run.call_deferred()

func dispatch_pass(commands: int, mode: int, axis: int, parity: int) -> void:
	var invocations := page_count * (512 if mode == 0 else 256)
	var groups := ceili(float(invocations) / GROUP_WIDTH)
	var group_x := mini(GROUP_LIMIT_X, groups)
	push.encode_s32(16, mode)
	push.encode_s32(20, axis)
	push.encode_s32(24, parity)
	push.encode_s32(28, group_x * GROUP_WIDTH)
	rd.compute_list_set_push_constant(commands, push, push.size())
	rd.compute_list_dispatch(commands, group_x, ceili(float(groups) / group_x), 1)
	rd.compute_list_add_barrier(commands)

func run() -> void:
	var args := OS.get_cmdline_user_args()
	if args.size() != 1:
		push_error("Pass a fixture directory after --.")
		quit(1)
		return
	var fixture := args[0]
	var metadata: Dictionary = JSON.parse_string(FileAccess.get_file_as_string(fixture.path_join("metadata.json")))
	var shape: Array = metadata.shape
	var focus: Array = metadata.focus
	var updates: int = metadata.updates
	var field_radius: float = metadata.get("field_radius", 85.0)
	page_count = metadata.page_count
	rd = RenderingServer.create_local_rendering_device()
	if rd == null:
		push_error("A Vulkan RenderingDevice is required.")
		quit(1)
		return
	var source := RDShaderSource.new()
	source.source_compute = FileAccess.get_file_as_string("res://paged_material.glsl.txt")
	var spirv := rd.shader_compile_spirv_from_source(source)
	var error := spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_COMPUTE)
	if not error.is_empty():
		push_error(error)
		rd.free()
		quit(1)
		return
	var shader := rd.shader_create_from_spirv(spirv)
	var pipeline := rd.compute_pipeline_create(shader)
	var resources: Array[RID] = []
	var bindings: Array[RDUniform] = []
	var material_bytes := 0
	var tracked_bytes := 0
	for name in ["directory", "initial", "origins", "writable", "support"]:
		var bytes := PackedByteArray()
		if name != "support":
			bytes = FileAccess.get_file_as_bytes(fixture.path_join(name + ".i32"))
		var count := material_bytes if name == "support" else bytes.size()
		if name == "initial":
			material_bytes = count
		if count == 0:
			push_error("Empty buffer: " + name)
			rd.free()
			quit(1)
			return
		var buffer := rd.storage_buffer_create(count, bytes)
		var uniform := RDUniform.new()
		uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
		uniform.binding = resources.size()
		uniform.add_id(buffer)
		resources.append(buffer)
		bindings.append(uniform)
		tracked_bytes += count
	var uniform_set := rd.uniform_set_create(bindings, shader, 0)
	push = PackedInt32Array([shape[0], shape[1], shape[2], page_count]).to_byte_array()
	push.resize(48)
	var samples: Array[float] = []
	for tick in updates:
		push.encode_float(32, float(focus[0]) + 0.75 * mini(tick, updates >> 1))
		push.encode_float(36, focus[1])
		push.encode_float(40, float(focus[2]) + 0.25 * tick)
		push.encode_float(44, field_radius * field_radius if tick < floori(2.0 * updates / 3.0) else 0.0)
		var start := Time.get_ticks_usec()
		var commands := rd.compute_list_begin()
		rd.compute_list_bind_compute_pipeline(commands, pipeline)
		rd.compute_list_bind_uniform_set(commands, uniform_set, 0)
		for axis in [1, 0, 2]:
			if axis != 1:
				dispatch_pass(commands, 0, axis, 0)
			dispatch_pass(commands, 1, axis, tick & 1)
			dispatch_pass(commands, 1, axis, (tick & 1) ^ 1)
		rd.compute_list_end()
		rd.submit()
		rd.sync()
		if tick >= 4:
			samples.append((Time.get_ticks_usec() - start) / 1000.0)
	var output := FileAccess.open(fixture.path_join("vulkan.i32"), FileAccess.WRITE)
	output.store_buffer(rd.buffer_get_data(resources[1]))
	output.close()
	rd.free_rid(uniform_set)
	for buffer in resources:
		rd.free_rid(buffer)
	rd.free_rid(pipeline)
	rd.free_rid(shader)
	rd.free()
	samples.sort()
	print(JSON.stringify({
		"layout": metadata.layout, "device": RenderingServer.get_video_adapter_name(),
		"updates": updates, "tracked_mib": tracked_bytes / 1048576.0,
		"fenced_p50_ms": samples[ceili(samples.size() * 0.50) - 1],
		"fenced_p95_ms": samples[ceili(samples.size() * 0.95) - 1],
	}))
	quit()
