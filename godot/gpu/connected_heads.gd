extends RefCounted
## Synchronous Vulkan accelerator for the ordinary water-head rebuild.
## The caller owns the authoritative snapshot and commits the returned heads.

const Kernel = preload("connected_heads.glsl")
const GROUP_WIDTH := 128
const DISPATCH_WIDTH := 16384
const MAX_ROWS := 2_000_000
const CAPACITY_STEP := 4096
const HEAD_BYTES_PER_ROW := 8
const RUN_BYTES_PER_ROW := 32

var _rd: RenderingDevice
var _shader := RID()
var _pipeline := RID()
var _uniforms := RID()
var _buffers: Array[RID] = []
var _capacity := 0

## Pay device and shader startup during map loading on eligible worlds.
func prepare() -> bool:
	if _capacity > 0:
		return true
	return solve_compact(PackedByteArray([1, 0, 0, 0]), Vector3i.ONE) == PackedByteArray([1])

## shape.x is the number of packed 8-cell rows along X. Input is padded
## to four bytes. Each output row contains four little-endian uint16 heads
## in the order of its occupied X runs. Unused entries are zero.
func solve(rows: PackedByteArray, shape: Vector3i) -> PackedByteArray:
	return _solve(rows, shape, false)

## Returns one shared byte per row, four bytes per row when runs have
## different heads, or four uint16 heads for tall water volumes.
func solve_compact(rows: PackedByteArray, shape: Vector3i) -> PackedByteArray:
	return _solve(rows, shape, shape.y < 255)

func _solve(rows: PackedByteArray, shape: Vector3i, compact: bool) -> PackedByteArray:
	var empty := PackedByteArray()
	for axis in 3:
		if shape[axis] <= 0 or shape[axis] > MAX_ROWS:
			return empty
	var count := shape.x * shape.y * shape.z
	if count > MAX_ROWS or shape.y > 65535 or rows.size() != ((count + 3) & ~3):
		return empty
	if not _ensure_buffers(count):
		return empty
	if _rd.buffer_update(_buffers[0], 0, rows.size(), rows) != OK:
		return empty
	var group_count := ceili(float(count) / GROUP_WIDTH)
	var constants := PackedInt32Array([shape.x, shape.y, shape.z, 0]).to_byte_array()
	var commands := _rd.compute_list_begin()
	_rd.compute_list_bind_compute_pipeline(commands, _pipeline)
	_rd.compute_list_bind_uniform_set(commands, _uniforms, 0)
	for mode in 4:
		var groups := group_count
		var publish_compact := compact and mode == 3
		if publish_compact:
			groups = ceili(float(count) / (4 * GROUP_WIDTH))
		constants.encode_s32(12, 4 if publish_compact else mode)
		_rd.compute_list_set_push_constant(commands, constants, constants.size())
		_rd.compute_list_dispatch(commands, mini(groups, DISPATCH_WIDTH), ceili(float(groups) / DISPATCH_WIDTH), 1)
		if mode < 3:
			_rd.compute_list_add_barrier(commands)
	_rd.compute_list_end()
	# Queue the transfer before the fence. Keep the destination local so a
	# callback delivered later by another Godot version cannot replace it.
	var read_bytes := ((count + 3) & ~3) if compact else count * HEAD_BYTES_PER_ROW
	var result: Array[PackedByteArray] = [PackedByteArray()]
	var receive := func(data: PackedByteArray) -> void: result[0] = data
	_rd.buffer_get_data_async(_buffers[2], receive, 0, read_bytes)
	_rd.submit()
	_rd.sync()
	var heads := result[0]
	if heads.size() != read_bytes:
		heads = _rd.buffer_get_data(_buffers[2], 0, read_bytes)
	if heads.size() != read_bytes:
		return empty
	if compact:
		if heads.has(255):
			return _rd.buffer_get_data(_buffers[2], read_bytes, count * 4)
		heads.resize(count)
	return heads

func close() -> void:
	if _rd == null:
		return
	_free_buffers()
	if _pipeline.is_valid():
		_rd.free_rid(_pipeline)
	if _shader.is_valid():
		_rd.free_rid(_shader)
	_rd.free()
	_rd = null
	_shader = RID()
	_pipeline = RID()

func _notification(what: int) -> void:
	if what == NOTIFICATION_PREDELETE and _rd != null:
		for resource: RID in [_uniforms, _pipeline, _shader]:
			if resource.is_valid():
				_rd.free_rid(resource)
		for buffer in _buffers:
			_rd.free_rid(buffer)
		_rd.free()

func _ensure_buffers(count: int) -> bool:
	if _rd == null:
		_rd = RenderingServer.create_local_rendering_device()
		if _rd == null:
			return false
		var spirv: RDShaderSPIRV = Kernel.get_spirv()
		if spirv == null or not spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_COMPUTE).is_empty():
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
	if count <= _capacity:
		return true
	_free_buffers()
	var capacity := mini(MAX_ROWS, (count + CAPACITY_STEP - 1) & ~(CAPACITY_STEP - 1))
	var bindings: Array[RDUniform] = []
	for bytes in [capacity, capacity * RUN_BYTES_PER_ROW, capacity * HEAD_BYTES_PER_ROW]:
		var buffer := _rd.storage_buffer_create(bytes)
		if not buffer.is_valid():
			_free_buffers()
			return false
		var uniform := RDUniform.new()
		uniform.binding = bindings.size()
		uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
		uniform.add_id(buffer)
		_buffers.append(buffer)
		bindings.append(uniform)
	_uniforms = _rd.uniform_set_create(bindings, _shader, 0)
	if not _uniforms.is_valid():
		_free_buffers()
		return false
	_capacity = capacity
	return true

func _free_buffers() -> void:
	if _uniforms.is_valid():
		_rd.free_rid(_uniforms)
	for buffer in _buffers:
		_rd.free_rid(buffer)
	_buffers.clear()
	_uniforms = RID()
	_capacity = 0
