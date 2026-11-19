extends SceneTree

const Heads = preload("res://addons/somevoxel/gpu/connected_heads.gd")

func _initialize() -> void:
	call_deferred("verify")

func verify() -> void:
	var solver := Heads.new()
	var ready := solver.prepare() and solver.prepare()
	# Both X runs have the same head. The final row ends in a partial word.
	var shared_heads := solver.solve_compact(PackedByteArray([5, 5, 0, 0]), Vector3i(1, 3, 1))
	var ok := ready and shared_heads == PackedByteArray([2, 2, 0])
	# The two runs in the first row drain into columns of different heights.
	var split := solver.solve_compact(PackedByteArray([5, 4, 4, 0]), Vector3i(1, 3, 1))
	ok = ok and split == PackedByteArray([1, 3, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0])
	var tall := PackedByteArray()
	tall.resize(256)
	tall.fill(1)
	var wide := solver.solve_compact(tall, Vector3i(1, 256, 1))
	ok = ok and wide.size() == 256 * 8 and wide.decode_u16(0) == 256
	ok = ok and wide == solver.solve(tall, Vector3i(1, 256, 1))
	solver.close()
	if not ok:
		push_error("GPU water-head encodings disagreed with the expected hydraulic heights.")
	quit(0 if ok else 1)
