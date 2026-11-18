extends SceneTree

const World = preload("res://addons/somevoxel/gpu/paged_world.gd")
const Layout = preload("res://addons/somevoxel/gpu/page_layout.gd")

var failed := false

func _initialize() -> void:
	run.call_deferred()

func check(condition: bool, description: String) -> void:
	if not condition:
		push_error(description)
		failed = true

func run() -> void:
	var args := OS.get_cmdline_user_args()
	if args.size() == 2 and args[0] == "--boxes":
		benchmark_boxes(args[1])
		quit(int(failed))
		return
	if not args.is_empty():
		for fixture in args:
			verify_fixture(fixture)
		quit(int(failed))
		return
	var world := World.new()
	if not world.load_boxes(Vector3i(19, 30, 9), PackedInt32Array([
		0, 0, 0, 19, 1, 9, World.Kind.STONE,
		7, 25, 7, 9, 29, 9, World.Kind.SAND,
		7, 12, 7, 9, 15, 9, World.Kind.WATER,
		8, 26, 8, 9, 27, 9, World.Kind.AIR,
	])):
		push_error("Failed to load GPU world.")
		quit(1)
		return
	check(world.sample_cells(PackedInt32Array([8, 26, 8, 8, 27, 8, 19, 0, 0])) == PackedInt32Array([0, 2, -1]), "Box order and exact bounds")
	var initial_pages: int = world.status().resident_pages
	check(world.step(36), "Batched GPU updates")
	var count := material_counts(world, Vector3i(19, 30, 9))
	check(count[1] == 171 and count[2] == 15 and count[3] == 12, "Material count after cross-page simulation")
	check(world.sample_cells(PackedInt32Array([8, 27, 8]))[0] == World.Kind.AIR, "Sand falls through page boundaries")
	check(world.status().resident_pages > initial_pages and world.status().capacity_misses == 0, "Page allocation on moving frontier")
	check(world.set_cells(PackedInt32Array([18, 28, 8, World.Kind.SAND, 1, 0, 1, World.Kind.AIR])), "Edit across pages")
	check(world.sample_cells(PackedInt32Array([18, 28, 8, 1, 0, 1])) == PackedInt32Array([2, 0]), "Edits update the same GPU state")
	check(not world.set_cells(PackedInt32Array([18, 28, 8, 0, 18, 28, 8, 1])), "Reject non-unique edits")
	check(not world.load_boxes(Vector3i.ONE, PackedInt32Array([0, 0, 0, 2, 1, 1, 1])), "Reject bad input")
	check(world.status().tick == 36, "Invalid load preserves existing world")
	check(world.load_boxes(Vector3i(24, 16, 8), PackedInt32Array([
		8, 1, 1, 9, 2, 2, World.Kind.SAND,
	]), 6), "Load field scenario")
	world.step(10, Vector3(1, 6, 1), 20.0)
	count = material_counts(world, Vector3i(24, 16, 8))
	var in_field := PackedInt32Array()
	for x in 8:
		for y in range(2, 16):
			in_field.append_array(PackedInt32Array([x, y, 1]))
	check(count[2] == 1 and world.sample_cells(in_field).has(2), "Attraction crosses a negative page boundary and lifts a conserved grain")
	check(world.load_boxes(Vector3i(24, 8, 8), PackedInt32Array([
		0, 0, 0, 8, 8, 8, World.Kind.STONE,
	]), 1), "Load constrained world")
	check(world.set_cells(PackedInt32Array([0, 0, 0, World.Kind.AIR])), "Edit a uniform static page")
	check(world.surface_cells(Vector3i(1, 0, 0), Vector3i.ONE).cells == PackedInt32Array([1, 0, 0, 1 | 21 << 8]), "Visible face mask after an edit")
	var surface := world.surface_cells(Vector3i.ZERO, Vector3i(8, 8, 8), 1)
	check(surface.total > 1 and surface.cells.size() == 4, "Bounded surface readback reports total exposed cells")
	check(not world.set_cells(PackedInt32Array([0, 0, 0, World.Kind.WATER, 8, 0, 0, World.Kind.SAND])), "Report page pool exhaustion")
	check(world.sample_cells(PackedInt32Array([0, 0, 0, 8, 0, 0])) == PackedInt32Array([0, 0]), "Failed allocation leaves materials unchanged")
	check(world.status().capacity_misses > 0, "Count rejected allocations")
	check(world.load_boxes(Vector3i(24, 16, 8), PackedInt32Array([
		7, 1, 2, 8, 2, 3, World.Kind.SAND,
	])), "Load impulse scenario")
	check(world.apply_impulse(Vector3(7, 1, 2), 2.0, Vector3(0.9, 0.6, 0)), "Apply regional impulse")
	check(world.sample_cells(PackedInt32Array([7, 1, 2]))[0] == 2, "Impulse preserves position until stepping")
	check(world.submit_steps(6), "Submit GPU steps without waiting")
	var trajectory := PackedInt32Array()
	for x in range(9, 16):
		for y in 8:
			trajectory.append_array(PackedInt32Array([x, y, 2]))
	check(world.sample_cells(trajectory).has(2), "Query waits for the submitted cross-page impulse trajectory")
	world.close()
	check(world.status().is_empty(), "Close releases world")
	if not failed:
		print("Sparse GPU world smoke test passed.")
	quit(int(failed))

func material_counts(world: RefCounted, bounds: Vector3i) -> PackedInt32Array:
	var points := PackedInt32Array()
	for y in bounds.y:
		for z in bounds.z:
			for x in bounds.x:
				points.append_array(PackedInt32Array([x, y, z]))
	var counts := PackedInt32Array([0, 0, 0, 0])
	for kind in world.sample_cells(points):
		counts[kind] += 1
	return counts

func verify_fixture(fixture: String) -> void:
	var meta: Dictionary = JSON.parse_string(FileAccess.get_file_as_string(fixture.path_join("metadata.json")))
	var layout := Layout.new()
	layout.bounds = Vector3i(meta.bounds[0], meta.bounds[1], meta.bounds[2])
	layout.shape = Vector3i(meta.shape[0], meta.shape[1], meta.shape[2])
	layout.directory = read_ints(fixture, "directory")
	layout.atlas = read_ints(fixture, "initial")
	layout.origins = read_ints(fixture, "origins")
	layout.writable = read_ints(fixture, "writable")
	var world := World.new()
	if not world._load_layout(layout, layout.writable.size()):
		check(false, "Failed to load fixture.")
		return
	if float(meta.get("field_radius", 85.0)) != 0.0:
		check(false, "Reference fixture must use field_radius = 0.")
		world.close()
		return
	var updates: int = meta.updates
	var samples: Array[float] = []
	for tick in updates:
		var start := Time.get_ticks_usec()
		world.step(1, Vector3.ZERO, 0.0, false)
		if tick >= 4:
			samples.append((Time.get_ticks_usec() - start) / 1000.0)
	var atlas: PackedByteArray = world._rd.buffer_get_data(world._buffers[World.Buffer.ATLAS])
	var expected := FileAccess.get_file_as_bytes(fixture.path_join("vulkan.i32"))
	check(not expected.is_empty() and atlas == expected, "Full fixture matches reference atlas: " + fixture)
	samples.sort()
	print(JSON.stringify({"fixture": fixture, "p50_ms": samples[ceili(samples.size() * 0.50) - 1],
		"p95_ms": samples[ceili(samples.size() * 0.95) - 1], "status": world.status()}))
	world.close()

func read_ints(fixture: String, name: String) -> PackedInt32Array:
	return FileAccess.get_file_as_bytes(fixture.path_join(name + ".i32")).to_int32_array()

func benchmark_boxes(path: String) -> void:
	var source := FileAccess.open(path, FileAccess.READ)
	if source == null:
		check(false, "Could not read box benchmark input.")
		return
	var fixture := 0
	while not source.eof_reached():
		var line := source.get_line().strip_edges()
		if line.is_empty():
			continue
		var meta: Dictionary = JSON.parse_string(line)
		var bounds := Vector3i(meta.bounds[0], meta.bounds[1], meta.bounds[2])
		var focus: Array = meta.get("focus", [0, 0, 0])
		var start_focus := Vector3(focus[0], focus[1], focus[2])
		var radius: float = meta.get("field_radius", 0.0)
		var world := World.new()
		var start := Time.get_ticks_usec()
		if not world.load_boxes(bounds, PackedInt32Array(meta.boxes), int(meta.get("max_pages", 0))):
			check(false, "Could not load box benchmark fixture.")
			return
		var load_ms := (Time.get_ticks_usec() - start) / 1000.0
		var before := world.status()
		var samples: Array[float] = []
		for tick in 48:
			start = Time.get_ticks_usec()
			world.step(1, start_focus + Vector3(0.75 * mini(tick, 24), 0, 0.25 * tick),
				radius if tick < 32 else 0.0)
			if tick >= 4:
				samples.append((Time.get_ticks_usec() - start) / 1000.0)
		samples.sort()
		var after := world.status()
		check(after.capacity_misses == 0, "Box benchmark exhausted page capacity.")
		print(JSON.stringify({"fixture": fixture, "load_ms": load_ms, "before": before, "after": after,
			"p50_ms": samples[ceili(samples.size() * 0.50) - 1],
			"p95_ms": samples[ceili(samples.size() * 0.95) - 1]}))
		world.close()
		fixture += 1
	check(fixture > 0, "No box benchmarks found.")
