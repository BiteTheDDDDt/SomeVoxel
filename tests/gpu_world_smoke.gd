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
	world.step(24, Vector3(3.5, 12.5, 1.5), 20.0, true, 8.0)
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
		7, 4, 2, 8, 6, 3, World.Kind.WATER,
		7, 9, 2, 8, 10, 3, World.Kind.SAND,
	])), "Load impulse scenario")
	var columns := world.column_map(1, 2)
	var spans := columns.get_pixel(7, 2)
	check(Vector4(spans.r, spans.g, spans.b, spans.a).is_equal_approx(Vector4(9, 10, 4, 6)), "Column snapshot preserves the upper layer")
	var lower := columns.get_pixel(7, 10)
	check(lower.r == 1.0 and lower.g == 2.0, "A high grain does not hide a lower span")
	var hit := world.ray_hit(Vector3(7.5, 20, 2.5), Vector3.DOWN, 22.0, 1 << World.Kind.SAND)
	check(hit.is_equal_approx(Vector3(7.5, 10, 2.5)), "Ray enters the world and respects the material mask")
	check(not world.ray_hit(Vector3(9, 20, 2.5), Vector3.DOWN, 22.0).is_finite(), "Empty column is a ray miss")
	check(world.apply_impulse(Vector3(7, 1, 2), 2.0, Vector3(0.9, 0.6, 0)), "Apply regional impulse")
	check(world.sample_cells(PackedInt32Array([7, 1, 2]))[0] == 2, "Impulse preserves position until stepping")
	check(world.submit_steps(6), "Submit GPU steps without waiting")
	var trajectory := PackedInt32Array()
	for x in range(9, 16):
		for y in 8:
			trajectory.append_array(PackedInt32Array([x, y, 2]))
	check(world.sample_cells(trajectory).has(2), "Query waits for the submitted cross-page impulse trajectory")
	check(world.load_boxes(Vector3i(16, 16, 16), PackedInt32Array([
		2, 2, 2, 3, 3, 3, World.Kind.WATER,
	])), "Load water impulse scenario")
	check(world.apply_impulse(Vector3(2, 2, 2), 2.0, Vector3(1.0, 0.75, 0), World.Kind.WATER), "Apply impulse to water")
	world.step(5)
	var water_column := world.column_map(1)
	var moved_water := false
	for x in range(4, 8):
		moved_water = moved_water or water_column.get_pixel(x, 2).a > 0
	check(moved_water and material_counts(world, Vector3i(16, 16, 16))[3] == 1, "Water carries directional momentum and conserves mass")
	check(world.load_boxes(Vector3i(16, 8, 16), PackedInt32Array([
		0, 0, 0, 16, 1, 16, World.Kind.STONE,
		6, 1, 6, 9, 2, 9, World.Kind.WATER,
		3, 1, 3, 4, 5, 4, World.Kind.SAND,
	])), "Load shallow contact and slope scenario")
	world.granular_sliding = true
	check(world.set_kinematic_box(Vector3i(7, 1, 7), 2, 5), "Place kinematic obstacle in water")
	world.step(24)
	var shallow := world.column_map(1)
	var inside_water := 0
	var maximum_water_height := 0.0
	for x in 16:
		for z in 16:
			var surface_height := shallow.get_pixel(x, z).a
			maximum_water_height = maxf(maximum_water_height, surface_height)
			if x >= 6 and x <= 8 and z >= 6 and z <= 8 and surface_height > 0.0:
				inside_water += 1
	var final := material_counts(world, Vector3i(16, 8, 16))
	check(inside_water == 0 and maximum_water_height <= 2.0 and final[3] == 9,
		"Shallow water moves sideways without losing mass or rising above the body")
	check(shallow.get_pixel(3, 3).g < 5.0 and final[2] == 4, "Unsupported slope edges slide without losing sand")
	world.clear_kinematic_box()
	check(world.load_boxes(Vector3i(32, 16, 8), PackedInt32Array([
		7, 9, 2, 8, 10, 3, World.Kind.SAND,
		19, 0, 0, 20, 16, 8, World.Kind.STONE,
	])), "Load swept contact scenario")
	check(world.set_cells(PackedInt32Array([23, 9, 2, World.Kind.AIR])), "Reserve the destination page")
	check(world.apply_impulse(Vector3(7.5, 9.5, 2.5), 1.0, Vector3(15, 0, 0)), "Apply high-speed impulse")
	world.step()
	check(world.sample_cells(PackedInt32Array([7, 9, 2, 18, 9, 2, 19, 9, 2, 20, 9, 2])) == PackedInt32Array([0, 2, 1, 0]),
		"Swept motion crosses more than one page in one step without tunnelling through the wall")
	check(world.load_boxes(Vector3i(16, 16, 16), PackedInt32Array([
		4, 9, 3, 6, 10, 4, World.Kind.SAND,
	])), "Load two touching sand grains")
	check(world.apply_impulse(Vector3(4.5, 9.5, 3.5), 0.75, Vector3(6.0, 0.0, 0.0)), "Push only the rear grain")
	world.step(2)
	var contact_hits := PackedInt32Array()
	for x in range(6, 16):
		contact_hits.append_array(PackedInt32Array([x, 9, 3]))
	check(world.sample_cells(contact_hits).has(2) and material_counts(world, Vector3i(16, 16, 16))[2] == 2,
		"Sand transfers contact momentum to the other grain without changing occupied mass")
	world.close()
	check(world.status().is_empty(), "Close releases world")
	verify_connected_pressure()
	verify_growth_and_sweep()
	if not failed:
		print("Sparse GPU world smoke test passed.")
	quit(int(failed))

# Compare exact material trajectories across scheduler and pool configurations.
func verify_growth_and_sweep() -> void:
	var bounds := Vector3i(56, 24, 24)
	var boxes := PackedInt32Array([7, 15, 7, 8, 16, 8, World.Kind.SAND])
	var points := PackedInt32Array()
	for y in bounds.y:
		for z in bounds.z:
			for x in bounds.x:
				points.append_array(PackedInt32Array([x, y, z]))
	var reference: Array[PackedInt32Array] = []
	for option in 3:
		var world := World.new()
		world.connected_water_pressure = true
		world.grow_page_pool = option == 2
		world.compact_sweep = option != 0
		var capacity := 1 if world.grow_page_pool else 63
		var loaded := world.load_boxes(bounds, boxes, capacity)
		check(loaded, "Load each pool and sweep configuration")
		if not loaded:
			return
		check(world.apply_impulse(Vector3(7.5, 15.5, 7.5), 0.75, Vector3(13, 0, -4)), "Apply fast boundary impulse")
		check(world.step(1, Vector3.ZERO, 0.0, false), "A step without frontier expansion keeps the pool fixed")
		check(world.status().page_capacity == capacity, "Disabled frontier does not grow the pool")
		for sample in 3:
			check(world.submit_steps(8), "Submit a batch that may grow the pool")
			# This changes the next batch while the current submission is pending.
			check(world.set_kinematic_box(Vector3i(18 + sample, 2, 7), 2, 6), "Set next batch's obstacle")
			var materials := world.sample_cells(points)
			if option == 0:
				reference.append(materials)
			else:
				check(reference[sample] == materials, "Exact cell trajectory with compact sweep and pool growth")
		var stats := world.status()
		check(stats.tick == 25 and stats.capacity_misses == 0, "Every submitted tick completed without rejected pages")
		if world.compact_sweep:
			check(stats.sweep_pairs < stats.sweep_full_pairs, "Compact sweep reduces submitted page pairs")
		if world.grow_page_pool:
			check(stats.page_growths >= 2 and stats.page_capacity >= stats.resident_pages, "Replay supports multiple pool growths")
		world.close()
	var edits := World.new()
	edits.grow_page_pool = true
	check(edits.load_boxes(Vector3i(24, 8, 8), PackedInt32Array(), 1), "Load an empty pool for edit growth")
	check(edits.set_cells(PackedInt32Array([1, 1, 1, 1, 9, 1, 1, 2, 17, 1, 1, 3])), "A single edit batch grows across three pages")
	check(edits.sample_cells(PackedInt32Array([1, 1, 1, 9, 1, 1, 17, 1, 1])) == PackedInt32Array([1, 2, 3]), "All edits committed after growth")
	check(edits.status().page_growths == 2 and edits.status().capacity_misses == 0, "Edit growth statistics")
	edits.close()

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

func verify_connected_pressure() -> void:
	var world := World.new()
	world.connected_water_pressure = true
	world.configure_motion(0.03, 1.0 / 120.0)
	var loaded := world.load_boxes(Vector3i(64, 48, 64), PackedInt32Array([
		0, 0, 0, 64, 2, 64, World.Kind.STONE,
		25, 2, 24, 26, 25, 32, World.Kind.STONE,
		39, 2, 24, 40, 25, 32, World.Kind.STONE,
		26, 2, 24, 39, 25, 25, World.Kind.STONE,
		26, 2, 31, 39, 25, 32, World.Kind.STONE,
		32, 2, 25, 33, 25, 31, World.Kind.STONE,
		26, 2, 25, 32, 18, 31, World.Kind.WATER,
	]))
	check(loaded, "Load communicating vessels")
	if not loaded:
		return
	var left := PackedInt32Array()
	var right := PackedInt32Array()
	for y in range(2, 25):
		for z in range(25, 31):
			for x in range(26, 32):
				left.append_array(PackedInt32Array([x, y, z]))
			for x in range(33, 39):
				right.append_array(PackedInt32Array([x, y, z]))
	check(world.step(24) and world.sample_cells(left).count(World.Kind.WATER) == 576
		and not world.sample_cells(right).has(World.Kind.WATER), "Sealed partition blocks pressure")
	check(world.set_cells(PackedInt32Array([32, 2, 28, World.Kind.AIR])) and world.step(480),
		"Open bottom connection and run connected pressure")
	var low := world.sample_cells(left)
	var high := world.sample_cells(right)
	var gate := world.sample_cells(PackedInt32Array([32, 2, 28]))
	check(low.count(World.Kind.WATER) + high.count(World.Kind.WATER) + gate.count(World.Kind.WATER) == 576,
		"Connected pressure conserves the vessel mass")
	check(absi(low.count(World.Kind.WATER) - high.count(World.Kind.WATER)) <= 16
		and world.status().pressure_transfers > 0, "Communicating vessels fill and equalize across a page boundary")
	world.close()

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
	world.ballistic_sand = false
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
