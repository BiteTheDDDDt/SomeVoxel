extends RefCounted

const TILE := 8
const PAGE_CELLS := 512

var bounds := Vector3i.ZERO
var shape := Vector3i.ZERO
var directory := PackedInt32Array()
var atlas := PackedInt32Array()
var origins := PackedInt32Array()
var writable := PackedInt32Array()

func key(tile: Vector3i) -> int:
	return tile.x + shape.x * (tile.z + shape.z * tile.y)

func tile_at(index: int) -> Vector3i:
	@warning_ignore("integer_division")
	return Vector3i(index % shape.x, index / (shape.x * shape.z), (index / shape.x) % shape.z)

func build(size: Vector3i, boxes: PackedInt32Array, max_pages: int) -> bool:
	if boxes.size() % 7 != 0 or max_pages < 0 or size.x <= 0 or size.y <= 0 or size.z <= 0:
		return false
	if size.x > 16_000_000 or size.y > 16_000_000 or size.z > 16_000_000:
		return false
	bounds = size
	shape = (size + Vector3i(7, 7, 7)) / TILE
	if shape.x * shape.y * shape.z > 2_000_000:
		return false
	for i in range(0, boxes.size(), 7):
		for axis in 3:
			if boxes[i + axis] < 0 or boxes[i + axis] >= boxes[i + axis + 3] or boxes[i + axis + 3] > size[axis]:
				return false
		if boxes[i + 6] < 0 or boxes[i + 6] > 3:
			return false
	directory.resize(shape.x * shape.y * shape.z)
	directory.fill(-1)
	var partial := {}
	var moving := {}
	for i in range(0, boxes.size(), 7):
		var first := Vector3i(boxes[i], boxes[i + 1], boxes[i + 2])
		var stop := Vector3i(boxes[i + 3], boxes[i + 4], boxes[i + 5])
		var kind := boxes[i + 6]
		var begin_tile := first / TILE
		var end_tile := (stop - Vector3i.ONE) / TILE
		for y in range(begin_tile.y, end_tile.y + 1):
			for z in range(begin_tile.z, end_tile.z + 1):
				for x in range(begin_tile.x, end_tile.x + 1):
					var tile := Vector3i(x, y, z)
					var origin := tile * TILE
					var id := key(tile)
					if kind >= 2:
						moving[id] = true
					var low := first.max(origin) - origin
					var high := stop.min(origin + Vector3i(8, 8, 8)) - origin
					if low == Vector3i.ZERO and high == Vector3i(8, 8, 8):
						directory[id] = -1 - kind
						partial.erase(id)
						continue
					var cells: PackedInt32Array
					if partial.has(id):
						cells = partial[id]
					else:
						cells.resize(PAGE_CELLS)
						cells.fill(-1 - directory[id])
					for ly in range(low.y, high.y):
						for lz in range(low.z, high.z):
							var row := 8 * (lz + 8 * ly)
							for lx in range(low.x, high.x):
								cells[row + lx] = kind
					partial[id] = cells
	var resident := partial.duplicate()
	for id: int in moving:
		if directory[id] <= -3:
			resident[id] = true
	var keys := resident.keys()
	if max_pages > 0 and keys.size() > max_pages:
		return false
	keys.sort()
	writable.resize(keys.size())
	var sand := PackedInt32Array()
	var water := PackedInt32Array()
	sand.resize(PAGE_CELLS)
	water.resize(PAGE_CELLS)
	sand.fill(2)
	water.fill(3)
	for slot in keys.size():
		var id: int = keys[slot]
		var active := true
		if partial.has(id):
			var cells: PackedInt32Array = partial[id]
			atlas.append_array(cells)
			active = moving.has(id) and (cells.has(2) or cells.has(3))
		else:
			atlas.append_array(sand if directory[id] == -3 else water)
		directory[id] = slot
		if active:
			var origin := tile_at(id) * TILE
			origins.append_array(PackedInt32Array([origin.x, origin.y, origin.z, slot]))
			writable[slot] = 1
	return true
