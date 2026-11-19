#pragma once

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

namespace somevoxel {

// The caller participates in each phase and waits before sharing results.
class HeadWorkers {
public:
    explicit HeadWorkers(int count) : count(count) {
        threads.reserve(count - 1);
        try {
            for (int id = 1; id < count; ++id)
                threads.emplace_back([this, id] { loop(id); });
        } catch (...) {
            finish();
            throw;
        }
    }

    ~HeadWorkers() { finish(); }

    template <typename Job> void run(Job job) {
        if (count == 1) { job(0); return; }
        {
            std::lock_guard lock(mutex);
            context = &job;
            work = [](void *data, int id) { (*static_cast<Job *>(data))(id); };
            pending = count - 1;
            ++generation;
        }
        ready.notify_all();
        job(0);
        std::unique_lock lock(mutex);
        done.wait(lock, [&] { return pending == 0; });
    }

    const int count;

private:
    void finish() {
        {
            std::lock_guard lock(mutex);
            stop = true;
        }
        ready.notify_all();
        for (auto &thread : threads) thread.join();
    }

    void loop(int id) {
        std::unique_lock lock(mutex);
        unsigned seen = 0;
        while (true) {
            ready.wait(lock, [&] { return stop || seen != generation; });
            if (stop) return;
            seen = generation;
            lock.unlock();
            work(context, id);
            lock.lock();
            if (!--pending) done.notify_one();
        }
    }

    std::mutex mutex;
    std::condition_variable ready, done;
    std::vector<std::thread> threads;
    void *context = nullptr;
    void (*work)(void *, int) = nullptr;
    unsigned generation = 0;
    int pending = 0;
    bool stop = false;
};

template <int X, int Y, int Z> class WaterSpans {
    static_assert(X % 8 == 0 && Z % 8 == 0);
    static constexpr int x_rows = X / 8;
    static constexpr int z_blocks = Z / 8;
    static constexpr int slice_rows = X * Z / 8;

    struct Span {
        int cell, width, parent, head;
    };
    struct Row {
        int begin = 0;
        uint8_t occupied = 0, count = 0;
        int end() const { return begin + count; }
    };

public:
    struct SurfaceRun {
        int component;
        int cell;
        unsigned mask;
    };
    using Surfaces = std::vector<std::vector<SurfaceRun>>;

    explicit WaterSpans(int count) : workers(count) {}

    HeadWorkers &worker_pool() { return workers; }

    template <typename Apply> void rebuild(const std::vector<int> &water_positions, Apply apply) {
        if (water_positions.empty()) return;
        const int highest = *std::max_element(water_positions.begin(), water_positions.end());
        ensure_rows(highest / (X * Z) + 1);
        for (int i : water_positions) mark_water(i);
        propagate_heads(apply);
    }

    template <typename LiquidMotion, typename Apply>
    void rebuild_rows(const std::vector<uint8_t> &bulk, const LiquidMotion &moving, int height, Apply apply) {
        mark_rows(bulk, moving, height);
        connect_runs();
        publish_heads(apply);
    }

    template <typename LiquidMotion, typename Apply, typename Decide>
    bool try_pressure_rows(const std::vector<uint8_t> &bulk, const LiquidMotion &moving, int height,
            Surfaces &surfaces, Apply apply, Decide can_publish) {
        mark_rows(bulk, moving, height);
        connect_runs();
        surfaces.resize(workers.count);
        const auto visit_surface_runs = [&](int k, auto visit) {
            for (int z = bands[k]; z < bands[k + 1]; ++z) {
                for (int row_id : rows_by_z[z]) {
                    if (row_id / slice_rows == Y - 1) continue;
                    const Row &row = rows[row_id];
                    const unsigned open = row.occupied & ~unsigned(rows[row_id + slice_rows].occupied);
                    if (!open) continue;
                    for (int id = row.begin; id < row.end(); ++id) {
                        const Span &span = spans[id];
                        const unsigned mask = open & (((1u << span.width) - 1u) << (span.cell % 8));
                        if (mask) visit(id, row_id * 8, mask);
                    }
                }
            }
        };
        std::vector<size_t> counts(workers.count);
        workers.run([&](int k) {
            visit_surface_runs(k, [&](int, int, unsigned) { ++counts[k]; });
        });
        for (int k = 0; k < workers.count; ++k) surfaces[k].resize(counts[k]);
        workers.run([&](int k) {
            auto output = surfaces[k].begin();
            visit_surface_runs(k, [&](int id, int cell, unsigned mask) {
                *output++ = {root(id), cell, mask};
            });
        });
        if (can_publish(surfaces)) {
            publish_heads(apply);
            return true;
        }
        workers.run([&](int k) {
            for (int z = bands[k]; z < bands[k + 1]; ++z) {
                for (int row_id : rows_by_z[z]) rows[row_id].occupied = 0;
                rows_by_z[z].clear();
            }
        });
        return false;
    }

private:
    void ensure_rows(int occupied_layers) {
        // Joining and scanning the highest occupied layer may read an empty
        // row immediately above it. Keep that sentinel when below the ceiling.
        const size_t extent = size_t(std::min(Y, occupied_layers + 1)) * slice_rows;
        if (rows.size() < extent) rows.resize(extent);
    }

    template <typename LiquidMotion>
    void mark_rows(const std::vector<uint8_t> &bulk, const LiquidMotion &moving, int height) {
        int occupied_layers = height;
        for (const auto &entry : moving)
            occupied_layers = std::max(occupied_layers, entry.first / (X * Z) + 1);
        ensure_rows(occupied_layers);
        // Entire Z bands belong to one worker. The packed source rows are
        // already available from the simulation's live occupancy masks.
        const auto run_bands = [&](auto visit) {
            workers.run([&](int k) {
                const int first = z_blocks * k / workers.count;
                const int last = z_blocks * (k + 1) / workers.count;
                for (int z = first; z < last; ++z) visit(z);
            });
        };
        std::array<int, z_blocks> occupied_counts{};
        run_bands([&](int z) {
            int count = 0;
            for (int y = 0; y < height; ++y) {
                auto first = bulk.begin() + y * slice_rows + z * 8 * x_rows;
                count += int(std::count_if(first, first + 8 * x_rows, [](uint8_t mask) { return mask != 0; }));
            }
            occupied_counts[z] = count;
        });
        // Allocate on the caller. The workers only touch pre-sized vectors.
        for (int z = 0; z < z_blocks; ++z) rows_by_z[z].resize(occupied_counts[z]);
        run_bands([&](int z) {
            auto output = rows_by_z[z].begin();
            for (int y = 0; y < height; ++y) {
                int row = y * slice_rows + z * 8 * x_rows;
                const int end = row + 8 * x_rows;
                // Every band contains a multiple of eight bytes.
                for (; row < end; row += 8) {
                    uint64_t packed;
                    std::memcpy(&packed, bulk.data() + row, sizeof(packed));
                    if (!packed) continue;
                    for (int offset = 0; offset < 8; ++offset) {
                        const uint8_t mask = bulk[row + offset];
                        if (!mask) continue;
                        rows[row + offset].occupied = mask;
                        *output++ = row + offset;
                    }
                }
            }
        });
        // Moving parcels are absent from bulk rows but participate in the
        // same face-connected components and hydraulic heads.
        for (const auto &entry : moving) mark_water(entry.first);
    }

public:
    template <typename LiquidMotion, typename Accelerate, typename Apply>
    bool try_accelerated_rows(const std::vector<uint8_t> &bulk, const LiquidMotion &moving, int height,
            Accelerate accelerate, Apply apply) {
        struct Bounds {
            int x0 = x_rows, y0 = Y, z0 = Z;
            int x1 = 0, y1 = 0, z1 = 0;
            void include(int first, int last, int y, int z) {
                x0 = std::min(x0, first);
                x1 = std::max(x1, last);
                y0 = std::min(y0, y);
                y1 = std::max(y1, y + 1);
                z0 = std::min(z0, z);
                z1 = std::max(z1, z + 1);
            }
        };
        std::vector<Bounds> bounds(workers.count);
        workers.run([&](int id) {
            Bounds &box = bounds[id];
            for (int y = 0; y < height; ++y) {
                for (int z = Z * id / workers.count; z < Z * (id + 1) / workers.count; ++z) {
                    const uint8_t *line = bulk.data() + y * slice_rows + z * x_rows;
                    const auto wet = [](uint8_t mask) { return mask != 0; };
                    const auto first = std::find_if(line, line + x_rows, wet);
                    if (first == line + x_rows) continue;
                    const auto last = std::find_if(std::make_reverse_iterator(line + x_rows),
                        std::make_reverse_iterator(first), wet).base();
                    box.include(int(first - line), int(last - line), y, z);
                }
            }
        });
        Bounds box;
        for (const auto &part : bounds) {
            if (part.x1 == 0) continue;
            box.include(part.x0, part.x1, part.y0, part.z0);
            box.include(part.x0, part.x1, part.y1 - 1, part.z1 - 1);
        }
        for (const auto &entry : moving) {
            const int cell = entry.first;
            const int x = cell / 8 % x_rows;
            box.include(x, x + 1, cell / (X * Z), cell / X % Z);
        }
        const int width = box.x1 - box.x0, levels = box.y1 - box.y0, depth = box.z1 - box.z0;
        if (width <= 0 || levels <= 0 || depth <= 0) return false;
        const size_t count = size_t(width) * depth * levels;
        if (count > 2000000) return false;
        accelerator_masks.resize((count + 3) & ~size_t(3));
        std::fill(accelerator_masks.begin() + count, accelerator_masks.end(), 0);
        workers.run([&](int id) {
            for (int y = levels * id / workers.count; y < levels * (id + 1) / workers.count; ++y) {
                for (int z = 0; z < depth; ++z) {
                    const size_t dest = size_t(y * depth + z) * width;
                    const int source = (box.y0 + y) * slice_rows + (box.z0 + z) * x_rows + box.x0;
                    std::memcpy(accelerator_masks.data() + dest, bulk.data() + source, width);
                }
            }
        });
        for (const auto &entry : moving) {
            const int cell = entry.first;
            const int row = ((cell / (X * Z) - box.y0) * depth + cell / X % Z - box.z0) * width
                + cell / 8 % x_rows - box.x0;
            accelerator_masks[row] |= uint8_t(1u << (cell & 7));
        }
        if (!accelerate(accelerator_masks, width, levels, depth, accelerator_heads)) return false;
        const size_t bytes_per_row = accelerator_heads.size() / count;
        if ((bytes_per_row != 1 && bytes_per_row != 4 && bytes_per_row != 8)
                || accelerator_heads.size() != count * bytes_per_row) return false;

        const auto head_at = [&](size_t row, int run) {
            if (bytes_per_row == 1) return int(accelerator_heads[row]);
            if (bytes_per_row == 4) return int(accelerator_heads[row * 4 + run]);
            const size_t offset = row * 8 + run * 2;
            return int(accelerator_heads[offset]) | (int(accelerator_heads[offset + 1]) << 8);
        };
        // Validate every occupied run before touching the live world's heads.
        std::vector<uint8_t> invalid(workers.count);
        workers.run([&](int id) {
            for (int y = levels * id / workers.count; y < levels * (id + 1) / workers.count; ++y) {
                const size_t begin = size_t(y) * depth * width;
                for (size_t row = begin; row < begin + depth * width; ++row) {
                    unsigned starts = accelerator_masks[row] & ~(unsigned(accelerator_masks[row]) << 1);
                    int run = 0;
                    for (; starts != 0; starts &= starts - 1) {
                        const int head = head_at(row, run++);
                        if (head <= y || head > levels) {
                            invalid[id] = 1;
                            return;
                        }
                    }
                }
            }
        });
        if (std::find(invalid.begin(), invalid.end(), 1) != invalid.end()) return false;

        // Keep a whole Z chunk on one worker so a wake flag has one writer.
        const int first_band = box.z0 / 8;
        const int bands = (box.z1 + 7) / 8 - first_band;
        workers.run([&](int id) {
            const int z0 = std::max(box.z0, (first_band + bands * id / workers.count) * 8);
            const int z1 = std::min(box.z1, (first_band + bands * (id + 1) / workers.count) * 8);
            for (int y = 0; y < levels; ++y) {
                for (int z = z0; z < z1; ++z) {
                    const int source = (box.y0 + y) * slice_rows + z * x_rows + box.x0;
                    for (int x = 0; x < width; ++x) {
                        const size_t row = size_t(y * depth + z - box.z0) * width + x;
                        const int cell = (source + x) * 8;
                        unsigned mask = accelerator_masks[row];
                        int run = 0;
                        while (mask != 0) {
                            const int first = first_bit(mask);
                            const int length = first_bit(~(mask >> first));
                            apply_run(apply, cell + first, length, head_at(row, run++) + box.y0);
                            mask &= ~(((1u << length) - 1u) << first);
                        }
                    }
                }
            }
        });
        return true;
    }

    size_t scratch_bytes() const {
        size_t bytes = rows.capacity() * sizeof(Row) + spans.capacity() * sizeof(Span);
        for (const auto &rows : rows_by_z) bytes += rows.capacity() * sizeof(int);
        return bytes;
    }

private:
    void mark_water(int i) {
        Row &row = rows[i / 8];
        if (!row.occupied) rows_by_z[i / X % Z / 8].push_back(i / 8);
        row.occupied |= uint8_t(1u << (i % 8));
    }

    template <typename Apply> void propagate_heads(Apply apply) {
        connect_runs();
        publish_heads(apply);
    }

    void connect_runs() {
        std::array<int, z_blocks + 1> offsets{};
        std::array<int, z_blocks + 1> weights{};
        for (int z = 0; z < z_blocks; ++z) {
            offsets[z + 1] = offsets[z];
            weights[z + 1] = weights[z] + int(rows_by_z[z].size());
            for (int id : rows_by_z[z]) {
                const unsigned mask = rows[id].occupied;
                offsets[z + 1] += popcount(mask & ~(mask << 1));
            }
        }
        spans.resize(offsets.back());
        bands.resize(workers.count + 1);
        bands.front() = 0;
        bands.back() = z_blocks;
        for (int k = 1; k < workers.count; ++k)
            bands[k] = int(std::lower_bound(weights.begin(), weights.end(), weights.back() * k / workers.count) - weights.begin());

        workers.run([&](int k) {
            for (int z = bands[k]; z < bands[k + 1]; ++z) {
                int id = offsets[z];
                for (int row_id : rows_by_z[z]) {
                    Row &row = rows[row_id];
                    row.begin = id;
                    unsigned mask = row.occupied;
                    while (mask != 0) {
                        const int start = first_bit(mask);
                        const int width = first_bit(~(mask >> start));
                        spans[id] = {row_id * 8 + start, width, id, row_id / slice_rows + 1};
                        ++id;
                        mask &= ~(((1u << width) - 1) << start);
                    }
                    row.count = uint8_t(id - row.begin);
                }
            }
            // The ownership band includes complete voxel chunks, so head
            // changes can later wake chunks without atomics or copies.
            for (int z = bands[k]; z < bands[k + 1]; ++z)
                for (int row_id : rows_by_z[z])
                    join_inside(row_id, bands[k + 1] * 8);
        });

        for (int k = 1; k < workers.count; ++k) {
            const int seam = bands[k] * 8;
            if (seam == 0 || seam == Z || (k > 1 && bands[k] == bands[k - 1])) continue;
            for (int row : rows_by_z[bands[k] - 1])
                if (row / x_rows % Z == seam - 1) pair_rows(row, row + x_rows);
        }
    }

    template <typename Apply> void publish_heads(Apply apply) {
        workers.run([&](int k) {
            for (int z = bands[k]; z < bands[k + 1]; ++z) {
                for (int row_id : rows_by_z[z]) {
                    Row &row = rows[row_id];
                    for (int id = row.begin; id < row.end(); ++id) {
                        const Span &span = spans[id];
                        const int head = spans[root(id)].head;
                        apply_run(apply, span.cell, span.width, head);
                    }
                    row.occupied = 0;
                }
                rows_by_z[z].clear();
            }
        });
    }

    template <typename Apply>
    static void apply_run(Apply &apply, int first, int width, int head) {
        if constexpr (std::is_invocable_v<Apply, int, int, int>) {
            apply(first, width, head);
        } else {
            for (int i = first; i < first + width; ++i) apply(i, head);
        }
    }

    HeadWorkers workers;
    std::vector<Row> rows;
    std::vector<Span> spans;
    std::vector<uint8_t> accelerator_masks, accelerator_heads;
    std::array<std::vector<int>, z_blocks> rows_by_z;
    std::vector<int> bands;

    static int first_bit(unsigned bits) {
#if defined(__GNUC__) || defined(__clang__)
        return __builtin_ctz(bits);
#else
        int n = 0;
        while (!(bits & 1)) { ++n; bits >>= 1; }
        return n;
#endif
    }
    static int popcount(unsigned bits) {
#if defined(__GNUC__) || defined(__clang__)
        return __builtin_popcount(bits);
#else
        int n = 0;
        while (bits) { bits &= bits - 1; ++n; }
        return n;
#endif
    }

    int root(int id) const {
        while (spans[id].parent != id) id = spans[id].parent;
        return id;
    }

    int compress(int id) {
        while (spans[id].parent != id) {
            spans[id].parent = spans[spans[id].parent].parent;
            id = spans[id].parent;
        }
        return id;
    }

    void join(int a, int b) {
        a = compress(a);
        b = compress(b);
        if (a == b) return;
        if (b < a) std::swap(a, b);
        spans[b].parent = a;
        spans[a].head = std::max(spans[a].head, spans[b].head);
    }

    void pair_rows(int a, int b) {
        const Row &first = rows[a], &second = rows[b];
        if (!(first.occupied & second.occupied)) return;
        int ai = first.begin, bi = second.begin;
        while (ai < first.end() && bi < second.end()) {
            const Span &left = spans[ai], &right = spans[bi];
            const int start_left = left.cell % 8, start_right = right.cell % 8;
            const int end_left = start_left + left.width, end_right = start_right + right.width;
            if (start_left < end_right && start_right < end_left) join(ai, bi);
            if (end_left <= end_right) ++ai;
            if (end_right <= end_left) ++bi;
        }
    }

    void join_inside(int row_id, int last_z) {
        const Row &row = rows[row_id];
        if ((row.occupied & 128) && row_id % x_rows != x_rows - 1 && (rows[row_id + 1].occupied & 1))
            join(row.end() - 1, rows[row_id + 1].begin);
        if (row_id / x_rows % Z + 1 < last_z) pair_rows(row_id, row_id + x_rows);
        if (row_id / slice_rows + 1 < Y) pair_rows(row_id, row_id + slice_rows);
    }
};

} // namespace somevoxel
