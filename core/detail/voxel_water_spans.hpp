#pragma once

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
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
    explicit WaterSpans(int count) : workers(count) {}

    template <typename Apply> void rebuild(const std::vector<int> &water_positions, Apply apply) {
        if (rows.empty()) rows.resize(X * Y * Z / 8);
        for (int i : water_positions) {
            Row &row = rows[i / 8];
            if (!row.occupied) rows_by_z[i / X % Z / 8].push_back(i / 8);
            row.occupied |= uint8_t(1u << (i % 8));
        }

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

        workers.run([&](int k) {
            for (int z = bands[k]; z < bands[k + 1]; ++z) {
                for (int row_id : rows_by_z[z]) {
                    Row &row = rows[row_id];
                    for (int id = row.begin; id < row.end(); ++id) {
                        const Span &span = spans[id];
                        const int head = spans[root(id)].head;
                        for (int i = span.cell; i < span.cell + span.width; ++i) apply(i, head);
                    }
                    row.occupied = 0;
                }
                rows_by_z[z].clear();
            }
        });
    }

    size_t scratch_bytes() const {
        size_t bytes = rows.capacity() * sizeof(Row) + spans.capacity() * sizeof(Span);
        for (const auto &rows : rows_by_z) bytes += rows.capacity() * sizeof(int);
        return bytes;
    }

private:
    HeadWorkers workers;
    std::vector<Row> rows;
    std::vector<Span> spans;
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
