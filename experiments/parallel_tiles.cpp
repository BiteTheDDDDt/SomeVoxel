#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;

class Pool {
public:
    explicit Pool(int count) : count(count) {
        threads.reserve(count - 1);
        try {
            for (int i = 1; i < count; ++i)
                threads.emplace_back([this, i] { worker(i); });
        } catch (...) {
            shutdown();
            throw;
        }
    }

    ~Pool() { shutdown(); }
    template <class F> void run(F f) {
        if (count == 1) { f(0); return; }
        {
            std::lock_guard lock(mutex);
            context = &f;
            task = [](void *p, int i) { (*static_cast<F *>(p))(i); };
            pending = count - 1;
            ++serial;
        }
        start.notify_all();
        f(0);
        std::unique_lock lock(mutex);
        end.wait(lock, [&] { return pending == 0; });
    }

    const int count;
private:
    std::vector<std::thread> threads;
    std::mutex mutex;
    std::condition_variable start, end;
    unsigned serial = 0;
    int pending = 0;
    bool stop = false;
    void *context = nullptr;
    void (*task)(void *, int) = nullptr;

    void worker(int index) {
        std::unique_lock lock(mutex);
        unsigned previous = 0;
        while (true) {
            start.wait(lock, [&] { return stop || serial != previous; });
            if (stop) return;
            previous = serial;
            lock.unlock();
            task(context, index);
            lock.lock();
            if (!--pending) end.notify_one();
        }
    }

    void shutdown() {
        {
            std::lock_guard lock(mutex);
            stop = true;
        }
        start.notify_all();
        for (auto &thread : threads) thread.join();
    }
};

struct Grain {
    int x, y, z;
    float vx = 0.f, vy = 0.f, vz = 0.f;
    float rx = 0.f, ry = 0.f, rz = 0.f;
};

struct World {
    static constexpr int nx = 288, ny = 144, nz = 224, tile = 16, wall = 157;
    static constexpr float dt = 1.f / 120.f, cell = .03f;
    std::vector<int> occupancy = std::vector<int>(nx * ny * nz);
    std::vector<Grain> grains;
    std::vector<std::vector<int>> bins;
    std::vector<std::vector<int>> seams;
    std::vector<int> order;
    std::vector<int> crossings;
    std::vector<std::vector<int>> incoming;
    std::vector<uint8_t> outgoing;
    int migrations = 0;
    bool persistent, pull;
    static constexpr int release_tick = 48, ticks = 144;
    static constexpr float radius = 5.5f, stiffness = 9.f;
    static constexpr float restitution = .25f;

    struct Stats {
        double x = 0, y = 0, z = 0, vx = 0, vy = 0, vz = 0;
        int wall = 0, floor = 0, rebound = 0;
    };

    Stats stats() const {
        Stats result;
        for (const auto &g : grains) {
            result.x += (g.x + .5f) * cell + g.rx;
            result.y += (g.y + .5f) * cell + g.ry;
            result.z += (g.z + .5f) * cell + g.rz;
            result.vx += g.vx;
            result.vy += g.vy;
            result.vz += g.vz;
            result.wall += g.x == wall - 1;
            result.floor += g.y == 1;
            result.rebound += g.x == wall - 1 && g.vx < 0.f;
        }
        return result;
    }

    static float field(const Grain &g, int axis, int tick) {
        if (tick >= release_tick) return 0.f;
        const float a = 6.5f - (g.x + .5f) * cell;
        const float b = 3.8f - (g.y + .5f) * cell;
        const float c = (tick < 24 ? 2.25f : 4.65f) - (g.z + .5f) * cell;
        if (a*a + b*b + c*c > radius*radius) return 0.f;
        return stiffness * (axis == 0 ? a : axis == 1 ? b : c);
    }
    double bin_ms = 0., inside_ms = 0., seam_ms = 0.;
    static int flat(int x, int y, int z) { return x + nx * (z + nz * y); }
    static int bin(const Grain &p) { return (p.x/tile) + (nx/tile) * ((p.z/tile)+(nz/tile)*(p.y/tile)); }

    explicit World(int threads, bool dense, bool keep_bins, bool use_pull)
        : bins((nx/tile)*(ny/tile)*(nz/tile)), seams(threads),
          incoming(bins.size()), outgoing(bins.size()), persistent(keep_bins), pull(use_pull) {
        for (int z = 0; z < nz; ++z)
            for (int x = 0; x < nx; ++x) occupancy[flat(x, 0, z)] = -1;
        for (int y = 1; y < ny; ++y)
            for (int z = 0; z < nz; ++z) occupancy[flat(wall, y, z)] = -1;
        for (int y = 5; y < (dense ? 63 : 35); ++y)
            for (int z = 28; z < (dense ? 176 : 98); ++z)
                for (int x = 30; x < (dense ? 153 : 82); ++x) {
                    grains.push_back({x, y, z});
                    occupancy[flat(x, y, z)] = int(grains.size());
                }
        if (persistent) group_grains();
    }

    bool try_move(int id, int axis) {
        Grain &g = grains[id];
        const int old = flat(g.x, g.y, g.z);
        float &v = axis == 0 ? g.vx : axis == 1 ? g.vy : g.vz;
        float &r = axis == 0 ? g.rx : axis == 1 ? g.ry : g.rz;
        int &coord = axis == 0 ? g.x : axis == 1 ? g.y : g.z;
        if (std::abs(r) < cell) return false;
        const int d = r > 0.f ? 1 : -1;
        const int next = old + d * (axis == 0 ? 1 : axis == 1 ? nx * nz : nx);
        const int dest = coord + d;
        const bool in_bounds = dest >= 0 && dest < (axis == 0 ? nx : axis == 1 ? ny : nz);
        if (in_bounds && occupancy[next] == 0) {
            occupancy[old] = 0;
            occupancy[next] = id + 1;
            coord = dest;
            r -= float(d) * cell;
            return true;
        }
        v = pull && (!in_bounds || occupancy[next] < 0) ? -restitution * v : 0.f;
        r = 0.f;
        return false;
    }

    void group_grains() {
        for (auto &b : bins) b.clear();
        for (int id = 0; id < int(grains.size()); ++id) bins[bin(grains[id])].push_back(id);
        refresh_order();
    }

    void refresh_order() {
        order.clear();
        for (int i = 0; i < int(bins.size()); ++i) if (!bins[i].empty()) order.push_back(i);
    }

    int step(Pool &pool, int tick) {
        int seam_attempts = 0;
        for (int sweep = 0; sweep < (pull ? 3 : 2); ++sweep) {
            const int axis = sweep == 0 ? 1 : sweep == 1 ? 0 : 2;
            const auto start = Clock::now();
            // A seam move during the previous sweep may change tile ownership.
            if (!persistent) group_grains();
            const auto grouped = Clock::now();
            for (auto &s : seams) s.clear();
            pool.run([&](int worker) {
                auto &seam = seams[worker];
                for (size_t i = worker; i < order.size(); i += pool.count) {
                    for (int id : bins[order[i]]) {
                        Grain &g = grains[id];
                        float &v = axis == 0 ? g.vx : axis == 1 ? g.vy : g.vz;
                        float &r = axis == 0 ? g.rx : axis == 1 ? g.ry : g.rz;
                        float acceleration = axis == 1 ? -9.81f : 0.f;
                        if (pull) acceleration += field(g, axis, tick);
                        else if (axis == 0 && tick < 75) acceleration = 7.f;
                        v = std::clamp(v + acceleration * dt, -cell / dt, cell / dt);
                        r += v * dt;
                        const int coord = axis == 0 ? g.x : axis == 1 ? g.y : g.z;
                        if (std::abs(r) < cell) continue;
                        const int dest = coord + (r > 0.f ? 1 : -1);
                        if (dest >= 0 && dest < (axis == 0 ? nx : axis == 1 ? ny : nz) && dest / tile == coord / tile) try_move(id, axis);
                        else seam.push_back(id);
                    }
                }
            });
            const auto solved = Clock::now();
            crossings.clear();
            for (const auto &items : seams) crossings.insert(crossings.end(), items.begin(), items.end());
            std::sort(crossings.begin(), crossings.end());
            for (int id : crossings) {
                const int from = bin(grains[id]);
                if (!try_move(id, axis)) continue;
                const int to = bin(grains[id]);
                if (persistent && from != to) {
                    outgoing[from] = 1;
                    incoming[to].push_back(id);
                    ++migrations;
                }
            }
            const auto resolved = Clock::now();
            if (persistent) {
                for (int i = 0; i < int(bins.size()); ++i) {
                    auto &b = bins[i];
                    if (outgoing[i]) {
                        b.erase(std::remove_if(b.begin(), b.end(), [&](int id) {
                            return bin(grains[id]) != i;
                        }), b.end());
                        outgoing[i] = 0;
                    }
                    auto &arrivals = incoming[i];
                    if (!arrivals.empty()) {
                        const auto middle = b.size();
                        b.insert(b.end(), arrivals.begin(), arrivals.end());
                        std::inplace_merge(b.begin(), b.begin() + middle, b.end());
                        arrivals.clear();
                    }
                }
                refresh_order();
            }
            seam_attempts += int(crossings.size());
            if (tick >= 15) {
                bin_ms += std::chrono::duration<double, std::milli>(grouped - start + (Clock::now() - resolved)).count();
                inside_ms += std::chrono::duration<double, std::milli>(solved - grouped).count();
                seam_ms += std::chrono::duration<double, std::milli>(resolved - solved).count();
            }
        }
        return seam_attempts;
    }
};

int main(int argc, char **argv) {
    const int threads = argc > 1 ? std::clamp(std::atoi(argv[1]), 1, 32) : 1;
    const bool dense = argc > 2 && std::string(argv[2]) == "dense";
    const bool persistent = argc > 3 && std::string(argv[3]) == "persistent";
    const bool pull = argc > 4 && std::string(argv[4]) == "pull";
    World world(threads, dense, persistent, pull);
    Pool pool(threads);
    std::vector<double> samples;
    int seam_attempts = 0;
    const int ticks = pull ? World::ticks : 180;
    const int batch = pull ? 2 : 1;
    std::array<World::Stats, 5> checkpoints;
    checkpoints[0] = world.stats();
    for (int tick = 0; tick < ticks; tick += batch) {
        const auto start = Clock::now();
        for (int i = 0; i < batch; ++i) seam_attempts += world.step(pool, tick + i);
        if (tick >= 16) samples.push_back(std::chrono::duration<double, std::milli>(Clock::now()-start).count());
        if (pull && (tick + batch) % 36 == 0)
            checkpoints[(tick + batch) / 36] = world.stats();
    }
    std::sort(samples.begin(), samples.end());
    uint64_t hash = 14695981039346656037ull;
    const auto append = [&](uint64_t value) { hash = (hash ^ value) * 1099511628211ull; };
    int occupancy_count = 0, bad = 0, through_wall = 0, wall_contacts = 0;
    for (int cell = 0; cell < int(world.occupancy.size()); ++cell) {
        const int id = world.occupancy[cell];
        if (id <= 0) continue;
        ++occupancy_count;
        const Grain g = world.grains[id-1];
        if (World::flat(g.x,g.y,g.z) != cell) ++bad;
        if (g.x >= World::wall) ++through_wall;
        if (g.x == World::wall - 1) ++wall_contacts;
        append(uint64_t(cell));
        append(uint64_t(id));
        for (float component : {g.vx, g.vy, g.vz, g.rx, g.ry, g.rz}) {
            uint32_t bits;
            std::memcpy(&bits, &component, sizeof(bits));
            append(bits);
        }
    }
    std::cout << std::fixed << std::setprecision(3)
        << "{\"threads\":" << threads << ",\"mode\":\"" << (persistent ? "persistent" : "regroup") << "\""
        << ",\"fixture\":\"" << (pull ? "pull" : "push") << "\""
        << ",\"grains\":" << world.grains.size()
        << ",\"p50_ms\":" << samples[samples.size()/2]
        << ",\"p95_ms\":" << samples[size_t(samples.size()*.95)]
        << ",\"bin_mean_ms\":" << world.bin_ms / samples.size()
        << ",\"inside_mean_ms\":" << world.inside_ms / samples.size()
        << ",\"seam_mean_ms\":" << world.seam_ms / samples.size()
        << ",\"seam_attempts\":" << seam_attempts << ",\"occupancy_errors\":" << bad
        << ",\"through_wall\":" << through_wall << ",\"wall_contacts\":" << wall_contacts
        << ",\"migrations\":" << world.migrations
        << ",\"hash\":\"" << hash << "\"";
    if (pull) {
        std::cout << ",\"checkpoints\":[";
        const double n = double(world.grains.size());
        for (size_t i = 0; i < checkpoints.size(); ++i) {
            const auto &c = checkpoints[i];
            if (i) std::cout << ',';
            std::cout << "[" << c.x/n << ',' << c.y/n << ',' << c.z/n << ','
                      << c.vx/n << ',' << c.vy/n << ',' << c.vz/n << ','
                      << c.wall << ',' << c.floor << ',' << c.rebound << "]";
        }
        std::cout << ']';
    }
    std::cout << "}\n";
    return bad || through_wall || occupancy_count != int(world.grains.size()) || (dense && !wall_contacts);
}
