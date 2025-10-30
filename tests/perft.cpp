// perft.cpp
#include "../src/position.hpp"
#include "../src/movegen.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>
#include <algorithm>

using clock_type = std::chrono::steady_clock;

// -------- single-thread recursive perft --------
uint64_t perft(Position &pos, int depth)
{
    if (depth == 0)
        return 1;

    MoveList moves;
    generateLegalMoves(pos, moves);

    uint64_t nodes = 0;
    for (const auto &m : moves.moves)
    {
        Position next = pos; // copy
        next.makeMove(m);    // sideToMove flips inside
        nodes += perft(next, depth - 1);
    }
    return nodes;
}

// -------- parallelized root perft --------
// Splits root moves across N threads; each thread does a normal (recursive) perft.
uint64_t perft_root_parallel(const Position &pos, int depth, int threads)
{
    MoveList root;
    generateLegalMoves(pos, root);
    const size_t R = root.moves.size();
    if (R == 0 || depth <= 1 || threads <= 1)
    {
        Position p = pos;
        return perft(p, depth);
    }

    threads = std::max(1, std::min<int>(threads, static_cast<int>(R)));
    std::atomic<uint64_t> total{0};

    auto worker = [&](size_t begin, size_t end)
    {
        uint64_t local = 0;
        for (size_t i = begin; i < end; ++i)
        {
            const Move m = root.moves[i]; // copy
            Position p = pos;             // copy base
            p.makeMove(m);
            local += perft(p, depth - 1);
        }
        total.fetch_add(local, std::memory_order_relaxed);
    };

    std::vector<std::thread> pool;
    pool.reserve(threads);

    // chunk the root move list
    size_t chunk = (R + threads - 1) / threads;
    size_t start = 0;
    for (int t = 0; t < threads && start < R; ++t)
    {
        size_t end = std::min(R, start + chunk);
        pool.emplace_back(worker, start, end);
        start = end;
    }
    for (auto &th : pool)
        th.join();
    return total.load(std::memory_order_relaxed);
}

// -------- small timing helper --------
template <class F>
uint64_t time_run(const char *label, F &&fn, double *out_sec = nullptr)
{
    auto t0 = clock_type::now();
    uint64_t nodes = fn();
    auto t1 = clock_type::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    double mnps = (sec > 0.0) ? (nodes / sec / 1e6) : 0.0;
    std::printf("%s: %llu nodes in %.3f s  (%.2f Mn/s)\n",
                label, (unsigned long long)nodes, sec, mnps);
    if (out_sec)
        *out_sec = sec;
    return nodes;
}

int main(int argc, char **argv)
{
    initAttackTables();

    Position pos;
    pos.setStartPos();

    // Args: ./perft [depth=8] [threads=hardware_concurrency]
    int depth = (argc >= 2 ? std::max(1, std::atoi(argv[1])) : 6);
    int threads = (argc >= 3 ? std::max(1, std::atoi(argv[2]))
                             : (int)std::max(1u, std::thread::hardware_concurrency()));

    std::printf("Threads: %d\n", threads);

    // Time each depth up to requested 'depth'.
    for (int d = 1; d <= depth; ++d)
    {
        if (d <= 6)
        {
            // single-thread is fine (and good for baseline)
            Position p = pos;
            char label[64];
            std::snprintf(label, sizeof(label), "Depth %d (single)", d);
            time_run(label, [&]
                     { return perft(p, d); });
        }
        else
        {
            // parallelize the root for deep depths
            char label[64];
            std::snprintf(label, sizeof(label), "Depth %d (root-parallel x%d)", d, threads);
            time_run(label, [&]
                     { return perft_root_parallel(pos, d, threads); });
        }
    }

    return 0;
}