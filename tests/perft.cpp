// perft.cpp
#include "../src/position.hpp"
#include "../src/movegen.hpp"
#include "../src/zobrist.hpp"

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
    Zobrist::init();

    // Depth and threading settings
    int depth = (argc >= 2 ? std::max(1, std::atoi(argv[1])) : 6);
    int threads = (argc >= 3 ? std::max(1, std::atoi(argv[2]))
                             : (int)std::max(1u, std::thread::hardware_concurrency()));

    std::printf("Threads: %d\n\n", threads);

    // Official ChessProgramming.org perft suite (6 positions)
    struct Test
    {
        const char *name;
        const char *fen;
    };

    static const Test tests[] = {
        {"Position 1 – Initial",
         "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"},

        {"Position 2 – Kiwipete",
         "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"},

        {"Position 3 – EP Validation",
         "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"},

        {"Position 4 – EP Pin / Discovered Check",
         "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"},

        {"Position 5 – Castling",
         "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"},

        {"Position 6 – Tricky Knight / Stalemate Patterns",
         "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"}};

    constexpr int N = sizeof(tests) / sizeof(Test);

    // Loop and run each test
    for (int t = 0; t < N; ++t)
    {
        std::printf("=== %s ===\nFEN: %s\n", tests[t].name, tests[t].fen);

        Position pos;
        pos.fromFEN(tests[t].fen);

        for (int d = 1; d <= 5; ++d)
        {
            if (d <= 5)
            {
                Position p = pos;
                char label[64];
                std::snprintf(label, sizeof(label), "Depth %d (single)", d);
                time_run(label, [&]
                         { return perft(p, d); });
            }
            else
            {
                char label[64];
                std::snprintf(label, sizeof(label), "Depth %d (root-parallel x%d)", d, threads);
                time_run(label, [&]
                         { return perft_root_parallel(pos, d, threads); });
            }
        }
        std::printf("\n");
    }

    return 0;
}
