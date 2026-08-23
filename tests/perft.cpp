#include "position.hpp"
#include "movegen.hpp"
#include "zobrist.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

using clock_type = std::chrono::steady_clock;

namespace
{
    constexpr int kMaxDepth = 5;
    constexpr int kDefaultSmokeDepth = 3;

    struct Test
    {
        const char *name;
        const char *fen;
        std::array<uint64_t, kMaxDepth> expected;
    };

    // Standard CPW perft positions. Each array is indexed by depth - 1.
    const std::array<Test, 6> kTests = {{
        {"Position 1 - Initial",
         "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
         {20ULL, 400ULL, 8902ULL, 197281ULL, 4865609ULL}},

        {"Position 2 - Kiwipete",
         "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
         {48ULL, 2039ULL, 97862ULL, 4085603ULL, 193690690ULL}},

        {"Position 3 - EP Validation",
         "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
         {14ULL, 191ULL, 2812ULL, 43238ULL, 674624ULL}},

        {"Position 4 - EP Pin / Discovered Check",
         "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
         {6ULL, 264ULL, 9467ULL, 422333ULL, 15833292ULL}},

        {"Position 5 - Promotion / Castling",
         "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
         {44ULL, 1486ULL, 62379ULL, 2103487ULL, 89941194ULL}},

        {"Position 6 - Tricky Knight / Stalemate Patterns",
         "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
         {46ULL, 2079ULL, 89890ULL, 3894594ULL, 164075551ULL}},
    }};

    struct Options
    {
        int depth = kDefaultSmokeDepth;
        int threads = 1;
        bool full = false;
        bool showHelp = false;
    };

    void printUsage(const char *program)
    {
        std::printf(
            "Usage: %s [options]\n"
            "\n"
            "Runs the six standard perft positions and validates every result.\n"
            "By default, the fast smoke suite runs depths 1 through 3.\n"
            "\n"
            "Options:\n"
            "  --smoke              Run the default fast suite (depths 1-3)\n"
            "  --full               Run the complete suite (depths 1-5)\n"
            "  --depth N            Run depths 1 through N (N: 1-%d)\n"
            "  --threads N          Use N root-level worker threads (default: 1)\n"
            "  -h, --help           Show this help\n"
            "\n"
            "Exit status: 0 = pass, 1 = perft mismatch, 2 = invalid arguments.\n",
            program, kMaxDepth);
    }

    bool parsePositiveInt(const char *text, int &value)
    {
        char *end = nullptr;
        long parsed = std::strtol(text, &end, 10);
        if (end == text || *end != '\0' || parsed < 1 || parsed > 1000000)
            return false;
        value = static_cast<int>(parsed);
        return true;
    }

    bool parseOptions(int argc, char **argv, Options &options)
    {
        bool sawDepth = false;
        bool sawThreads = false;

        for (int i = 1; i < argc; ++i)
        {
            const std::string arg = argv[i];
            if (arg == "-h" || arg == "--help")
            {
                options.showHelp = true;
                continue;
            }
            if (arg == "--smoke")
            {
                options.depth = kDefaultSmokeDepth;
                options.full = false;
                sawDepth = true;
                continue;
            }
            if (arg == "--full")
            {
                options.depth = kMaxDepth;
                options.full = true;
                sawDepth = true;
                continue;
            }

            int value = 0;
            if (arg == "--depth" || arg == "--threads")
            {
                if (++i >= argc || !parsePositiveInt(argv[i], value))
                    return false;
                if (arg == "--depth")
                {
                    if (value > kMaxDepth)
                        return false;
                    options.depth = value;
                    options.full = value == kMaxDepth;
                    sawDepth = true;
                }
                else
                {
                    options.threads = value;
                    sawThreads = true;
                }
                continue;
            }

            // Preserve the old `perft depth threads` form for local workflows.
            if (!arg.empty() && arg[0] != '-' && !sawDepth && parsePositiveInt(arg.c_str(), value))
            {
                if (value > kMaxDepth)
                    return false;
                options.depth = value;
                options.full = value == kMaxDepth;
                sawDepth = true;
                continue;
            }
            if (!arg.empty() && arg[0] != '-' && sawDepth && !sawThreads && parsePositiveInt(arg.c_str(), value))
            {
                options.threads = value;
                sawThreads = true;
                continue;
            }
            return false;
        }
        return true;
    }

    uint64_t perft(Position &pos, int depth)
    {
        if (depth == 0)
            return 1;

        MoveList moves;
        generateLegalMoves(pos, moves);

        uint64_t nodes = 0;
        for (const auto &move : moves.moves)
        {
            Position next = pos;
            next.makeMove(move);
            nodes += perft(next, depth - 1);
        }
        return nodes;
    }

    uint64_t perftRootParallel(const Position &pos, int depth, int threads)
    {
        MoveList root;
        generateLegalMoves(pos, root);
        const size_t rootCount = root.moves.size();
        if (rootCount == 0 || depth <= 1 || threads <= 1)
        {
            Position copy = pos;
            return perft(copy, depth);
        }

        threads = std::max(1, std::min<int>(threads, static_cast<int>(rootCount)));
        std::atomic<uint64_t> total{0};
        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(threads));

        const size_t chunk = (rootCount + static_cast<size_t>(threads) - 1) /
                             static_cast<size_t>(threads);
        for (int worker = 0; worker < threads; ++worker)
        {
            const size_t begin = static_cast<size_t>(worker) * chunk;
            const size_t end = std::min(rootCount, begin + chunk);
            if (begin >= end)
                break;
            workers.emplace_back([&, begin, end]
            {
                uint64_t local = 0;
                for (size_t i = begin; i < end; ++i)
                {
                    Position copy = pos;
                    copy.makeMove(root.moves[i]);
                    local += perft(copy, depth - 1);
                }
                total.fetch_add(local, std::memory_order_relaxed);
            });
        }

        for (auto &worker : workers)
            worker.join();
        return total.load(std::memory_order_relaxed);
    }

    uint64_t runPerft(const Position &position, int depth, int threads)
    {
        if (threads <= 1)
        {
            Position copy = position;
            return perft(copy, depth);
        }
        return perftRootParallel(position, depth, threads);
    }
}

int main(int argc, char **argv)
{
    Options options;
    if (!parseOptions(argc, argv, options))
    {
        std::fprintf(stderr, "Invalid arguments. Use --help for usage.\n");
        return 2;
    }
    if (options.showHelp)
    {
        printUsage(argv[0]);
        return 0;
    }

    initAttackTables();
    Zobrist::init();

    std::printf("Perft %s: depths 1-%d, threads %d\n",
                options.full ? "full suite" : "smoke suite",
                options.depth, options.threads);
    std::fflush(stdout);

    int failures = 0;
    int completed = 0;
    const int total = static_cast<int>(kTests.size()) * options.depth;

    for (size_t testIndex = 0; testIndex < kTests.size(); ++testIndex)
    {
        const Test &test = kTests[testIndex];
        Position position;
        position.fromFEN(test.fen);

        for (int depth = 1; depth <= options.depth; ++depth)
        {
            const auto started = clock_type::now();
            const uint64_t actual = runPerft(position, depth, options.threads);
            const auto finished = clock_type::now();
            const double seconds = std::chrono::duration<double>(finished - started).count();
            const uint64_t expected = test.expected[static_cast<size_t>(depth - 1)];
            const bool passed = actual == expected;
            const double mnps = seconds > 0.0 ? actual / seconds / 1e6 : 0.0;

            ++completed;
            if (!passed)
                ++failures;

            std::printf("[%2d/%d] %-42s depth %d: %s\n"
                        "        expected %llu, actual %llu, %.3f s (%.2f Mn/s)\n",
                        completed, total, test.name, depth, passed ? "PASS" : "FAIL",
                        static_cast<unsigned long long>(expected),
                        static_cast<unsigned long long>(actual), seconds, mnps);
            std::fflush(stdout);
        }
    }

    if (failures == 0)
    {
        std::printf("PASS: %d/%d perft checks matched.\n", completed, total);
        return 0;
    }

    std::printf("FAIL: %d/%d perft checks mismatched.\n", failures, total);
    return 1;
}
