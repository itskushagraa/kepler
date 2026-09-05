#include "movegen.hpp"
#include "position.hpp"
#include "search.hpp"
#include "zobrist.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

namespace
{
    bool sameMove(const Move &left, const Move &right)
    {
        return left.from == right.from && left.to == right.to &&
               left.isCapture == right.isCapture &&
               left.isPromotion == right.isPromotion &&
               left.isCastle == right.isCastle &&
               left.promoPiece == right.promoPiece;
    }

    bool testHashSizingAndRoundTrip()
    {
        TranspositionTable tt;
        for (const int requestedMb : {1, 3, 64, 256})
        {
            tt.resizeMB(requestedMb);
            const size_t requestedBytes = static_cast<size_t>(requestedMb) * 1024 * 1024;
            if (tt.sizeBytes() == 0 || tt.sizeBytes() > requestedBytes ||
                (tt.bucketCount() & (tt.bucketCount() - 1)) != 0)
            {
                std::cerr << "invalid TT allocation for Hash " << requestedMb
                          << ": bytes=" << tt.sizeBytes()
                          << " buckets=" << tt.bucketCount() << "\n";
                return false;
            }
        }

        tt.resizeMB(1);
        if (!tt.isLockFree())
        {
            std::cerr << "64-bit TT atomics are not lock-free on this target\n";
            return false;
        }

        const Move expected{12, 28, true, true, false, PROMO_KNIGHT};
        constexpr uint64_t key = 0x123456789abcdef0ULL;
        tt.store(key, 17, -321, 2, expected);
        TTEntry actual;
        if (!tt.probe(key, actual) || actual.key != key || actual.depth != 17 ||
            actual.score != -321 || actual.bound != 2 || !sameMove(actual.bestMove, expected))
        {
            std::cerr << "TT entry did not survive pack/publish/probe\n";
            return false;
        }

        const Move exactMove{4, 6, false, false, true, PROMO_NONE};
        tt.store(key, 20, 88, 3, exactMove);
        tt.store(key, 8, -50, 1, Move{});
        if (!tt.probe(key, actual) || actual.depth != 20 || actual.score != 88 ||
            actual.bound != 3 || !sameMove(actual.bestMove, exactMove))
        {
            std::cerr << "shallower non-exact TT entry replaced a deeper exact entry\n";
            return false;
        }

        tt.store(0, 5, 12, 3, Move{1, 2});
        if (!tt.probe(0, actual) || actual.key != 0 || actual.depth != 5)
        {
            std::cerr << "zero hash key was confused with an empty TT slot\n";
            return false;
        }
        return true;
    }

    bool testConcurrentTranspositionAccess()
    {
        TranspositionTable tt;
        tt.resizeMB(4);
        std::atomic<bool> failed{false};
        std::vector<std::thread> workers;
        for (int worker = 0; worker < 4; ++worker)
        {
            workers.emplace_back([&, worker]
            {
                for (int iteration = 0; iteration < 50000; ++iteration)
                {
                    const uint64_t key =
                        ((static_cast<uint64_t>(worker * 50000 + iteration + 1) *
                          tt.bucketCount()) |
                         7ULL);
                    const Move move{iteration & 63, (iteration * 7) & 63,
                                    (iteration & 1) != 0, false, false, PROMO_NONE};
                    tt.store(key, iteration & 63, iteration - 25000,
                             static_cast<uint8_t>((iteration % 3) + 1), move);
                    TTEntry entry;
                    if (tt.probe(key, entry) && entry.key != key)
                        failed.store(true, std::memory_order_relaxed);
                }
            });
        }
        for (auto &worker : workers)
            worker.join();
        if (failed.load(std::memory_order_relaxed))
        {
            std::cerr << "concurrent TT probe accepted an unverified entry\n";
            return false;
        }
        return true;
    }

    bool testTimeBudgets()
    {
        Position opening;
        opening.setStartPos();

        SearchLimits explicitMoveTime;
        explicitMoveTime.movetimeMs = 1000;
        const TimeBudget explicitBudget = calculateTimeBudget(opening, explicitMoveTime);
        if (explicitBudget.optimumMs != 950 || explicitBudget.maximumMs != 1000 ||
            explicitBudget.adaptive)
        {
            std::cerr << "explicit movetime budget changed semantics\n";
            return false;
        }

        SearchLimits clock;
        clock.wtimeMs = 180000;
        clock.wincMs = 2000;
        clock.moveOverheadMs = 150;
        const TimeBudget openingBudget = calculateTimeBudget(opening, clock);
        if (!openingBudget.adaptive || openingBudget.optimumMs <= 0 ||
            openingBudget.maximumMs < openingBudget.optimumMs ||
            openingBudget.maximumMs > openingBudget.optimumMs * 2 ||
            openingBudget.maximumMs > clock.wtimeMs - clock.moveOverheadMs)
        {
            std::cerr << "invalid adaptive opening time budget\n";
            return false;
        }

        SearchLimits nearControl = clock;
        nearControl.movesToGo = 5;
        const TimeBudget nearControlBudget = calculateTimeBudget(opening, nearControl);
        if (nearControlBudget.optimumMs <= openingBudget.optimumMs)
        {
            std::cerr << "moves-to-go did not increase the per-move allocation\n";
            return false;
        }

        Position endgame;
        endgame.fromFEN("8/8/8/3k4/8/4K3/4P3/8 w - - 0 35");
        const TimeBudget endgameBudget = calculateTimeBudget(endgame, clock);
        if (endgameBudget.optimumMs != openingBudget.optimumMs ||
            endgameBudget.maximumMs != openingBudget.maximumMs)
        {
            std::cerr << "endgame incorrectly shortened the sudden-death horizon\n";
            return false;
        }

        SearchLimits bullet;
        bullet.wtimeMs = 57000; // 1+0 after the bridge's three-second reserve.
        bullet.moveOverheadMs = 900;
        bullet.threads = 4;
        bullet.printInfo = false;
        const TimeBudget bulletBudget = calculateTimeBudget(opening, bullet);
        if (bulletBudget.optimumMs < 50 || bulletBudget.optimumMs > 150 ||
            bulletBudget.maximumMs > 300)
        {
            std::cerr << "unsafe 1+0 budget: optimum=" << bulletBudget.optimumMs
                      << " maximum=" << bulletBudget.maximumMs << "\n";
            return false;
        }

        SearchLimits emergency = bullet;
        emergency.wtimeMs = 2500;
        const TimeBudget emergencyBudget = calculateTimeBudget(endgame, emergency);
        if (emergencyBudget.optimumMs > 10 || emergencyBudget.maximumMs > 20)
        {
            std::cerr << "low-clock emergency budget was not activated\n";
            return false;
        }

        SearchLimits exhausted = bullet;
        exhausted.wtimeMs = 2000;
        exhausted.moveOverheadMs = 5000;
        const TimeBudget exhaustedBudget = calculateTimeBudget(endgame, exhausted);
        if (exhaustedBudget.optimumMs != 1 || exhaustedBudget.maximumMs != 1)
        {
            std::cerr << "overhead-exhausted clock did not fall back to one millisecond\n";
            return false;
        }

        SearchLimits fixedDepth = clock;
        fixedDepth.depth = 8;
        const TimeBudget noClockBudget = calculateTimeBudget(opening, fixedDepth);
        if (noClockBudget.optimumMs != 0 || noClockBudget.maximumMs != 0)
        {
            std::cerr << "clock budget overrode a fixed-depth search\n";
            return false;
        }
        return true;
    }

    bool testThreadedClockDeadline()
    {
        Position position;
        position.setStartPos();

        SearchLimits limits;
        limits.wtimeMs = 57000;
        limits.moveOverheadMs = 900;
        limits.threads = 4;
        limits.printInfo = false;
        const TimeBudget budget = calculateTimeBudget(position, limits);

        TranspositionTable tt;
        tt.resizeMB(16);
        std::atomic<bool> externalStop{false};
        const auto start = std::chrono::steady_clock::now();
        const SearchResult result = search(position, limits, tt, externalStop);
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();

        if (elapsed > budget.maximumMs + 300)
        {
            std::cerr << "threaded search overran shared hard deadline: elapsed="
                      << elapsed << " maximum=" << budget.maximumMs << "\n";
            return false;
        }
        const auto helperJoinMs = elapsed - result.elapsedMs;
        if (helperJoinMs > 100)
        {
            std::cerr << "helper threads delayed the main result by "
                      << helperJoinMs << " ms\n";
            return false;
        }
        if (externalStop.load(std::memory_order_relaxed))
        {
            std::cerr << "normal timed completion polluted the UCI stop flag\n";
            return false;
        }
        if (result.depth <= 0)
        {
            std::cerr << "bullet deadline returned without a completed iteration\n";
            return false;
        }
        return true;
    }
}

int main()
{
    initAttackTables();
    Zobrist::init();
    if (!testHashSizingAndRoundTrip() ||
        !testConcurrentTranspositionAccess() ||
        !testTimeBudgets() ||
        !testThreadedClockDeadline())
    {
        return EXIT_FAILURE;
    }
    std::cout << "PASS: lock-free TT, bounded hash sizing, and time budgets are correct.\n";
    return EXIT_SUCCESS;
}
