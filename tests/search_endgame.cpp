#include <atomic>
#include <iostream>

#include "movegen.hpp"
#include "position.hpp"
#include "search.hpp"
#include "zobrist.hpp"

namespace
{
    constexpr const char *kLichessCrashFen = "8/7k/2p5/2K5/8/8/8/8 b - - 1 75";

    bool sameMove(const Move &a, const Move &b)
    {
        return a.from == b.from && a.to == b.to &&
               a.isPromotion == b.isPromotion && a.promoPiece == b.promoPiece &&
               a.isCastle == b.isCastle;
    }

    bool isLegalMove(const Position &position, const Move &move)
    {
        MoveList legal;
        generateLegalMoves(position, legal);
        for (const Move &candidate : legal.moves)
        {
            if (sameMove(candidate, move))
                return true;
        }
        return false;
    }

    bool runCase(int threads, int iteration)
    {
        Position position;
        position.fromFEN(kLichessCrashFen);
        const std::string originalFen = position.toFEN();

        TranspositionTable tt;
        tt.resizeMB(16);

        SearchLimits limits;
        limits.depth = 64;
        limits.threads = threads;
        limits.printInfo = false;

        std::atomic<bool> stopFlag{false};
        const SearchResult result = search(position, limits, tt, stopFlag);

        if (result.depth != 64)
        {
            std::cerr << "threads=" << threads << " iteration=" << iteration
                      << " completed_depth=" << result.depth << " expected=64\n";
            return false;
        }
        if (result.score != 0)
        {
            std::cerr << "threads=" << threads << " iteration=" << iteration
                      << " score=" << result.score << " expected=0\n";
            return false;
        }
        if (!isLegalMove(position, result.bestMove))
        {
            std::cerr << "threads=" << threads << " iteration=" << iteration
                      << " returned an illegal or null best move\n";
            return false;
        }
        if (position.toFEN() != originalFen)
        {
            std::cerr << "threads=" << threads << " iteration=" << iteration
                      << " search did not restore the root position\n";
            return false;
        }
        return true;
    }
}

int main()
{
    Zobrist::init();
    initAttackTables();

    if (!runCase(1, 1))
        return 1;

    // The live failure occurred in a four-thread worker at ply 63. Run the
    // exact position more than once to exercise all lazy-SMP worker paths.
    for (int iteration = 1; iteration <= 3; ++iteration)
    {
        if (!runCase(4, iteration))
            return 1;
    }

    std::cout << "deep king-and-pawn search passed at depth 64 with 1 and 4 threads\n";
    return 0;
}
