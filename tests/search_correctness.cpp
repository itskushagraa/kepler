#include <atomic>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "movegen.hpp"
#include "position.hpp"
#include "search.hpp"
#include "zobrist.hpp"

namespace
{
    bool sameMove(const Move &a, const Move &b)
    {
        return a.from == b.from && a.to == b.to && a.isPromotion == b.isPromotion &&
               a.promoPiece == b.promoPiece && a.isCastle == b.isCastle;
    }

    bool legalMove(const Position &position, const Move &move)
    {
        MoveList legal;
        generateLegalMoves(position, legal);
        for (const Move &candidate : legal.moves)
            if (sameMove(candidate, move))
                return true;
        return false;
    }

    Move findMove(const Position &position, int from, int to)
    {
        MoveList legal;
        generateLegalMoves(position, legal);
        for (const Move &move : legal.moves)
            if (move.from == from && move.to == to)
                return move;
        return Move{};
    }

    SearchResult run(Position &position, int depth, int threads, bool pruning,
                     const std::vector<uint64_t> &history = {})
    {
        TranspositionTable tt;
        tt.resizeMB(16);
        SearchLimits limits;
        limits.depth = depth;
        limits.threads = threads;
        limits.usePruning = pruning;
        limits.printInfo = false;
        limits.positionHistory = history;
        std::atomic<bool> stop{false};
        return search(position, limits, tt, stop);
    }

    bool checkFalseMateRegression(int threads, bool pruning)
    {
        Position position;
        position.fromFEN("r3kb1r/ppp1pppp/2n5/4P3/6b1/5N2/PPP2PP1/RNBK1B1R b q - 0 8");
        const std::string before = position.toFEN();
        const uint64_t hashBefore = position.hashKey;
        const SearchResult result = run(position, 4, threads, pruning);
        if (isSearchMateScore(result.score) || !legalMove(position, result.bestMove) ||
            position.toFEN() != before || position.hashKey != hashBefore)
        {
            std::cerr << "false-mate regression failed threads=" << threads
                      << " pruning=" << pruning << " score=" << result.score << "\n";
            return false;
        }
        return true;
    }

    bool checkMaterialRules()
    {
        const char *draws[] = {
            "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            "4k3/8/8/8/8/8/8/3BK3 w - - 0 1",
            "4k3/8/8/8/8/8/8/3NK3 w - - 0 1",
            "4kb2/8/8/8/8/8/8/2B1K3 w - - 0 1",
        };
        for (const char *fen : draws)
        {
            Position position;
            position.fromFEN(fen);
            if (!position.isInsufficientMaterial())
            {
                std::cerr << "missed insufficient material: " << fen << "\n";
                return false;
            }
            SearchResult result = run(position, 4, 1, true);
            if (result.score != 0 || result.depth != 0)
            {
                std::cerr << "insufficient material was not adjudicated at root: " << fen << "\n";
                return false;
            }
        }

        const char *playable[] = {
            "4k3/8/8/8/8/8/8/2BNK3 w - - 0 1",
            "4k3/8/8/8/8/8/8/2NNK3 w - - 0 1",
            "2b1k3/8/8/8/8/8/8/2B1K3 w - - 0 1",
            "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
        };
        for (const char *fen : playable)
        {
            Position position;
            position.fromFEN(fen);
            if (position.isInsufficientMaterial())
            {
                std::cerr << "false insufficient-material draw: " << fen << "\n";
                return false;
            }
        }
        return true;
    }

    bool checkThreefoldHistory()
    {
        Position position;
        position.setStartPos();
        std::vector<uint64_t> history{position.hashKey};
        const int sequence[][2] = {
            {G1, F3}, {G8, F6}, {F3, G1}, {F6, G8},
            {G1, F3}, {G8, F6}, {F3, G1}, {F6, G8},
        };
        for (const auto &squares : sequence)
        {
            Move move = findMove(position, squares[0], squares[1]);
            if (move.from != squares[0] || move.to != squares[1])
            {
                std::cerr << "could not construct repetition history\n";
                return false;
            }
            position.makeMove(move);
            history.push_back(position.hashKey);
        }
        SearchResult result = run(position, 4, 1, true, history);
        if (result.score != 0 || result.depth != 0)
        {
            std::cerr << "threefold repetition was not adjudicated from UCI history\n";
            return false;
        }
        return true;
    }

    bool checkMatePrecedesFiftyMoveDraw()
    {
        Position position;
        position.fromFEN("7k/8/5KQ1/8/8/8/8/8 w - - 99 1");
        const SearchResult result = run(position, 3, 1, false);
        if (!isSearchMateScore(result.score) || result.score <= 0)
        {
            std::cerr << "quiet mate on halfmove 100 was scored as a draw: "
                      << result.score << "\n";
            return false;
        }
        return true;
    }
}

int main()
{
    Zobrist::init();
    initAttackTables();

    for (int threads : {1, 4})
        for (bool pruning : {false, true})
            if (!checkFalseMateRegression(threads, pruning))
                return EXIT_FAILURE;
    if (!checkMaterialRules() || !checkThreefoldHistory() || !checkMatePrecedesFiftyMoveDraw())
        return EXIT_FAILURE;
    if (!isSearchMateScore(29999) || searchMateMoves(29999) != 1 ||
        !isSearchMateScore(-29997) || searchMateMoves(-29997) != -2 ||
        isSearchMateScore(1000))
    {
        std::cerr << "mate score conversion failed\n";
        return EXIT_FAILURE;
    }

    std::cout << "search correctness regressions passed\n";
    return EXIT_SUCCESS;
}
