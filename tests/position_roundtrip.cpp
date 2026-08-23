#include <cstdlib>
#include <iostream>
#include <random>
#include <string>

#include "eval.hpp"
#include "movegen.hpp"
#include "nnue.hpp"
#include "position.hpp"
#include "zobrist.hpp"

namespace
{
    struct Snapshot
    {
        std::string fen;
        std::array<Bitboard, 12> pieces{};
        Bitboard whiteOccupancy = 0;
        Bitboard blackOccupancy = 0;
        Bitboard allPieces = 0;
        uint64_t hash = 0;
        Nnue::Accumulator accumulator{};
    };

    Snapshot snapshot(Position &position)
    {
        (void)evaluate(position);
        return {
            position.toFEN(),
            position.pieceBB,
            position.occupancy[WHITE],
            position.occupancy[BLACK],
            position.allPieces,
            position.hashKey,
            position.nnueAccumulator,
        };
    }

    bool sameAccumulator(const Nnue::Accumulator &a, const Nnue::Accumulator &b)
    {
        return a.classicHidden == b.classicHidden && a.halfHidden == b.halfHidden &&
               a.classicValid == b.classicValid && a.halfValid == b.halfValid &&
               a.kingSq == b.kingSq && a.positionKey == b.positionKey;
    }

    bool matches(Position &position, const Snapshot &before, const std::string &context)
    {
        const bool ok = position.toFEN() == before.fen && position.pieceBB == before.pieces &&
                        position.occupancy[WHITE] == before.whiteOccupancy &&
                        position.occupancy[BLACK] == before.blackOccupancy &&
                        position.allPieces == before.allPieces && position.hashKey == before.hash &&
                        sameAccumulator(position.nnueAccumulator, before.accumulator);
        if (!ok)
            std::cerr << "round-trip mismatch after " << context << "\n"
                      << "before: " << before.fen << "\n"
                      << "after:  " << position.toFEN() << "\n";
        return ok;
    }

    bool checkEveryLegalMove(Position &position, const std::string &context)
    {
        const Snapshot before = snapshot(position);
        MoveList legal;
        generateLegalMoves(position, legal);
        for (const Move &move : legal.moves)
        {
            Undo undo;
            position.makeMove(move, undo);
            (void)evaluate(position);
            position.unmakeMove(move, undo);
            if (!matches(position, before, context + " move " + std::to_string(move.from) + "-" +
                                             std::to_string(move.to)))
                return false;
        }
        return true;
    }

    bool checkCastlingFlags()
    {
        Position position;
        position.fromFEN("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
        MoveList legal;
        generateLegalMoves(position, legal);
        int castles = 0;
        for (const Move &move : legal.moves)
        {
            if (move.isCastle)
                castles++;
            if (move.from == E1 && (move.to == G1 || move.to == C1) && !move.isCastle)
            {
                std::cerr << "legal move generation dropped a white castling flag\n";
                return false;
            }
        }
        if (castles != 2)
        {
            std::cerr << "expected two legal white castling moves, got " << castles << "\n";
            return false;
        }
        return true;
    }
}

int main(int argc, char **argv)
{
    if (argc != 2 || !Nnue::loadFromFile(argv[1]))
    {
        std::cerr << "usage: position_roundtrip <model.nnue>\n";
        return EXIT_FAILURE;
    }
    Zobrist::init();
    initAttackTables();

    if (!checkCastlingFlags())
        return EXIT_FAILURE;

    const char *edgeCases[] = {
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        "4k3/P7/8/8/8/8/7p/4K3 w - - 0 1",
        "r3kb1r/ppp1pppp/2n5/4P3/6b1/5N2/PPP2PP1/RNBK1B1R b q - 0 8",
    };
    for (const char *fen : edgeCases)
    {
        Position position;
        position.fromFEN(fen);
        if (!checkEveryLegalMove(position, fen))
            return EXIT_FAILURE;
    }

    std::mt19937_64 rng(0x4B45504C4552ULL);
    Position position;
    position.setStartPos();
    int checkedPositions = 0;
    for (int ply = 0; ply < 512; ++ply)
    {
        if (!checkEveryLegalMove(position, "random walk ply " + std::to_string(ply)))
            return EXIT_FAILURE;
        checkedPositions++;
        MoveList legal;
        generateLegalMoves(position, legal);
        if (legal.moves.empty() || position.halfmoveClock >= 100 || position.isInsufficientMaterial())
        {
            position.setStartPos();
            continue;
        }
        std::uniform_int_distribution<std::size_t> pick(0, legal.moves.size() - 1);
        position.makeMove(legal.moves[pick(rng)]);
    }

    std::cout << "position round trips passed across edge cases and " << checkedPositions
              << " randomized positions\n";
    return EXIT_SUCCESS;
}
