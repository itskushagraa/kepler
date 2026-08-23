#include "eval.hpp"
#include "movegen.hpp"
#include "nnue.hpp"
#include "position.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

namespace
{
    bool sameEvaluation(const Position &incremental, int expected, const std::string &context)
    {
        const int actual = evaluate(incremental);
        if (actual == expected)
            return true;

        std::cerr << "evaluation changed after " << context
                  << ": expected " << expected << ", got " << actual << "\n";
        return false;
    }

    bool walk(Position &pos, int depth, int &checked)
    {
        const int before = evaluate(pos);
        MoveList legal;
        generateLegalMoves(pos, legal);

        for (const Move &move : legal.moves)
        {
            const std::string moveText = pos.toFEN() + " move " +
                                          std::to_string(move.from) + "-" +
                                          std::to_string(move.to);
            Undo undo;
            pos.makeMove(move, undo);

            Position rebuilt;
            rebuilt.fromFEN(pos.toFEN());
            const int rebuiltScore = evaluate(rebuilt);
            const int incrementalScore = evaluate(pos);
            ++checked;
            if (incrementalScore != rebuiltScore)
            {
                std::cerr << "incremental evaluation mismatch after " << moveText
                          << ": incremental " << incrementalScore
                          << ", rebuilt " << rebuiltScore << "\n";
                return false;
            }

            if (depth > 1 && !walk(pos, depth - 1, checked))
                return false;

            pos.unmakeMove(move, undo);
            if (!sameEvaluation(pos, before, "unmake " + moveText))
                return false;
        }

        return true;
    }
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: eval_consistency <model.nnue>\n";
        return EXIT_FAILURE;
    }

    if (!Nnue::loadFromFile(argv[1]))
    {
        std::cerr << "Could not load NNUE model: " << argv[1] << "\n";
        return EXIT_FAILURE;
    }

    Position position;
    position.setStartPos();
    int checked = 0;
    if (!walk(position, 3, checked))
        return EXIT_FAILURE;

    std::cout << "PASS: NNUE incremental evaluation matched rebuilt positions across "
              << checked << " moves.\n";
    return EXIT_SUCCESS;
}
