#include "eval.hpp"
#include "movegen.hpp"
#include "nnue.hpp"
#include "position.hpp"

#include <cstdlib>
#include <iostream>

int main(int argc, char **argv)
{
    initAttackTables();
    if (argc != 2 || !Nnue::loadFromFile(argv[1]))
    {
        std::cerr << "Usage: eval_blend <model.nnue>\n";
        return EXIT_FAILURE;
    }

    Position position;
    position.fromFEN("r1bq1rk1/ppp2ppp/2n2n2/2bp4/8/2P1PN2/PP1N1PPP/R1BQKB1R w KQ - 1 8");
    setNnueClamp(0);
    setNnueWeight(0);
    const int classical = evaluate(position);
    setNnueWeight(100);
    const int neural = evaluate(position);

    for (const int weight : {25, 50, 75})
    {
        setNnueWeight(weight);
        const int actual = evaluate(position);
        const int expected = (classical * (100 - weight) + neural * weight) / 100;
        if (actual != expected)
        {
            std::cerr << "blend mismatch at " << weight << "%: expected "
                      << expected << ", got " << actual << "\n";
            return EXIT_FAILURE;
        }
    }

    setNnueWeight(-10);
    if (nnueWeight() != 0)
        return EXIT_FAILURE;
    setNnueWeight(1000);
    if (nnueWeight() != 100)
        return EXIT_FAILURE;
    setNnueClamp(-1);
    if (nnueClamp() != 0)
        return EXIT_FAILURE;
    setNnueClamp(20000);
    if (nnueClamp() != 10000)
        return EXIT_FAILURE;

    std::cout << "PASS: configurable NNUE blending and bounds are correct.\n";
    return EXIT_SUCCESS;
}
