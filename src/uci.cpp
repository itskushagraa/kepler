#include <iostream>
#include <string>
#include "uci.hpp"
#include "position.hpp"
#include "movegen.hpp"

void runUciLoop()
{
    std::string cmd;
    Position pos;
    pos.setStartPos();

    while (std::getline(std::cin, cmd))
    {
        if (cmd == "uci")
        {
            std::cout << "id name Kepler\n";
            std::cout << "id author Kush Sharma\n";
            std::cout << "uciok\n";
        }
        else if (cmd == "isready")
        {
            std::cout << "readyok\n";
        }
        else if (cmd.rfind("position", 0) == 0)
        {
            pos.setStartPos();
        }
        else if (cmd == "d")
        { // like Stockfish's debug print
            pos.printBoard();
        }
        // else if (cmd.rfind("testmoves", 0) == 0) {
        //     int square = 28; // c3 for example
        //     Bitboard n = KNIGHT_ATTACKS[square];
        //     Bitboard k = KING_ATTACKS[square];
        //     std::cout << "Knight attacks from e4:\n";
        //     for (int r = 7; r >= 0; --r) {
        //         for (int f = 0; f < 8; ++f) {
        //             int sq = r * 8 + f;
        //             std::cout << (get_bit(n, sq) ? 'x' : '.') << ' ';
        //         }
        //         std::cout << '\n';
        //     }
        //     std::cout << "King attacks from e4:\n";
        //     for (int r = 7; r >= 0; --r) {
        //         for (int f = 0; f < 8; ++f) {
        //             int sq = r * 8 + f;
        //             std::cout << (get_bit(k, sq) ? 'x' : '.') << ' ';
        //         }
        //         std::cout << '\n';
        //     }
        // }
        // else if (cmd == "pawns") {
        //     MoveList ml;
        //     pos.setStartPos();
        //     generatePawnMoves(pos, ml);
        //     ml.print();
        // }
        // else if (cmd == "sliders") {
        //     MoveList ml;

        //     // Example test: rook on d5 with a few blockers/captures
        //     Position test;
        //     test.fromFEN("8/8/3P4/2pRP3/3P4/8/8/8 w - - 0 1"); // white rook d5; black pawn d7 & e4

        //     generateSliderMoves(test, ml);
        //     ml.print();
        // }
        else if (cmd == "allmoves")
        {
            MoveList ml;
            //pos.fromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK2R w KQkq - 0 1");
            pos.fromFEN("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");
            generateLegalMoves(pos, ml);
            ml.print();
        }
        else if (cmd.rfind("go", 0) == 0)
        {
            std::cout << "bestmove e2e4\n";
        }
        else if (cmd == "quit")
        {
            break;
        }
    }
}
