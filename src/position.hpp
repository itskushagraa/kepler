#pragma once
#include "bitboard.hpp"
#include <array>
#include <string>
#include <sstream>
#include <iostream>
#include <cctype>

struct Move;

class Position
{
public:
    std::array<Bitboard, 12> pieceBB{};
    Bitboard occupancy[2]{};
    Bitboard allPieces{};
    Side sideToMove = WHITE;
    bool canCastleKingside[2];
    bool canCastleQueenside[2];
    int enPassantSquare = -1;
    int kingSquare[2]; // kingSquare[WHITE], kingSquare[BLACK]

    void setStartPos();
    void fromFEN(const std::string &fen);
    void printBoard() const;
    void makeMove(const Move &m);
    bool isSquareAttacked(int sq, Side bySide) const;

private:
    static int pieceIndex(char c);
    char pieceAt(int sq) const;
};
