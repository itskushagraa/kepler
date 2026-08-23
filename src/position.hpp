#pragma once
#include "bitboard.hpp"
#include "nnue.hpp"
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
    int halfmoveClock = 0;
    int fullmoveNumber = 1;
    uint64_t hashKey = 0;
    mutable Nnue::Accumulator nnueAccumulator{};

    void setStartPos();
    void fromFEN(const std::string &fen);
    std::string toFEN() const;
    void printBoard() const;
    void makeMove(const Move &m);
    void makeMove(const Move &m, struct Undo &u);
    void unmakeMove(const Move &m, const struct Undo &u);
    bool isSquareAttacked(int sq, Side bySide) const;
    bool isMoveLegal(const Move &m);
    bool isInsufficientMaterial() const;
    int pieceIndexAt(int sq) const;

private:
    static int pieceIndex(char c);
    char pieceAt(int sq) const;
    int castlingRightsMask() const;
};

struct Undo
{
    int movedIndex = -1;
    int capturedIndex = -1;
    int capturedSquare = -1;
    int promotionIndex = -1;
    int castleRookPiece = -1;
    int castleRookFrom = -1;
    int castleRookTo = -1;
    int prevEnPassant = -1;
    bool prevCastleK[2]{};
    bool prevCastleQ[2]{};
    int prevKingSquare[2]{};
    int prevHalfmoveClock = 0;
    int prevFullmoveNumber = 1;
    uint64_t prevHash = 0;
};
