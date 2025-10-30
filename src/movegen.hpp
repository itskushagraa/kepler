#pragma once
#include "position.hpp"

// attack tables
extern Bitboard KNIGHT_ATTACKS[64];
extern Bitboard KING_ATTACKS[64];

// init function
void initAttackTables();

// move structure
struct Move
{
    int from, to;
    bool isCapture = false;
    bool isPromotion = false;
    bool isCastle = false;
};

struct MoveList
{
    std::vector<Move> moves;

    void add(int from, int to, bool capture = false, bool promo = false);
    void print() const;
};

// generators
void generatePawnMoves(const Position &pos, MoveList &ml);
void generateSliderMoves(const Position &pos, MoveList &ml);
Bitboard rookAttacks(int sq, Bitboard occ);
Bitboard bishopAttacks(int sq, Bitboard occ);
void generateKnightMoves(const Position &pos, MoveList &ml);
void generateKingMoves(const Position &pos, MoveList &ml);
void generateAllMoves(const Position &pos, MoveList &ml);
void generateLegalMoves(const Position &pos, MoveList &legal);