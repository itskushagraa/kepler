#pragma once
#include "position.hpp"

// attack tables
extern Bitboard KNIGHT_ATTACKS[64];
extern Bitboard KING_ATTACKS[64];

// init function
void initAttackTables();

enum PromoPiece : uint8_t
{
    PROMO_NONE = 0,
    PROMO_QUEEN = 1,
    PROMO_ROOK = 2,
    PROMO_BISHOP = 3,
    PROMO_KNIGHT = 4,
};

// move structure
struct Move
{
    int from, to;
    bool isCapture = false;
    bool isPromotion = false;
    bool isCastle = false;
    uint8_t promoPiece = PROMO_NONE;
};

struct MoveList
{
    std::vector<Move> moves;

    inline void add(int from, int to, bool capture = false, bool promo = false)
    {
        moves.push_back({from, to, capture, promo, /*isCastle*/ false,
                         promo ? (uint8_t)PROMO_QUEEN : (uint8_t)PROMO_NONE});
    }

    inline void add(int from, int to, bool capture, bool promo, uint8_t promoPiece)
    {
        moves.push_back({from, to, capture, promo, /*isCastle*/ false, promo ? promoPiece : (uint8_t)PROMO_NONE});
    }

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