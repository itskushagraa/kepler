#pragma once
#include "position.hpp"
#include <array>
#include <cstddef>

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
    struct MoveBuffer
    {
        static constexpr std::size_t kMaxMoves = 256;
        std::array<Move, kMaxMoves> data{};
        std::size_t count = 0;

        void push_back(const Move &m)
        {
            if (count < kMaxMoves)
                data[count++] = m;
        }

        std::size_t size() const { return count; }
        bool empty() const { return count == 0; }
        Move &front() { return data[0]; }
        const Move &front() const { return data[0]; }
        Move &operator[](std::size_t i) { return data[i]; }
        const Move &operator[](std::size_t i) const { return data[i]; }
        Move *begin() { return data.data(); }
        Move *end() { return data.data() + count; }
        const Move *begin() const { return data.data(); }
        const Move *end() const { return data.data() + count; }
    };

    MoveBuffer moves;

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
