#pragma once
#include <cstdint>
#include <string>

// Each bit represents a square (0 = a1, 63 = h8)
using Bitboard = uint64_t;

constexpr Bitboard ONE = 1ULL;

// Enumerations for clarity
enum Piece
{
    WP,
    WN,
    WB,
    WR,
    WQ,
    WK,
    BP,
    BN,
    BB,
    BR,
    BQ,
    BK,
    NO_PIECE
};

enum Side
{
    WHITE,
    BLACK
};

enum Square
{
    A1,
    B1,
    C1,
    D1,
    E1,
    F1,
    G1,
    H1,
    A2,
    B2,
    C2,
    D2,
    E2,
    F2,
    G2,
    H2,
    A3,
    B3,
    C3,
    D3,
    E3,
    F3,
    G3,
    H3,
    A4,
    B4,
    C4,
    D4,
    E4,
    F4,
    G4,
    H4,
    A5,
    B5,
    C5,
    D5,
    E5,
    F5,
    G5,
    H5,
    A6,
    B6,
    C6,
    D6,
    E6,
    F6,
    G6,
    H6,
    A7,
    B7,
    C7,
    D7,
    E7,
    F7,
    G7,
    H7,
    A8,
    B8,
    C8,
    D8,
    E8,
    F8,
    G8,
    H8
};

// bitboard helpers
inline void set_bit(Bitboard &b, int sq) { b |= (ONE << sq); }
inline void clear_bit(Bitboard &b, int sq) { b &= ~(ONE << sq); }
inline bool get_bit(Bitboard b, int sq) { return b & (ONE << sq); }
inline int popcount(Bitboard b) { return __builtin_popcountll(b); }
inline int lsb(Bitboard b) { return __builtin_ctzll(b); }

// Files and ranks
constexpr Bitboard FILE_A = 0x0101010101010101ULL;
constexpr Bitboard FILE_H = 0x8080808080808080ULL;
constexpr Bitboard RANK_1 = 0x00000000000000FFULL;
constexpr Bitboard RANK_2 = 0x000000000000FF00ULL;
constexpr Bitboard RANK_4 = 0x00000000FF000000ULL;
constexpr Bitboard RANK_7 = 0x00FF000000000000ULL;
constexpr Bitboard RANK_8 = 0xFF00000000000000ULL;

// Direction shifts
inline Bitboard north(Bitboard b) { return b << 8; }
inline Bitboard south(Bitboard b) { return b >> 8; }
inline Bitboard east(Bitboard b) { return (b << 1) & ~FILE_A; }
inline Bitboard west(Bitboard b) { return (b >> 1) & ~FILE_H; }

// Knight and king move masks (precomputed on startup)
extern Bitboard KNIGHT_ATTACKS[64];
extern Bitboard KING_ATTACKS[64];

void initAttackTables();