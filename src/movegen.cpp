#include "movegen.hpp"
#include <iostream>
#include <vector>

Bitboard KNIGHT_ATTACKS[64];
Bitboard KING_ATTACKS[64];

void MoveList::print() const
{
    for (const auto &m : moves)
    {
        char ffile = 'a' + (m.from % 8);
        char frank = '1' + (m.from / 8);
        char tfile = 'a' + (m.to % 8);
        char trank = '1' + (m.to / 8);
        std::cout << ffile << frank << tfile << trank;
        if (m.isPromotion)
        {
            char pc = 'q';
            if (m.promoPiece == PROMO_ROOK)
                pc = 'r';
            else if (m.promoPiece == PROMO_BISHOP)
                pc = 'b';
            else if (m.promoPiece == PROMO_KNIGHT)
                pc = 'n';
            std::cout << "(promo=" << pc << ")";
        }
        if (m.isCapture)
            std::cout << "x";
        std::cout << "\n";
    }
}

// Offsets for piece jumps
const int knightOffsets[8] = {17, 15, 10, 6, -17, -15, -10, -6};
const int kingOffsets[8] = {8, -8, 1, -1, 9, 7, -9, -7};

void initAttackTables()
{
    for (int sq = 0; sq < 64; ++sq)
    {
        Bitboard b = ONE << sq;
        Bitboard attacks = 0ULL;

        int r = sq / 8, f = sq % 8;
        for (int off : knightOffsets)
        {
            int target = sq + off;
            int tr = target / 8, tf = target % 8;
            if (target >= 0 && target < 64 && std::max(abs(tr - r), abs(tf - f)) <= 2)
                attacks |= (ONE << target);
        }
        KNIGHT_ATTACKS[sq] = attacks;

        attacks = 0ULL;
        for (int off : kingOffsets)
        {
            int target = sq + off;
            int tr = target / 8, tf = target % 8;
            if (target >= 0 && target < 64 && std::max(abs(tr - r), abs(tf - f)) == 1)
                attacks |= (ONE << target);
        }
        KING_ATTACKS[sq] = attacks;
    }
}

// ------------------------------------------------------------
// Sliding-piece helpers (directional ray scans)
// ------------------------------------------------------------
static inline bool in_bounds(int r, int f) { return (r >= 0 && r < 8 && f >= 0 && f < 8); }

static inline void gen_ray_from_square(
    int from, int dr, int df,
    Bitboard friends, Bitboard enemies,
    MoveList &ml)
{
    int r = from / 8, f = from % 8;
    int tr = r + dr, tf = f + df;
    while (in_bounds(tr, tf))
    {
        int to = tr * 8 + tf;
        if (get_bit(friends, to))
            break; // blocked by our own piece
        if (get_bit(enemies, to))
        { // capture and stop
            ml.add(from, to, /*capture=*/true);
            break;
        }
        ml.add(from, to); // quiet
        tr += dr;
        tf += df;
    }
}

void generateSliderMoves(const Position &pos, MoveList &ml)
{
    // friends/enemies masks for each side
    Bitboard whiteFriends = pos.occupancy[WHITE];
    Bitboard blackFriends = pos.occupancy[BLACK];
    Bitboard whiteEnemies = blackFriends;
    Bitboard blackEnemies = whiteFriends;

    if (pos.sideToMove == WHITE)
    {
        // ----- WHITE SLIDERS -----
        // Rooks
        {
            Bitboard bb = pos.pieceBB[WR];
            while (bb)
            {
                int from = lsb(bb);
                bb &= bb - 1;

                // 4 rook rays
                gen_ray_from_square(from, +1, 0, whiteFriends, whiteEnemies, ml); // north
                gen_ray_from_square(from, -1, 0, whiteFriends, whiteEnemies, ml); // south
                gen_ray_from_square(from, 0, +1, whiteFriends, whiteEnemies, ml); // east
                gen_ray_from_square(from, 0, -1, whiteFriends, whiteEnemies, ml); // west
            }
        }
        // Bishops
        {
            Bitboard bb = pos.pieceBB[WB];
            while (bb)
            {
                int from = lsb(bb);
                bb &= bb - 1;

                // 4 bishop rays
                gen_ray_from_square(from, +1, +1, whiteFriends, whiteEnemies, ml); // NE
                gen_ray_from_square(from, +1, -1, whiteFriends, whiteEnemies, ml); // NW
                gen_ray_from_square(from, -1, +1, whiteFriends, whiteEnemies, ml); // SE
                gen_ray_from_square(from, -1, -1, whiteFriends, whiteEnemies, ml); // SW
            }
        }
        // Queens = rook + bishop rays
        {
            Bitboard bb = pos.pieceBB[WQ];
            while (bb)
            {
                int from = lsb(bb);
                bb &= bb - 1;

                // Rook-like
                gen_ray_from_square(from, +1, 0, whiteFriends, whiteEnemies, ml);
                gen_ray_from_square(from, -1, 0, whiteFriends, whiteEnemies, ml);
                gen_ray_from_square(from, 0, +1, whiteFriends, whiteEnemies, ml);
                gen_ray_from_square(from, 0, -1, whiteFriends, whiteEnemies, ml);
                // Bishop-like
                gen_ray_from_square(from, +1, +1, whiteFriends, whiteEnemies, ml);
                gen_ray_from_square(from, +1, -1, whiteFriends, whiteEnemies, ml);
                gen_ray_from_square(from, -1, +1, whiteFriends, whiteEnemies, ml);
                gen_ray_from_square(from, -1, -1, whiteFriends, whiteEnemies, ml);
            }
        }
    }

    else
    {
        // ----- BLACK SLIDERS -----
        // Rooks
        {
            Bitboard bb = pos.pieceBB[BR];
            while (bb)
            {
                int from = lsb(bb);
                bb &= bb - 1;

                gen_ray_from_square(from, +1, 0, blackFriends, blackEnemies, ml);
                gen_ray_from_square(from, -1, 0, blackFriends, blackEnemies, ml);
                gen_ray_from_square(from, 0, +1, blackFriends, blackEnemies, ml);
                gen_ray_from_square(from, 0, -1, blackFriends, blackEnemies, ml);
            }
        }
        // Bishops
        {
            Bitboard bb = pos.pieceBB[BB];
            while (bb)
            {
                int from = lsb(bb);
                bb &= bb - 1;

                gen_ray_from_square(from, +1, +1, blackFriends, blackEnemies, ml);
                gen_ray_from_square(from, +1, -1, blackFriends, blackEnemies, ml);
                gen_ray_from_square(from, -1, +1, blackFriends, blackEnemies, ml);
                gen_ray_from_square(from, -1, -1, blackFriends, blackEnemies, ml);
            }
        }
        // Queens
        {
            Bitboard bb = pos.pieceBB[BQ];
            while (bb)
            {
                int from = lsb(bb);
                bb &= bb - 1;

                gen_ray_from_square(from, +1, 0, blackFriends, blackEnemies, ml);
                gen_ray_from_square(from, -1, 0, blackFriends, blackEnemies, ml);
                gen_ray_from_square(from, 0, +1, blackFriends, blackEnemies, ml);
                gen_ray_from_square(from, 0, -1, blackFriends, blackEnemies, ml);

                gen_ray_from_square(from, +1, +1, blackFriends, blackEnemies, ml);
                gen_ray_from_square(from, +1, -1, blackFriends, blackEnemies, ml);
                gen_ray_from_square(from, -1, +1, blackFriends, blackEnemies, ml);
                gen_ray_from_square(from, -1, -1, blackFriends, blackEnemies, ml);
            }
        }
    }

    // ---------- CASTLING (TEMP BASIC VERSION) ----------
    if (pos.sideToMove == WHITE)
    {
        if (get_bit(pos.pieceBB[WK], 4))
        {
            // White kingside (e1g1) and queenside (e1c1)
            if (pos.canCastleKingside[WHITE] &&
                !get_bit(pos.allPieces, 5) && !get_bit(pos.allPieces, 6))
            {
                Move m;
                m.from = 4;
                m.to = 6;
                m.isCastle = true;
                ml.moves.push_back(m); // e1 → g1
            }
            if (pos.canCastleQueenside[WHITE] &&
                !get_bit(pos.allPieces, 3) && !get_bit(pos.allPieces, 2) && !get_bit(pos.allPieces, 1))
            {
                Move m;
                m.from = 4;
                m.to = 2;
                m.isCastle = true;
                ml.moves.push_back(m); // e1 → c1
            }
        }
    }
    else
    {
        if (get_bit(pos.pieceBB[BK], 60))
        {
            // Black kingside (e8g8) and queenside (e8c8)
            if (pos.canCastleKingside[BLACK] &&
                !get_bit(pos.allPieces, 61) && !get_bit(pos.allPieces, 62))
            {
                Move m;
                m.from = 60;
                m.to = 62;
                m.isCastle = true;
                ml.moves.push_back(m); // e8 → g8
            }
            if (pos.canCastleQueenside[BLACK] &&
                !get_bit(pos.allPieces, 59) && !get_bit(pos.allPieces, 58) && !get_bit(pos.allPieces, 57))
            {
                Move m;
                m.from = 60;
                m.to = 58;
                m.isCastle = true;
                ml.moves.push_back(m);
                ; // e8 → c8
            }
        }
    }
}

// Returns all squares a bishop attacks from `sq` given current occupancy
Bitboard bishopAttacks(int sq, Bitboard occ)
{
    Bitboard attacks = 0ULL;
    int rank = sq / 8, file = sq % 8;

    // NE
    for (int r = rank + 1, f = file + 1; r < 8 && f < 8; r++, f++)
    {
        attacks |= (1ULL << (r * 8 + f));
        if (occ & (1ULL << (r * 8 + f)))
            break;
    }
    // NW
    for (int r = rank + 1, f = file - 1; r < 8 && f >= 0; r++, f--)
    {
        attacks |= (1ULL << (r * 8 + f));
        if (occ & (1ULL << (r * 8 + f)))
            break;
    }
    // SE
    for (int r = rank - 1, f = file + 1; r >= 0 && f < 8; r--, f++)
    {
        attacks |= (1ULL << (r * 8 + f));
        if (occ & (1ULL << (r * 8 + f)))
            break;
    }
    // SW
    for (int r = rank - 1, f = file - 1; r >= 0 && f >= 0; r--, f--)
    {
        attacks |= (1ULL << (r * 8 + f));
        if (occ & (1ULL << (r * 8 + f)))
            break;
    }

    return attacks;
}

// Returns all squares a rook attacks from `sq` given current occupancy
Bitboard rookAttacks(int sq, Bitboard occ)
{
    Bitboard attacks = 0ULL;
    int rank = sq / 8, file = sq % 8;

    // North
    for (int r = rank + 1; r < 8; r++)
    {
        attacks |= (1ULL << (r * 8 + file));
        if (occ & (1ULL << (r * 8 + file)))
            break;
    }
    // South
    for (int r = rank - 1; r >= 0; r--)
    {
        attacks |= (1ULL << (r * 8 + file));
        if (occ & (1ULL << (r * 8 + file)))
            break;
    }
    // East
    for (int f = file + 1; f < 8; f++)
    {
        attacks |= (1ULL << (rank * 8 + f));
        if (occ & (1ULL << (rank * 8 + f)))
            break;
    }
    // West
    for (int f = file - 1; f >= 0; f--)
    {
        attacks |= (1ULL << (rank * 8 + f));
        if (occ & (1ULL << (rank * 8 + f)))
            break;
    }

    return attacks;
}

void generatePawnMoves(const Position &pos, MoveList &ml)
{
    Bitboard wpawns = pos.pieceBB[WP];
    Bitboard bpawns = pos.pieceBB[BP];
    Bitboard empty = ~pos.allPieces;

    if (pos.sideToMove == WHITE)
    {
        // ---------- WHITE PAWNS ----------
        // single pushes
        Bitboard singlePush = north(wpawns) & empty;
        Bitboard promotionPush = singlePush & RANK_8;
        Bitboard quietPush = singlePush & ~RANK_8;

        // add normal pushes
        Bitboard tmp = quietPush;
        while (tmp)
        {
            int to = lsb(tmp);
            int from = to - 8;
            ml.add(from, to);
            tmp &= tmp - 1;
        }

        // add promotion pushes
        tmp = promotionPush;
        while (tmp)
        {
            int to = lsb(tmp);
            int from = to - 8;
            ml.add(from, to, false, true, PROMO_QUEEN);
            ml.add(from, to, false, true, PROMO_ROOK);
            ml.add(from, to, false, true, PROMO_BISHOP);
            ml.add(from, to, false, true, PROMO_KNIGHT);
            tmp &= tmp - 1;
        }

        // double pushes (rank 2 -> 4)
        Bitboard rank2 = wpawns & RANK_2;
        Bitboard singlePush2 = north(rank2) & empty;
        Bitboard doublePush = north(singlePush2) & empty;

        tmp = doublePush;
        while (tmp)
        {
            int to = lsb(tmp);
            int from = to - 16;
            ml.add(from, to);
            tmp &= tmp - 1;
        }

        // ----- EN PASSANT -----
        if (pos.enPassantSquare != -1)
        {
            int ep = pos.enPassantSquare;
            Bitboard epTarget = 1ULL << ep;

            // Pawns that could capture onto ep square:
            // For white, a pawn captures NE or NW to reach the ep target,
            // so the *from* squares are one rank south.
            Bitboard fromLeft = south(west(epTarget));  // pawn from right of target
            Bitboard fromRight = south(east(epTarget)); // pawn from left of target
            Bitboard candidates = (fromLeft | fromRight) & wpawns;

            while (candidates)
            {
                int from = lsb(candidates);
                ml.add(from, ep, true); // mark as capture
                candidates &= candidates - 1;
            }
        }

        // WHITE captures — handle each ray separately
        Bitboard capL = north(west(wpawns)) & pos.occupancy[BLACK]; // from = to - 7
        Bitboard capR = north(east(wpawns)) & pos.occupancy[BLACK]; // from = to - 9

        Bitboard t = capL;
        while (t)
        {
            int to = lsb(t);
            int from = to - 7;
            bool promo = (to >= 56);
            if (!promo)
            {
                ml.add(from, to, /*capture=*/true, /*promo=*/false);
            }
            else
            {
                ml.add(from, to, true, true, PROMO_QUEEN);
                ml.add(from, to, true, true, PROMO_ROOK);
                ml.add(from, to, true, true, PROMO_BISHOP);
                ml.add(from, to, true, true, PROMO_KNIGHT);
            }
            t &= t - 1;
        }

        t = capR;
        while (t)
        {
            int to = lsb(t);
            int from = to - 9;
            bool promo = (to >= 56);
            if (!promo)
            {
                ml.add(from, to, /*capture=*/true, /*promo=*/false);
            }
            else
            {
                ml.add(from, to, true, true, PROMO_QUEEN);
                ml.add(from, to, true, true, PROMO_ROOK);
                ml.add(from, to, true, true, PROMO_BISHOP);
                ml.add(from, to, true, true, PROMO_KNIGHT);
            }
            t &= t - 1;
        }
    }
    else
    {
        // ---------- BLACK PAWNS ----------
        // single pushes
        Bitboard singlePushB = south(bpawns) & empty;
        Bitboard promotionPushB = singlePushB & RANK_1;
        Bitboard quietPushB = singlePushB & ~RANK_1;

        Bitboard tmp = quietPushB;
        while (tmp)
        {
            int to = lsb(tmp);
            int from = to + 8;
            ml.add(from, to);
            tmp &= tmp - 1;
        }

        tmp = promotionPushB;
        while (tmp)
        {
            int to = lsb(tmp);
            int from = to + 8;
            ml.add(from, to, false, true, PROMO_QUEEN);
            ml.add(from, to, false, true, PROMO_ROOK);
            ml.add(from, to, false, true, PROMO_BISHOP);
            ml.add(from, to, false, true, PROMO_KNIGHT);
            tmp &= tmp - 1;
        }

        // double pushes (rank 7 -> 5)
        Bitboard rank7 = bpawns & RANK_7;
        Bitboard singlePushB2 = south(rank7) & empty;
        Bitboard doublePushB = south(singlePushB2) & empty;

        tmp = doublePushB;
        while (tmp)
        {
            int to = lsb(tmp);
            int from = to + 16;
            ml.add(from, to);
            tmp &= tmp - 1;
        }

        // ----- EN PASSANT -----
        if (pos.enPassantSquare != -1)
        {
            int ep = pos.enPassantSquare;
            Bitboard epTarget = 1ULL << ep;

            // For black, a pawn captures SE or SW to reach ep target,
            // so the *from* squares are one rank north.
            Bitboard fromLeft = north(west(epTarget));
            Bitboard fromRight = north(east(epTarget));
            Bitboard candidates = (fromLeft | fromRight) & bpawns;

            while (candidates)
            {
                int from = lsb(candidates);
                ml.add(from, ep, true);
                candidates &= candidates - 1;
            }
        }

        // BLACK captures — handle each ray separately
        Bitboard capL = south(west(bpawns)) & pos.occupancy[WHITE]; // from = to + 9 (black SW)
        Bitboard capR = south(east(bpawns)) & pos.occupancy[WHITE]; // from = to + 7 (black SE)

        // BLACK capture promos — left
        Bitboard t = capL;
        while (t)
        {
            int to = lsb(t);
            int from = to + 9;
            bool promo = (to <= 7);
            if (!promo)
            {
                ml.add(from, to, /*capture=*/true, /*promo=*/false);
            }
            else
            {
                ml.add(from, to, true, true, PROMO_QUEEN);
                ml.add(from, to, true, true, PROMO_ROOK);
                ml.add(from, to, true, true, PROMO_BISHOP);
                ml.add(from, to, true, true, PROMO_KNIGHT);
            }
            t &= t - 1;
        }

        // BLACK capture promos — right
        t = capR;
        while (t)
        {
            int to = lsb(t);
            int from = to + 7;
            bool promo = (to <= 7);
            if (!promo)
            {
                ml.add(from, to, /*capture=*/true, /*promo=*/false);
            }
            else
            {
                ml.add(from, to, true, true, PROMO_QUEEN);
                ml.add(from, to, true, true, PROMO_ROOK);
                ml.add(from, to, true, true, PROMO_BISHOP);
                ml.add(from, to, true, true, PROMO_KNIGHT);
            }
            t &= t - 1;
        }
    }
}

void generateKnightMoves(const Position &pos, MoveList &ml)
{
    Bitboard friends = pos.occupancy[pos.sideToMove];
    Bitboard enemies = pos.occupancy[1 - pos.sideToMove];

    Bitboard knights = (pos.sideToMove == WHITE) ? pos.pieceBB[WN] : pos.pieceBB[BN];

    while (knights)
    {
        int from = lsb(knights);
        knights &= knights - 1;

        Bitboard attacks = KNIGHT_ATTACKS[from] & ~friends;
        Bitboard tmp = attacks;
        while (tmp)
        {
            int to = lsb(tmp);
            bool cap = get_bit(enemies, to);
            ml.add(from, to, cap);
            tmp &= tmp - 1;
        }
    }
}

void generateKingMoves(const Position &pos, MoveList &ml)
{
    Bitboard friends = pos.occupancy[pos.sideToMove];
    Bitboard enemies = pos.occupancy[1 - pos.sideToMove];

    Bitboard king = (pos.sideToMove == WHITE) ? pos.pieceBB[WK] : pos.pieceBB[BK];
    if (!king)
        return;

    int from = lsb(king);
    Bitboard attacks = KING_ATTACKS[from] & ~friends;
    Bitboard tmp = attacks;
    while (tmp)
    {
        int to = lsb(tmp);
        bool cap = get_bit(enemies, to);
        ml.add(from, to, cap);
        tmp &= tmp - 1;
    }
}

void generateAllMoves(const Position &pos, MoveList &ml)
{
    generatePawnMoves(pos, ml);
    generateSliderMoves(pos, ml);
    generateKnightMoves(pos, ml);
    generateKingMoves(pos, ml);
}

void generateLegalMoves(const Position &pos, MoveList &legal)
{
    MoveList pseudo;
    generateAllMoves(pos, pseudo);

    Side us = pos.sideToMove;
    Side them = (us == WHITE ? BLACK : WHITE);

    for (auto &m : pseudo.moves)
    {
        Position next = pos;
        next.makeMove(m);

        // Determine our king square AFTER the move
        int kingSq = next.kingSquare[us];

        // If we haven't tracked kingSquare correctly yet, fallback to scanning:
        if (kingSq == -1)
        {
            Bitboard kingBB = next.pieceBB[(us == WHITE) ? WK : BK];
            kingSq = kingBB ? lsb(kingBB) : -1;
        }

        // illegal if king still ends in check
        if (kingSq != -1 && !next.isSquareAttacked(kingSq, them))
        {
            // special case: castling must NOT pass through check
            if (m.isCastle)
            {
                // starting king square in the original position
                int startK = pos.kingSquare[us];
                if (startK == -1)
                {
                    Bitboard kbb = pos.pieceBB[(us == WHITE) ? WK : BK];
                    startK = kbb ? lsb(kbb) : -1;
                }
                if (startK != -1 && pos.isSquareAttacked(startK, them))
                    continue; // can't castle out of check

                int f = -1, g = -1;
                if (us == WHITE && m.to == 6)
                {
                    f = 5;
                    g = 6;
                } // e1->g1
                else if (us == WHITE && m.to == 2)
                {
                    f = 3;
                    g = 2;
                } // e1->c1
                else if (us == BLACK && m.to == 62)
                {
                    f = 61;
                    g = 62;
                } // e8->g8
                else if (us == BLACK && m.to == 58)
                {
                    f = 59;
                    g = 58;
                } // e8->c8

                if (f != -1 && (pos.isSquareAttacked(f, them) || pos.isSquareAttacked(g, them)))
                    continue; // can't pass through or into check
            }

            // Preserve every semantic flag. In particular, dropping isCastle
            // here makes root search move the rook on makeMove(), but not put
            // it back on unmakeMove(), corrupting every later root branch.
            legal.moves.push_back(m);
        }
    }
}
