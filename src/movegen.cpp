#include "movegen.hpp"
#include <array>
#include <cstdlib>
#include <iostream>

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

namespace
{
    constexpr std::array<Bitboard, 64> kRookMagics = {
        0x8a80104000800020ULL, 0x0140002000100040ULL, 0x02801880a0017001ULL, 0x0100081001000420ULL,
        0x0200020010080420ULL, 0x03001c0002010008ULL, 0x8480008002000100ULL, 0x2080088004402900ULL,
        0x0000800098204000ULL, 0x2024401000200040ULL, 0x0100802000801000ULL, 0x0120800800801000ULL,
        0x0208808088000400ULL, 0x0002802200800400ULL, 0x2200800100020080ULL, 0x0801000060821100ULL,
        0x0080044006422000ULL, 0x0100808020004000ULL, 0x12108a0010204200ULL, 0x0140848010000802ULL,
        0x0481828014002800ULL, 0x8094004002004100ULL, 0x4010040010010802ULL, 0x0000020008806104ULL,
        0x0100400080208000ULL, 0x2040002120081000ULL, 0x0021200680100081ULL, 0x0020100080080080ULL,
        0x0002000a00200410ULL, 0x0000020080800400ULL, 0x0080088400100102ULL, 0x0080004600042881ULL,
        0x4040008040800020ULL, 0x0440003000200801ULL, 0x0004200011004500ULL, 0x0188020010100100ULL,
        0x0014800401802800ULL, 0x2080040080800200ULL, 0x0124080204001001ULL, 0x0200046502000484ULL,
        0x0480400080088020ULL, 0x1000422010034000ULL, 0x0030200100110040ULL, 0x0000100021010009ULL,
        0x2002080100110004ULL, 0x0202008004008002ULL, 0x0020020004010100ULL, 0x2048440040820001ULL,
        0x0101002200408200ULL, 0x0040802000401080ULL, 0x4008142004410100ULL, 0x02060820c0120200ULL,
        0x0001001004080100ULL, 0x020c020080040080ULL, 0x2935610830022400ULL, 0x0044440041009200ULL,
        0x0280001040802101ULL, 0x2100190040002085ULL, 0x80c0084100102001ULL, 0x4024081001000421ULL,
        0x00020030a0244872ULL, 0x0012001008414402ULL, 0x02006104900a0804ULL, 0x0001004081002402ULL};

    constexpr std::array<Bitboard, 64> kBishopMagics = {
        0x0040040844404084ULL, 0x002004208a004208ULL, 0x0010190041080202ULL, 0x0108060845042010ULL,
        0x0581104180800210ULL, 0x2112080446200010ULL, 0x1080820820060210ULL, 0x03c0808410220200ULL,
        0x0004050404440404ULL, 0x0000021001420088ULL, 0x24d0080801082102ULL, 0x0001020a0a020400ULL,
        0x0000040308200402ULL, 0x0004011002100800ULL, 0x0401484104104005ULL, 0x0801010402020200ULL,
        0x00400210c3880100ULL, 0x0404022024108200ULL, 0x0810018200204102ULL, 0x0004002801a02003ULL,
        0x0085040820080400ULL, 0x810102c808880400ULL, 0x000e900410884800ULL, 0x8002020480840102ULL,
        0x0220200865090201ULL, 0x2010100a02021202ULL, 0x0152048408022401ULL, 0x0020080002081110ULL,
        0x4001001021004000ULL, 0x800040400a011002ULL, 0x00e4004081011002ULL, 0x001c004001012080ULL,
        0x8004200962a00220ULL, 0x8422100208500202ULL, 0x2000402200300c08ULL, 0x8646020080080080ULL,
        0x80020a0200100808ULL, 0x2010004880111000ULL, 0x623000a080011400ULL, 0x42008c0340209202ULL,
        0x0209188240001000ULL, 0x400408a884001800ULL, 0x00110400a6080400ULL, 0x1840060a44020800ULL,
        0x0090080104000041ULL, 0x0201011000808101ULL, 0x1a2208080504f080ULL, 0x8012020600211212ULL,
        0x0500861011240000ULL, 0x0180806108200800ULL, 0x4000020e01040044ULL, 0x300000261044000aULL,
        0x0802241102020002ULL, 0x0020906061210001ULL, 0x5a84841004010310ULL, 0x0004010801011c04ULL,
        0x000a010109502200ULL, 0x0000004a02012000ULL, 0x500201010098b028ULL, 0x8040002811040900ULL,
        0x0028000010020204ULL, 0x06000020202d0240ULL, 0x8918844842082200ULL, 0x4010011029020020ULL};

    std::array<Bitboard, 64> rookMasks{};
    std::array<Bitboard, 64> bishopMasks{};
    std::array<unsigned, 64> rookShifts{};
    std::array<unsigned, 64> bishopShifts{};
    std::array<std::array<Bitboard, 4096>, 64> rookAttackTable{};
    std::array<std::array<Bitboard, 512>, 64> bishopAttackTable{};

    bool inBounds(int rank, int file)
    {
        return rank >= 0 && rank < 8 && file >= 0 && file < 8;
    }

    Bitboard slidingAttacksSlow(int sq, Bitboard occupancy, const int directions[][2], int directionCount)
    {
        Bitboard attacks = 0;
        const int rank = sq >> 3;
        const int file = sq & 7;
        for (int direction = 0; direction < directionCount; ++direction)
        {
            int targetRank = rank + directions[direction][0];
            int targetFile = file + directions[direction][1];
            while (inBounds(targetRank, targetFile))
            {
                const int target = targetRank * 8 + targetFile;
                attacks |= ONE << target;
                if (occupancy & (ONE << target))
                    break;
                targetRank += directions[direction][0];
                targetFile += directions[direction][1];
            }
        }
        return attacks;
    }

    Bitboard relevantMask(int sq, const int directions[][2], int directionCount)
    {
        Bitboard mask = 0;
        const int rank = sq >> 3;
        const int file = sq & 7;
        for (int direction = 0; direction < directionCount; ++direction)
        {
            int targetRank = rank + directions[direction][0];
            int targetFile = file + directions[direction][1];
            while (inBounds(targetRank, targetFile))
            {
                const int nextRank = targetRank + directions[direction][0];
                const int nextFile = targetFile + directions[direction][1];
                if (!inBounds(nextRank, nextFile))
                    break;
                mask |= ONE << (targetRank * 8 + targetFile);
                targetRank = nextRank;
                targetFile = nextFile;
            }
        }
        return mask;
    }

    template <size_t TableSize>
    void initializeSliderTable(
        int sq,
        Bitboard mask,
        Bitboard magic,
        unsigned shift,
        const int directions[][2],
        int directionCount,
        std::array<Bitboard, TableSize> &table)
    {
        std::array<bool, TableSize> initialized{};
        Bitboard subset = 0;
        do
        {
            const size_t index = static_cast<size_t>((subset * magic) >> shift);
            const Bitboard attacks = slidingAttacksSlow(sq, subset, directions, directionCount);
            if (index >= TableSize || (initialized[index] && table[index] != attacks))
            {
                std::cerr << "invalid sliding magic at square " << sq
                          << " index " << index << "\n";
                std::abort();
            }
            table[index] = attacks;
            initialized[index] = true;
            subset = (subset - mask) & mask;
        } while (subset != 0);
    }
}

void initAttackTables()
{
    static const bool initialized = []
    {
        constexpr int rookDirections[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        constexpr int bishopDirections[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
        for (int sq = 0; sq < 64; ++sq)
        {
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

            rookMasks[sq] = relevantMask(sq, rookDirections, 4);
            bishopMasks[sq] = relevantMask(sq, bishopDirections, 4);
            rookShifts[sq] = 64U - static_cast<unsigned>(popcount(rookMasks[sq]));
            bishopShifts[sq] = 64U - static_cast<unsigned>(popcount(bishopMasks[sq]));
            initializeSliderTable(
                sq, rookMasks[sq], kRookMagics[sq], rookShifts[sq],
                rookDirections, 4, rookAttackTable[sq]);
            initializeSliderTable(
                sq, bishopMasks[sq], kBishopMagics[sq], bishopShifts[sq],
                bishopDirections, 4, bishopAttackTable[sq]);
        }
        return true;
    }();
    (void)initialized;
}

// ------------------------------------------------------------
// Sliding-piece helpers (magic-bitboard attack tables)
// ------------------------------------------------------------
void generateSliderMoves(const Position &pos, MoveList &ml)
{
    const Side side = pos.sideToMove;
    const Bitboard friends = pos.occupancy[side];
    const Bitboard enemies = pos.occupancy[side == WHITE ? BLACK : WHITE];

    auto addMoves = [&](Bitboard pieces, bool diagonal, bool orthogonal)
    {
        while (pieces)
        {
            const int from = lsb(pieces);
            pieces &= pieces - 1;
            Bitboard destinations = 0;
            if (diagonal)
                destinations |= bishopAttacks(from, pos.allPieces);
            if (orthogonal)
                destinations |= rookAttacks(from, pos.allPieces);
            destinations &= ~friends;
            while (destinations)
            {
                const int to = lsb(destinations);
                destinations &= destinations - 1;
                ml.add(from, to, get_bit(enemies, to));
            }
        }
    };

    const int rook = side == WHITE ? WR : BR;
    const int bishop = side == WHITE ? WB : BB;
    const int queen = side == WHITE ? WQ : BQ;
    addMoves(pos.pieceBB[rook], false, true);
    addMoves(pos.pieceBB[bishop], true, false);
    addMoves(pos.pieceBB[queen], true, true);

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
    const Bitboard blockers = occ & bishopMasks[sq];
    const size_t index = static_cast<size_t>((blockers * kBishopMagics[sq]) >> bishopShifts[sq]);
    return bishopAttackTable[sq][index];
}

// Returns all squares a rook attacks from `sq` given current occupancy
Bitboard rookAttacks(int sq, Bitboard occ)
{
    const Bitboard blockers = occ & rookMasks[sq];
    const size_t index = static_cast<size_t>((blockers * kRookMagics[sq]) >> rookShifts[sq]);
    return rookAttackTable[sq][index];
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
