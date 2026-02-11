#include "eval.hpp"
#include "movegen.hpp"
#include "nnue.hpp"
#include <algorithm>
#include <cstdlib>

namespace
{
    // Tuned-ish values and PSQT baseline adapted from a standard PeSTO-style eval.
    constexpr int kMgValue[6] = {82, 337, 365, 477, 1025, 0};
    constexpr int kEgValue[6] = {94, 281, 297, 512, 936, 0};
    constexpr int kPhaseInc[6] = {0, 1, 1, 2, 4, 0};

    constexpr int kMgPst[6][64] = {
        {
            0, 0, 0, 0, 0, 0, 0, 0,
            98, 134, 61, 95, 68, 126, 34, -11,
            -6, 7, 26, 31, 65, 56, 25, -20,
            -14, 13, 6, 21, 23, 12, 17, -23,
            -27, -2, -5, 12, 17, 6, 10, -25,
            -26, -4, -4, -10, 3, 3, 33, -12,
            -35, -1, -20, -23, -15, 24, 38, -22,
            0, 0, 0, 0, 0, 0, 0, 0},
        {
            -167, -89, -34, -49, 61, -97, -15, -107,
            -73, -41, 72, 36, 23, 62, 7, -17,
            -47, 60, 37, 65, 84, 129, 73, 44,
            -9, 17, 19, 53, 37, 69, 18, 22,
            -13, 4, 16, 13, 28, 19, 21, -8,
            -23, -9, 12, 10, 19, 17, 25, -16,
            -29, -53, -12, -3, -1, 18, -14, -19,
            -105, -21, -58, -33, -17, -28, -19, -23},
        {
            -29, 4, -82, -37, -25, -42, 7, -8,
            -26, 16, -18, -13, 30, 59, 18, -47,
            -16, 37, 43, 40, 35, 50, 37, -2,
            -4, 5, 19, 50, 37, 37, 7, -2,
            -6, 13, 13, 26, 34, 12, 10, 4,
            0, 15, 15, 15, 14, 27, 18, 10,
            4, 15, 16, 0, 7, 21, 33, 1,
            -33, -3, -14, -21, -13, -12, -39, -21},
        {
            32, 42, 32, 51, 63, 9, 31, 43,
            27, 32, 58, 62, 80, 67, 26, 44,
            -5, 19, 26, 36, 17, 45, 61, 16,
            -24, -11, 7, 26, 24, 35, -8, -20,
            -36, -26, -12, -1, 9, -7, 6, -23,
            -45, -25, -16, -17, 3, 0, -5, -33,
            -44, -16, -20, -9, -1, 11, -6, -71,
            -19, -13, 1, 17, 16, 7, -37, -26},
        {
            -28, 0, 29, 12, 59, 44, 43, 45,
            -24, -39, -5, 1, -16, 57, 28, 54,
            -13, -17, 7, 8, 29, 56, 47, 57,
            -27, -27, -16, -16, -1, 17, -2, 1,
            -9, -26, -9, -10, -2, -4, 3, -3,
            -14, 2, -11, -2, -5, 2, 14, 5,
            -35, -8, 11, 2, 8, 15, -3, 1,
            -1, -18, -9, 10, -15, -25, -31, -50},
        {
            -65, 23, 16, -15, -56, -34, 2, 13,
            29, -1, -20, -7, -8, -4, -38, -29,
            -9, 24, 2, -16, -20, 6, 22, -22,
            -17, -20, -12, -27, -30, -25, -14, -36,
            -49, -1, -27, -39, -46, -44, -33, -51,
            -14, -14, -22, -46, -44, -30, -15, -27,
            1, 7, -8, -64, -43, -16, 9, 8,
            -15, 36, 12, -54, 8, -28, 24, 14},
    };

    constexpr int kEgPst[6][64] = {
        {
            0, 0, 0, 0, 0, 0, 0, 0,
            178, 173, 158, 134, 147, 132, 165, 187,
            94, 100, 85, 67, 56, 53, 82, 84,
            32, 24, 13, 5, -2, 4, 17, 17,
            13, 9, -3, -7, -7, -8, 3, -1,
            4, 7, -6, 1, 0, -5, -1, -8,
            13, 8, 8, 10, 13, 0, 2, -7,
            0, 0, 0, 0, 0, 0, 0, 0},
        {
            -58, -38, -13, -28, -31, -27, -63, -99,
            -25, -8, -25, -2, -9, -25, -24, -52,
            -24, -20, 10, 9, -1, -9, -19, -41,
            -17, 3, 22, 22, 22, 11, 8, -18,
            -18, -6, 16, 25, 16, 17, 4, -18,
            -23, -3, -1, 15, 10, -3, -20, -22,
            -42, -20, -10, -5, -2, -20, -23, -44,
            -29, -51, -23, -15, -22, -18, -50, -64},
        {
            -14, -21, -11, -8, -7, -9, -17, -24,
            -8, -4, 7, -12, -3, -13, -4, -14,
            2, -8, 0, -1, -2, 6, 0, 4,
            -3, 9, 12, 9, 14, 10, 3, 2,
            -6, 3, 13, 19, 7, 10, -3, -9,
            -12, -3, 8, 10, 13, 3, -7, -15,
            -14, -18, -7, -1, 4, -9, -15, -27,
            -23, -9, -23, -5, -9, -16, -5, -17},
        {
            13, 10, 18, 15, 12, 12, 8, 5,
            11, 13, 13, 11, -3, 3, 8, 3,
            7, 7, 7, 5, 4, -3, -5, -3,
            4, 3, 13, 1, 2, 1, -1, 2,
            3, 5, 8, 4, -5, -6, -8, -11,
            -4, 0, -5, -1, -7, -12, -8, -16,
            -6, -6, 0, 2, -9, -9, -11, -3,
            -9, 2, 3, -1, -5, -13, 4, -20},
        {
            -9, 22, 22, 27, 27, 19, 10, 20,
            -17, 20, 32, 41, 58, 25, 30, 0,
            -20, 6, 9, 49, 47, 35, 19, 9,
            3, 22, 24, 45, 57, 40, 57, 36,
            -18, 28, 19, 47, 31, 34, 39, 23,
            -16, -27, 15, 6, 9, 17, 10, 5,
            -22, -23, -30, -16, -16, -23, -36, -32,
            -33, -28, -22, -43, -5, -32, -20, -41},
        {
            -74, -35, -18, -18, -11, 15, 4, -17,
            -12, 17, 14, 17, 17, 38, 23, 11,
            10, 17, 23, 15, 20, 45, 44, 13,
            -8, 22, 24, 27, 26, 33, 26, 3,
            -18, -4, 21, 24, 27, 23, 9, -11,
            -19, -3, 11, 21, 23, 16, 7, -9,
            -27, -11, 4, 13, 14, 4, -5, -17,
            -53, -34, -21, -11, -28, -14, -24, -43},
    };

    constexpr int kMaxPhase = 24;
    constexpr int kTempoBonus = 10;
    constexpr int kPassedBonusByRank[8] = {0, 6, 12, 22, 38, 60, 92, 0};

    int mirrorSq(int sq)
    {
        return sq ^ 56;
    }

    int pieceTypeIndex(int piece)
    {
        if (piece == WP || piece == BP)
            return 0;
        if (piece == WN || piece == BN)
            return 1;
        if (piece == WB || piece == BB)
            return 2;
        if (piece == WR || piece == BR)
            return 3;
        if (piece == WQ || piece == BQ)
            return 4;
        return 5;
    }

    Bitboard fileMask(int file)
    {
        return FILE_A << file;
    }

    int manhattanDist(int a, int b)
    {
        return std::abs((a & 7) - (b & 7)) + std::abs((a >> 3) - (b >> 3));
    }

    int materialWithoutKing(const Position &pos, Side side)
    {
        if (side == WHITE)
        {
            return popcount(pos.pieceBB[WP]) * kEgValue[0] +
                   popcount(pos.pieceBB[WN]) * kEgValue[1] +
                   popcount(pos.pieceBB[WB]) * kEgValue[2] +
                   popcount(pos.pieceBB[WR]) * kEgValue[3] +
                   popcount(pos.pieceBB[WQ]) * kEgValue[4];
        }
        return popcount(pos.pieceBB[BP]) * kEgValue[0] +
               popcount(pos.pieceBB[BN]) * kEgValue[1] +
               popcount(pos.pieceBB[BB]) * kEgValue[2] +
               popcount(pos.pieceBB[BR]) * kEgValue[3] +
               popcount(pos.pieceBB[BQ]) * kEgValue[4];
    }

    int bishopPairScore(const Position &pos)
    {
        int score = 0;
        if (popcount(pos.pieceBB[WB]) >= 2)
            score += 28;
        if (popcount(pos.pieceBB[BB]) >= 2)
            score -= 28;
        return score;
    }

    int rookActivityScore(const Position &pos)
    {
        const Bitboard whitePawns = pos.pieceBB[WP];
        const Bitboard blackPawns = pos.pieceBB[BP];
        int score = 0;

        Bitboard wr = pos.pieceBB[WR];
        while (wr)
        {
            int sq = lsb(wr);
            wr &= wr - 1;
            int file = sq & 7;
            Bitboard fm = fileMask(file);
            bool ownPawn = (whitePawns & fm) != 0ULL;
            bool oppPawn = (blackPawns & fm) != 0ULL;
            if (!ownPawn && !oppPawn)
                score += 18;
            else if (!ownPawn)
                score += 10;
            if ((sq >> 3) == 6)
                score += 14;
        }

        Bitboard br = pos.pieceBB[BR];
        while (br)
        {
            int sq = lsb(br);
            br &= br - 1;
            int file = sq & 7;
            Bitboard fm = fileMask(file);
            bool ownPawn = (blackPawns & fm) != 0ULL;
            bool oppPawn = (whitePawns & fm) != 0ULL;
            if (!ownPawn && !oppPawn)
                score -= 18;
            else if (!ownPawn)
                score -= 10;
            if ((sq >> 3) == 1)
                score -= 14;
        }

        return score;
    }

    int mopUpScore(const Position &pos)
    {
        const int whiteMat = materialWithoutKing(pos, WHITE);
        const int blackMat = materialWithoutKing(pos, BLACK);
        const int diff = whiteMat - blackMat;
        if (std::abs(diff) < 220)
            return 0;

        const Side winner = (diff > 0) ? WHITE : BLACK;
        const Side loser = (winner == WHITE) ? BLACK : WHITE;
        const int winKing = pos.kingSquare[winner];
        const int loseKing = pos.kingSquare[loser];
        if (winKing < 0 || loseKing < 0)
            return 0;

        // Pull enemy king toward the edge and bring our king closer in won endings.
        const int enemyEdge =
            std::abs((loseKing & 7) - 3) + std::abs((loseKing >> 3) - 3);
        const int kingApproach = 14 - manhattanDist(winKing, loseKing);
        const int bonus = enemyEdge * 8 + kingApproach * 6;
        return (diff > 0) ? bonus : -bonus;
    }

    int mobilityScore(const Position &pos, Side side)
    {
        const Bitboard ownOcc = pos.occupancy[side];
        const Bitboard occ = pos.allPieces;

        int score = 0;
        Bitboard bb = pos.pieceBB[(side == WHITE) ? WN : BN];
        while (bb)
        {
            int sq = lsb(bb);
            bb &= bb - 1;
            score += popcount(KNIGHT_ATTACKS[sq] & ~ownOcc) * 4;
        }

        bb = pos.pieceBB[(side == WHITE) ? WB : BB];
        while (bb)
        {
            int sq = lsb(bb);
            bb &= bb - 1;
            score += popcount(bishopAttacks(sq, occ) & ~ownOcc) * 4;
        }

        bb = pos.pieceBB[(side == WHITE) ? WR : BR];
        while (bb)
        {
            int sq = lsb(bb);
            bb &= bb - 1;
            score += popcount(rookAttacks(sq, occ) & ~ownOcc) * 2;
        }

        bb = pos.pieceBB[(side == WHITE) ? WQ : BQ];
        while (bb)
        {
            int sq = lsb(bb);
            bb &= bb - 1;
            score += popcount((rookAttacks(sq, occ) | bishopAttacks(sq, occ)) & ~ownOcc);
        }

        return score;
    }

    int pawnStructureScore(const Position &pos)
    {
        const Bitboard whitePawns = pos.pieceBB[WP];
        const Bitboard blackPawns = pos.pieceBB[BP];
        int score = 0;

        for (int file = 0; file < 8; ++file)
        {
            int whiteOnFile = popcount(whitePawns & fileMask(file));
            int blackOnFile = popcount(blackPawns & fileMask(file));
            if (whiteOnFile > 1)
                score -= (whiteOnFile - 1) * 14;
            if (blackOnFile > 1)
                score += (blackOnFile - 1) * 14;
        }

        Bitboard wp = whitePawns;
        while (wp)
        {
            int sq = lsb(wp);
            wp &= wp - 1;
            const int f = sq & 7;
            const int r = sq >> 3;

            Bitboard adjacent = 0ULL;
            if (f > 0)
                adjacent |= fileMask(f - 1);
            if (f < 7)
                adjacent |= fileMask(f + 1);
            if ((whitePawns & adjacent) == 0ULL)
                score -= 10;

            bool passed = true;
            for (int rr = r + 1; rr < 8 && passed; ++rr)
            {
                for (int ff = std::max(0, f - 1); ff <= std::min(7, f + 1); ++ff)
                {
                    if (blackPawns & (ONE << (rr * 8 + ff)))
                    {
                        passed = false;
                        break;
                    }
                }
            }
            if (passed)
                score += kPassedBonusByRank[r];
        }

        Bitboard bp = blackPawns;
        while (bp)
        {
            int sq = lsb(bp);
            bp &= bp - 1;
            const int f = sq & 7;
            const int r = sq >> 3;

            Bitboard adjacent = 0ULL;
            if (f > 0)
                adjacent |= fileMask(f - 1);
            if (f < 7)
                adjacent |= fileMask(f + 1);
            if ((blackPawns & adjacent) == 0ULL)
                score += 10;

            bool passed = true;
            for (int rr = r - 1; rr >= 0 && passed; --rr)
            {
                for (int ff = std::max(0, f - 1); ff <= std::min(7, f + 1); ++ff)
                {
                    if (whitePawns & (ONE << (rr * 8 + ff)))
                    {
                        passed = false;
                        break;
                    }
                }
            }
            if (passed)
                score -= kPassedBonusByRank[7 - r];
        }

        return score;
    }

    int kingSafetyScore(const Position &pos, Side side)
    {
        const int kingSq = pos.kingSquare[side];
        if (kingSq < 0 || kingSq >= 64)
            return 0;

        const int kf = kingSq & 7;
        const int kr = kingSq >> 3;
        const int dir = (side == WHITE) ? 1 : -1;
        const Bitboard ownPawns = pos.pieceBB[(side == WHITE) ? WP : BP];
        const Bitboard oppPawns = pos.pieceBB[(side == WHITE) ? BP : WP];

        int score = 0;
        int shield = 0;
        const int sr = kr + dir;
        if (sr >= 0 && sr < 8)
        {
            for (int df = -1; df <= 1; ++df)
            {
                int sf = kf + df;
                if (sf < 0 || sf > 7)
                    continue;
                int sq = sr * 8 + sf;
                if (ownPawns & (ONE << sq))
                    shield++;
            }
        }
        score += shield * 12;
        if (shield == 0)
            score -= 28;
        else if (shield == 1)
            score -= 10;

        Bitboard kingFile = fileMask(kf);
        bool ownOnKingFile = (ownPawns & kingFile) != 0ULL;
        bool oppOnKingFile = (oppPawns & kingFile) != 0ULL;
        if (!ownOnKingFile)
            score -= 12;
        if (!ownOnKingFile && !oppOnKingFile)
            score -= 8;

        for (int af = kf - 1; af <= kf + 1; af += 2)
        {
            if (af < 0 || af > 7)
                continue;
            if ((ownPawns & fileMask(af)) == 0ULL)
                score -= 4;
        }

        return score;
    }

    int evaluateClassical(const Position &pos)
    {
        int mg = 0;
        int eg = 0;
        int phase = 0;

        for (int piece = 0; piece < 12; ++piece)
        {
            const int pType = pieceTypeIndex(piece);
            const int sign = (piece < 6) ? 1 : -1;
            Bitboard bb = pos.pieceBB[piece];
            while (bb)
            {
                int sq = lsb(bb);
                bb &= bb - 1;
                int psq = (sign > 0) ? sq : mirrorSq(sq);
                mg += sign * (kMgValue[pType] + kMgPst[pType][psq]);
                eg += sign * (kEgValue[pType] + kEgPst[pType][psq]);
                phase += kPhaseInc[pType];
            }
        }

        if (phase > kMaxPhase)
            phase = kMaxPhase;

        int score = (mg * phase + eg * (kMaxPhase - phase)) / kMaxPhase;
        score += mobilityScore(pos, WHITE) - mobilityScore(pos, BLACK);
        score += pawnStructureScore(pos);
        score += bishopPairScore(pos);
        score += rookActivityScore(pos);

        // De-emphasize king safety as pieces come off.
        int kingSafety = kingSafetyScore(pos, WHITE) - kingSafetyScore(pos, BLACK);
        score += (kingSafety * phase) / kMaxPhase;
        score += (mopUpScore(pos) * (kMaxPhase - phase)) / kMaxPhase;

        score += (pos.sideToMove == WHITE) ? kTempoBonus : -kTempoBonus;
        return score;
    }
}

int evaluate(const Position &pos, bool *usedNnue)
{
    const int classical = evaluateClassical(pos);
    const int classicalStm = (pos.sideToMove == WHITE) ? classical : -classical;

    int nnueScore = 0;
    auto &acc = const_cast<Position &>(pos).nnueAccumulator;
    if (Nnue::evaluate(pos, acc, nnueScore))
    {
        if (usedNnue)
            *usedNnue = true;

        // Keep NNUE in the loop but anchor to a stable classical signal
        // while our current net is still bootstrapping.
        int nnueClamped = std::clamp(nnueScore, classicalStm - 300, classicalStm + 300);
        return (classicalStm * 3 + nnueClamped) / 4;
    }

    if (usedNnue)
        *usedNnue = false;
    return classicalStm;
}

int evaluate(const Position &pos)
{
    return evaluate(pos, nullptr);
}
