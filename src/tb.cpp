#include "tb.hpp"
#include "position.hpp"
#include "bitboard.hpp"

#include <algorithm>
#include <string>

#ifdef KEPLER_SYZYGY
    #include "tbprobe.h"
#endif

namespace
{
    std::string g_path;
    int g_probeDepth = 1;
    int g_probeLimit = 6;
    bool g_enabled = false;

#ifdef KEPLER_SYZYGY
    int tbWdlToScore(int wdl, int ply)
    {
        constexpr int TB_WIN_SCORE = 20000;
        if (wdl == TB_WIN || wdl == TB_CURSED_WIN)
            return TB_WIN_SCORE - ply;
        if (wdl == TB_LOSS || wdl == TB_BLESSED_LOSS)
            return -TB_WIN_SCORE + ply;
        return 0;
    }
#endif
}

namespace TB
{
    bool isEnabled()
    {
        return g_enabled;
    }

    const std::string &path()
    {
        return g_path;
    }

    bool setPath(const std::string &path)
    {
        g_path = path;
#ifdef KEPLER_SYZYGY
        if (!g_path.empty())
        {
            g_enabled = tb_init(g_path.c_str());
            return g_enabled;
        }
#endif
        g_enabled = false;
        return false;
    }

    void setProbeDepth(int depth)
    {
        g_probeDepth = std::max(1, depth);
    }

    int probeDepth()
    {
        return g_probeDepth;
    }

    void setProbeLimit(int pieces)
    {
        g_probeLimit = std::clamp(pieces, 0, 7);
    }

    int probeLimit()
    {
        return g_probeLimit;
    }

    bool probeWDL(const Position &pos, int &scoreOut)
    {
#ifdef KEPLER_SYZYGY
        if (!g_enabled)
            return false;
        if (g_probeLimit <= 0)
            return false;
        if (pos.canCastleKingside[WHITE] || pos.canCastleQueenside[WHITE] ||
            pos.canCastleKingside[BLACK] || pos.canCastleQueenside[BLACK])
            return false;

        const int pieces = popcount(pos.allPieces);
        if (pieces > g_probeLimit)
            return false;

        const Bitboard white = pos.occupancy[WHITE];
        const Bitboard black = pos.occupancy[BLACK];
        const Bitboard kings = pos.pieceBB[WK] | pos.pieceBB[BK];
        const Bitboard queens = pos.pieceBB[WQ] | pos.pieceBB[BQ];
        const Bitboard rooks = pos.pieceBB[WR] | pos.pieceBB[BR];
        const Bitboard bishops = pos.pieceBB[WB] | pos.pieceBB[BB];
        const Bitboard knights = pos.pieceBB[WN] | pos.pieceBB[BN];
        const Bitboard pawns = pos.pieceBB[WP] | pos.pieceBB[BP];

        const int ep = pos.enPassantSquare >= 0 ? pos.enPassantSquare : 0;
        const int wdl = tb_probe_wdl(
            white,
            black,
            kings,
            queens,
            rooks,
            bishops,
            knights,
            pawns,
            pos.halfmoveClock,
            ep,
            pos.sideToMove == WHITE ? 0 : 1);
        if (wdl == TB_RESULT_FAILED)
            return false;

        scoreOut = tbWdlToScore(wdl, 0);
        return true;
#else
        (void)pos;
        (void)scoreOut;
        return false;
#endif
    }
}
