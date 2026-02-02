#include "zobrist.hpp"
#include <random>

namespace
{
    uint64_t pieceKeys[12][64];
    uint64_t castlingKeys[16];
    uint64_t epKeys[8];
    uint64_t sideKey = 0;

    uint64_t rand64(std::mt19937_64 &rng)
    {
        std::uniform_int_distribution<uint64_t> dist;
        return dist(rng);
    }
}

namespace Zobrist
{
    void init()
    {
        std::mt19937_64 rng(0xC0FFEE1234ULL);
        for (int p = 0; p < 12; ++p)
        {
            for (int sq = 0; sq < 64; ++sq)
            {
                pieceKeys[p][sq] = rand64(rng);
            }
        }
        for (int i = 0; i < 16; ++i)
        {
            castlingKeys[i] = rand64(rng);
        }
        for (int f = 0; f < 8; ++f)
        {
            epKeys[f] = rand64(rng);
        }
        sideKey = rand64(rng);
    }

    uint64_t piece(int piece, int sq) { return pieceKeys[piece][sq]; }
    uint64_t castling(int rights) { return castlingKeys[rights & 0xF]; }
    uint64_t epFile(int file) { return epKeys[file & 7]; }
    uint64_t side() { return sideKey; }
}
