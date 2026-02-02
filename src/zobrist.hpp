#pragma once
#include <cstdint>

namespace Zobrist
{
    void init();
    uint64_t piece(int piece, int sq);
    uint64_t castling(int rights);
    uint64_t epFile(int file);
    uint64_t side();
}
