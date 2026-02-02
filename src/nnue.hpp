#pragma once
#include <array>
#include <cstdint>
#include <string>

class Position;

namespace Nnue
{
    constexpr int kInputSize = 768;  // 12 pieces * 64 squares
    constexpr int kHiddenSize = 128; // starter scaffold size

    struct Accumulator
    {
        std::array<int32_t, kHiddenSize> hidden{};
        bool initialized = false;
        uint64_t positionKey = 0;
    };

    bool loadFromFile(const std::string &path);
    bool isLoaded();
    void clear();
    void invalidate(Accumulator &acc);
    void applyMove(
        Accumulator &acc,
        int movedPiece, int from, int to,
        int capturedPiece, int capturedSquare,
        int promotionPiece,
        int castleRookPiece, int castleRookFrom, int castleRookTo,
        uint64_t newPositionKey);
    bool evaluate(const Position &pos, Accumulator &acc, int &scoreOut);
}
