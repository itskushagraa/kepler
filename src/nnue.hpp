#pragma once
#include <array>
#include <cstdint>
#include <string>

class Position;

namespace Nnue
{
    constexpr int kPsqtInputSize = 768;     // 12 pieces * 64 squares
    constexpr int kHalfKPInputSize = 41024; // 64 * (10 * 64 + 1)
    constexpr int kMaxHiddenSize = 1536;    // supports larger HalfKP nets

    struct Accumulator
    {
        std::array<int32_t, kMaxHiddenSize> classicHidden{};
        std::array<std::array<int32_t, kMaxHiddenSize>, 2> halfHidden{};
        bool classicValid = false;
        std::array<bool, 2> halfValid{false, false};
        std::array<int, 2> kingSq{-1, -1};
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
    void unapplyMove(
        Accumulator &acc,
        int movedPiece, int from, int to,
        int capturedPiece, int capturedSquare,
        int promotionPiece,
        int castleRookPiece, int castleRookFrom, int castleRookTo,
        uint64_t previousPositionKey);
    bool evaluate(const Position &pos, Accumulator &acc, int &scoreOut);
}
