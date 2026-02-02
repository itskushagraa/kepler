#include "nnue.hpp"
#include "position.hpp"
#include <algorithm>
#include <cstring>
#include <fstream>

namespace
{
    struct Model
    {
        bool loaded = false;
        int32_t scale = 256;
        std::array<int16_t, Nnue::kInputSize * Nnue::kHiddenSize> featureWeights{};
        std::array<int16_t, Nnue::kHiddenSize> hiddenBias{};
        std::array<int16_t, Nnue::kHiddenSize> outputWeights{};
        int32_t outputBias = 0;
    };

    Model g_model;

    int featureIndex(int piece, int sq)
    {
        return piece * 64 + sq;
    }

    void addFeature(Nnue::Accumulator &acc, int piece, int sq)
    {
        int fi = featureIndex(piece, sq);
        int offset = fi * Nnue::kHiddenSize;
        for (int h = 0; h < Nnue::kHiddenSize; ++h)
        {
            acc.hidden[h] += g_model.featureWeights[offset + h];
        }
    }

    void removeFeature(Nnue::Accumulator &acc, int piece, int sq)
    {
        int fi = featureIndex(piece, sq);
        int offset = fi * Nnue::kHiddenSize;
        for (int h = 0; h < Nnue::kHiddenSize; ++h)
        {
            acc.hidden[h] -= g_model.featureWeights[offset + h];
        }
    }

    bool readExact(std::ifstream &in, char *dst, std::streamsize size)
    {
        in.read(dst, size);
        return static_cast<std::streamsize>(in.gcount()) == size;
    }

    void refreshAccumulator(const Position &pos, Nnue::Accumulator &acc)
    {
        for (int h = 0; h < Nnue::kHiddenSize; ++h)
        {
            acc.hidden[h] = g_model.hiddenBias[h];
        }

        for (int p = 0; p < 12; ++p)
        {
            Bitboard bb = pos.pieceBB[p];
            while (bb)
            {
                int sq = lsb(bb);
                bb &= bb - 1;

                addFeature(acc, p, sq);
            }
        }

        acc.initialized = true;
        acc.positionKey = pos.hashKey;
    }
}

namespace Nnue
{
    bool loadFromFile(const std::string &path)
    {
        std::ifstream in(path, std::ios::binary);
        if (!in)
            return false;

        char magic[8]{};
        if (!readExact(in, magic, 8))
            return false;
        const char expected[8] = {'K', 'N', 'N', 'U', 'E', 'v', '1', '\0'};
        if (std::memcmp(magic, expected, 8) != 0)
            return false;

        int32_t inputSize = 0;
        int32_t hiddenSize = 0;
        int32_t scale = 0;
        if (!readExact(in, reinterpret_cast<char *>(&inputSize), sizeof(inputSize)))
            return false;
        if (!readExact(in, reinterpret_cast<char *>(&hiddenSize), sizeof(hiddenSize)))
            return false;
        if (!readExact(in, reinterpret_cast<char *>(&scale), sizeof(scale)))
            return false;

        if (inputSize != kInputSize || hiddenSize != kHiddenSize || scale <= 0)
            return false;

        if (!readExact(in, reinterpret_cast<char *>(g_model.featureWeights.data()),
                       sizeof(int16_t) * g_model.featureWeights.size()))
            return false;
        if (!readExact(in, reinterpret_cast<char *>(g_model.hiddenBias.data()),
                       sizeof(int16_t) * g_model.hiddenBias.size()))
            return false;
        if (!readExact(in, reinterpret_cast<char *>(g_model.outputWeights.data()),
                       sizeof(int16_t) * g_model.outputWeights.size()))
            return false;
        if (!readExact(in, reinterpret_cast<char *>(&g_model.outputBias), sizeof(g_model.outputBias)))
            return false;

        g_model.scale = scale;
        g_model.loaded = true;
        return true;
    }

    bool isLoaded()
    {
        return g_model.loaded;
    }

    void clear()
    {
        g_model = Model{};
    }

    void invalidate(Accumulator &acc)
    {
        acc.initialized = false;
        acc.positionKey = 0;
    }

    void applyMove(
        Accumulator &acc,
        int movedPiece, int from, int to,
        int capturedPiece, int capturedSquare,
        int promotionPiece,
        int castleRookPiece, int castleRookFrom, int castleRookTo,
        uint64_t newPositionKey)
    {
        if (!g_model.loaded || !acc.initialized)
            return;
        if (movedPiece < 0 || from < 0 || to < 0)
        {
            invalidate(acc);
            return;
        }

        removeFeature(acc, movedPiece, from);
        if (promotionPiece >= 0)
            addFeature(acc, promotionPiece, to);
        else
            addFeature(acc, movedPiece, to);

        if (capturedPiece >= 0 && capturedSquare >= 0)
            removeFeature(acc, capturedPiece, capturedSquare);

        if (castleRookPiece >= 0 && castleRookFrom >= 0 && castleRookTo >= 0)
        {
            removeFeature(acc, castleRookPiece, castleRookFrom);
            addFeature(acc, castleRookPiece, castleRookTo);
        }

        acc.positionKey = newPositionKey;
    }

    bool evaluate(const Position &pos, Accumulator &acc, int &scoreOut)
    {
        if (!g_model.loaded)
            return false;

        if (!acc.initialized || acc.positionKey != pos.hashKey)
        {
            refreshAccumulator(pos, acc);
        }

        int64_t sum = g_model.outputBias;
        for (int h = 0; h < kHiddenSize; ++h)
        {
            int32_t v = std::clamp(acc.hidden[h], 0, 255);
            sum += static_cast<int64_t>(v) * g_model.outputWeights[h];
        }
        int score = static_cast<int>(sum / g_model.scale);
        scoreOut = (pos.sideToMove == WHITE) ? score : -score;
        return true;
    }
}
