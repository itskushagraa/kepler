#include "nnue.hpp"
#include "position.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <vector>

namespace
{
    constexpr int kHalfKPStride = 641;

    enum class ModelKind
    {
        NONE,
        CLASSIC_PSQT,
        HALFKP,
    };

    struct Model
    {
        bool loaded = false;
        ModelKind kind = ModelKind::NONE;
        int32_t inputSize = 0;
        int32_t hiddenSize = 0;
        int32_t outputSize = 0;
        int32_t scale = 256;
        std::vector<int16_t> featureWeights;
        std::vector<int16_t> hiddenBias;
        std::vector<int16_t> outputWeights;
        int32_t outputBias = 0;
    };

    Model g_model;

    bool readExact(std::ifstream &in, char *dst, std::streamsize size)
    {
        in.read(dst, size);
        return static_cast<std::streamsize>(in.gcount()) == size;
    }

    int orientedSq(int sq, Side perspective)
    {
        return (perspective == WHITE) ? sq : (sq ^ 56);
    }

    bool isKingPiece(int piece)
    {
        return piece == WK || piece == BK;
    }

    int halfKPPieceIndex(int piece, Side perspective)
    {
        static constexpr std::array<int, 12> kWhiteMap = {
            0, 1, 2, 3, 4, -1,
            5, 6, 7, 8, 9, -1};
        static constexpr std::array<int, 12> kBlackMap = {
            5, 6, 7, 8, 9, -1,
            0, 1, 2, 3, 4, -1};

        if (piece < 0 || piece >= 12)
            return -1;
        return (perspective == WHITE) ? kWhiteMap[static_cast<size_t>(piece)] : kBlackMap[static_cast<size_t>(piece)];
    }

    int classicFeatureIndex(int piece, int sq)
    {
        if (piece < 0 || piece >= 12 || sq < 0 || sq >= 64)
            return -1;
        return piece * 64 + sq;
    }

    int halfKPFeatureIndex(int kingSq, Side perspective, int piece, int sq)
    {
        if (kingSq < 0 || kingSq >= 64 || sq < 0 || sq >= 64)
            return -1;

        const int mappedPiece = halfKPPieceIndex(piece, perspective);
        if (mappedPiece < 0)
            return -1;

        const int orientedKing = orientedSq(kingSq, perspective);
        const int orientedPieceSq = orientedSq(sq, perspective);
        return orientedKing * kHalfKPStride + mappedPiece * 64 + orientedPieceSq;
    }

    void addFeature(std::array<int32_t, Nnue::kMaxHiddenSize> &hidden, int fi)
    {
        if (fi < 0 || fi >= g_model.inputSize)
            return;

        const size_t offset = static_cast<size_t>(fi) * static_cast<size_t>(g_model.hiddenSize);
        const int16_t *weights = g_model.featureWeights.data() + offset;
        for (int h = 0; h < g_model.hiddenSize; ++h)
            hidden[static_cast<size_t>(h)] += weights[static_cast<size_t>(h)];
    }

    void removeFeature(std::array<int32_t, Nnue::kMaxHiddenSize> &hidden, int fi)
    {
        if (fi < 0 || fi >= g_model.inputSize)
            return;

        const size_t offset = static_cast<size_t>(fi) * static_cast<size_t>(g_model.hiddenSize);
        const int16_t *weights = g_model.featureWeights.data() + offset;
        for (int h = 0; h < g_model.hiddenSize; ++h)
            hidden[static_cast<size_t>(h)] -= weights[static_cast<size_t>(h)];
    }

    void refreshClassicAccumulator(const Position &pos, Nnue::Accumulator &acc)
    {
        for (int h = 0; h < g_model.hiddenSize; ++h)
            acc.classicHidden[static_cast<size_t>(h)] = g_model.hiddenBias[static_cast<size_t>(h)];
        for (int h = g_model.hiddenSize; h < Nnue::kMaxHiddenSize; ++h)
            acc.classicHidden[static_cast<size_t>(h)] = 0;

        for (int p = 0; p < 12; ++p)
        {
            Bitboard bb = pos.pieceBB[static_cast<size_t>(p)];
            while (bb)
            {
                const int sq = lsb(bb);
                bb &= bb - 1;
                addFeature(acc.classicHidden, classicFeatureIndex(p, sq));
            }
        }

        acc.classicValid = true;
        acc.halfValid[WHITE] = false;
        acc.halfValid[BLACK] = false;
        acc.kingSq[WHITE] = pos.kingSquare[WHITE];
        acc.kingSq[BLACK] = pos.kingSquare[BLACK];
        acc.positionKey = pos.hashKey;
    }

    bool refreshHalfPerspective(const Position &pos, Nnue::Accumulator &acc, Side perspective)
    {
        const int kingSq = pos.kingSquare[perspective];
        if (kingSq < 0 || kingSq >= 64)
        {
            acc.halfValid[perspective] = false;
            return false;
        }

        acc.kingSq[perspective] = kingSq;

        auto &target = acc.halfHidden[perspective];
        for (int h = 0; h < g_model.hiddenSize; ++h)
            target[static_cast<size_t>(h)] = g_model.hiddenBias[static_cast<size_t>(h)];
        for (int h = g_model.hiddenSize; h < Nnue::kMaxHiddenSize; ++h)
            target[static_cast<size_t>(h)] = 0;

        for (int piece = 0; piece < 12; ++piece)
        {
            if (isKingPiece(piece))
                continue;

            Bitboard bb = pos.pieceBB[static_cast<size_t>(piece)];
            while (bb)
            {
                const int sq = lsb(bb);
                bb &= bb - 1;
                addFeature(target, halfKPFeatureIndex(kingSq, perspective, piece, sq));
            }
        }

        acc.halfValid[perspective] = true;
        return true;
    }

    int halfKPFeatureIndexFromAcc(const Nnue::Accumulator &acc, Side perspective, int piece, int sq)
    {
        return halfKPFeatureIndex(acc.kingSq[perspective], perspective, piece, sq);
    }

    void applyClassicMove(
        Nnue::Accumulator &acc,
        int movedPiece, int from, int to,
        int capturedPiece, int capturedSquare,
        int promotionPiece,
        int castleRookPiece, int castleRookFrom, int castleRookTo)
    {
        if (!acc.classicValid)
            return;

        removeFeature(acc.classicHidden, classicFeatureIndex(movedPiece, from));
        if (promotionPiece >= 0)
            addFeature(acc.classicHidden, classicFeatureIndex(promotionPiece, to));
        else
            addFeature(acc.classicHidden, classicFeatureIndex(movedPiece, to));

        if (capturedPiece >= 0 && capturedSquare >= 0)
            removeFeature(acc.classicHidden, classicFeatureIndex(capturedPiece, capturedSquare));

        if (castleRookPiece >= 0 && castleRookFrom >= 0 && castleRookTo >= 0)
        {
            removeFeature(acc.classicHidden, classicFeatureIndex(castleRookPiece, castleRookFrom));
            addFeature(acc.classicHidden, classicFeatureIndex(castleRookPiece, castleRookTo));
        }
    }

    void unapplyClassicMove(
        Nnue::Accumulator &acc,
        int movedPiece, int from, int to,
        int capturedPiece, int capturedSquare,
        int promotionPiece,
        int castleRookPiece, int castleRookFrom, int castleRookTo)
    {
        if (!acc.classicValid)
            return;

        if (castleRookPiece >= 0 && castleRookFrom >= 0 && castleRookTo >= 0)
        {
            removeFeature(acc.classicHidden, classicFeatureIndex(castleRookPiece, castleRookTo));
            addFeature(acc.classicHidden, classicFeatureIndex(castleRookPiece, castleRookFrom));
        }

        if (capturedPiece >= 0 && capturedSquare >= 0)
            addFeature(acc.classicHidden, classicFeatureIndex(capturedPiece, capturedSquare));

        if (promotionPiece >= 0)
        {
            removeFeature(acc.classicHidden, classicFeatureIndex(promotionPiece, to));
            addFeature(acc.classicHidden, classicFeatureIndex(movedPiece, from));
        }
        else
        {
            removeFeature(acc.classicHidden, classicFeatureIndex(movedPiece, to));
            addFeature(acc.classicHidden, classicFeatureIndex(movedPiece, from));
        }
    }

    void applyHalfMove(
        Nnue::Accumulator &acc,
        int movedPiece, int from, int to,
        int capturedPiece, int capturedSquare,
        int promotionPiece,
        int castleRookPiece, int castleRookFrom, int castleRookTo)
    {
        const Side movedSide = (movedPiece < 6) ? WHITE : BLACK;
        const bool kingMoved = (movedPiece == WK || movedPiece == BK);

        for (int p = 0; p < 2; ++p)
        {
            const Side perspective = static_cast<Side>(p);
            if (!acc.halfValid[perspective])
                continue;

            if (kingMoved && perspective == movedSide)
            {
                acc.halfValid[perspective] = false;
                continue;
            }

            auto &target = acc.halfHidden[perspective];

            if (!kingMoved)
            {
                removeFeature(target, halfKPFeatureIndexFromAcc(acc, perspective, movedPiece, from));
                if (promotionPiece >= 0)
                    addFeature(target, halfKPFeatureIndexFromAcc(acc, perspective, promotionPiece, to));
                else
                    addFeature(target, halfKPFeatureIndexFromAcc(acc, perspective, movedPiece, to));
            }

            if (capturedPiece >= 0 && capturedSquare >= 0 && !isKingPiece(capturedPiece))
                removeFeature(target, halfKPFeatureIndexFromAcc(acc, perspective, capturedPiece, capturedSquare));

            if (castleRookPiece >= 0 && castleRookFrom >= 0 && castleRookTo >= 0)
            {
                removeFeature(target, halfKPFeatureIndexFromAcc(acc, perspective, castleRookPiece, castleRookFrom));
                addFeature(target, halfKPFeatureIndexFromAcc(acc, perspective, castleRookPiece, castleRookTo));
            }
        }

        if (kingMoved)
            acc.kingSq[movedSide] = to;
    }

    void unapplyHalfMove(
        Nnue::Accumulator &acc,
        int movedPiece, int from, int to,
        int capturedPiece, int capturedSquare,
        int promotionPiece,
        int castleRookPiece, int castleRookFrom, int castleRookTo)
    {
        const Side movedSide = (movedPiece < 6) ? WHITE : BLACK;
        const bool kingMoved = (movedPiece == WK || movedPiece == BK);

        if (kingMoved)
            acc.kingSq[movedSide] = from;

        for (int p = 0; p < 2; ++p)
        {
            const Side perspective = static_cast<Side>(p);
            if (!acc.halfValid[perspective])
                continue;

            if (kingMoved && perspective == movedSide)
            {
                acc.halfValid[perspective] = false;
                continue;
            }

            auto &target = acc.halfHidden[perspective];

            if (castleRookPiece >= 0 && castleRookFrom >= 0 && castleRookTo >= 0)
            {
                removeFeature(target, halfKPFeatureIndexFromAcc(acc, perspective, castleRookPiece, castleRookTo));
                addFeature(target, halfKPFeatureIndexFromAcc(acc, perspective, castleRookPiece, castleRookFrom));
            }

            if (capturedPiece >= 0 && capturedSquare >= 0 && !isKingPiece(capturedPiece))
                addFeature(target, halfKPFeatureIndexFromAcc(acc, perspective, capturedPiece, capturedSquare));

            if (!kingMoved)
            {
                if (promotionPiece >= 0)
                {
                    removeFeature(target, halfKPFeatureIndexFromAcc(acc, perspective, promotionPiece, to));
                    addFeature(target, halfKPFeatureIndexFromAcc(acc, perspective, movedPiece, from));
                }
                else
                {
                    removeFeature(target, halfKPFeatureIndexFromAcc(acc, perspective, movedPiece, to));
                    addFeature(target, halfKPFeatureIndexFromAcc(acc, perspective, movedPiece, from));
                }
            }
        }
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

        Model next;
        next.inputSize = inputSize;
        next.hiddenSize = hiddenSize;
        next.scale = scale;

        if (inputSize == kPsqtInputSize)
        {
            next.kind = ModelKind::CLASSIC_PSQT;
            next.outputSize = hiddenSize;
        }
        else if (inputSize == kHalfKPInputSize)
        {
            next.kind = ModelKind::HALFKP;
            next.outputSize = hiddenSize * 2;
        }
        else
        {
            return false;
        }

        if (hiddenSize <= 0 || hiddenSize > kMaxHiddenSize || scale <= 0 || next.outputSize <= 0)
            return false;

        const size_t featureCount = static_cast<size_t>(inputSize) * static_cast<size_t>(hiddenSize);
        const size_t biasCount = static_cast<size_t>(hiddenSize);
        const size_t outCount = static_cast<size_t>(next.outputSize);

        next.featureWeights.resize(featureCount);
        next.hiddenBias.resize(biasCount);
        next.outputWeights.resize(outCount);

        if (!readExact(in, reinterpret_cast<char *>(next.featureWeights.data()),
                       static_cast<std::streamsize>(sizeof(int16_t) * featureCount)))
            return false;
        if (!readExact(in, reinterpret_cast<char *>(next.hiddenBias.data()),
                       static_cast<std::streamsize>(sizeof(int16_t) * biasCount)))
            return false;
        if (!readExact(in, reinterpret_cast<char *>(next.outputWeights.data()),
                       static_cast<std::streamsize>(sizeof(int16_t) * outCount)))
            return false;
        if (!readExact(in, reinterpret_cast<char *>(&next.outputBias), sizeof(next.outputBias)))
            return false;

        next.loaded = true;
        g_model = std::move(next);
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
        acc.classicValid = false;
        acc.halfValid[WHITE] = false;
        acc.halfValid[BLACK] = false;
        acc.kingSq[WHITE] = -1;
        acc.kingSq[BLACK] = -1;
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
        if (!g_model.loaded)
            return;

        if (movedPiece < 0 || from < 0 || to < 0)
        {
            invalidate(acc);
            return;
        }

        if (g_model.kind == ModelKind::CLASSIC_PSQT)
            applyClassicMove(acc, movedPiece, from, to, capturedPiece, capturedSquare, promotionPiece, castleRookPiece, castleRookFrom, castleRookTo);
        else if (g_model.kind == ModelKind::HALFKP)
            applyHalfMove(acc, movedPiece, from, to, capturedPiece, capturedSquare, promotionPiece, castleRookPiece, castleRookFrom, castleRookTo);

        acc.positionKey = newPositionKey;
    }

    void unapplyMove(
        Accumulator &acc,
        int movedPiece, int from, int to,
        int capturedPiece, int capturedSquare,
        int promotionPiece,
        int castleRookPiece, int castleRookFrom, int castleRookTo,
        uint64_t previousPositionKey)
    {
        if (!g_model.loaded)
            return;

        if (movedPiece < 0 || from < 0 || to < 0)
        {
            invalidate(acc);
            return;
        }

        if (g_model.kind == ModelKind::CLASSIC_PSQT)
            unapplyClassicMove(acc, movedPiece, from, to, capturedPiece, capturedSquare, promotionPiece, castleRookPiece, castleRookFrom, castleRookTo);
        else if (g_model.kind == ModelKind::HALFKP)
            unapplyHalfMove(acc, movedPiece, from, to, capturedPiece, capturedSquare, promotionPiece, castleRookPiece, castleRookFrom, castleRookTo);

        acc.positionKey = previousPositionKey;
    }

    bool evaluate(const Position &pos, Accumulator &acc, int &scoreOut)
    {
        if (!g_model.loaded)
            return false;

        if (g_model.kind == ModelKind::CLASSIC_PSQT)
        {
            if (!acc.classicValid || acc.positionKey != pos.hashKey)
                refreshClassicAccumulator(pos, acc);

            int64_t sum = g_model.outputBias;
            for (int h = 0; h < g_model.hiddenSize; ++h)
            {
                const int32_t v = std::clamp(acc.classicHidden[static_cast<size_t>(h)], 0, 255);
                sum += static_cast<int64_t>(v) * g_model.outputWeights[static_cast<size_t>(h)];
            }
            const int score = static_cast<int>(sum / g_model.scale);
            scoreOut = (pos.sideToMove == WHITE) ? score : -score;
            return true;
        }

        if (g_model.kind == ModelKind::HALFKP)
        {
            if (acc.positionKey != pos.hashKey)
            {
                acc.halfValid[WHITE] = false;
                acc.halfValid[BLACK] = false;
            }

            if (!acc.halfValid[WHITE] && !refreshHalfPerspective(pos, acc, WHITE))
                return false;
            if (!acc.halfValid[BLACK] && !refreshHalfPerspective(pos, acc, BLACK))
                return false;

            const Side us = pos.sideToMove;
            const Side them = (us == WHITE ? BLACK : WHITE);
            int64_t sum = g_model.outputBias;

            for (int h = 0; h < g_model.hiddenSize; ++h)
            {
                const int32_t v = std::clamp(acc.halfHidden[us][static_cast<size_t>(h)], 0, 255);
                sum += static_cast<int64_t>(v) * g_model.outputWeights[static_cast<size_t>(h)];
            }

            const size_t secondOffset = static_cast<size_t>(g_model.hiddenSize);
            for (int h = 0; h < g_model.hiddenSize; ++h)
            {
                const int32_t v = std::clamp(acc.halfHidden[them][static_cast<size_t>(h)], 0, 255);
                sum += static_cast<int64_t>(v) * g_model.outputWeights[secondOffset + static_cast<size_t>(h)];
            }

            scoreOut = static_cast<int>(sum / g_model.scale);
            acc.positionKey = pos.hashKey;
            return true;
        }

        return false;
    }
}
