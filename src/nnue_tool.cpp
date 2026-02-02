#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

namespace
{
    constexpr int kInputSize = 768;
    constexpr int kHiddenSize = 128;

    int pieceValueUnit(int piece)
    {
        switch (piece)
        {
        case 0: // WP
        case 6: // BP
            return 1;
        case 1: // WN
        case 7: // BN
            return 3;
        case 2: // WB
        case 8: // BB
            return 3;
        case 3: // WR
        case 9: // BR
            return 5;
        case 4: // WQ
        case 10: // BQ
            return 9;
        default:
            return 0;
        }
    }
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: nnue_tool <output.nnue>\n";
        return 1;
    }

    std::string outPath = argv[1];
    std::ofstream out(outPath, std::ios::binary);
    if (!out)
    {
        std::cerr << "Failed to open output file: " << outPath << "\n";
        return 1;
    }

    const char magic[8] = {'K', 'N', 'N', 'U', 'E', 'v', '1', '\0'};
    const int32_t inputSize = kInputSize;
    const int32_t hiddenSize = kHiddenSize;
    const int32_t scale = 1;

    std::array<int16_t, kInputSize * kHiddenSize> featureWeights{};
    std::array<int16_t, kHiddenSize> hiddenBias{};
    std::array<int16_t, kHiddenSize> outputWeights{};
    int32_t outputBias = 0;

    // Build a tiny deterministic test network:
    // hidden[0] tracks white material units, hidden[1] tracks black material units.
    for (int piece = 0; piece < 12; ++piece)
    {
        int unit = pieceValueUnit(piece);
        for (int sq = 0; sq < 64; ++sq)
        {
            int fi = piece * 64 + sq;
            int offset = fi * kHiddenSize;
            if (piece <= 5)
                featureWeights[offset + 0] = static_cast<int16_t>(unit);
            else
                featureWeights[offset + 1] = static_cast<int16_t>(unit);
        }
    }

    outputWeights[0] = 1;
    outputWeights[1] = -1;

    out.write(magic, sizeof(magic));
    out.write(reinterpret_cast<const char *>(&inputSize), sizeof(inputSize));
    out.write(reinterpret_cast<const char *>(&hiddenSize), sizeof(hiddenSize));
    out.write(reinterpret_cast<const char *>(&scale), sizeof(scale));
    out.write(reinterpret_cast<const char *>(featureWeights.data()), sizeof(int16_t) * featureWeights.size());
    out.write(reinterpret_cast<const char *>(hiddenBias.data()), sizeof(int16_t) * hiddenBias.size());
    out.write(reinterpret_cast<const char *>(outputWeights.data()), sizeof(int16_t) * outputWeights.size());
    out.write(reinterpret_cast<const char *>(&outputBias), sizeof(outputBias));

    if (!out)
    {
        std::cerr << "Failed while writing output file.\n";
        return 1;
    }

    std::cout << "Wrote test NNUE file: " << outPath << "\n";
    return 0;
}
