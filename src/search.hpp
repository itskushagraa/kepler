#pragma once
#include <atomic>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include "movegen.hpp"

struct SearchLimits
{
    int depth = 0;
    int movetimeMs = 0;
    int wtimeMs = 0;
    int btimeMs = 0;
    int wincMs = 0;
    int bincMs = 0;
    int movesToGo = 0;
    int moveOverheadMs = 10;
    int nodes = 0;
    int threads = 1;
    int contempt = 0;
    bool usePruning = true;
    bool infinite = false;
    bool printInfo = true;
    // Complete game hash history, including the current root position.
    // Search extends this vector as it descends so threefold repetition can
    // be distinguished from a single earlier occurrence.
    std::vector<uint64_t> positionHistory;
};

struct SearchResult
{
    Move bestMove{};
    int score = 0;
    int depth = 0;
    int elapsedMs = 0;
    uint64_t nodes = 0;
    uint64_t qnodes = 0;
};

struct TimeBudget
{
    int optimumMs = 0;
    int maximumMs = 0;
    bool adaptive = false;
};

struct TTEntry
{
    uint64_t key = 0;
    int depth = -1;
    int score = 0;
    uint8_t bound = 0;
    uint8_t generation = 0;
    Move bestMove{};
};

class TranspositionTable
{
public:
    void resizeMB(int mb);
    void clear();
    void newSearch();
    bool probe(uint64_t key, TTEntry &out) const;
    void store(uint64_t key, int depth, int score, uint8_t bound, const Move &bestMove);
    size_t sizeBytes() const;
    size_t bucketCount() const;
    bool isLockFree() const;

private:
    // Each slot is published as an atomic payload plus an XOR verification
    // word. Readers validate the pair and reject inconsistent publications
    // without serializing every node on a mutex or creating a C++ data race.
    struct TTSlot
    {
        std::atomic<uint64_t> verification{0};
        std::atomic<uint64_t> payload{0};

        TTSlot() noexcept = default;
        TTSlot(const TTSlot &other) noexcept
            : verification(other.verification.load(std::memory_order_relaxed)),
              payload(other.payload.load(std::memory_order_relaxed)) {}
        TTSlot &operator=(const TTSlot &other) noexcept
        {
            verification.store(other.verification.load(std::memory_order_relaxed), std::memory_order_relaxed);
            payload.store(other.payload.load(std::memory_order_relaxed), std::memory_order_relaxed);
            return *this;
        }
        TTSlot(TTSlot &&other) noexcept : TTSlot(other) {}
        TTSlot &operator=(TTSlot &&other) noexcept { return operator=(other); }
    };

    struct TTBucket
    {
        static constexpr int kClusterSize = 4;
        std::array<TTSlot, kClusterSize> entries{};
    };

    std::vector<TTBucket> table;
    size_t mask = 0;
    std::atomic<uint32_t> generation{1};
};

SearchResult search(Position &pos, const SearchLimits &limits, TranspositionTable &tt, std::atomic<bool> &stopFlag);
TimeBudget calculateTimeBudget(const Position &pos, const SearchLimits &limits);

constexpr int SEARCH_MATE_SCORE = 30000;
bool isSearchMateScore(int score);
int searchMateMoves(int score);
