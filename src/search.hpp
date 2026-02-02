#pragma once
#include <atomic>
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
    int nodes = 0;
    bool infinite = false;
    bool printInfo = true;
};

struct SearchResult
{
    Move bestMove{};
    int score = 0;
    int depth = 0;
    uint64_t nodes = 0;
    uint64_t qnodes = 0;
};

struct TTEntry
{
    uint64_t key = 0;
    int depth = -1;
    int score = 0;
    uint8_t bound = 0;
    Move bestMove{};
};

class TranspositionTable
{
public:
    void resizeMB(int mb);
    void clear();
    bool probe(uint64_t key, TTEntry &out) const;
    void store(uint64_t key, int depth, int score, uint8_t bound, const Move &bestMove);

private:
    std::vector<TTEntry> table;
    size_t mask = 0;
};

SearchResult search(Position &pos, const SearchLimits &limits, TranspositionTable &tt, std::atomic<bool> &stopFlag);
