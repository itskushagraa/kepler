#include <atomic>
#include <cstdint>
#include <iostream>

#include "position.hpp"
#include "search.hpp"
#include "zobrist.hpp"

int main()
{
    Zobrist::init();

    constexpr int kBudget = 20000;
    for (int threads : {1, 2, 4, 8})
    {
        Position position;
        position.setStartPos();

        TranspositionTable tt;
        tt.resizeMB(16);

        SearchLimits limits;
        limits.nodes = kBudget;
        limits.threads = threads;
        limits.printInfo = false;

        std::atomic<bool> stopFlag{false};
        const SearchResult result = search(position, limits, tt, stopFlag);
        const uint64_t searched = result.nodes + result.qnodes;
        if (searched != static_cast<uint64_t>(kBudget))
        {
            std::cerr << "threads=" << threads
                      << " searched=" << searched
                      << " expected=" << kBudget << "\n";
            return 1;
        }
    }

    std::cout << "search node budget passed for 1, 2, 4, and 8 threads\n";
    return 0;
}
