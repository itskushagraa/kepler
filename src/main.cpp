#include <iostream>
#include "uci.hpp"
#include "bitboard.hpp"

int main()
{
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    initAttackTables();
    runUciLoop();
    return 0;
}
