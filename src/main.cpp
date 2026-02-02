#include <iostream>
#include "uci.hpp"
#include "bitboard.hpp"
#include "zobrist.hpp"

int main()
{
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    initAttackTables();
    Zobrist::init();
    runUciLoop();
    return 0;
}
