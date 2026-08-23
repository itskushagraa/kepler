#pragma once
#include "position.hpp"

int evaluate(const Position &pos);
int evaluate(const Position &pos, bool *usedNnue);

void setNnueWeight(int percent);
int nnueWeight();
void setNnueClamp(int centipawns);
int nnueClamp();
