#pragma once
#include <string>

class Position;

namespace TB
{
    bool isEnabled();
    bool setPath(const std::string &path);
    const std::string &path();

    void setProbeDepth(int depth);
    int probeDepth();

    void setProbeLimit(int pieces);
    int probeLimit();

    bool probeWDL(const Position &pos, int &scoreOut);
}
