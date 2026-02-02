#pragma once
#include <fstream>
#include <mutex>
#include <string>

class DataWriter
{
public:
    bool open(const std::string &path);
    void close();
    bool isOpen() const;
    bool appendSample(const std::string &fen, int result, int scoreCp);

private:
    std::ofstream out;
    mutable std::mutex mtx;
};
