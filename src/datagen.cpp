#include "datagen.hpp"

bool DataWriter::open(const std::string &path)
{
    std::lock_guard<std::mutex> lock(mtx);
    if (out.is_open())
        out.close();
    out.open(path, std::ios::out | std::ios::app);
    return out.good();
}

void DataWriter::close()
{
    std::lock_guard<std::mutex> lock(mtx);
    if (out.is_open())
        out.close();
}

bool DataWriter::isOpen() const
{
    std::lock_guard<std::mutex> lock(mtx);
    return out.is_open();
}

bool DataWriter::appendSample(const std::string &fen, int result, int scoreCp)
{
    std::lock_guard<std::mutex> lock(mtx);
    if (!out.is_open())
        return false;
    out << result << '\t' << scoreCp << '\t' << fen << '\n';
    return out.good();
}
