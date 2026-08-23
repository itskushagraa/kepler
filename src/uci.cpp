#include <atomic>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include "uci.hpp"
#include "datagen.hpp"
#include "position.hpp"
#include "movegen.hpp"
#include "search.hpp"
#include "nnue.hpp"
#include "eval.hpp"
#include "tb.hpp"

namespace
{
    const std::string kDefaultBaselineEvalFile = "models/kepler_baseline_pst_v1.nnue";

    int squareFromString(const std::string &s)
    {
        if (s.size() < 2)
            return -1;
        int file = s[0] - 'a';
        int rank = s[1] - '1';
        if (file < 0 || file > 7 || rank < 0 || rank > 7)
            return -1;
        return rank * 8 + file;
    }

    std::string moveToUci(const Move &m)
    {
        std::string out;
        out += char('a' + (m.from % 8));
        out += char('1' + (m.from / 8));
        out += char('a' + (m.to % 8));
        out += char('1' + (m.to / 8));
        if (m.isPromotion)
        {
            char pc = 'q';
            if (m.promoPiece == PROMO_ROOK)
                pc = 'r';
            else if (m.promoPiece == PROMO_BISHOP)
                pc = 'b';
            else if (m.promoPiece == PROMO_KNIGHT)
                pc = 'n';
            out += pc;
        }
        return out;
    }

    bool parseMoveUci(const Position &pos, const std::string &str, Move &out)
    {
        if (str.size() < 4)
            return false;
        int from = squareFromString(str.substr(0, 2));
        int to = squareFromString(str.substr(2, 2));
        if (from < 0 || to < 0)
            return false;

        uint8_t promo = PROMO_NONE;
        if (str.size() >= 5)
        {
            char c = str[4];
            if (c == 'q')
                promo = PROMO_QUEEN;
            else if (c == 'r')
                promo = PROMO_ROOK;
            else if (c == 'b')
                promo = PROMO_BISHOP;
            else if (c == 'n')
                promo = PROMO_KNIGHT;
        }

        MoveList legal;
        generateLegalMoves(pos, legal);
        for (const auto &m : legal.moves)
        {
            if (m.from == from && m.to == to)
            {
                if (m.isPromotion && promo != PROMO_NONE && m.promoPiece != promo)
                    continue;
                if (m.isPromotion && promo == PROMO_NONE)
                    continue;
                out = m;
                return true;
            }
        }
        return false;
    }

    std::string trim(std::string s)
    {
        auto isNotSpace = [](unsigned char c)
        { return !std::isspace(c); };
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), isNotSpace));
        s.erase(std::find_if(s.rbegin(), s.rend(), isNotSpace).base(), s.end());
        return s;
    }

    bool parseBool(const std::string &v, bool fallback)
    {
        if (v == "true" || v == "1" || v == "on")
            return true;
        if (v == "false" || v == "0" || v == "off")
            return false;
        return fallback;
    }

    struct SelfplaySample
    {
        std::string fen;
        Side sideToMove = WHITE;
        int scoreCp = 0;
    };

    bool isGameTerminal(Position &p, const std::unordered_map<uint64_t, int> &reps, int ply, int maxPly, int &whiteResult)
    {
        if (p.halfmoveClock >= 100 || ply >= maxPly)
        {
            whiteResult = 0;
            return true;
        }
        auto it = reps.find(p.hashKey);
        if (it != reps.end() && it->second >= 3)
        {
            whiteResult = 0;
            return true;
        }

        MoveList legal;
        generateLegalMoves(p, legal);
        if (!legal.moves.empty())
            return false;

        Side us = p.sideToMove;
        Side them = (us == WHITE ? BLACK : WHITE);
        int ksq = p.kingSquare[us];
        bool inCheck = (ksq != -1) && p.isSquareAttacked(ksq, them);
        if (!inCheck)
            whiteResult = 0;
        else
            whiteResult = (us == WHITE ? -1 : 1);
        return true;
    }
}

void runUciLoop()
{
    // Ensure UCI responses are flushed when stdout is piped (GUI/python-chess).
    std::cout << std::unitbuf;

    std::string cmd;
    Position pos;
    pos.setStartPos();

    TranspositionTable tt;
    tt.resizeMB(64);

    std::atomic<bool> stopFlag{false};
    std::atomic<bool> searching{false};
    std::thread searchThread;
    std::vector<uint64_t> lastBenchNodes;
    std::vector<uint64_t> lastBenchTimes;
    uint64_t lastBenchTotalNodes = 0;
    uint64_t lastBenchTotalTimeMs = 0;
    bool hasLastBench = false;
    std::string baselineEvalFile = kDefaultBaselineEvalFile;
    bool useBaseline = true;
    int searchThreads = 1;
    int contemptCp = 0;
    int moveOverheadMs = 10;
    bool usePruning = true;

    auto loadEvalFile = [&](const std::string &path)
    {
        // Accumulators contain sums of the currently loaded model's weights.
        // They must not survive a model replacement, even when the board hash
        // itself has not changed.
        Nnue::invalidate(pos.nnueAccumulator);
        if (!path.empty() && Nnue::loadFromFile(path))
        {
            std::cout << "info string nnue loaded " << path << "\n";
            return true;
        }
        Nnue::clear();
        std::cout << "info string nnue load failed " << path << "\n";
        return false;
    };

    if (useBaseline)
        Nnue::loadFromFile(baselineEvalFile);

    auto stopSearch = [&]()
    {
        stopFlag.store(true, std::memory_order_relaxed);
        if (searchThread.joinable())
            searchThread.join();
        searching.store(false, std::memory_order_relaxed);
    };

    while (std::getline(std::cin, cmd))
    {
        if (cmd == "uci")
        {
            std::cout << "id name Kepler\n";
            std::cout << "id author Kush Sharma\n";
            std::cout << "option name Hash type spin default 64 min 1 max 1024\n";
            std::cout << "option name Threads type spin default 1 min 1 max 128\n";
            std::cout << "option name Contempt type spin default 0 min -100 max 100\n";
            std::cout << "option name MoveOverhead type spin default 10 min 0 max 500\n";
            std::cout << "option name UsePruning type check default true\n";
            std::cout << "option name NNUEWeight type spin default 25 min 0 max 100\n";
            std::cout << "option name NNUEClamp type spin default 300 min 0 max 10000\n";
            std::cout << "option name EvalFile type string default <empty>\n";
            std::cout << "option name BaselineEvalFile type string default " << kDefaultBaselineEvalFile << "\n";
            std::cout << "option name UseBaseline type check default true\n";
            std::cout << "option name SyzygyPath type string default <empty>\n";
            std::cout << "option name SyzygyProbeDepth type spin default 1 min 1 max 64\n";
            std::cout << "option name SyzygyProbeLimit type spin default 6 min 0 max 7\n";
            std::cout << "uciok\n";
        }
        else if (cmd == "isready")
        {
            std::cout << "readyok\n";
        }
        else if (cmd.rfind("setoption", 0) == 0)
        {
            std::istringstream iss(cmd);
            std::string token, name, value;
            iss >> token; // setoption
            iss >> token; // name
            if (token != "name")
                continue;
            while (iss >> token && token != "value")
            {
                if (!name.empty())
                    name += " ";
                name += token;
            }
            if (token == "value")
                std::getline(iss, value);
            value = trim(value);
            if (name == "Hash")
            {
                int mb = std::stoi(value);
                tt.resizeMB(mb);
            }
            else if (name == "Threads")
            {
                int t = 1;
                try
                {
                    t = std::stoi(value);
                }
                catch (const std::exception &)
                {
                    t = searchThreads;
                }
                searchThreads = std::clamp(t, 1, 128);
            }
            else if (name == "Contempt")
            {
                int c = 0;
                try
                {
                    c = std::stoi(value);
                }
                catch (const std::exception &)
                {
                    c = contemptCp;
                }
                contemptCp = std::clamp(c, -100, 100);
            }
            else if (name == "MoveOverhead")
            {
                int overhead = moveOverheadMs;
                try
                {
                    overhead = std::stoi(value);
                }
                catch (const std::exception &)
                {
                    overhead = moveOverheadMs;
                }
                moveOverheadMs = std::clamp(overhead, 0, 500);
            }
            else if (name == "UsePruning")
            {
                usePruning = parseBool(value, usePruning);
            }
            else if (name == "NNUEWeight")
            {
                int weight = nnueWeight();
                try
                {
                    weight = std::stoi(value);
                }
                catch (const std::exception &)
                {
                }
                setNnueWeight(weight);
            }
            else if (name == "NNUEClamp")
            {
                int clampCp = nnueClamp();
                try
                {
                    clampCp = std::stoi(value);
                }
                catch (const std::exception &)
                {
                }
                setNnueClamp(clampCp);
            }
            else if (name == "EvalFile")
            {
                if (!value.empty())
                {
                    useBaseline = false;
                    loadEvalFile(value);
                }
                else if (useBaseline)
                {
                    loadEvalFile(baselineEvalFile);
                }
                else
                {
                    Nnue::clear();
                    Nnue::invalidate(pos.nnueAccumulator);
                }
            }
            else if (name == "SyzygyPath")
            {
                TB::setPath(value);
            }
            else if (name == "SyzygyProbeDepth")
            {
                int d = 1;
                try
                {
                    d = std::stoi(value);
                }
                catch (const std::exception &)
                {
                    d = TB::probeDepth();
                }
                TB::setProbeDepth(d);
            }
            else if (name == "SyzygyProbeLimit")
            {
                int p = 6;
                try
                {
                    p = std::stoi(value);
                }
                catch (const std::exception &)
                {
                    p = TB::probeLimit();
                }
                TB::setProbeLimit(p);
            }
            else if (name == "BaselineEvalFile")
            {
                if (!value.empty())
                    baselineEvalFile = value;
                if (useBaseline)
                    loadEvalFile(baselineEvalFile);
            }
            else if (name == "UseBaseline")
            {
                const bool requestedBaseline = parseBool(value, useBaseline);
                if (requestedBaseline)
                {
                    useBaseline = true;
                    loadEvalFile(baselineEvalFile);
                }
                else if (useBaseline)
                {
                    useBaseline = false;
                    Nnue::clear();
                    Nnue::invalidate(pos.nnueAccumulator);
                }
            }
        }
        else if (cmd == "usebaseline")
        {
            useBaseline = true;
            loadEvalFile(baselineEvalFile);
        }
        else if (cmd == "ucinewgame")
        {
            stopSearch();
            tt.clear();
            // A new game resets search and position-derived state, not the
            // selected evaluation model. EvalFile/UseBaseline are persistent
            // UCI options and remain in force until explicitly changed.
            Nnue::invalidate(pos.nnueAccumulator);
        }
        else if (cmd.rfind("position", 0) == 0)
        {
            std::istringstream iss(cmd);
            std::string token;
            iss >> token; // position
            iss >> token;

            if (token == "startpos")
            {
                pos.setStartPos();
            }
            else if (token == "fen")
            {
                std::string fen, part;
                for (int i = 0; i < 6; ++i)
                {
                    if (!(iss >> part))
                        break;
                    if (i)
                        fen += " ";
                    fen += part;
                }
                if (!fen.empty())
                    pos.fromFEN(fen);
            }

            while (iss >> token)
            {
                if (token == "moves")
                {
                    std::string moveStr;
                    while (iss >> moveStr)
                    {
                        Move m;
                        if (parseMoveUci(pos, moveStr, m))
                            pos.makeMove(m);
                    }
                }
            }
        }
        else if (cmd == "d")
        {
            pos.printBoard();
        }
        else if (cmd == "eval")
        {
            bool usedNnue = false;
            int score = evaluate(pos, &usedNnue);
            std::cout << "info string eval score " << score
                      << " source " << (usedNnue ? "nnue" : "fallback") << "\n";
        }
        else if (cmd.rfind("bench", 0) == 0)
        {
            stopSearch();

            int depth = 6;
            std::istringstream iss(cmd);
            std::string token;
            iss >> token; // bench
            bool showDiff = false;
            while (iss >> token)
            {
                if (token == "diff")
                {
                    showDiff = true;
                    continue;
                }
                try
                {
                    depth = std::max(1, std::stoi(token));
                }
                catch (const std::exception &)
                {
                    // Ignore unknown bench tokens for now.
                }
            }

            const std::vector<std::string> benchFens = {
                "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
                "4rrk1/2p1qppp/p1np1n2/1pb1p1B1/4P1b1/1NNP1N2/PPP1QPPP/2KR1B1R w - - 0 1",
                "r2q1rk1/pb1nbppp/1pn1p3/3pP3/3P1P2/2N1BN2/PPQ1B1PP/2RR2K1 w - - 0 1",
                "2r3k1/5ppp/p2p4/1p1P4/1P2n3/P3PN2/5PPP/2R3K1 w - - 0 1",
                "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
                "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"};

            uint64_t totalNodes = 0;
            uint64_t totalTimeMs = 0;
            std::vector<uint64_t> currNodes;
            std::vector<uint64_t> currTimes;
            currNodes.reserve(benchFens.size());
            currTimes.reserve(benchFens.size());
            stopFlag.store(false, std::memory_order_relaxed);
            tt.clear();

            std::cout << "info string bench start depth " << depth << " positions " << benchFens.size() << "\n";

            for (size_t i = 0; i < benchFens.size(); ++i)
            {
                Position benchPos;
                benchPos.fromFEN(benchFens[i]);

                SearchLimits limits;
                limits.depth = depth;
                limits.threads = searchThreads;
                limits.contempt = contemptCp;
                limits.moveOverheadMs = moveOverheadMs;
                limits.printInfo = false;

                auto t0 = std::chrono::steady_clock::now();
                SearchResult result = search(benchPos, limits, tt, stopFlag);
                auto t1 = std::chrono::steady_clock::now();
                uint64_t ms = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());

                uint64_t nodes = result.nodes + result.qnodes;
                totalNodes += nodes;
                totalTimeMs += ms;
                currNodes.push_back(nodes);
                currTimes.push_back(ms);

                uint64_t nps = (ms > 0) ? (nodes * 1000ULL / ms) : nodes;
                std::cout << "info string bench pos " << (i + 1)
                          << " nodes " << nodes
                          << " time " << ms
                          << " nps " << nps << "\n";
            }

            uint64_t totalNps = (totalTimeMs > 0) ? (totalNodes * 1000ULL / totalTimeMs) : totalNodes;
            std::cout << "info string bench total nodes " << totalNodes
                      << " time " << totalTimeMs
                      << " nps " << totalNps << "\n";

            if (showDiff)
            {
                if (!hasLastBench || lastBenchNodes.size() != currNodes.size())
                {
                    std::cout << "info string bench diff unavailable (no previous benchmark in this session)\n";
                }
                else
                {
                    auto pct = [](uint64_t oldVal, uint64_t newVal) -> double
                    {
                        if (oldVal == 0)
                            return 0.0;
                        return (100.0 * (static_cast<double>(newVal) - static_cast<double>(oldVal))) / static_cast<double>(oldVal);
                    };

                    std::cout << std::fixed << std::setprecision(2);
                    for (size_t i = 0; i < currNodes.size(); ++i)
                    {
                        double nodeDelta = pct(lastBenchNodes[i], currNodes[i]);
                        double timeDelta = pct(lastBenchTimes[i], currTimes[i]);
                        std::cout << "info string bench diff pos " << (i + 1)
                                  << " nodes " << (nodeDelta >= 0.0 ? "+" : "") << nodeDelta << "%"
                                  << " time " << (timeDelta >= 0.0 ? "+" : "") << timeDelta << "%\n";
                    }
                    double totalNodeDelta = pct(lastBenchTotalNodes, totalNodes);
                    double totalTimeDelta = pct(lastBenchTotalTimeMs, totalTimeMs);
                    std::cout << "info string bench diff total nodes "
                              << (totalNodeDelta >= 0.0 ? "+" : "") << totalNodeDelta << "%"
                              << " time " << (totalTimeDelta >= 0.0 ? "+" : "") << totalTimeDelta << "%\n";
                    std::cout.unsetf(std::ios::floatfield);
                }
            }

            lastBenchNodes = currNodes;
            lastBenchTimes = currTimes;
            lastBenchTotalNodes = totalNodes;
            lastBenchTotalTimeMs = totalTimeMs;
            hasLastBench = true;
            std::cout << "bestmove 0000\n";
        }
        else if (cmd.rfind("selfplay", 0) == 0)
        {
            stopSearch();

            int games = 1;
            int movetimeMs = 30;
            int maxPly = 300;
            int depth = 0;
            int nodes = 0;
            int randomPlies = 0;
            int sampleEvery = 1;
            int minSamplePly = 0;
            uint64_t seed = 1;
            std::string outFile;

            std::istringstream iss(cmd);
            std::string token;
            iss >> token; // selfplay
            while (iss >> token)
            {
                if (token == "games")
                    iss >> games;
                else if (token == "movetime")
                    iss >> movetimeMs;
                else if (token == "maxply")
                    iss >> maxPly;
                else if (token == "depth")
                    iss >> depth;
                else if (token == "nodes")
                    iss >> nodes;
                else if (token == "randomplies")
                    iss >> randomPlies;
                else if (token == "sampleevery")
                    iss >> sampleEvery;
                else if (token == "minsampleply")
                    iss >> minSamplePly;
                else if (token == "seed")
                    iss >> seed;
                else if (token == "outfile")
                    iss >> outFile;
            }

            games = std::max(1, games);
            movetimeMs = std::max(1, movetimeMs);
            maxPly = std::max(20, maxPly);
            randomPlies = std::max(0, randomPlies);
            sampleEvery = std::max(1, sampleEvery);
            minSamplePly = std::max(0, minSamplePly);

            if (outFile.empty())
            {
                std::cout << "info string selfplay error missing outfile\n";
                std::cout << "bestmove 0000\n";
                continue;
            }

            DataWriter writer;
            if (!writer.open(outFile))
            {
                std::cout << "info string selfplay error could not open outfile " << outFile << "\n";
                std::cout << "bestmove 0000\n";
                continue;
            }

            int whiteWins = 0, blackWins = 0, draws = 0;
            uint64_t writtenSamples = 0;
            stopFlag.store(false, std::memory_order_relaxed);
            std::mt19937_64 rng(seed);

            for (int g = 1; g <= games; ++g)
            {
                Position gamePos;
                gamePos.setStartPos();
                tt.clear();

                std::unordered_map<uint64_t, int> reps;
                reps.reserve(1024);
                reps[gamePos.hashKey] = 1;

                std::vector<SelfplaySample> samples;
                samples.reserve(maxPly);

                int ply = 0;
                int whiteResult = 0;

                // Optional randomized opening plies for better data diversity.
                for (int rp = 0; rp < randomPlies; ++rp)
                {
                    if (isGameTerminal(gamePos, reps, ply, maxPly, whiteResult))
                        break;
                    MoveList legal;
                    generateLegalMoves(gamePos, legal);
                    if (legal.moves.empty())
                        break;
                    std::uniform_int_distribution<size_t> dist(0, legal.moves.size() - 1);
                    Move m = legal.moves[dist(rng)];
                    gamePos.makeMove(m);
                    reps[gamePos.hashKey]++;
                    ply++;
                }

                while (true)
                {
                    if (isGameTerminal(gamePos, reps, ply, maxPly, whiteResult))
                        break;

                    SearchLimits limits;
                    limits.movetimeMs = movetimeMs;
                    limits.depth = depth;
                    limits.nodes = nodes;
                    limits.printInfo = false;
                    limits.threads = searchThreads;
                    limits.contempt = contemptCp;
                    limits.moveOverheadMs = moveOverheadMs;

                    SearchResult sr = search(gamePos, limits, tt, stopFlag);
                    if (ply >= minSamplePly && (ply % sampleEvery) == 0)
                    {
                        samples.push_back({gamePos.toFEN(), gamePos.sideToMove, sr.score});
                    }

                    MoveList legal;
                    generateLegalMoves(gamePos, legal);
                    if (legal.moves.empty())
                    {
                        whiteResult = 0;
                        break;
                    }

                    Move best = sr.bestMove;
                    bool found = false;
                    for (const auto &m : legal.moves)
                    {
                        if (m.from == best.from && m.to == best.to && m.isPromotion == best.isPromotion &&
                            m.promoPiece == best.promoPiece)
                        {
                            found = true;
                            break;
                        }
                    }
                    if (!found)
                        best = legal.moves.front();

                    gamePos.makeMove(best);
                    reps[gamePos.hashKey]++;
                    ply++;
                }

                if (whiteResult > 0)
                    whiteWins++;
                else if (whiteResult < 0)
                    blackWins++;
                else
                    draws++;

                for (const auto &s : samples)
                {
                    int stmResult = (s.sideToMove == WHITE) ? whiteResult : -whiteResult;
                    if (writer.appendSample(s.fen, stmResult, s.scoreCp))
                        writtenSamples++;
                }

                std::cout << "info string selfplay game " << g << "/" << games
                          << " result " << (whiteResult > 0 ? "1-0" : (whiteResult < 0 ? "0-1" : "1/2-1/2"))
                          << " plies " << ply
                          << " samples " << samples.size() << "\n";
            }

            writer.close();
            std::cout << "info string selfplay done games " << games
                      << " whitewins " << whiteWins
                      << " blackwins " << blackWins
                      << " draws " << draws
                      << " sampleevery " << sampleEvery
                      << " minsampleply " << minSamplePly
                      << " written " << writtenSamples
                      << " outfile " << outFile << "\n";
            std::cout << "bestmove 0000\n";
        }
        else if (cmd.rfind("probe", 0) == 0)
        {
            stopSearch();
            int depth = 8;
            int movetimeMs = 0;

            std::istringstream iss(cmd);
            std::string token;
            iss >> token; // probe
            while (iss >> token)
            {
                if (token == "depth")
                    iss >> depth;
                else if (token == "movetime")
                    iss >> movetimeMs;
            }
            depth = std::max(1, depth);

            SearchLimits limits;
            limits.depth = depth;
            limits.movetimeMs = std::max(0, movetimeMs);
            limits.threads = searchThreads;
            limits.contempt = contemptCp;
            limits.moveOverheadMs = moveOverheadMs;
            limits.printInfo = false;

            stopFlag.store(false, std::memory_order_relaxed);
            Position probePos = pos;
            auto t0 = std::chrono::steady_clock::now();
            SearchResult sr = search(probePos, limits, tt, stopFlag);
            auto t1 = std::chrono::steady_clock::now();
            uint64_t ms = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());
            uint64_t nodes = sr.nodes + sr.qnodes;
            uint64_t nps = (ms > 0) ? (nodes * 1000ULL / ms) : nodes;
            std::string best = (sr.bestMove.from == 0 && sr.bestMove.to == 0 && !sr.bestMove.isPromotion &&
                                !sr.bestMove.isCapture && !sr.bestMove.isCastle)
                                   ? "0000"
                                   : moveToUci(sr.bestMove);
            std::cout << "info string probe depth " << sr.depth
                      << " scorecp " << sr.score
                      << " nodes " << nodes
                      << " time " << ms
                      << " nps " << nps
                      << " best " << best << "\n";
            std::cout << "bestmove " << best << "\n";
        }
        else if (cmd.rfind("go", 0) == 0)
        {
            stopSearch();
            SearchLimits limits;
            std::istringstream iss(cmd);
            std::string token;
            iss >> token; // go
            while (iss >> token)
            {
                if (token == "wtime")
                    iss >> limits.wtimeMs;
                else if (token == "btime")
                    iss >> limits.btimeMs;
                else if (token == "winc")
                    iss >> limits.wincMs;
                else if (token == "binc")
                    iss >> limits.bincMs;
                else if (token == "movetime")
                    iss >> limits.movetimeMs;
                else if (token == "depth")
                    iss >> limits.depth;
                else if (token == "nodes")
                    iss >> limits.nodes;
                else if (token == "movestogo")
                    iss >> limits.movesToGo;
                else if (token == "infinite")
                    limits.infinite = true;
            }

            limits.moveOverheadMs = moveOverheadMs;

            if (limits.movetimeMs == 0 && !limits.infinite && limits.depth == 0 && limits.nodes == 0)
            {
                int remain = (pos.sideToMove == WHITE) ? limits.wtimeMs : limits.btimeMs;
                int inc = (pos.sideToMove == WHITE) ? limits.wincMs : limits.bincMs;
                if (remain > 0)
                {
                    int safeRemain = std::max(1, remain - limits.moveOverheadMs);
                    int mtg = (limits.movesToGo > 0) ? limits.movesToGo : 30;
                    int target = safeRemain / std::max(10, mtg) + (inc * 3) / 4;
                    int minSpend = std::max(5, safeRemain / 80);
                    int maxSpend = std::max(20, safeRemain / 2);
                    limits.movetimeMs = std::clamp(target, minSpend, maxSpend);
                }
            }

            stopFlag.store(false, std::memory_order_relaxed);
            searching.store(true, std::memory_order_relaxed);
            Position searchPos = pos;
            SearchLimits actualLimits = limits;
            actualLimits.threads = searchThreads;
            actualLimits.contempt = contemptCp;
            actualLimits.usePruning = usePruning;
            actualLimits.moveOverheadMs = moveOverheadMs;
            searchThread = std::thread([searchPos, actualLimits, &tt, &stopFlag, &searching]() mutable
                                       {
                                           SearchResult result = search(searchPos, actualLimits, tt, stopFlag);
                                           std::string best;
                                           if (!(result.bestMove.from == 0 && result.bestMove.to == 0 && !result.bestMove.isPromotion &&
                                                 !result.bestMove.isCapture && !result.bestMove.isCastle))
                                           {
                                               best = moveToUci(result.bestMove);
                                           }
                                           if (best.empty())
                                               best = "0000";
                                           std::cout << "bestmove " << best << "\n";
                                           searching.store(false, std::memory_order_relaxed);
                                       });
        }
        else if (cmd == "stop")
        {
            stopSearch();
        }
        else if (cmd == "quit")
        {
            stopSearch();
            break;
        }
    }
}
