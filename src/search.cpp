#include "search.hpp"
#include "eval.hpp"
#include "zobrist.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>

namespace
{
    constexpr int MATE_SCORE = 30000;
    constexpr int INF_SCORE = 32000;
    constexpr int MAX_PLY = 64;

    constexpr int pieceValuesAbs[12] = {
        100, 320, 330, 500, 900, 0,
        100, 320, 330, 500, 900, 0};

    bool sameMove(const Move &a, const Move &b)
    {
        return a.from == b.from && a.to == b.to && a.isPromotion == b.isPromotion &&
               a.promoPiece == b.promoPiece && a.isCastle == b.isCastle;
    }

    int scoreToTT(int score, int ply)
    {
        if (score > MATE_SCORE - 1000)
            return score + ply;
        if (score < -MATE_SCORE + 1000)
            return score - ply;
        return score;
    }

    int scoreFromTT(int score, int ply)
    {
        if (score > MATE_SCORE - 1000)
            return score - ply;
        if (score < -MATE_SCORE + 1000)
            return score + ply;
        return score;
    }
}

void TranspositionTable::resizeMB(int mb)
{
    if (mb <= 0)
        mb = 1;
    size_t bytes = static_cast<size_t>(mb) * 1024 * 1024;
    size_t count = std::max<size_t>(1, bytes / sizeof(TTEntry));
    size_t pow2 = 1;
    while (pow2 < count)
        pow2 <<= 1;
    table.clear();
    table.resize(pow2);
    mask = pow2 - 1;
}

void TranspositionTable::clear()
{
    std::fill(table.begin(), table.end(), TTEntry{});
}

bool TranspositionTable::probe(uint64_t key, TTEntry &out) const
{
    if (table.empty())
        return false;
    const TTEntry &e = table[key & mask];
    if (e.key == key)
    {
        out = e;
        return true;
    }
    return false;
}

void TranspositionTable::store(uint64_t key, int depth, int score, uint8_t bound, const Move &bestMove)
{
    if (table.empty())
        return;
    TTEntry &e = table[key & mask];
    if (e.key == key && e.depth > depth && bound != 0)
        return;
    e.key = key;
    e.depth = depth;
    e.score = score;
    e.bound = bound;
    e.bestMove = bestMove;
}

struct SearchState
{
    std::atomic<bool> *stopFlag = nullptr;
    TranspositionTable *tt = nullptr;
    uint64_t nodes = 0;
    uint64_t qnodes = 0;
    std::chrono::steady_clock::time_point start;
    int hardTimeLimitMs = 0;
    int softTimeLimitMs = 0;
    int maxDepth = 0;
    int nodeLimit = 0;
    int killerMoves[MAX_PLY][2]{};
    int history[12][64]{};
    std::vector<uint64_t> repHistory;
};

struct NullUndo
{
    Side prevSide = WHITE;
    int prevEnPassant = -1;
    uint64_t prevHash = 0;
    int prevHalfmoveClock = 0;
    int prevFullmoveNumber = 1;
    Nnue::Accumulator prevAccumulator{};
};

uint64_t totalNodes(const SearchState &st)
{
    return st.nodes + st.qnodes;
}

int elapsedMs(const SearchState &st)
{
    auto now = std::chrono::steady_clock::now();
    return static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(now - st.start).count());
}

void decayHistory(SearchState &st)
{
    for (int p = 0; p < 12; ++p)
    {
        for (int sq = 0; sq < 64; ++sq)
        {
            st.history[p][sq] /= 2;
        }
    }
}

bool hasNonPawnMaterial(const Position &pos, Side side)
{
    if (side == WHITE)
    {
        return (pos.pieceBB[WN] | pos.pieceBB[WB] | pos.pieceBB[WR] | pos.pieceBB[WQ]) != 0ULL;
    }
    return (pos.pieceBB[BN] | pos.pieceBB[BB] | pos.pieceBB[BR] | pos.pieceBB[BQ]) != 0ULL;
}

void doNullMove(Position &pos, NullUndo &u)
{
    u.prevSide = pos.sideToMove;
    u.prevEnPassant = pos.enPassantSquare;
    u.prevHash = pos.hashKey;
    u.prevHalfmoveClock = pos.halfmoveClock;
    u.prevFullmoveNumber = pos.fullmoveNumber;
    u.prevAccumulator = pos.nnueAccumulator;

    if (pos.enPassantSquare != -1)
    {
        pos.hashKey ^= Zobrist::epFile(pos.enPassantSquare & 7);
        pos.enPassantSquare = -1;
    }

    if (pos.sideToMove == BLACK)
        pos.fullmoveNumber++;
    pos.halfmoveClock++;

    pos.sideToMove = (pos.sideToMove == WHITE ? BLACK : WHITE);
    pos.hashKey ^= Zobrist::side();
    if (pos.nnueAccumulator.initialized)
        pos.nnueAccumulator.positionKey = pos.hashKey;
}

void undoNullMove(Position &pos, const NullUndo &u)
{
    pos.sideToMove = u.prevSide;
    pos.enPassantSquare = u.prevEnPassant;
    pos.hashKey = u.prevHash;
    pos.halfmoveClock = u.prevHalfmoveClock;
    pos.fullmoveNumber = u.prevFullmoveNumber;
    pos.nnueAccumulator = u.prevAccumulator;
}

bool castlePathSafe(const Position &pos, const Move &m, Side us, Side them)
{
    if (!m.isCastle)
        return true;

    int startK = pos.kingSquare[us];
    if (startK == -1)
    {
        Bitboard kbb = pos.pieceBB[(us == WHITE) ? WK : BK];
        startK = kbb ? lsb(kbb) : -1;
    }
    if (startK == -1 || pos.isSquareAttacked(startK, them))
        return false;

    int f = -1, g = -1;
    if (us == WHITE && m.to == 6)
    {
        f = 5;
        g = 6;
    }
    else if (us == WHITE && m.to == 2)
    {
        f = 3;
        g = 2;
    }
    else if (us == BLACK && m.to == 62)
    {
        f = 61;
        g = 62;
    }
    else if (us == BLACK && m.to == 58)
    {
        f = 59;
        g = 58;
    }
    if (f == -1)
        return false;
    return !pos.isSquareAttacked(f, them) && !pos.isSquareAttacked(g, them);
}

int scoreMove(const Position &pos, const Move &m, const Move &ttMove, int ply, const SearchState &st)
{
    if (sameMove(m, ttMove))
        return 1000000;

    if (m.isCapture)
    {
        int captured = pos.pieceIndexAt(m.to);
        int attacker = pos.pieceIndexAt(m.from);
        int victimVal = (captured >= 0) ? pieceValuesAbs[captured] : 100;
        int attackerVal = (attacker >= 0) ? pieceValuesAbs[attacker] : 100;
        return 100000 + (victimVal * 10 - attackerVal);
    }

    if (m.isPromotion)
        return 90000 + m.promoPiece * 10;

    int killer1 = st.killerMoves[ply][0];
    int killer2 = st.killerMoves[ply][1];
    int moveId = (m.from << 6) | m.to;
    if (moveId == killer1)
        return 80000;
    if (moveId == killer2)
        return 70000;

    int attacker = pos.pieceIndexAt(m.from);
    if (attacker >= 0)
        return st.history[attacker][m.to];

    return 0;
}

bool shouldStop(SearchState &st)
{
    if (st.stopFlag && st.stopFlag->load(std::memory_order_relaxed))
        return true;
    if (st.nodeLimit > 0 && (int)totalNodes(st) >= st.nodeLimit)
        return true;
    if (st.hardTimeLimitMs > 0 && (totalNodes(st) & 4095) == 0)
    {
        if (elapsedMs(st) >= st.hardTimeLimitMs)
        {
            if (st.stopFlag)
                st.stopFlag->store(true, std::memory_order_relaxed);
            return true;
        }
    }
    return false;
}

int quiescence(Position &pos, SearchState &st, int alpha, int beta, int ply)
{
    if (shouldStop(st))
        return 0;

    if (ply >= MAX_PLY - 1)
        return evaluate(pos);

    // 50-move rule draw.
    if (pos.halfmoveClock >= 100)
        return 0;

    st.qnodes++;

    Side us = pos.sideToMove;
    Side them = (us == WHITE ? BLACK : WHITE);
    int kingSq = pos.kingSquare[us];
    bool inCheck = (kingSq != -1) && pos.isSquareAttacked(kingSq, them);

    int stand = evaluate(pos);
    if (!inCheck)
    {
        if (stand >= beta)
            return beta;
        if (stand > alpha)
            alpha = stand;
    }

    MoveList moves;
    generateAllMoves(pos, moves);
    std::vector<Move> moveVec = moves.moves;
    Move emptyMove{};
    std::sort(moveVec.begin(), moveVec.end(), [&](const Move &a, const Move &b)
              { return scoreMove(pos, a, emptyMove, ply, st) > scoreMove(pos, b, emptyMove, ply, st); });

    int legalMoves = 0;
    for (const auto &m : moveVec)
    {
        if (!inCheck && !(m.isCapture || m.isPromotion))
            continue;

        if (!inCheck && m.isCapture)
        {
            int captured = pos.pieceIndexAt(m.to);
            int gain = (captured >= 0) ? pieceValuesAbs[captured] : 0;
            if (stand + gain + 120 < alpha)
                continue;
        }

        if (!castlePathSafe(pos, m, us, them))
            continue;

        Undo u;
        pos.makeMove(m, u);
        if (pos.isSquareAttacked(pos.kingSquare[us], them))
        {
            pos.unmakeMove(m, u);
            continue;
        }
        legalMoves++;
        st.repHistory.push_back(pos.hashKey);
        int score = -quiescence(pos, st, -beta, -alpha, ply + 1);
        st.repHistory.pop_back();
        pos.unmakeMove(m, u);

        if (score >= beta)
            return beta;
        if (score > alpha)
            alpha = score;
    }

    if (inCheck && legalMoves == 0)
        return -MATE_SCORE + ply;

    return alpha;
}

int negamax(Position &pos, SearchState &st, int depth, int alpha, int beta, int ply)
{
    if (shouldStop(st))
        return 0;

    if (ply >= MAX_PLY - 1)
        return evaluate(pos);

    // 50-move rule draw.
    if (pos.halfmoveClock >= 100)
        return 0;

    st.nodes++;
    int alphaOrig = alpha;

    Side us = pos.sideToMove;
    Side them = (us == WHITE ? BLACK : WHITE);
    int kingSq = pos.kingSquare[us];
    bool inCheck = (kingSq != -1) && pos.isSquareAttacked(kingSq, them);
    bool pvNode = (beta - alpha) > 1;

    // Mate distance pruning.
    alpha = std::max(alpha, -MATE_SCORE + ply);
    beta = std::min(beta, MATE_SCORE - ply - 1);
    if (alpha >= beta)
        return alpha;

    // Check extension.
    if (inCheck && depth > 0 && ply < MAX_PLY - 1)
        depth++;

    if (depth == 0)
        return quiescence(pos, st, alpha, beta, ply);

    if (!st.repHistory.empty())
    {
        uint64_t key = pos.hashKey;
        for (size_t i = 0; i + 1 < st.repHistory.size(); ++i)
        {
            if (st.repHistory[i] == key)
                return 0;
        }
    }

    TTEntry tte;
    Move ttMove{};
    if (st.tt && st.tt->probe(pos.hashKey, tte))
    {
        ttMove = tte.bestMove;
        if (tte.depth >= depth)
        {
            int ttScore = scoreFromTT(tte.score, ply);
            if (tte.bound == 1 && ttScore <= alpha)
                return ttScore;
            if (tte.bound == 2 && ttScore >= beta)
                return ttScore;
            if (tte.bound == 3)
                return ttScore;
        }
    }

    int staticEval = INF_SCORE;
    bool haveStaticEval = false;

    // Reverse futility pruning for shallow, non-check, non-PV nodes.
    if (depth <= 3 && !pvNode && !inCheck)
    {
        staticEval = evaluate(pos);
        haveStaticEval = true;
        int margin = 120 * depth + 60;
        if (staticEval - margin >= beta)
            return staticEval;
    }

    // Null move pruning: try passing the move to prove a beta cutoff quickly.
    if (depth >= 3 && !inCheck && ply > 0 && hasNonPawnMaterial(pos, us))
    {
        if (!haveStaticEval)
        {
            staticEval = evaluate(pos);
            haveStaticEval = true;
        }
        if (staticEval >= beta)
        {
            int reduction = 2 + depth / 4;
            NullUndo nu;
            doNullMove(pos, nu);
            st.repHistory.push_back(pos.hashKey);
            int score = -negamax(pos, st, std::max(0, depth - 1 - reduction), -beta, -beta + 1, ply + 1);
            st.repHistory.pop_back();
            undoNullMove(pos, nu);

            if (shouldStop(st))
                return 0;
            if (score >= beta)
            {
                if (st.tt)
                    st.tt->store(pos.hashKey, depth, scoreToTT(beta, ply), 2, Move{});
                return beta;
            }
        }
    }

    MoveList moves;
    generateAllMoves(pos, moves);
    std::vector<Move> moveVec = moves.moves;
    std::sort(moveVec.begin(), moveVec.end(), [&](const Move &a, const Move &b)
              { return scoreMove(pos, a, ttMove, ply, st) > scoreMove(pos, b, ttMove, ply, st); });

    int bestScore = -INF_SCORE;
    Move bestMove{};
    int legalMoves = 0;
    bool searchedPvMove = false;

    for (const auto &m : moveVec)
    {
        if (!castlePathSafe(pos, m, us, them))
            continue;

        Undo u;
        pos.makeMove(m, u);
        if (pos.isSquareAttacked(pos.kingSquare[us], them))
        {
            pos.unmakeMove(m, u);
            continue;
        }

        legalMoves++;
        bool givesCheck = false;
        int oppKingSq = pos.kingSquare[them];
        if (oppKingSq != -1)
            givesCheck = pos.isSquareAttacked(oppKingSq, us);

        // Late move pruning (LMP): skip very late quiets at low depth.
        if (!pvNode && !inCheck && !givesCheck && depth <= 2 && !m.isCapture && !m.isPromotion && !m.isCastle)
        {
            int lmpThreshold = 14 + 6 * depth; // depth1:20 depth2:26
            if (legalMoves > lmpThreshold)
            {
                pos.unmakeMove(m, u);
                continue;
            }
        }

        st.repHistory.push_back(pos.hashKey);
        int score = 0;
        int nextDepth = depth - 1;

        bool lmrCandidate = !pvNode && !inCheck && !givesCheck && !m.isCapture && !m.isPromotion && !m.isCastle &&
                            depth >= 3 && legalMoves >= 4;
        if (lmrCandidate)
        {
            int reduction = 1;
            if (depth >= 8)
                reduction++;
            int reducedDepth = std::max(0, nextDepth - reduction);

            score = -negamax(pos, st, reducedDepth, -alpha - 1, -alpha, ply + 1);
            if (score > alpha)
            {
                score = -negamax(pos, st, nextDepth, -alpha - 1, -alpha, ply + 1);
                if (score > alpha && score < beta)
                {
                    score = -negamax(pos, st, nextDepth, -beta, -alpha, ply + 1);
                }
            }
        }
        else
        {
            if (!searchedPvMove)
            {
                score = -negamax(pos, st, nextDepth, -beta, -alpha, ply + 1);
            }
            else
            {
                score = -negamax(pos, st, nextDepth, -alpha - 1, -alpha, ply + 1);
                if (score > alpha && score < beta)
                {
                    score = -negamax(pos, st, nextDepth, -beta, -alpha, ply + 1);
                }
            }
        }
        st.repHistory.pop_back();
        pos.unmakeMove(m, u);
        searchedPvMove = true;

        if (shouldStop(st))
            return 0;

        if (score > bestScore)
        {
            bestScore = score;
            bestMove = m;
        }
        if (score > alpha)
        {
            alpha = score;
        }
        if (alpha >= beta)
        {
            if (!m.isCapture)
            {
                int moveId = (m.from << 6) | m.to;
                st.killerMoves[ply][1] = st.killerMoves[ply][0];
                st.killerMoves[ply][0] = moveId;
                int attacker = pos.pieceIndexAt(m.from);
                if (attacker >= 0)
                {
                    st.history[attacker][m.to] += depth * depth;
                    if (st.history[attacker][m.to] > 32767)
                        st.history[attacker][m.to] = 32767;
                }
            }
            if (st.tt)
                st.tt->store(pos.hashKey, depth, scoreToTT(beta, ply), 2, m);
            return beta;
        }
    }

    if (legalMoves == 0)
    {
        if (inCheck)
            return -MATE_SCORE + ply;
        return 0;
    }

    uint8_t bound = 3;
    if (bestScore <= alphaOrig)
        bound = 1;
    if (st.tt)
        st.tt->store(pos.hashKey, depth, scoreToTT(bestScore, ply), bound, bestMove);

    return bestScore;
}

SearchResult search(Position &pos, const SearchLimits &limits, TranspositionTable &tt, std::atomic<bool> &stopFlag)
{
    SearchState st;
    st.stopFlag = &stopFlag;
    st.tt = &tt;
    st.nodes = 0;
    st.qnodes = 0;
    st.start = std::chrono::steady_clock::now();
    st.hardTimeLimitMs = limits.movetimeMs;
    st.softTimeLimitMs = (limits.movetimeMs > 0) ? (limits.movetimeMs * 7) / 10 : 0;
    st.maxDepth = limits.depth;
    st.nodeLimit = limits.nodes;
    st.repHistory.clear();
    st.repHistory.push_back(pos.hashKey);

    SearchResult result;
    result.bestMove = Move{};
    result.score = 0;
    result.depth = 0;
    int lastScore = 0;

    int maxDepth = (limits.depth > 0) ? limits.depth : 64;
    for (int d = 1; d <= maxDepth; ++d)
    {
        if (shouldStop(st))
            break;

        int score = 0;
        int alpha = -INF_SCORE;
        int beta = INF_SCORE;
        int window = 25;
        int reSearches = 0;

        if (d >= 4)
        {
            alpha = std::max(-INF_SCORE, lastScore - window);
            beta = std::min(INF_SCORE, lastScore + window);
        }

        while (true)
        {
            score = negamax(pos, st, d, alpha, beta, 0);
            if (shouldStop(st))
                break;

            if (score <= alpha)
            {
                reSearches++;
                window = std::min(2000, window * 2);
                alpha = std::max(-INF_SCORE, score - window);
                beta = std::min(INF_SCORE, score + window);
            }
            else if (score >= beta)
            {
                reSearches++;
                window = std::min(2000, window * 2);
                alpha = std::max(-INF_SCORE, score - window);
                beta = std::min(INF_SCORE, score + window);
            }
            else
            {
                break;
            }

            if (reSearches >= 4)
            {
                alpha = -INF_SCORE;
                beta = INF_SCORE;
            }
        }

        if (shouldStop(st))
            break;

        result.score = score;
        result.depth = d;
        result.nodes = st.nodes;
        result.qnodes = st.qnodes;
        lastScore = score;

        if ((d % 4) == 0)
            decayHistory(st);

        TTEntry tte;
        if (tt.probe(pos.hashKey, tte))
            result.bestMove = tte.bestMove;

        int elapsed = elapsedMs(st);
        uint64_t allNodes = totalNodes(st);
        int nps = (elapsed > 0) ? (int)(allNodes * 1000 / elapsed) : (int)allNodes;
        if (limits.printInfo)
        {
            std::cout << "info depth " << d << " score cp " << score
                      << " nodes " << allNodes << " nps " << nps
                      << " time " << elapsed << " string qnodes " << st.qnodes << "\n";
        }

        if (st.softTimeLimitMs > 0 && elapsed >= st.softTimeLimitMs)
            break;
    }

    result.nodes = st.nodes;
    result.qnodes = st.qnodes;

    // Safety fallback: never return null move when legal moves exist.
    if (result.bestMove.from == 0 && result.bestMove.to == 0 &&
        !result.bestMove.isPromotion && !result.bestMove.isCapture && !result.bestMove.isCastle)
    {
        MoveList legal;
        generateLegalMoves(pos, legal);
        if (!legal.moves.empty())
            result.bestMove = legal.moves.front();
    }

    return result;
}
