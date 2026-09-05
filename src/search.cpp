#include "search.hpp"
#include "eval.hpp"
#include "tb.hpp"
#include "zobrist.hpp"
#include <algorithm>
#include <array>
#include <climits>
#include <chrono>
#include <iostream>
#include <thread>

namespace
{
    constexpr int MATE_SCORE = 30000;
    constexpr int INF_SCORE = 32000;
    constexpr int MAX_PLY = 64;
    constexpr int MAX_HISTORY = 32767;

    constexpr int pieceValuesAbs[12] = {
        100, 320, 330, 500, 900, 0,
        100, 320, 330, 500, 900, 0};
    constexpr int pieceValuesSee[12] = {
        100, 320, 330, 500, 900, 20000,
        100, 320, 330, 500, 900, 20000};

    bool sameMove(const Move &a, const Move &b)
    {
        return a.from == b.from && a.to == b.to && a.isPromotion == b.isPromotion &&
               a.promoPiece == b.promoPiece && a.isCastle == b.isCastle;
    }

    bool isNullMove(const Move &m)
    {
        return m.from == 0 && m.to == 0 && !m.isPromotion && !m.isCapture && !m.isCastle;
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

    void updateHistory(int &slot, int delta)
    {
        slot += delta;
        if (slot > MAX_HISTORY)
            slot = MAX_HISTORY;
        if (slot < -MAX_HISTORY)
            slot = -MAX_HISTORY;
    }

    int capturedPieceAt(const Position &pos, const Move &m, Side us)
    {
        int captured = pos.pieceIndexAt(m.to);
        if (captured >= 0)
            return captured;
        if (m.isCapture && (pos.pieceIndexAt(m.from) == WP || pos.pieceIndexAt(m.from) == BP) && m.to == pos.enPassantSquare)
            return (us == WHITE) ? BP : WP;
        return -1;
    }

    int moveId(const Move &m)
    {
        return (m.from << 6) | m.to;
    }

    Bitboard pawnAttackersTo(int sq, Side side, Bitboard pawns)
    {
        int file = sq & 7;
        Bitboard attackers = 0ULL;
        if (side == WHITE)
        {
            if (file < 7 && sq >= 9)
                attackers |= (ONE << (sq - 9));
            if (file > 0 && sq >= 7)
                attackers |= (ONE << (sq - 7));
        }
        else
        {
            if (file < 7 && sq <= 56)
                attackers |= (ONE << (sq + 7));
            if (file > 0 && sq <= 54)
                attackers |= (ONE << (sq + 9));
        }
        return attackers & pawns;
    }

    int promotedPieceIndex(Side side, uint8_t promo)
    {
        if (promo == PROMO_QUEEN)
            return (side == WHITE) ? WQ : BQ;
        if (promo == PROMO_ROOK)
            return (side == WHITE) ? WR : BR;
        if (promo == PROMO_BISHOP)
            return (side == WHITE) ? WB : BB;
        if (promo == PROMO_KNIGHT)
            return (side == WHITE) ? WN : BN;
        return -1;
    }

    bool tryTablebaseProbe(Position &pos, int depth, int ply, int &scoreOut)
    {
        if (!TB::isEnabled())
            return false;
        if (depth < TB::probeDepth())
            return false;
        int tbScore = 0;
        if (!TB::probeWDL(pos, tbScore))
            return false;
        if (tbScore > 0)
            tbScore = std::min(tbScore, MATE_SCORE - ply - 1);
        else if (tbScore < 0)
            tbScore = std::max(tbScore, -MATE_SCORE + ply + 1);
        scoreOut = tbScore;
        return true;
    }

    Bitboard attackersToSq(
        int sq,
        Side side,
        Bitboard occ,
        const std::array<Bitboard, 12> &pieceBB)
    {
        if (side == WHITE)
        {
            Bitboard attackers = 0ULL;
            attackers |= pawnAttackersTo(sq, WHITE, pieceBB[WP]);
            attackers |= KNIGHT_ATTACKS[sq] & pieceBB[WN];
            attackers |= bishopAttacks(sq, occ) & (pieceBB[WB] | pieceBB[WQ]);
            attackers |= rookAttacks(sq, occ) & (pieceBB[WR] | pieceBB[WQ]);
            attackers |= KING_ATTACKS[sq] & pieceBB[WK];
            return attackers;
        }
        Bitboard attackers = 0ULL;
        attackers |= pawnAttackersTo(sq, BLACK, pieceBB[BP]);
        attackers |= KNIGHT_ATTACKS[sq] & pieceBB[BN];
        attackers |= bishopAttacks(sq, occ) & (pieceBB[BB] | pieceBB[BQ]);
        attackers |= rookAttacks(sq, occ) & (pieceBB[BR] | pieceBB[BQ]);
        attackers |= KING_ATTACKS[sq] & pieceBB[BK];
        return attackers;
    }

    bool leastValuableAttacker(
        int sq,
        Side side,
        Bitboard occ,
        const std::array<Bitboard, 12> &pieceBB,
        int &pieceOut,
        int &fromOut)
    {
        static constexpr int whiteOrder[6] = {WP, WN, WB, WR, WQ, WK};
        static constexpr int blackOrder[6] = {BP, BN, BB, BR, BQ, BK};
        const int *order = (side == WHITE) ? whiteOrder : blackOrder;
        Bitboard allAttackers = attackersToSq(sq, side, occ, pieceBB);
        if (!allAttackers)
            return false;
        for (int i = 0; i < 6; ++i)
        {
            int p = order[i];
            Bitboard bb = allAttackers & pieceBB[p];
            if (bb)
            {
                pieceOut = p;
                fromOut = lsb(bb);
                return true;
            }
        }
        return false;
    }

    int staticExchangeEval(const Position &pos, const Move &m)
    {
        if (!m.isCapture && !m.isPromotion)
            return 0;

        Side us = pos.sideToMove;
        Side them = (us == WHITE ? BLACK : WHITE);
        int from = m.from;
        int to = m.to;
        if (from < 0 || from >= 64 || to < 0 || to >= 64)
            return 0;

        int movedPiece = pos.pieceIndexAt(from);
        if (movedPiece < 0)
            return 0;

        int capturedSq = to;
        int capturedPiece = pos.pieceIndexAt(to);
        if (m.isCapture && capturedPiece < 0)
        {
            if ((movedPiece == WP || movedPiece == BP) && to == pos.enPassantSquare)
            {
                capturedSq = to + ((us == WHITE) ? -8 : 8);
                capturedPiece = (us == WHITE) ? BP : WP;
            }
        }
        if (capturedPiece < 0)
            return 0;

        std::array<Bitboard, 12> bb = pos.pieceBB;
        Bitboard occ = pos.allPieces;
        int gain[32]{};
        int d = 0;
        gain[0] = pieceValuesSee[capturedPiece];

        int placedPiece = movedPiece;
        if (m.isPromotion)
        {
            int promoPiece = promotedPieceIndex(us, m.promoPiece);
            if (promoPiece >= 0)
                placedPiece = promoPiece;
        }
        gain[0] += pieceValuesSee[placedPiece] - pieceValuesSee[movedPiece];

        clear_bit(bb[movedPiece], from);
        clear_bit(occ, from);
        clear_bit(bb[capturedPiece], capturedSq);
        clear_bit(occ, capturedSq);
        set_bit(bb[placedPiece], to);
        set_bit(occ, to);

        Side stm = them;
        int currentPieceOnTo = placedPiece;

        while (true)
        {
            int attackerPiece = -1;
            int attackerSq = -1;
            if (!leastValuableAttacker(to, stm, occ, bb, attackerPiece, attackerSq))
                break;
            ++d;
            if (d >= 31)
                break;
            gain[d] = pieceValuesSee[currentPieceOnTo] - gain[d - 1];

            clear_bit(bb[attackerPiece], attackerSq);
            clear_bit(occ, attackerSq);
            clear_bit(bb[currentPieceOnTo], to);
            set_bit(bb[attackerPiece], to);
            set_bit(occ, to);

            currentPieceOnTo = attackerPiece;
            stm = (stm == WHITE ? BLACK : WHITE);
        }

        while (--d >= 0)
            gain[d] = -std::max(-gain[d], gain[d + 1]);

        return gain[0];
    }

}

void TranspositionTable::resizeMB(int mb)
{
    if (mb <= 0)
        mb = 1;
    size_t bytes = static_cast<size_t>(mb) * 1024 * 1024;
    size_t count = std::max<size_t>(1, bytes / sizeof(TTBucket));
    size_t pow2 = 1;
    while (pow2 <= count / 2)
        pow2 <<= 1;
    std::vector<TTBucket> replacement(pow2);
    table.swap(replacement);
    mask = pow2 - 1;
    generation.store(1, std::memory_order_relaxed);
}

void TranspositionTable::clear()
{
    for (auto &bucket : table)
    {
        for (auto &entry : bucket.entries)
        {
            entry.payload.store(0, std::memory_order_relaxed);
            entry.verification.store(0, std::memory_order_relaxed);
        }
    }
    generation.store(1, std::memory_order_relaxed);
}

void TranspositionTable::newSearch()
{
    generation.fetch_add(1, std::memory_order_relaxed);
}

bool TranspositionTable::probe(uint64_t key, TTEntry &out) const
{
    if (table.empty())
        return false;
    const size_t index = static_cast<size_t>(key & mask);
    const TTBucket &bucket = table[index];
    bool found = false;
    int bestDepth = -INF_SCORE;

    auto unpack = [](uint64_t entryKey, uint64_t payload, TTEntry &entry)
    {
        entry.key = entryKey;
        const uint16_t packedScore = static_cast<uint16_t>(payload & 0xFFFFULL);
        entry.score = packedScore < 0x8000U
                          ? static_cast<int>(packedScore)
                          : static_cast<int>(packedScore) - 65536;
        entry.depth = static_cast<int>((payload >> 16) & 0xFFULL) - 1;
        entry.bound = static_cast<uint8_t>((payload >> 24) & 0x3ULL);
        entry.generation = static_cast<uint8_t>((payload >> 26) & 0xFFULL);
        const uint32_t move = static_cast<uint32_t>((payload >> 34) & 0x3FFFFULL);
        entry.bestMove.from = static_cast<int>(move & 0x3F);
        entry.bestMove.to = static_cast<int>((move >> 6) & 0x3F);
        entry.bestMove.promoPiece = static_cast<uint8_t>((move >> 12) & 0x7);
        entry.bestMove.isCapture = ((move >> 15) & 1U) != 0;
        entry.bestMove.isPromotion = ((move >> 16) & 1U) != 0;
        entry.bestMove.isCastle = ((move >> 17) & 1U) != 0;
    };

    for (const auto &slot : bucket.entries)
    {
        const uint64_t verification = slot.verification.load(std::memory_order_acquire);
        const uint64_t payload = slot.payload.load(std::memory_order_relaxed);
        if ((payload & (1ULL << 63)) == 0 ||
            slot.verification.load(std::memory_order_acquire) != verification ||
            (verification ^ payload) != key)
        {
            continue;
        }

        TTEntry entry;
        unpack(key, payload, entry);
        if (!found || entry.depth > bestDepth)
        {
            found = true;
            bestDepth = entry.depth;
            out = entry;
        }
    }
    return found;
}

void TranspositionTable::store(uint64_t key, int depth, int score, uint8_t bound, const Move &bestMove)
{
    if (table.empty())
        return;
    const size_t index = static_cast<size_t>(key & mask);
    const uint8_t gen = static_cast<uint8_t>(generation.load(std::memory_order_relaxed));
    TTBucket &bucket = table[index];

    auto read = [](const TTSlot &slot, TTEntry &entry) -> bool
    {
        const uint64_t verification = slot.verification.load(std::memory_order_acquire);
        const uint64_t payload = slot.payload.load(std::memory_order_relaxed);
        if ((payload & (1ULL << 63)) == 0 ||
            slot.verification.load(std::memory_order_acquire) != verification)
        {
            return false;
        }
        entry.key = verification ^ payload;
        const uint16_t packedScore = static_cast<uint16_t>(payload & 0xFFFFULL);
        entry.score = packedScore < 0x8000U
                          ? static_cast<int>(packedScore)
                          : static_cast<int>(packedScore) - 65536;
        entry.depth = static_cast<int>((payload >> 16) & 0xFFULL) - 1;
        entry.bound = static_cast<uint8_t>((payload >> 24) & 0x3ULL);
        entry.generation = static_cast<uint8_t>((payload >> 26) & 0xFFULL);
        const uint32_t move = static_cast<uint32_t>((payload >> 34) & 0x3FFFFULL);
        entry.bestMove.from = static_cast<int>(move & 0x3F);
        entry.bestMove.to = static_cast<int>((move >> 6) & 0x3F);
        entry.bestMove.promoPiece = static_cast<uint8_t>((move >> 12) & 0x7);
        entry.bestMove.isCapture = ((move >> 15) & 1U) != 0;
        entry.bestMove.isPromotion = ((move >> 16) & 1U) != 0;
        entry.bestMove.isCastle = ((move >> 17) & 1U) != 0;
        return true;
    };

    auto pack = [](int entryDepth, int entryScore, uint8_t entryBound,
                   uint8_t entryGeneration, const Move &move) -> uint64_t
    {
        const uint64_t packedScore = static_cast<uint16_t>(
            std::clamp(entryScore, -32768, 32767));
        const uint64_t packedDepth = static_cast<uint64_t>(
            std::clamp(entryDepth, -1, 254) + 1);
        uint64_t packedMove = static_cast<uint64_t>(move.from & 63);
        packedMove |= static_cast<uint64_t>(move.to & 63) << 6;
        packedMove |= static_cast<uint64_t>(move.promoPiece & 7) << 12;
        packedMove |= static_cast<uint64_t>(move.isCapture) << 15;
        packedMove |= static_cast<uint64_t>(move.isPromotion) << 16;
        packedMove |= static_cast<uint64_t>(move.isCastle) << 17;
        return packedScore |
               (packedDepth << 16) |
               (static_cast<uint64_t>(entryBound & 3) << 24) |
               (static_cast<uint64_t>(entryGeneration) << 26) |
               (packedMove << 34) |
               (1ULL << 63);
    };

    TTSlot *replace = &bucket.entries[0];
    int replaceScore = INT_MAX;

    for (auto &slot : bucket.entries)
    {
        TTEntry e;
        const bool occupied = read(slot, e);
        if (!occupied)
        {
            replace = &slot;
            replaceScore = INT_MIN;
            break;
        }

        if (e.key == key)
        {
            if (e.depth > depth && e.bound == 3 && bound != 3)
                return;
            const uint64_t payload = pack(depth, score, bound, gen, bestMove);
            slot.payload.store(payload, std::memory_order_relaxed);
            slot.verification.store(key ^ payload, std::memory_order_release);
            return;
        }

        const int age = static_cast<int>((gen - e.generation) & 0xFF);
        const int currentScore = e.depth - age * 2;
        if (currentScore < replaceScore)
        {
            replaceScore = currentScore;
            replace = &slot;
        }
    }

    const uint64_t payload = pack(depth, score, bound, gen, bestMove);
    replace->payload.store(payload, std::memory_order_relaxed);
    replace->verification.store(key ^ payload, std::memory_order_release);
}

size_t TranspositionTable::sizeBytes() const
{
    return table.size() * sizeof(TTBucket);
}

size_t TranspositionTable::bucketCount() const
{
    return table.size();
}

bool TranspositionTable::isLockFree() const
{
    if (table.empty())
        return std::atomic<uint64_t>{}.is_lock_free();
    return table.front().entries.front().payload.is_lock_free() &&
           table.front().entries.front().verification.is_lock_free();
}

struct ScoredMove
{
    int score = 0;
    Move move{};
};

// Move generation and ordering need roughly 7 KiB of scratch space per ply.
// Keeping those buffers in recursive negamax frames exhausts the 512 KiB
// worker-thread stack on macOS near MAX_PLY.  Allocate one reusable buffer per
// ply in SearchState instead: this preserves the search ceiling without a heap
// allocation at every node.
struct MoveOrderingBuffer
{
    MoveList generated;
    std::array<ScoredMove, MoveList::MoveBuffer::kMaxMoves> ordered{};
};

struct SearchState
{
    struct EvalCacheEntry
    {
        uint64_t key = 0;
        int score = 0;
    };

    // The caller-owned flag handles UCI `stop`; sharedStopFlag is private to
    // one search and lets any worker stop the whole Lazy-SMP team.  Keeping
    // them separate means a normally completed timed search does not look
    // externally cancelled.
    std::atomic<bool> *externalStopFlag = nullptr;
    std::atomic<bool> *sharedStopFlag = nullptr;
    std::atomic<uint64_t> *sharedNodeCounter = nullptr;
    uint64_t sharedNodeLimit = 0;
    TranspositionTable *tt = nullptr;
    uint64_t nodes = 0;
    uint64_t qnodes = 0;
    std::chrono::steady_clock::time_point start;
    int hardTimeLimitMs = 0;
    int softTimeLimitMs = 0;
    bool adaptiveTimeManagement = false;
    int nodeLimit = 0;
    int contempt = 0;
    bool usePruning = true;
    Side rootSide = WHITE;
    int workerId = 0;
    int killerMoves[MAX_PLY][2]{};
    int history[12][64]{};
    int counterMoves[64][64]{};
    int continuation[64][64]{};
    int moveStack[MAX_PLY]{};
    std::vector<uint64_t> repHistory;
    std::vector<EvalCacheEntry> evalCache;
    std::vector<MoveOrderingBuffer> moveOrderingBuffers;
};

struct NullUndo
{
    Side prevSide = WHITE;
    int prevEnPassant = -1;
    uint64_t prevHash = 0;
    int prevHalfmoveClock = 0;
    int prevFullmoveNumber = 1;
};

uint64_t totalNodes(const SearchState &st)
{
    return st.nodes + st.qnodes;
}

bool reserveSearchNode(SearchState &st)
{
    if (!st.sharedNodeCounter || st.sharedNodeLimit == 0)
        return true;

    uint64_t current = st.sharedNodeCounter->load(std::memory_order_relaxed);
    while (current < st.sharedNodeLimit)
    {
        if (st.sharedNodeCounter->compare_exchange_weak(
                current,
                current + 1,
                std::memory_order_relaxed,
                std::memory_order_relaxed))
        {
            if (current + 1 >= st.sharedNodeLimit && st.sharedStopFlag)
                st.sharedStopFlag->store(true, std::memory_order_relaxed);
            return true;
        }
    }

    if (st.sharedStopFlag)
        st.sharedStopFlag->store(true, std::memory_order_relaxed);
    return false;
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

int drawScore(const Position &pos, const SearchState &st)
{
    if (st.contempt == 0)
        return 0;
    return (pos.sideToMove == st.rootSide) ? st.contempt : -st.contempt;
}

bool isThreefoldRepetition(const Position &pos, const SearchState &st)
{
    int occurrences = 0;
    for (uint64_t key : st.repHistory)
    {
        if (key == pos.hashKey && ++occurrences >= 3)
            return true;
    }
    return false;
}

bool isRuleDraw(const Position &pos, const SearchState &st)
{
    return pos.halfmoveClock >= 100 || pos.isInsufficientMaterial() ||
           isThreefoldRepetition(pos, st);
}

bool adjudicateRuleDraw(Position &pos, const SearchState &st, int ply, int &score)
{
    if (!isRuleDraw(pos, st))
        return false;

    // Checkmate ends the game before a fifty-move or repetition draw can be
    // claimed.  Generating legal moves here is only necessary for the rare
    // case where a draw condition and check occur at the same node.
    const Side us = pos.sideToMove;
    const Side them = (us == WHITE ? BLACK : WHITE);
    const int kingSq = pos.kingSquare[us];
    const bool inCheck = kingSq != -1 && pos.isSquareAttacked(kingSq, them);
    if (inCheck)
    {
        MoveList legal;
        generateLegalMoves(pos, legal);
        if (legal.moves.empty())
        {
            score = -MATE_SCORE + ply;
            return true;
        }
    }

    score = drawScore(pos, st);
    return true;
}

int evaluateCached(Position &pos, SearchState &st)
{
    if (st.evalCache.empty())
        return evaluate(pos);

    const uint64_t key = pos.hashKey;
    const size_t idx = static_cast<size_t>(key) & (st.evalCache.size() - 1);
    auto &entry = st.evalCache[idx];
    if (entry.key == key)
        return entry.score;

    const int score = evaluate(pos);
    entry.key = key;
    entry.score = score;
    return score;
}

void doNullMove(Position &pos, NullUndo &u)
{
    u.prevSide = pos.sideToMove;
    u.prevEnPassant = pos.enPassantSquare;
    u.prevHash = pos.hashKey;
    u.prevHalfmoveClock = pos.halfmoveClock;
    u.prevFullmoveNumber = pos.fullmoveNumber;

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
    pos.nnueAccumulator.positionKey = pos.hashKey;
}

void undoNullMove(Position &pos, const NullUndo &u)
{
    pos.sideToMove = u.prevSide;
    pos.enPassantSquare = u.prevEnPassant;
    pos.hashKey = u.prevHash;
    pos.halfmoveClock = u.prevHalfmoveClock;
    pos.fullmoveNumber = u.prevFullmoveNumber;
    pos.nnueAccumulator.positionKey = pos.hashKey;
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

    int id = moveId(m);

    if (m.isCapture)
    {
        Side us = pos.sideToMove;
        int captured = capturedPieceAt(pos, m, us);
        int attacker = pos.pieceIndexAt(m.from);
        int victimVal = (captured >= 0) ? pieceValuesAbs[captured] : 100;
        int attackerVal = (attacker >= 0) ? pieceValuesAbs[attacker] : 100;
        const int mvvLva = victimVal * 12 - attackerVal;
        return 100000 + mvvLva;
    }

    if (m.isPromotion)
        return 90000 + m.promoPiece * 10;

    int killer1 = st.killerMoves[ply][0];
    int killer2 = st.killerMoves[ply][1];
    if (id == killer1)
        return 80000;
    if (id == killer2)
        return 70000;

    if (ply > 0)
    {
        int prevId = st.moveStack[ply - 1];
        if (prevId >= 0)
        {
            int prevFrom = (prevId >> 6) & 63;
            int prevTo = prevId & 63;
            int continuationBonus = st.continuation[prevTo][m.to] / 4;
            if (st.counterMoves[prevFrom][prevTo] == id)
                return 75000 + continuationBonus;
            if (continuationBonus > 0)
                return continuationBonus;
        }
    }

    int attacker = pos.pieceIndexAt(m.from);
    if (attacker >= 0)
    {
        int base = st.history[attacker][m.to];
        if (st.workerId > 0 && ply == 0)
        {
            const int jitter = ((m.from * 17 + m.to * 13 + st.workerId * 23) & 15);
            base += jitter;
        }
        return base;
    }

    return 0;
}

bool shouldStop(SearchState &st)
{
    if (st.externalStopFlag && st.externalStopFlag->load(std::memory_order_relaxed))
        return true;
    if (st.sharedStopFlag && st.sharedStopFlag->load(std::memory_order_relaxed))
        return true;
    if (st.nodeLimit > 0 && (int)totalNodes(st) >= st.nodeLimit)
        return true;
    if (st.hardTimeLimitMs > 0 && (totalNodes(st) & 1023) == 0)
    {
        if (elapsedMs(st) >= st.hardTimeLimitMs)
        {
            if (st.sharedStopFlag)
                st.sharedStopFlag->store(true, std::memory_order_relaxed);
            return true;
        }
    }
    return false;
}

int quiescence(Position &pos, SearchState &st, int alpha, int beta, int ply)
{
    if (shouldStop(st))
        return 0;

    if (!reserveSearchNode(st))
        return 0;

    if (ply >= MAX_PLY - 1)
        return evaluateCached(pos, st);

    int terminalScore = 0;
    if (adjudicateRuleDraw(pos, st, ply, terminalScore))
        return terminalScore;

    int tbScore = 0;
    if (tryTablebaseProbe(pos, 1, ply, tbScore))
        return tbScore;

    st.qnodes++;

    Side us = pos.sideToMove;
    Side them = (us == WHITE ? BLACK : WHITE);
    int kingSq = pos.kingSquare[us];
    bool inCheck = (kingSq != -1) && pos.isSquareAttacked(kingSq, them);

    int stand = evaluateCached(pos, st);
    if (!inCheck)
    {
        if (stand >= beta)
            return stand;
        if (stand > alpha)
            alpha = stand;

        // Delta pruning: if even the best plausible tactical swing cannot raise alpha,
        // fail low immediately.
        if (stand + pieceValuesAbs[WQ] + 160 < alpha)
            return alpha;
    }

    MoveOrderingBuffer &moveBuffer = st.moveOrderingBuffers[ply];
    MoveList &moves = moveBuffer.generated;
    moves.moves.clear();
    generateAllMoves(pos, moves);
    auto &moveVec = moveBuffer.ordered;
    std::size_t moveCount = 0;
    Move emptyMove{};
    for (const auto &m : moves.moves)
        moveVec[moveCount++] = {scoreMove(pos, m, emptyMove, ply, st), m};
    std::sort(moveVec.begin(), moveVec.begin() + moveCount, [](const auto &a, const auto &b)
              { return a.score > b.score; });

    int legalMoves = 0;
    for (std::size_t i = 0; i < moveCount; ++i)
    {
        const Move &m = moveVec[i].move;
        if (!inCheck && !(m.isCapture || m.isPromotion))
            continue;

        int see = 0;
        bool haveSee = false;
        if (!inCheck && m.isCapture && !m.isPromotion)
        {
            see = staticExchangeEval(pos, m);
            haveSee = true;
        }

        if (!inCheck && m.isCapture)
        {
            int captured = capturedPieceAt(pos, m, us);
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

        // Check whether a capture gives check before applying the losing-SEE
        // filter.  Sacrificing checking captures can be forcing and must stay
        // visible to quiescence.
        bool givesCheck = false;
        const int opponentKingSq = pos.kingSquare[them];
        if (opponentKingSq != -1)
            givesCheck = pos.isSquareAttacked(opponentKingSq, us);

        // A losing capture can still be a forcing check or a mating
        // sacrifice.  Only prune it after checking whether it gives check.
        if (!inCheck && haveSee && see < 0 && !givesCheck)
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
            return score;
        if (score > alpha)
            alpha = score;
    }

    if (inCheck && legalMoves == 0)
        return -MATE_SCORE + ply;

    return alpha;
}

int negamax(Position &pos, SearchState &st, int depth, int alpha, int beta, int ply, bool allowNullMove = true)
{
    if (shouldStop(st))
        return 0;

    if (!reserveSearchNode(st))
        return 0;

    if (ply >= MAX_PLY - 1)
        return evaluateCached(pos, st);

    int terminalScore = 0;
    if (adjudicateRuleDraw(pos, st, ply, terminalScore))
        return terminalScore;

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

    int tbScore = 0;
    if (tryTablebaseProbe(pos, depth, ply, tbScore))
        return tbScore;

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

    // Internal iterative deepening: recover a useful TT move when ordering is empty.
    if (pvNode && !inCheck && isNullMove(ttMove) && st.tt && depth >= 7)
    {
        (void)negamax(pos, st, depth - 2, alpha, beta, ply, allowNullMove);
        if (shouldStop(st))
            return 0;
        TTEntry iid;
        if (st.tt->probe(pos.hashKey, iid))
            ttMove = iid.bestMove;
    }

    int staticEval = INF_SCORE;
    bool haveStaticEval = false;

    // Reverse futility pruning for shallow, non-check, non-PV nodes.
    if (st.usePruning && depth <= 3 && !pvNode && !inCheck)
    {
        staticEval = evaluateCached(pos, st);
        haveStaticEval = true;
        int margin = 120 * depth + 60;
        if (staticEval - margin >= beta)
            return staticEval;
    }

    // Null move pruning: try passing the move to prove a beta cutoff quickly.
    if (st.usePruning && allowNullMove && depth >= 3 && !inCheck && ply > 0 && hasNonPawnMaterial(pos, us))
    {
        if (!haveStaticEval)
        {
            staticEval = evaluateCached(pos, st);
            haveStaticEval = true;
        }
        if (staticEval >= beta)
        {
            int reduction = 2 + depth / 4;
            NullUndo nu;
            doNullMove(pos, nu);
            int score = -negamax(pos, st, std::max(0, depth - 1 - reduction), -beta, -beta + 1, ply + 1, false);
            undoNullMove(pos, nu);

            if (shouldStop(st))
                return 0;
            if (score >= beta)
            {
                if (depth >= 7)
                {
                    int verifyDepth = std::max(0, depth - 1 - reduction);
                    int verify = negamax(pos, st, verifyDepth, beta - 1, beta, ply, false);
                    if (shouldStop(st))
                        return 0;
                    if (verify < beta)
                        goto skip_null_cutoff;
                }
                if (st.tt)
                    st.tt->store(pos.hashKey, depth, scoreToTT(score, ply), 2, Move{});
                return score;
            }
        skip_null_cutoff:
            ;
        }
    }

    MoveOrderingBuffer &moveBuffer = st.moveOrderingBuffers[ply];
    MoveList &moves = moveBuffer.generated;
    moves.moves.clear();
    generateAllMoves(pos, moves);
    auto &moveVec = moveBuffer.ordered;
    std::size_t moveCount = 0;
    for (const auto &m : moves.moves)
        moveVec[moveCount++] = {scoreMove(pos, m, ttMove, ply, st), m};
    std::sort(moveVec.begin(), moveVec.begin() + moveCount, [](const auto &a, const auto &b)
              { return a.score > b.score; });

    int bestScore = -INF_SCORE;
    Move bestMove{};
    int legalMoves = 0;
    int searchedMoves = 0;
    bool searchedPvMove = false;
    std::vector<std::pair<int, int>> quietTried;
    quietTried.reserve(48);

    for (std::size_t i = 0; i < moveCount; ++i)
    {
        const Move &m = moveVec[i].move;
        if (!castlePathSafe(pos, m, us, them))
            continue;

        int attackerFrom = pos.pieceIndexAt(m.from);
        bool isQuiet = !m.isCapture && !m.isPromotion && !m.isCastle;
        int see = 0;
        bool haveSee = false;
        if (m.isCapture && !m.isPromotion)
        {
            see = staticExchangeEval(pos, m);
            haveSee = true;
        }

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
        if (st.usePruning && !pvNode && !inCheck && !givesCheck && depth <= 3 && isQuiet)
        {
            int lmpThreshold = 12 + 8 * depth + depth * depth; // depth1:21 depth2:32 depth3:45
            if (searchedMoves > 0 && legalMoves > lmpThreshold)
            {
                pos.unmakeMove(m, u);
                continue;
            }
        }

        if (st.usePruning && !pvNode && !inCheck && !givesCheck && isQuiet && depth <= 3 && searchedMoves > 0)
        {
            if (!haveStaticEval)
            {
                staticEval = evaluateCached(pos, st);
                haveStaticEval = true;
            }
            int hist = (attackerFrom >= 0) ? st.history[attackerFrom][m.to] : 0;
            int futilityMargin = 95 + 125 * depth;
            if (hist > 3000)
                futilityMargin += 80;
            if (staticEval + futilityMargin <= alpha)
            {
                pos.unmakeMove(m, u);
                continue;
            }
        }

        st.repHistory.push_back(pos.hashKey);
        int prevMoveStack = st.moveStack[ply];
        st.moveStack[ply] = moveId(m);
        int score = 0;
        int nextDepth = depth - 1;

        if (st.usePruning && !pvNode && !inCheck && !givesCheck && depth <= 3 &&
            haveSee && see < -(95 * depth) && searchedMoves > 0)
        {
            st.moveStack[ply] = prevMoveStack;
            st.repHistory.pop_back();
            pos.unmakeMove(m, u);
            continue;
        }

        bool lmrCandidate = !pvNode && !inCheck && !givesCheck && isQuiet &&
                            depth >= 3 && legalMoves >= 4;
        searchedMoves++;
        if (lmrCandidate)
        {
            int reduction = 1;
            if (depth >= 6)
                reduction++;
            if (legalMoves >= 8)
                reduction++;
            if (depth >= 10 && legalMoves >= 14)
                reduction++;
            int hist = (attackerFrom >= 0) ? st.history[attackerFrom][m.to] : 0;
            if (hist > 6000)
                reduction--;
            else if (hist < -3000)
                reduction++;
            reduction = std::clamp(reduction, 1, std::max(1, nextDepth - 1));
            int reducedDepth = std::max(0, nextDepth - reduction);

            score = -negamax(pos, st, reducedDepth, -alpha - 1, -alpha, ply + 1, allowNullMove);
            if (score > alpha)
            {
                score = -negamax(pos, st, nextDepth, -alpha - 1, -alpha, ply + 1, allowNullMove);
                if (score > alpha && score < beta)
                {
                    score = -negamax(pos, st, nextDepth, -beta, -alpha, ply + 1, allowNullMove);
                }
            }
        }
        else
        {
            if (!searchedPvMove)
            {
                score = -negamax(pos, st, nextDepth, -beta, -alpha, ply + 1, allowNullMove);
            }
            else
            {
                score = -negamax(pos, st, nextDepth, -alpha - 1, -alpha, ply + 1, allowNullMove);
                if (score > alpha && score < beta)
                {
                    score = -negamax(pos, st, nextDepth, -beta, -alpha, ply + 1, allowNullMove);
                }
            }
        }
        st.moveStack[ply] = prevMoveStack;
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
            if (isQuiet)
            {
                int id = moveId(m);
                int bonus = depth * depth + depth * 2;
                st.killerMoves[ply][1] = st.killerMoves[ply][0];
                st.killerMoves[ply][0] = id;
                if (attackerFrom >= 0)
                {
                    updateHistory(st.history[attackerFrom][m.to], bonus);
                    for (const auto &q : quietTried)
                        updateHistory(st.history[q.first][q.second], -bonus);
                }
                if (ply > 0)
                {
                    int prevId = st.moveStack[ply - 1];
                    if (prevId >= 0)
                    {
                        int prevFrom = (prevId >> 6) & 63;
                        int prevTo = prevId & 63;
                        st.counterMoves[prevFrom][prevTo] = id;
                        updateHistory(st.continuation[prevTo][m.to], bonus);
                        for (const auto &q : quietTried)
                            updateHistory(st.continuation[prevTo][q.second], -bonus);
                    }
                }
            }
            if (st.tt)
                st.tt->store(pos.hashKey, depth, scoreToTT(score, ply), 2, m);
            return score;
        }

        if (isQuiet && attackerFrom >= 0)
            quietTried.push_back({attackerFrom, m.to});
    }

    if (legalMoves == 0)
    {
        if (inCheck)
            return -MATE_SCORE + ply;
        return drawScore(pos, st);
    }

    // Every nonterminal node must search at least one legal move. Returning
    // -INF here would be interpreted as a mate score by the parent.
    if (searchedMoves == 0)
        return evaluateCached(pos, st);

    uint8_t bound = 3;
    if (bestScore <= alphaOrig)
        bound = 1;

    if (bound != 1 && !bestMove.isCapture && !bestMove.isPromotion && !bestMove.isCastle)
    {
        int attacker = pos.pieceIndexAt(bestMove.from);
        if (attacker >= 0)
            updateHistory(st.history[attacker][bestMove.to], depth);
    }

    if (st.tt)
        st.tt->store(pos.hashKey, depth, scoreToTT(bestScore, ply), bound, bestMove);

    return bestScore;
}

SearchResult searchSingle(
    Position &pos,
    const SearchLimits &limits,
    TranspositionTable &tt,
    std::atomic<bool> &externalStopFlag,
    std::atomic<bool> &sharedStopFlag,
    std::chrono::steady_clock::time_point sharedStart,
    const TimeBudget &timeBudget,
    int workerId,
    std::atomic<uint64_t> *sharedNodeCounter)
{
    SearchState st;
    st.externalStopFlag = &externalStopFlag;
    st.sharedStopFlag = &sharedStopFlag;
    st.sharedNodeCounter = sharedNodeCounter;
    st.sharedNodeLimit = limits.nodes > 0 ? static_cast<uint64_t>(limits.nodes) : 0;
    st.tt = &tt;
    st.nodes = 0;
    st.qnodes = 0;
    st.start = sharedStart;
    st.hardTimeLimitMs = timeBudget.maximumMs;
    st.softTimeLimitMs = timeBudget.optimumMs;
    st.adaptiveTimeManagement = timeBudget.adaptive;
    st.nodeLimit = limits.nodes;
    st.contempt = limits.contempt;
    st.usePruning = limits.usePruning;
    st.rootSide = pos.sideToMove;
    st.workerId = workerId;
    st.evalCache.assign(1u << 15, SearchState::EvalCacheEntry{});
    st.moveOrderingBuffers.resize(MAX_PLY);
    st.repHistory = limits.positionHistory;
    if (st.repHistory.empty() || st.repHistory.back() != pos.hashKey)
        st.repHistory.push_back(pos.hashKey);
    std::fill(std::begin(st.moveStack), std::end(st.moveStack), -1);
    for (int from = 0; from < 64; ++from)
    {
        for (int to = 0; to < 64; ++to)
            st.counterMoves[from][to] = -1;
    }

    SearchResult result;
    result.bestMove = Move{};
    result.score = 0;
    result.depth = 0;
    int lastScore = 0;

    MoveList rootLegal;
    generateLegalMoves(pos, rootLegal);
    if (rootLegal.moves.empty())
    {
        Side us = pos.sideToMove;
        Side them = (us == WHITE) ? BLACK : WHITE;
        bool inCheck = (pos.kingSquare[us] != -1) && pos.isSquareAttacked(pos.kingSquare[us], them);
        result.score = inCheck ? -MATE_SCORE : drawScore(pos, st);
        return result;
    }

    if (isRuleDraw(pos, st))
    {
        result.bestMove = rootLegal.moves.front();
        result.score = drawScore(pos, st);
        return result;
    }

    struct RootMoveEntry
    {
        Move move{};
        int score = 0;
    };
    std::vector<RootMoveEntry> rootMoves;
    rootMoves.reserve(rootLegal.moves.size());
    for (const auto &m : rootLegal.moves)
        rootMoves.push_back({m, 0});

    TTEntry rootTte;
    if (tt.probe(pos.hashKey, rootTte) && !isNullMove(rootTte.bestMove))
    {
        for (auto &rm : rootMoves)
        {
            if (sameMove(rm.move, rootTte.bestMove))
            {
                rm.score = 1000000;
                break;
            }
        }
        std::stable_sort(rootMoves.begin(), rootMoves.end(), [](const RootMoveEntry &a, const RootMoveEntry &b)
                         { return a.score > b.score; });
    }

    if (workerId > 0)
    {
        for (auto &rm : rootMoves)
        {
            const int jitter = ((rm.move.from * 31 + rm.move.to * 7 + rm.move.promoPiece * 19 + workerId * 29) & 31);
            rm.score += jitter;
        }
        std::stable_sort(rootMoves.begin(), rootMoves.end(), [](const RootMoveEntry &a, const RootMoveEntry &b)
                         { return a.score > b.score; });
    }

    auto rootSearch = [&](int depth, int alpha, int beta, Move &bestMoveOut,
                          int &bestScoreOut, int &secondBestScoreOut,
                          bool &lateBestMoveChangeOut) -> bool
    {
        Side us = pos.sideToMove;
        Side them = (us == WHITE ? BLACK : WHITE);

        int localAlpha = alpha;
        int bestScore = -INF_SCORE;
        int secondBestScore = -INF_SCORE;
        Move bestMove{};
        bool searchedAny = false;
        bool searchedPvMove = false;
        int searchedRootMoves = 0;
        bool lateBestMoveChange = false;

        for (auto &rm : rootMoves)
        {
            const Move &m = rm.move;
            if (!castlePathSafe(pos, m, us, them))
                continue;

            Undo u;
            pos.makeMove(m, u);
            if (pos.isSquareAttacked(pos.kingSquare[us], them))
            {
                pos.unmakeMove(m, u);
                continue;
            }

            searchedAny = true;
            ++searchedRootMoves;
            st.repHistory.push_back(pos.hashKey);
            int prevMoveStack = st.moveStack[0];
            st.moveStack[0] = moveId(m);

            int score = 0;
            if (!searchedPvMove)
            {
                score = -negamax(pos, st, depth - 1, -beta, -localAlpha, 1);
            }
            else
            {
                score = -negamax(pos, st, depth - 1, -localAlpha - 1, -localAlpha, 1);
                if (score > localAlpha && score < beta)
                    score = -negamax(pos, st, depth - 1, -beta, -localAlpha, 1);
            }

            st.moveStack[0] = prevMoveStack;
            st.repHistory.pop_back();
            pos.unmakeMove(m, u);
            searchedPvMove = true;

            if (shouldStop(st))
                return false;

            rm.score = score;
            if (score > bestScore)
            {
                if (bestScore != -INF_SCORE &&
                    searchedRootMoves * 2 >= static_cast<int>(rootMoves.size()))
                {
                    lateBestMoveChange = true;
                }
                secondBestScore = bestScore;
                bestScore = score;
                bestMove = m;
            }
            else if (score > secondBestScore)
            {
                secondBestScore = score;
            }
            if (score > localAlpha)
                localAlpha = score;
            if (localAlpha >= beta)
                break;
        }

        if (!searchedAny)
        {
            bestMoveOut = Move{};
            bestScoreOut = 0;
            secondBestScoreOut = -INF_SCORE;
            lateBestMoveChangeOut = false;
            return true;
        }

        bestMoveOut = bestMove;
        bestScoreOut = bestScore;
        secondBestScoreOut = secondBestScore;
        lateBestMoveChangeOut = lateBestMoveChange;
        std::stable_sort(rootMoves.begin(), rootMoves.end(), [](const RootMoveEntry &a, const RootMoveEntry &b)
                         { return a.score > b.score; });
        return true;
    };

    int maxDepth = (limits.depth > 0) ? limits.depth : 64;
    Move previousBestMove{};
    int stableBestMoveIterations = 0;
    for (int d = 1; d <= maxDepth; ++d)
    {
        if (shouldStop(st))
            break;

        int score = 0;
        int alpha = -INF_SCORE;
        int beta = INF_SCORE;
        // A slightly wider initial window avoids repeated full root searches
        // when tactical scores move by more than a quarter pawn between
        // completed iterations.
        int window = 50;
        int reSearches = 0;
        Move depthBestMove = result.bestMove;
        int secondBestScore = -INF_SCORE;
        bool lateBestMoveChange = false;

        if (d >= 4)
        {
            alpha = std::max(-INF_SCORE, lastScore - window);
            beta = std::min(INF_SCORE, lastScore + window);
        }

        while (true)
        {
            Move trialBestMove{};
            int trialScore = 0;
            int trialSecondBestScore = -INF_SCORE;
            bool trialLateBestMoveChange = false;
            if (!rootSearch(d, alpha, beta, trialBestMove, trialScore,
                            trialSecondBestScore, trialLateBestMoveChange))
                break;
            score = trialScore;
            depthBestMove = trialBestMove;
            secondBestScore = trialSecondBestScore;
            lateBestMoveChange = trialLateBestMoveChange;
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

        const int scoreChange = d > 1 ? std::abs(score - lastScore) : INF_SCORE;
        if (d > 1 && sameMove(depthBestMove, previousBestMove))
            ++stableBestMoveIterations;
        else
            stableBestMoveIterations = 1;
        previousBestMove = depthBestMove;

        result.bestMove = depthBestMove;
        result.score = score;
        result.depth = d;
        result.nodes = st.nodes;
        result.qnodes = st.qnodes;
        lastScore = score;

        if ((d % 4) == 0)
            decayHistory(st);

        tt.store(pos.hashKey, d, scoreToTT(score, 0), 3, result.bestMove);

        int elapsed = elapsedMs(st);
        uint64_t allNodes = totalNodes(st);
        int nps = (elapsed > 0) ? (int)(allNodes * 1000 / elapsed) : (int)allNodes;
        if (limits.printInfo)
        {
            std::cout << "info depth " << d << " score ";
            if (isSearchMateScore(score))
                std::cout << "mate " << searchMateMoves(score);
            else
                std::cout << "cp " << score;
            std::cout
                      << " nodes " << allNodes << " nps " << nps
                      << " time " << elapsed << " string qnodes " << st.qnodes << "\n";
        }

        if (st.softTimeLimitMs > 0)
        {
            int timeScale = 100;
            if (st.adaptiveTimeManagement)
            {
                const int scoreGap = secondBestScore <= -INF_SCORE / 2
                                         ? INF_SCORE
                                         : std::max(0, score - secondBestScore);

                // Stable, clearly separated choices can return time to the
                // clock. PV changes, score swings, close alternatives, and
                // aspiration failures earn progressively more of the hard
                // budget.
                if (stableBestMoveIterations >= 3 && scoreChange <= 12 &&
                    scoreGap >= 70 && reSearches == 0)
                {
                    timeScale = 55;
                }
                else if (stableBestMoveIterations >= 2 && scoreChange <= 20 &&
                         scoreGap >= 40 && reSearches == 0)
                {
                    timeScale = 75;
                }
                else
                {
                    if (stableBestMoveIterations == 1)
                        timeScale = std::max(timeScale, 135);
                    if (scoreChange > 80)
                        timeScale = std::max(timeScale, 145);
                    else if (scoreChange > 35)
                        timeScale = std::max(timeScale, 120);
                    if (scoreGap < 25)
                        timeScale = std::max(timeScale, 125);
                    if (lateBestMoveChange)
                        timeScale = std::max(timeScale, 140);
                    if (reSearches >= 2)
                        timeScale = std::max(timeScale, 150);
                    else if (reSearches == 1)
                        timeScale = std::max(timeScale, 125);
                }
            }

            const int iterationLimit = std::min(
                st.hardTimeLimitMs,
                std::max(1, (st.softTimeLimitMs * timeScale) / 100));
            if (elapsed >= iterationLimit)
                break;
        }
    }

    result.nodes = st.nodes;
    result.qnodes = st.qnodes;
    result.elapsedMs = elapsedMs(st);

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

SearchResult search(Position &pos, const SearchLimits &limits, TranspositionTable &tt, std::atomic<bool> &stopFlag)
{
    tt.newSearch();

    const auto sharedStart = std::chrono::steady_clock::now();
    const TimeBudget timeBudget = calculateTimeBudget(pos, limits);
    std::atomic<bool> sharedStopFlag{false};
    std::atomic<uint64_t> sharedNodeCounter{0};
    std::atomic<uint64_t> *sharedCounter = limits.nodes > 0 ? &sharedNodeCounter : nullptr;

    SearchLimits singleLimits = limits;
    singleLimits.threads = 1;

    int threadCount = std::max(1, limits.threads);
    const unsigned hw = std::thread::hardware_concurrency();
    if (hw > 0)
        threadCount = std::min(threadCount, static_cast<int>(hw));
    threadCount = std::min(threadCount, 32);

    if (threadCount <= 1)
        return searchSingle(
            pos, singleLimits, tt, stopFlag, sharedStopFlag,
            sharedStart, timeBudget, 0, sharedCounter);

    std::vector<SearchResult> workerResults(threadCount);
    std::vector<std::thread> workers;
    workers.reserve(threadCount - 1);

    auto workerFn = [&](int workerId)
    {
        Position localPos = pos;
        SearchLimits localLimits = singleLimits;
        localLimits.printInfo = (workerId == 0) ? limits.printInfo : false;
        workerResults[workerId] = searchSingle(
            localPos,
            localLimits,
            tt,
            stopFlag,
            sharedStopFlag,
            sharedStart,
            timeBudget,
            workerId,
            sharedCounter);
    };

    for (int id = 1; id < threadCount; ++id)
        workers.emplace_back(workerFn, id);
    workerFn(0);

    // Worker zero owns the published result.  Once it has completed a usable
    // iteration there is no reason to let a helper consume the rest of the
    // hard budget: request cancellation before joining them.  Previously a
    // helper in a large iteration could delay `bestmove` by many seconds.
    sharedStopFlag.store(true, std::memory_order_relaxed);
    for (auto &w : workers)
        w.join();

    // Lazy-SMP helpers improve the shared TT, but the main worker owns the
    // answer. Picking the highest equal-depth helper score introduces a
    // systematic optimistic bias and nondeterministic best moves.
    SearchResult best = workerResults[0];
    uint64_t totalNodes = 0;
    uint64_t totalQnodes = 0;
    for (const auto &r : workerResults)
    {
        totalNodes += r.nodes;
        totalQnodes += r.qnodes;
    }
    best.nodes = totalNodes;
    best.qnodes = totalQnodes;
    return best;
}

TimeBudget calculateTimeBudget(const Position &pos, const SearchLimits &limits)
{
    if (limits.movetimeMs > 0)
    {
        return {static_cast<int>(std::max<int64_t>(1, (static_cast<int64_t>(limits.movetimeMs) * 95) / 100)),
                limits.movetimeMs, false};
    }
    if (limits.infinite || limits.depth > 0 || limits.nodes > 0)
        return {};

    const int remaining = pos.sideToMove == WHITE ? limits.wtimeMs : limits.btimeMs;
    const int increment = pos.sideToMove == WHITE ? limits.wincMs : limits.bincMs;
    if (remaining <= 0)
        return {};

    const int overhead = std::max(0, limits.moveOverheadMs);
    const int safeRemaining = std::max(1, remaining - overhead);

    // In sudden death, never assume that a late or low-material position is
    // close to ending.  Technical endings can consume another fifty moves;
    // the old 12--16 move estimate deliberately spent *more* time in exactly
    // the positions where flag safety matters most.
    int movesHorizon = limits.movesToGo > 0 ? limits.movesToGo : 50;
    movesHorizon = std::clamp(movesHorizon, 1, 50);

    // Preserve ten percent of the clock and charge MoveOverhead for every
    // anticipated move, not merely once.  That distinction is essential for
    // remote bullet play where HTTP latency is often larger than search time.
    const int64_t clockShare =
        (static_cast<int64_t>(safeRemaining) * 9) / (10 * movesHorizon);
    const int64_t base = clockShare - overhead +
                         (static_cast<int64_t>(increment) * 3) / 4;
    const int64_t minimum = std::min<int64_t>(
        safeRemaining, std::clamp<int64_t>(remaining / 1000, 1, 10));
    const int64_t optimum = std::clamp<int64_t>(base, minimum, safeRemaining);
    const int64_t rawHardCap = movesHorizon == 1
                                   ? safeRemaining
                                   : std::max<int64_t>(optimum, safeRemaining / 5);
    const int64_t hardCap = std::max(optimum, rawHardCap);
    const int64_t maximum = std::clamp<int64_t>(
        optimum * 2,
        optimum,
        hardCap);
    return {static_cast<int>(optimum), static_cast<int>(maximum), true};
}

bool isSearchMateScore(int score)
{
    return score > SEARCH_MATE_SCORE - 1000 || score < -SEARCH_MATE_SCORE + 1000;
}

int searchMateMoves(int score)
{
    if (!isSearchMateScore(score))
        return 0;
    const int plies = std::max(0, SEARCH_MATE_SCORE - std::abs(score));
    const int moves = (plies + 1) / 2;
    return score >= 0 ? moves : -moves;
}
