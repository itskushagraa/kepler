#include "position.hpp"
#include "movegen.hpp"
#include "zobrist.hpp"

void Position::setStartPos()
{
    fromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

void Position::fromFEN(const std::string &fen)
{
    // Reset everything
    for (int i = 0; i < 12; ++i)
        pieceBB[i] = 0ULL;
    occupancy[WHITE] = occupancy[BLACK] = allPieces = 0ULL;
    enPassantSquare = -1;

    // Defaults
    canCastleKingside[WHITE] = canCastleQueenside[WHITE] = false;
    canCastleKingside[BLACK] = canCastleQueenside[BLACK] = false;

    std::istringstream ss(fen);
    std::string board, side, castling, ep;
    int halfmove = 0, fullmove = 1;
    ss >> board >> side >> castling >> ep >> halfmove >> fullmove;

    // --- Parse board ---
    int sq = 56; // start at a8
    for (char c : board)
    {
        if (c == '/')
            sq -= 16;
        else if (isdigit(c))
            sq += c - '0';
        else
        {
            int idx = -1;
            switch (c)
            {
            case 'P':
                idx = WP;
                break;
            case 'N':
                idx = WN;
                break;
            case 'B':
                idx = WB;
                break;
            case 'R':
                idx = WR;
                break;
            case 'Q':
                idx = WQ;
                break;
            case 'K':
                idx = WK;
                break;
            case 'p':
                idx = BP;
                break;
            case 'n':
                idx = BN;
                break;
            case 'b':
                idx = BB;
                break;
            case 'r':
                idx = BR;
                break;
            case 'q':
                idx = BQ;
                break;
            case 'k':
                idx = BK;
                break;
            }
            set_bit(pieceBB[idx], sq);
            ++sq;
        }
    }

    // --- Occupancy ---
    for (int i = 0; i < 6; ++i)
        occupancy[WHITE] |= pieceBB[i];
    for (int i = 6; i < 12; ++i)
        occupancy[BLACK] |= pieceBB[i];
    allPieces = occupancy[WHITE] | occupancy[BLACK];

    // --- Side to move ---
    sideToMove = (side == "w" ? WHITE : BLACK);

    // --- Castling rights ---
    for (char c : castling)
    {
        if (c == 'K')
            canCastleKingside[WHITE] = true;
        if (c == 'Q')
            canCastleQueenside[WHITE] = true;
        if (c == 'k')
            canCastleKingside[BLACK] = true;
        if (c == 'q')
            canCastleQueenside[BLACK] = true;
    }

    // --- En passant square ---
    if (ep != "-")
    {
        int file = ep[0] - 'a';
        int rank = ep[1] - '1';
        enPassantSquare = rank * 8 + file;
    }

    kingSquare[WHITE] = (pieceBB[WK] ? __builtin_ctzll(pieceBB[WK]) : -1);
    kingSquare[BLACK] = (pieceBB[BK] ? __builtin_ctzll(pieceBB[BK]) : -1);
    halfmoveClock = halfmove;
    fullmoveNumber = fullmove;

    // --- Zobrist hash ---
    hashKey = 0;
    for (int p = 0; p < 12; ++p)
    {
        Bitboard bb = pieceBB[p];
        while (bb)
        {
            int sq = lsb(bb);
            hashKey ^= Zobrist::piece(p, sq);
            bb &= bb - 1;
        }
    }
    int rights = castlingRightsMask();
    hashKey ^= Zobrist::castling(rights);
    if (sideToMove == BLACK)
        hashKey ^= Zobrist::side();
    if (enPassantSquare != -1)
        hashKey ^= Zobrist::epFile(enPassantSquare & 7);

    Nnue::invalidate(nnueAccumulator);
}

std::string Position::toFEN() const
{
    std::string fen;
    for (int rank = 7; rank >= 0; --rank)
    {
        int empty = 0;
        for (int file = 0; file < 8; ++file)
        {
            int sq = rank * 8 + file;
            char p = pieceAt(sq);
            if (!p)
            {
                empty++;
                continue;
            }
            if (empty)
            {
                fen += char('0' + empty);
                empty = 0;
            }
            fen += p;
        }
        if (empty)
            fen += char('0' + empty);
        if (rank)
            fen += '/';
    }

    fen += ' ';
    fen += (sideToMove == WHITE ? 'w' : 'b');
    fen += ' ';

    std::string castle;
    if (canCastleKingside[WHITE])
        castle += 'K';
    if (canCastleQueenside[WHITE])
        castle += 'Q';
    if (canCastleKingside[BLACK])
        castle += 'k';
    if (canCastleQueenside[BLACK])
        castle += 'q';
    fen += castle.empty() ? "-" : castle;

    fen += ' ';
    if (enPassantSquare == -1)
    {
        fen += '-';
    }
    else
    {
        fen += char('a' + (enPassantSquare % 8));
        fen += char('1' + (enPassantSquare / 8));
    }

    fen += ' ';
    fen += std::to_string(halfmoveClock);
    fen += ' ';
    fen += std::to_string(fullmoveNumber);
    return fen;
}

void Position::printBoard() const
{
    for (int rank = 7; rank >= 0; --rank)
    {
        for (int file = 0; file < 8; ++file)
        {
            int sq = rank * 8 + file;
            char piece = pieceAt(sq);
            std::cout << (piece ? piece : '.') << ' ';
        }
        std::cout << '\n';
    }
    std::cout << (sideToMove == WHITE ? "White" : "Black") << " to move\n";
}

int Position::pieceIndex(char c)
{
    switch (c)
    {
    case 'P':
        return WP;
    case 'N':
        return WN;
    case 'B':
        return WB;
    case 'R':
        return WR;
    case 'Q':
        return WQ;
    case 'K':
        return WK;
    case 'p':
        return BP;
    case 'n':
        return BN;
    case 'b':
        return BB;
    case 'r':
        return BR;
    case 'q':
        return BQ;
    case 'k':
        return BK;
    default:
        return -1;
    }
}

char Position::pieceAt(int sq) const
{
    const char pieceChars[12] = {'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k'};
    for (int i = 0; i < 12; ++i)
        if (get_bit(pieceBB[i], sq))
            return pieceChars[i];
    return 0;
}

void Position::makeMove(const Move &m)
{
    Undo dummy;
    makeMove(m, dummy);
}

void Position::makeMove(const Move &m, Undo &u)
{
    Side us = sideToMove;
    Side them = (us == WHITE ? BLACK : WHITE);
    int oldEPSq = enPassantSquare;

    u.prevEnPassant = enPassantSquare;
    u.prevCastleK[WHITE] = canCastleKingside[WHITE];
    u.prevCastleK[BLACK] = canCastleKingside[BLACK];
    u.prevCastleQ[WHITE] = canCastleQueenside[WHITE];
    u.prevCastleQ[BLACK] = canCastleQueenside[BLACK];
    u.prevKingSquare[WHITE] = kingSquare[WHITE];
    u.prevKingSquare[BLACK] = kingSquare[BLACK];
    u.prevHalfmoveClock = halfmoveClock;
    u.prevFullmoveNumber = fullmoveNumber;
    u.prevHash = hashKey;
    u.prevAccumulator = nnueAccumulator;

    // 1. Remove our piece from the origin and move it.
    int movedIndex = -1;
    for (int i = (us == WHITE ? 0 : 6); i < (us == WHITE ? 6 : 12); ++i)
    {
        if (get_bit(pieceBB[i], m.from))
        {
            movedIndex = i;
            break;
        }
    }
    u.movedIndex = movedIndex;

    // hash: clear old ep
    if (oldEPSq != -1)
        hashKey ^= Zobrist::epFile(oldEPSq & 7);

    // 2. Capture handling (including en passant)
    u.capturedIndex = -1;
    u.capturedSquare = -1;
    if (m.isCapture)
    {
        if (oldEPSq != -1 && m.to == oldEPSq && movedIndex == (us == WHITE ? WP : BP))
        {
            u.capturedSquare = (us == WHITE ? m.to - 8 : m.to + 8);
            u.capturedIndex = (us == WHITE ? BP : WP);
            clear_bit(pieceBB[u.capturedIndex], u.capturedSquare);
            hashKey ^= Zobrist::piece(u.capturedIndex, u.capturedSquare);
        }
        else
        {
            u.capturedSquare = m.to;
            u.capturedIndex = pieceIndexAt(m.to);
            if (u.capturedIndex != -1)
            {
                clear_bit(pieceBB[u.capturedIndex], m.to);
                hashKey ^= Zobrist::piece(u.capturedIndex, m.to);
            }
        }
    }

    // 3. Move our piece (and handle promotion)
    int promotionIndex = -1;
    if (m.isPromotion && movedIndex >= 0)
    {
        clear_bit(pieceBB[movedIndex], m.from);
        hashKey ^= Zobrist::piece(movedIndex, m.from);

        // choose piece (default to queen if none supplied)
        int addIdx = -1;
        uint8_t pp = (m.promoPiece == PROMO_NONE ? PROMO_QUEEN : m.promoPiece);

        if (us == WHITE)
        {
            switch (pp)
            {
            case PROMO_QUEEN:
                addIdx = WQ;
                break;
            case PROMO_ROOK:
                addIdx = WR;
                break;
            case PROMO_BISHOP:
                addIdx = WB;
                break;
            case PROMO_KNIGHT:
                addIdx = WN;
                break;
            default:
                addIdx = WQ;
                break;
            }
        }
        else
        {
            switch (pp)
            {
            case PROMO_QUEEN:
                addIdx = BQ;
                break;
            case PROMO_ROOK:
                addIdx = BR;
                break;
            case PROMO_BISHOP:
                addIdx = BB;
                break;
            case PROMO_KNIGHT:
                addIdx = BN;
                break;
            default:
                addIdx = BQ;
                break;
            }
        }
        set_bit(pieceBB[addIdx], m.to);
        hashKey ^= Zobrist::piece(addIdx, m.to);
        promotionIndex = addIdx;
    }
    else if (movedIndex >= 0)
    {
        clear_bit(pieceBB[movedIndex], m.from);
        set_bit(pieceBB[movedIndex], m.to);
        hashKey ^= Zobrist::piece(movedIndex, m.from);
        hashKey ^= Zobrist::piece(movedIndex, m.to);
    }

    // 3.5 Castling rook shift if the moved piece was a king and moved two squares
    int castleRookPiece = -1;
    int castleRookFrom = -1;
    int castleRookTo = -1;
    if (movedIndex == WK)
    {
        // O-O e1->g1 : move rook h1->f1
        if (m.from == 4 && m.to == 6)
        {
            clear_bit(pieceBB[WR], 7);
            set_bit(pieceBB[WR], 5);
            hashKey ^= Zobrist::piece(WR, 7);
            hashKey ^= Zobrist::piece(WR, 5);
            castleRookPiece = WR;
            castleRookFrom = 7;
            castleRookTo = 5;
        }
        // O-O-O e1->c1 : move rook a1->d1
        else if (m.from == 4 && m.to == 2)
        {
            clear_bit(pieceBB[WR], 0);
            set_bit(pieceBB[WR], 3);
            hashKey ^= Zobrist::piece(WR, 0);
            hashKey ^= Zobrist::piece(WR, 3);
            castleRookPiece = WR;
            castleRookFrom = 0;
            castleRookTo = 3;
        }
        kingSquare[WHITE] = m.to;
    }
    else if (movedIndex == BK)
    {
        // O-O e8->g8 : rook h8->f8
        if (m.from == 60 && m.to == 62)
        {
            clear_bit(pieceBB[BR], 63);
            set_bit(pieceBB[BR], 61);
            hashKey ^= Zobrist::piece(BR, 63);
            hashKey ^= Zobrist::piece(BR, 61);
            castleRookPiece = BR;
            castleRookFrom = 63;
            castleRookTo = 61;
        }
        // O-O-O e8->c8 : rook a8->d8
        else if (m.from == 60 && m.to == 58)
        {
            clear_bit(pieceBB[BR], 56);
            set_bit(pieceBB[BR], 59);
            hashKey ^= Zobrist::piece(BR, 56);
            hashKey ^= Zobrist::piece(BR, 59);
            castleRookPiece = BR;
            castleRookFrom = 56;
            castleRookTo = 59;
        }
        kingSquare[BLACK] = m.to;
    }

    // 3.7 En-passant target refresh: reset then set only for double push
    if (movedIndex == WP || movedIndex == BP || m.isCapture)
        halfmoveClock = 0;
    else
        halfmoveClock++;
    if (us == BLACK)
        fullmoveNumber++;

    enPassantSquare = -1;
    if (movedIndex == WP && (m.from / 8 == 1) && (m.to / 8 == 3))
        enPassantSquare = m.from + 8;
    else if (movedIndex == BP && (m.from / 8 == 6) && (m.to / 8 == 4))
        enPassantSquare = m.from - 8;

    // 3.8 Strip castling rights when king/rooks move or those squares are captured
    int oldRights = castlingRightsMask();
    if (m.from == 4 || m.to == 4)
    {
        canCastleKingside[WHITE] = false;
        canCastleQueenside[WHITE] = false;
    }
    if (m.from == 7 || m.to == 7)
        canCastleKingside[WHITE] = false;
    if (m.from == 0 || m.to == 0)
        canCastleQueenside[WHITE] = false;

    if (m.from == 60 || m.to == 60)
    {
        canCastleKingside[BLACK] = false;
        canCastleQueenside[BLACK] = false;
    }
    if (m.from == 63 || m.to == 63)
        canCastleKingside[BLACK] = false;
    if (m.from == 56 || m.to == 56)
        canCastleQueenside[BLACK] = false;

    int newRights = castlingRightsMask();
    if (oldRights != newRights)
    {
        hashKey ^= Zobrist::castling(oldRights);
        hashKey ^= Zobrist::castling(newRights);
    }

    if (enPassantSquare != -1)
        hashKey ^= Zobrist::epFile(enPassantSquare & 7);

    // 4. Recompute occupancy.
    occupancy[WHITE] = occupancy[BLACK] = 0ULL;
    for (int i = 0; i < 6; ++i)
        occupancy[WHITE] |= pieceBB[i];
    for (int i = 6; i < 12; ++i)
        occupancy[BLACK] |= pieceBB[i];
    allPieces = occupancy[WHITE] | occupancy[BLACK];

    // 5. Flip side
    sideToMove = them;
    hashKey ^= Zobrist::side();
    Nnue::applyMove(
        nnueAccumulator,
        movedIndex, m.from, m.to,
        u.capturedIndex, u.capturedSquare,
        promotionIndex,
        castleRookPiece, castleRookFrom, castleRookTo,
        hashKey);
}

void Position::unmakeMove(const Move &m, const Undo &u)
{
    sideToMove = (sideToMove == WHITE ? BLACK : WHITE);
    Side us = sideToMove;

    if (m.isPromotion && u.movedIndex != -1)
    {
        int promoIndex = -1;
        if (us == WHITE)
        {
            promoIndex = (m.promoPiece == PROMO_ROOK) ? WR : (m.promoPiece == PROMO_BISHOP) ? WB
                                                  : (m.promoPiece == PROMO_KNIGHT) ? WN
                                                                                   : WQ;
        }
        else
        {
            promoIndex = (m.promoPiece == PROMO_ROOK) ? BR : (m.promoPiece == PROMO_BISHOP) ? BB
                                                  : (m.promoPiece == PROMO_KNIGHT) ? BN
                                                                                   : BQ;
        }
        clear_bit(pieceBB[promoIndex], m.to);
        set_bit(pieceBB[u.movedIndex], m.from);
    }
    else if (u.movedIndex != -1)
    {
        clear_bit(pieceBB[u.movedIndex], m.to);
        set_bit(pieceBB[u.movedIndex], m.from);
    }

    if (u.capturedIndex != -1 && u.capturedSquare != -1)
    {
        set_bit(pieceBB[u.capturedIndex], u.capturedSquare);
    }

    if (m.isCastle)
    {
        if (us == WHITE)
        {
            if (m.from == 4 && m.to == 6)
            {
                clear_bit(pieceBB[WR], 5);
                set_bit(pieceBB[WR], 7);
            }
            else if (m.from == 4 && m.to == 2)
            {
                clear_bit(pieceBB[WR], 3);
                set_bit(pieceBB[WR], 0);
            }
        }
        else
        {
            if (m.from == 60 && m.to == 62)
            {
                clear_bit(pieceBB[BR], 61);
                set_bit(pieceBB[BR], 63);
            }
            else if (m.from == 60 && m.to == 58)
            {
                clear_bit(pieceBB[BR], 59);
                set_bit(pieceBB[BR], 56);
            }
        }
    }

    occupancy[WHITE] = occupancy[BLACK] = 0ULL;
    for (int i = 0; i < 6; ++i)
        occupancy[WHITE] |= pieceBB[i];
    for (int i = 6; i < 12; ++i)
        occupancy[BLACK] |= pieceBB[i];
    allPieces = occupancy[WHITE] | occupancy[BLACK];

    enPassantSquare = u.prevEnPassant;
    canCastleKingside[WHITE] = u.prevCastleK[WHITE];
    canCastleKingside[BLACK] = u.prevCastleK[BLACK];
    canCastleQueenside[WHITE] = u.prevCastleQ[WHITE];
    canCastleQueenside[BLACK] = u.prevCastleQ[BLACK];
    kingSquare[WHITE] = u.prevKingSquare[WHITE];
    kingSquare[BLACK] = u.prevKingSquare[BLACK];
    halfmoveClock = u.prevHalfmoveClock;
    fullmoveNumber = u.prevFullmoveNumber;
    hashKey = u.prevHash;
    nnueAccumulator = u.prevAccumulator;
}

bool Position::isSquareAttacked(int sq, Side bySide) const
{
    Bitboard occ = allPieces;

    // Pawn attacks
    if (bySide == WHITE)
    {
        Bitboard attacks = south(east(1ULL << sq)) | south(west(1ULL << sq));
        if (attacks & pieceBB[WP])
            return true;
    }
    else
    {
        Bitboard attacks = north(east(1ULL << sq)) | north(west(1ULL << sq));
        if (attacks & pieceBB[BP])
            return true;
    }

    // Knight attacks
    if (KNIGHT_ATTACKS[sq] & pieceBB[(bySide == WHITE) ? WN : BN])
        return true;

    // King attacks
    if (KING_ATTACKS[sq] & pieceBB[(bySide == WHITE) ? WK : BK])
        return true;

    // Sliding attacks (bishop/queen diagonals)
    Bitboard bishops = pieceBB[(bySide == WHITE) ? WB : BB] | pieceBB[(bySide == WHITE) ? WQ : BQ];
    Bitboard rooks = pieceBB[(bySide == WHITE) ? WR : BR] | pieceBB[(bySide == WHITE) ? WQ : BQ];
    if (bishopAttacks(sq, occ) & bishops)
        return true;
    if (rookAttacks(sq, occ) & rooks)
        return true;

    return false;
}

bool Position::isMoveLegal(const Move &m)
{
    Side us = sideToMove;
    Side them = (us == WHITE ? BLACK : WHITE);

    if (m.isCastle)
    {
        int startK = kingSquare[us];
        if (startK == -1)
        {
            Bitboard kbb = pieceBB[(us == WHITE) ? WK : BK];
            startK = kbb ? lsb(kbb) : -1;
        }
        if (startK != -1 && isSquareAttacked(startK, them))
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
        if (f != -1 && (isSquareAttacked(f, them) || isSquareAttacked(g, them)))
            return false;
    }

    Undo u;
    makeMove(m, u);
    int kingSq = kingSquare[us];
    bool ok = (kingSq != -1) && !isSquareAttacked(kingSq, them);
    unmakeMove(m, u);
    return ok;
}

int Position::pieceIndexAt(int sq) const
{
    for (int i = 0; i < 12; ++i)
    {
        if (get_bit(pieceBB[i], sq))
            return i;
    }
    return -1;
}

int Position::castlingRightsMask() const
{
    int rights = 0;
    if (canCastleKingside[WHITE])
        rights |= 1;
    if (canCastleQueenside[WHITE])
        rights |= 2;
    if (canCastleKingside[BLACK])
        rights |= 4;
    if (canCastleQueenside[BLACK])
        rights |= 8;
    return rights;
}
