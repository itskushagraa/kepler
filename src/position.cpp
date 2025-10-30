#include "position.hpp"
#include "movegen.hpp"

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
    int halfmove, fullmove;
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
    Side us = sideToMove;
    Side them = (us == WHITE ? BLACK : WHITE);
    int oldEPSq = enPassantSquare;

    // 1. Remove our piece from the origin and move it.
    int movedIndex = -1;
    for (int i = (us == WHITE ? 0 : 6); i < (us == WHITE ? 6 : 12); ++i)
    {
        if (get_bit(pieceBB[i], m.from))
        {
            movedIndex = i;
            clear_bit(pieceBB[i], m.from);
            set_bit(pieceBB[i], m.to);
            break;
        }
    }

    // 2. If capture, erase opponent piece at destination.
    if (m.isCapture)
    {
        for (int i = (them == WHITE ? 0 : 6); i < (them == WHITE ? 6 : 12); ++i)
            clear_bit(pieceBB[i], m.to);
    }

    // 3. Promotion — only for pawns that reached last rank.
    if (m.isPromotion && movedIndex >= 0)
    {
        // clear pawn we just moved
        clear_bit(pieceBB[movedIndex], m.to);
        // add a queen of the same color
        int qIndex = (us == WHITE ? WQ : BQ);
        set_bit(pieceBB[qIndex], m.to);
    }
    
    // 3.5 Castling rook shift if the moved piece was a king and moved two squares
    if (movedIndex == WK)
    {
        // O-O e1->g1 : move rook h1->f1
        if (m.from == 4 && m.to == 6)
        {
            clear_bit(pieceBB[WR], 7);
            set_bit(pieceBB[WR], 5);
        }
        // O-O-O e1->c1 : move rook a1->d1
        else if (m.from == 4 && m.to == 2)
        {
            clear_bit(pieceBB[WR], 0);
            set_bit(pieceBB[WR], 3);
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
        }
        // O-O-O e8->c8 : rook a8->d8
        else if (m.from == 60 && m.to == 58)
        {
            clear_bit(pieceBB[BR], 56);
            set_bit(pieceBB[BR], 59);
        }
        kingSquare[BLACK] = m.to;
    }

    // 3.6 En-passant capture: if pawn moved diagonally onto the *old* ep square, remove the pawn behind
    // Detect a diagonal pawn move landing on oldEPSq
    if (oldEPSq != -1 && m.isCapture && m.to == oldEPSq && movedIndex == (us == WHITE ? WP : BP))
    {
        // white captured upward, black downward
        int capSq = (us == WHITE ? m.to - 8 : m.to + 8);
        int victim = (us == WHITE ? BP : WP);
        clear_bit(pieceBB[victim], capSq);
    }

    // 3.7 En-passant target refresh: reset then set only for double push
    enPassantSquare = -1;
    if (movedIndex == WP && (m.from / 8 == 1) && (m.to / 8 == 3))
        enPassantSquare = m.from + 8;
    else if (movedIndex == BP && (m.from / 8 == 6) && (m.to / 8 == 4))
        enPassantSquare = m.from - 8;

    // 3.8 Strip castling rights when king/rooks move or those squares are captured
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

    // 4. Recompute occupancy.
    occupancy[WHITE] = occupancy[BLACK] = 0ULL;
    for (int i = 0; i < 6; ++i)
        occupancy[WHITE] |= pieceBB[i];
    for (int i = 6; i < 12; ++i)
        occupancy[BLACK] |= pieceBB[i];
    allPieces = occupancy[WHITE] | occupancy[BLACK];

    // 5. Flip side
    sideToMove = them;
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
