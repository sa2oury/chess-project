import os
import random
import chess
try:
    import chess.polyglot
    HAS_POLYGLOT = True
except ImportError:
    HAS_POLYGLOT = False


# Curated Opening Repertoire (UCI move sequences from startpos)
CURATED_OPENINGS = [
    # Ruy Lopez (Spanish)
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1"],
    # Italian Game / Giuoco Piano
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "c2c3", "g8f6", "d2d3"],
    # Sicilian Defense (Open / Najdorf)
    ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6"],
    # Sicilian Defense (Classical)
    ["e2e4", "c7c5", "g1f3", "b8c6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "e7e5"],
    # French Defense (Winawer)
    ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "f8b4", "e4e5", "c7c5"],
    # Caro-Kann Defense (Classical)
    ["e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4", "c3e4", "c8f5"],
    # Queen's Gambit Declined
    ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5", "f8e7", "e2e3"],
    # Queen's Gambit Accepted
    ["d2d4", "d7d5", "c2c4", "d5c4", "g1f3", "g8f6", "e2e3", "e7e6", "f1c4"],
    # King's Indian Defense
    ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6", "g1f3"],
    # Nimzo-Indian Defense
    ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "e2e3", "e8g8"],
    # English Opening
    ["c2c4", "e7e5", "b1c3", "g8f6", "g1f3", "b8c6", "g2g3", "f8b4"],
    # Reti Opening
    ["g1f3", "d7d5", "c2c4", "c7c6", "b2b3", "g8f6", "c1b2", "c8f5"],
]


def find_polyglot_book() -> str | None:
    """Finds standard polyglot book if present in books/ or models/."""
    candidates = [
        "books/opening_book.bin",
        "books/book.bin",
        "models/opening_book.bin",
        "models/book.bin",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def get_opening_move(board: chess.Board, max_ply: int = 12) -> chess.Move | None:
    """
    Returns a book move for the board position if available.
    1. Checks Polyglot .bin file if available.
    2. Otherwise checks built-in curated Grandmaster lines.
    Returns None if the board is out of book or past max_ply.
    """
    if board.ply() >= max_ply:
        return None

    # 1. Polyglot book reader
    if HAS_POLYGLOT:
        book_path = find_polyglot_book()
        if book_path:
            try:
                with chess.polyglot.open_reader(book_path) as reader:
                    entry = reader.weighted_choice(board)
                    if entry:
                        return entry.move
            except Exception:
                pass

    # 2. Curated openings tree
    current_moves = [m.uci() for m in board.move_stack]
    matching_continuations = []

    for line in CURATED_OPENINGS:
        if len(line) > len(current_moves):
            # Check if line matches history so far
            if line[:len(current_moves)] == current_moves:
                next_uci = line[len(current_moves)]
                try:
                    mv = chess.Move.from_uci(next_uci)
                    if mv in board.legal_moves:
                        matching_continuations.append(mv)
                except Exception:
                    pass

    if matching_continuations:
        # Choose randomly among theory continuations for variety
        return random.choice(matching_continuations)

    return None

