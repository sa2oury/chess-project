import chess


def find_immediate_mate_in_1(board: chess.Board) -> chess.Move | None:
    """
    Scans legal moves for an immediate checkmate.
    Finds forced wins instantly with zero simulation overhead.
    """
    for move in board.legal_moves:
        if board.gives_check(move):
            board.push(move)
            is_mate = board.is_checkmate()
            board.pop()
            if is_mate:
                return move
    return None


def allows_opponent_mate_in_1(board: chess.Board, move: chess.Move) -> bool:
    """Checks whether playing `move` gives opponent an immediate mate-in-1."""
    board.push(move)
    try:
        for opp_m in board.legal_moves:
            if board.gives_check(opp_m):
                board.push(opp_m)
                try:
                    if board.is_checkmate():
                        return True
                finally:
                    board.pop()
        return False
    finally:
        board.pop()


def tactical_filter_best_move(
    board: chess.Board,
    chosen_move: chess.Move | None,
    candidate_moves: list[tuple[chess.Move, float, float]]
) -> chess.Move | None:
    """
    Tactical post-processor:
    1. Checks if an immediate mate-in-1 is available.
    2. If the chosen move allows immediate mate-in-1, picks the best candidate that avoids it.
    """
    # 1. Immediate forced checkmate
    mate_move = find_immediate_mate_in_1(board)
    if mate_move:
        return mate_move

    if chosen_move is None:
        return None

    # 2. Check if chosen move blunders into mate-in-1
    if allows_opponent_mate_in_1(board, chosen_move):
        # Look for a candidate move that defends against mate
        for move, _, _ in candidate_moves:
            if not allows_opponent_mate_in_1(board, move):
                return move

    return chosen_move

