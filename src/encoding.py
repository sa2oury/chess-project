import numpy as np
import chess


# =========================================================
# MOVE ENCODING & ACTION SPACE (4,864 MOVES)
# =========================================================
KNIGHT_MOVES = [
    (2, 1), (1, 2), (-1, 2), (-2, 1),
    (-2, -1), (-1, -2), (1, -2), (2, -1)
]

DIRECTIONS = [
    (1, 0), (1, 1), (0, 1), (-1, 1),
    (-1, 0), (-1, -1), (0, -1), (1, -1)
]

PROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]


def move_to_index(move: chess.Move) -> int:
    """
    Encodes a python-chess Move into an action index in [0, 4863].
    76 planes x 64 squares:
      - Planes 0..55:  8 ray directions x 7 distances = 56 queen-like moves
      - Planes 56..63: 8 knight move offsets = 8 planes
      - Planes 64..75: 3 directions (ahead, capture left, capture right) x 4 promotion pieces = 12 planes
    """
    from_sq = move.from_square
    to_sq = move.to_square

    r0, c0 = divmod(from_sq, 8)
    r1, c1 = divmod(to_sq, 8)

    dr = r1 - r0
    dc = c1 - c0

    # 1. Knight moves
    if (dr, dc) in KNIGHT_MOVES:
        plane = 56 + KNIGHT_MOVES.index((dr, dc))
        return plane * 64 + from_sq

    # 2. Underpromotions and Queen promotions
    if move.promotion is not None:
        if dc == 0:
            dir_idx = 0
        elif dc == -1:
            dir_idx = 1
        else:
            dir_idx = 2

        piece_idx = PROMO_PIECES.index(move.promotion)
        plane = 64 + dir_idx * 4 + piece_idx
        return plane * 64 + from_sq

    # 3. Sliding / Queen-like moves
    for dir_idx, (drd, dcd) in enumerate(DIRECTIONS):
        for dist in range(1, 8):
            if dr == drd * dist and dc == dcd * dist:
                plane = dir_idx * 7 + (dist - 1)
                return plane * 64 + from_sq

    raise ValueError(f"Could not encode move: {move.uci()}")


def index_to_move(idx: int, board: chess.Board = None) -> chess.Move:
    """
    Decodes an action index in [0, 4863] back into a python-chess Move.
    """
    if not (0 <= idx < 4864):
        raise ValueError(f"Action index {idx} out of range [0, 4863]")

    plane = idx // 64
    from_sq = idx % 64
    r0, c0 = divmod(from_sq, 8)

    # 1. Sliding moves (planes 0..55)
    if plane < 56:
        dir_idx = plane // 7
        dist = (plane % 7) + 1
        drd, dcd = DIRECTIONS[dir_idx]
        r1 = r0 + drd * dist
        c1 = c0 + dcd * dist
        if not (0 <= r1 < 8 and 0 <= c1 < 8):
            raise ValueError(f"Decoded square out of bounds for index {idx}")
        to_sq = r1 * 8 + c1
        return chess.Move(from_sq, to_sq)

    # 2. Knight moves (planes 56..63)
    if plane < 64:
        dr, dc = KNIGHT_MOVES[plane - 56]
        r1 = r0 + dr
        c1 = c0 + dc
        if not (0 <= r1 < 8 and 0 <= c1 < 8):
            raise ValueError(f"Decoded knight square out of bounds for index {idx}")
        to_sq = r1 * 8 + c1
        return chess.Move(from_sq, to_sq)

    # 3. Promotions (planes 64..75)
    promo_plane = plane - 64
    dir_idx = promo_plane // 4
    piece_idx = promo_plane % 4
    promotion_piece = PROMO_PIECES[piece_idx]

    dc = 0 if dir_idx == 0 else (-1 if dir_idx == 1 else 1)
    dr = 1  # From white perspective, promotion advances rank +1
    r1 = r0 + dr
    c1 = c0 + dc
    if not (0 <= r1 < 8 and 0 <= c1 < 8):
        raise ValueError(f"Decoded promo square out of bounds for index {idx}")
    to_sq = r1 * 8 + c1
    return chess.Move(from_sq, to_sq, promotion=promotion_piece)


def mirror_move_if_needed(move: chess.Move, is_black: bool) -> chess.Move:
    """Mirrors a move vertically if evaluating from Black's perspective."""
    if not is_black:
        return move
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        move.promotion
    )


# =========================================================
# 1. LEGACY 12-PLANE BOARD REPRESENTATION (v1.2, v1.3, v1.4, blackie)
# =========================================================
def board_to_tensor_legacy(board: chess.Board) -> np.ndarray:
    """
    Produces the 12-channel (12, 8, 8) float32 representation
    matching the existing trained models and archive shards.
      - Planes 0..5: White (P, N, B, R, Q, K)
      - Planes 6..11: Black (P, N, B, R, Q, K)
    """
    tensor = np.zeros((12, 64), dtype=np.float32)

    for sq, piece in board.piece_map().items():
        layer = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
        tensor[layer][sq] = 1.0

    tensor = tensor.reshape(12, 8, 8)
    return np.flip(tensor, axis=1).copy()


# =========================================================
# 2. ENHANCED 18-PLANE BOARD REPRESENTATION (Area for Enhancement)
# =========================================================
def board_to_tensor_v2(board: chess.Board) -> np.ndarray:
    """
    Enhanced 18-channel (18, 8, 8) representation incorporating complete rules:
      - Planes 0..5:  Current player pieces (P, N, B, R, Q, K)
      - Planes 6..11: Opponent pieces (P, N, B, R, Q, K)
      - Plane 12: Current player Kingside castling right (all 1.0 or 0.0)
      - Plane 13: Current player Queenside castling right (all 1.0 or 0.0)
      - Plane 14: Opponent Kingside castling right (all 1.0 or 0.0)
      - Plane 15: Opponent Queenside castling right (all 1.0 or 0.0)
      - Plane 16: En passant target square (1.0 at square, 0.0 elsewhere)
      - Plane 17: Halfmove clock normalized (halfmove_clock / 100.0)
    Note: Always normalized so current player advances towards row 7.
    """
    is_black = (board.turn == chess.BLACK)
    eval_board = board.mirror() if is_black else board

    tensor = np.zeros((18, 8, 8), dtype=np.float32)

    # Current player pieces (always white in mirrored eval_board)
    for sq, piece in eval_board.piece_map().items():
        layer = (piece.piece_type - 1) if piece.color == chess.WHITE else (6 + piece.piece_type - 1)
        r, c = divmod(sq, 8)
        tensor[layer, 7 - r, c] = 1.0

    # Castling rights (planes 12-15)
    if eval_board.has_kingside_castling_rights(chess.WHITE):
        tensor[12, :, :] = 1.0
    if eval_board.has_queenside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1.0
    if eval_board.has_kingside_castling_rights(chess.BLACK):
        tensor[14, :, :] = 1.0
    if eval_board.has_queenside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1.0

    # En passant square (plane 16)
    if eval_board.ep_square is not None:
        ep_r, ep_c = divmod(eval_board.ep_square, 8)
        tensor[16, 7 - ep_r, ep_c] = 1.0

    # Halfmove clock (plane 17)
    tensor[17, :, :] = min(1.0, eval_board.halfmove_clock / 100.0)

    return tensor


# =========================================================
# INPUT PREPARATION DISPATCHER
# =========================================================
def prepare_model_input(board: chess.Board, mode: str = "legacy") -> tuple[np.ndarray, bool]:
    """
    Prepares tensor input for model forward pass.
    Returns:
      (input_tensor, is_black)
    """
    is_black = (board.turn == chess.BLACK)
    if mode == "v2":
        x = board_to_tensor_v2(board)
    else:
        eval_board = board.mirror() if is_black else board
        x = board_to_tensor_legacy(eval_board)
    return x.astype(np.float32), is_black

