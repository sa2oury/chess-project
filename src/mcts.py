import math
import numpy as np
import torch
import torch.nn.functional as F
import chess

from .encoding import prepare_model_input, mirror_move_if_needed, move_to_index
from .book import get_opening_move
from .tactics import tactical_filter_best_move, find_immediate_mate_in_1


# =========================================================
# DIRICHLET NOISE
# =========================================================
def add_dirichlet_noise(priors: np.ndarray, alpha: float = 0.3, epsilon: float = 0.25) -> np.ndarray:
    """Adds Dirichlet exploration noise to root node priors."""
    noise = np.random.dirichlet([alpha] * len(priors))
    return (1.0 - epsilon) * priors + epsilon * noise


# =========================================================
# TERMINAL EVALUATION
# =========================================================
def terminal_value(board: chess.Board) -> float:
    """
    Returns terminal game outcome from the perspective of the player to move.
    +1.0 if player to move is won, -1.0 if player to move is checkmated, 0.0 for draw.
    """
    if board.is_checkmate():
        # The player whose turn it is has no legal moves and is in check -> checkmated (-1.0)
        return -1.0
    return 0.0


# =========================================================
# MCTS NODE
# =========================================================
class MCTSNode:
    """A node in the Monte Carlo Search Tree."""
    def __init__(self, board: chess.Board, parent=None, prior: float = 0.0, move: chess.Move = None):
        self.board = board
        self.parent = parent
        self.prior = prior
        self.move = move

        self.children: dict[chess.Move, "MCTSNode"] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


# =========================================================
# PUCT SELECTION FORMULA
# =========================================================
def puct_score(parent: MCTSNode, child: MCTSNode, c_puct: float = 1.5) -> float:
    """
    Standard AlphaZero PUCT formula:
    Q(s, a) + U(s, a) where U(s, a) = c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
    Note: child.q_value is from the child's perspective (opponent), so parent sees -child.q_value.
    """
    q = -child.q_value
    parent_visits = max(1, parent.visit_count)
    u = c_puct * child.prior * math.sqrt(parent_visits) / (1.0 + child.visit_count)
    return q + u


def select_child(node: MCTSNode, c_puct: float = 1.5) -> tuple[chess.Move, MCTSNode]:
    best_move = None
    best_child = None
    best_score = -float("inf")

    for move, child in node.children.items():
        score = puct_score(node, child, c_puct)
        if score > best_score:
            best_score = score
            best_move = move
            best_child = child

    return best_move, best_child


# =========================================================
# NODE EXPANSION & EVALUATION
# =========================================================
def expand_node(
    node: MCTSNode,
    model: torch.nn.Module,
    device: str,
    policy_size: int = 4864,
    add_noise: bool = False,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
    transposition_cache: dict = None,
) -> float:
    board = node.board

    if board.is_game_over() or board.can_claim_draw():
        node.is_expanded = True
        return float(terminal_value(board))

    # Transposition cache lookup to avoid redundant NN passes
    fen_key = board.fen()
    if transposition_cache is not None and fen_key in transposition_cache:
        policy_logits_np, value = transposition_cache[fen_key]
    else:
        x_np, is_black = prepare_model_input(board, mode="legacy")

        # Fast ONNX pathway (Zero tensor overhead, direct NumPy)
        if getattr(model, "is_onnx", False):
            x_4d = np.ascontiguousarray(x_np[np.newaxis, ...], dtype=np.float32)
            p_out, v_out = model(x_4d)
            policy_logits_np = p_out[0]
            value = float(v_out.item() if hasattr(v_out, "item") else v_out[0])
        else:
            x = torch.from_numpy(x_np).unsqueeze(0).to(device)
            with torch.no_grad():
                policy_logits, value_tensor = model(x)
                policy_logits_np = policy_logits[0].cpu().numpy()
                value = float(value_tensor.item())

        if transposition_cache is not None and len(transposition_cache) < 50000:
            transposition_cache[fen_key] = (policy_logits_np, value)

    is_black = (board.turn == chess.BLACK)
    legal_moves = list(board.legal_moves)
    legal_indices = []
    filtered_moves = []

    for move in legal_moves:
        try:
            enc_move = mirror_move_if_needed(move, is_black)
            idx = move_to_index(enc_move)
            if 0 <= idx < policy_size:
                legal_indices.append(idx)
                filtered_moves.append(move)
        except Exception:
            continue

    if len(filtered_moves) == 0:
        node.is_expanded = True
        return value

    legal_logits = policy_logits_np[legal_indices]
    # Numerically stable softmax
    max_logit = np.max(legal_logits)
    exp_logits = np.exp(legal_logits - max_logit)
    legal_probs = exp_logits / np.sum(exp_logits)

    if add_noise:
        legal_probs = add_dirichlet_noise(
            legal_probs,
            alpha=dirichlet_alpha,
            epsilon=dirichlet_epsilon
        )

    for move, prob in zip(filtered_moves, legal_probs):
        next_board = board.copy()
        next_board.push(move)
        node.children[move] = MCTSNode(
            board=next_board,
            parent=node,
            prior=float(prob),
            move=move
        )

    node.is_expanded = True
    return value


# =========================================================
# BACKPROPAGATION
# =========================================================
def backup(search_path: list[MCTSNode], value: float):
    """Backpropagates leaf value up the search tree, negating at each alternating ply."""
    for node in reversed(search_path):
        node.visit_count += 1
        node.value_sum += value
        value = -value


# =========================================================
# MCTS SEARCH
# =========================================================
def run_mcts(
    root_board: chess.Board,
    model: torch.nn.Module,
    device: str,
    num_simulations: int = 100,
    c_puct: float = 1.5,
    policy_size: int = 4864,
    add_root_noise: bool = True,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
    transposition_cache: dict = None,
) -> MCTSNode:
    root = MCTSNode(root_board.copy())

    expand_node(
        root,
        model,
        device,
        policy_size=policy_size,
        add_noise=add_root_noise,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
        transposition_cache=transposition_cache,
    )

    for _ in range(num_simulations):
        node = root
        search_path = [node]

        while node.is_expanded and node.children:
            best_move, best_child = select_child(node, c_puct=c_puct)
            node = best_child
            search_path.append(node)

        if node.board.is_game_over() or node.board.can_claim_draw():
            value = float(terminal_value(node.board))
        else:
            value = expand_node(
                node,
                model=model,
                device=device,
                policy_size=policy_size,
                add_noise=False,
                transposition_cache=transposition_cache,
            )

        backup(search_path, value)

    return root


# =========================================================
# MOVE SELECTION
# =========================================================
def choose_move_with_mcts(
    board: chess.Board,
    model: torch.nn.Module,
    device: str,
    num_simulations: int = 200,
    c_puct: float = 1.5,
    policy_size: int = 4864,
    temperature: float = 0.0,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.0,
    add_root_noise: bool = False,
    return_debug: bool = False,
    transposition_cache: dict = None,
):
    """Executes MCTS and selects a move based on visit count distribution."""
    root = run_mcts(
        root_board=board,
        model=model,
        device=device,
        num_simulations=num_simulations,
        c_puct=c_puct,
        policy_size=policy_size,
        add_root_noise=add_root_noise,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
        transposition_cache=transposition_cache,
    )

    legal_moves = []
    move_visits = []

    for move, child in root.children.items():
        legal_moves.append(move)
        move_visits.append(child.visit_count)

    move_visits = np.array(move_visits, dtype=np.float32)

    if len(legal_moves) == 0:
        if return_debug:
            return None, root, {"root_value": 0.0, "top_moves": []}
        return None, root

    if temperature <= 1e-4:
        best_idx = int(np.argmax(move_visits))
        chosen_move = legal_moves[best_idx]
    else:
        # Temperature scaling with clipping to prevent numerical overflow
        safe_temp = max(0.05, float(temperature))
        visits_scaled = np.log(move_visits + 1e-8) / safe_temp
        visits_scaled -= np.max(visits_scaled)
        probs = np.exp(visits_scaled)
        probs /= np.sum(probs)
        chosen_move = np.random.choice(legal_moves, p=probs)

    if return_debug:
        scored_moves = []
        for move, child in root.children.items():
            u = c_puct * child.prior * math.sqrt(max(1, root.visit_count)) / (1 + child.visit_count)
            q_from_parent = -child.q_value
            score = q_from_parent + u
            prob = child.visit_count / max(1, root.visit_count)

            scored_moves.append({
                "move": move,
                "uci": move.uci(),
                "san": board.san(move),
                "visits": child.visit_count,
                "prior": float(child.prior),
                "q": float(q_from_parent),
                "prob": float(prob),
                "puct": float(score)
            })

        scored_moves.sort(key=lambda x: x["visits"], reverse=True)
        debug_info = {
            "root_value": float(root.q_value),
            "chosen_move": chosen_move.uci(),
            "top_moves": scored_moves
        }
        return chosen_move, root, debug_info

    return chosen_move, root


# =========================================================
# GET BEST MOVE (Missing in self-play.ipynb)
# =========================================================
def get_best_move(
    board: chess.Board,
    model: torch.nn.Module,
    device: str,
    num_simulations: int = 150,
    depth: int = None,
    resign_threshold: float = -0.95,
    temperature: float = 0.0,
    transposition_cache: dict = None,
    use_book: bool = True,
) -> tuple[chess.Move | None, float, list, bool]:
    """
    Evaluates board and returns the best move.
    1. Checks for immediate forced mate-in-1 (0 ms).
    2. Checks opening book for theory moves (0 ms).
    3. Runs high-speed MCTS search.
    4. Applies tactical sanity check to prevent blunders.
    """
    if board.is_game_over() or len(list(board.legal_moves)) == 0:
        return None, 0.0, [], False

    # 1. Immediate forced checkmate check (0 simulations needed!)
    mate_move = find_immediate_mate_in_1(board)
    if mate_move:
        return mate_move, 1.0, [(mate_move, 1.0, 1.0)], False

    # 2. Opening book lookup (play instant theory move if enabled)
    if use_book and board.ply() < 12:
        book_move = get_opening_move(board)
        if book_move:
            return book_move, 0.05, [(book_move, 0.05, 1.0)], False

    # If depth is given instead of simulations, convert depth proportionally
    if depth is not None:
        num_simulations = max(40, depth * 40)

    chosen_move, root, debug = choose_move_with_mcts(
        board=board,
        model=model,
        device=device,
        num_simulations=num_simulations,
        temperature=temperature,
        return_debug=True,
        transposition_cache=transposition_cache,
    )

    scored_moves = []
    for item in debug["top_moves"]:
        scored_moves.append((item["move"], item["q"], item["prob"]))

    # 3. Tactical sanity check (mate-in-1 prevention)
    chosen_move = tactical_filter_best_move(board, chosen_move, scored_moves)

    best_eval = debug["root_value"]
    resigned = (best_eval < resign_threshold) and (root.visit_count > 20)

    return chosen_move, best_eval, scored_moves, resigned

