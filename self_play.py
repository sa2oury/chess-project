#!/usr/bin/env python3
"""
Self-Play Game Generator for AlphaZero Chess
Generates reinforcement learning training data by having the model play against itself.
Saves .npz shards containing:
  - 'features': (N, 12, 8, 8) float32 board tensors
  - 'policy_labels': (N,) int64 selected move indices (or soft distribution)
  - 'value_labels': (N,) float32 terminal outcomes from current player's perspective

Usage:
  python self_play.py --num-games 10 --sims 60 --model models/chess_v1.4.onnx --output-dir data/self_play
"""

import os
import time
import argparse
import numpy as np
import chess
import chess.pgn
import torch

from src.models import load_model
from src.mcts import choose_move_with_mcts
from src.encoding import prepare_model_input, move_to_index, mirror_move_if_needed


def play_one_game(
    model,
    device: str,
    num_simulations: int = 60,
    temperature_drop_ply: int = 15,
    max_plies: int = 200,
    c_puct: float = 1.5,
    transposition_cache: dict = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    """
    Plays a single self-play game to completion.
    Returns:
      (features, policy_labels, value_labels, result, pgn_str)
    """
    board = chess.Board()
    game_history = []  # (feature_tensor, chosen_action_index, player_turn)
    pgn_game = chess.pgn.Game()
    pgn_game.headers["Event"] = "AlphaZero Self-Play"
    pgn_game.headers["Site"] = "Local Engine"
    pgn_game.headers["Date"] = time.strftime("%Y.%m.%d")
    pgn_node = pgn_game

    for ply in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break

        player_turn = board.turn
        feature_tensor, is_black = prepare_model_input(board, mode="legacy")

        # Exploration temperature for opening moves, greedy afterwards
        temp = 1.0 if ply < temperature_drop_ply else 0.05

        chosen_move, root, debug = choose_move_with_mcts(
            board=board,
            model=model,
            device=device,
            num_simulations=num_simulations,
            c_puct=c_puct,
            temperature=temp,
            add_root_noise=(ply < temperature_drop_ply),
            dirichlet_epsilon=0.25 if (ply < temperature_drop_ply) else 0.0,
            return_debug=True,
            transposition_cache=transposition_cache,
        )

        if chosen_move is None or chosen_move not in board.legal_moves:
            break

        # Record action from the canonical white perspective
        enc_move = mirror_move_if_needed(chosen_move, is_black)
        action_idx = move_to_index(enc_move)

        game_history.append((feature_tensor, action_idx, player_turn))
        pgn_node = pgn_node.add_variation(chosen_move)
        board.push(chosen_move)

    # Determine game outcome
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        white_val = 0.0
        result_str = "1/2-1/2"
    elif outcome.winner == chess.WHITE:
        white_val = 1.0
        result_str = "1-0"
    else:
        white_val = -1.0
        result_str = "0-1"

    pgn_game.headers["Result"] = result_str

    # Assign terminal values relative to player to move at each ply
    features = []
    policy_labels = []
    value_labels = []

    for state, act_idx, turn in game_history:
        v = white_val if turn == chess.WHITE else -white_val
        features.append(state)
        policy_labels.append(act_idx)
        value_labels.append(v)

    if len(features) == 0:
        return np.empty((0, 12, 8, 8), dtype=np.float32), np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32), result_str, str(pgn_game)

    features = np.array(features, dtype=np.float32)
    policy_labels = np.array(policy_labels, dtype=np.int64)
    value_labels = np.array(value_labels, dtype=np.float32)

    return features, policy_labels, value_labels, result_str, str(pgn_game)


def main():
    parser = argparse.ArgumentParser(description="AlphaZero Self-Play Data Generator")
    parser.add_argument("--model", type=str, default="models/chess_v1.4.onnx", help="Model path (.onnx or .pth)")
    parser.add_argument("--num-games", type=int, default=10, help="Number of games to generate")
    parser.add_argument("--sims", type=int, default=60, help="MCTS simulations per move")
    parser.add_argument("--temp-ply", type=int, default=15, help="Plies to explore with high temperature")
    parser.add_argument("--output-dir", type=str, default="data/self_play", help="Directory to save .npz files")
    parser.add_argument("--save-pgn", action="store_true", help="Save games as .pgn file")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model}")
    model = load_model(args.model, device=args.device)

    all_features = []
    all_policies = []
    all_values = []
    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    cache = {}

    pgn_lines = []

    print(f"Generating {args.num_games} self-play games ({args.sims} sims/move)...")
    t0 = time.time()

    for g_idx in range(args.num_games):
        t_game = time.time()
        f, p, v, res, pgn = play_one_game(
            model=model,
            device=args.device,
            num_simulations=args.sims,
            temperature_drop_ply=args.temp_ply,
            transposition_cache=cache,
        )

        if len(f) > 0:
            all_features.append(f)
            all_policies.append(p)
            all_values.append(v)
            results[res] = results.get(res, 0) + 1
            pgn_lines.append(pgn + "\n\n")

        elapsed = time.time() - t_game
        total_pos = sum(len(x) for x in all_features)
        print(f"  Game {g_idx+1:3d}/{args.num_games} | Result: {res:7s} | Plies: {len(f):3d} | Total Pos: {total_pos:5d} | Time: {elapsed:.1f}s")

    if all_features:
        features_np = np.concatenate(all_features, axis=0)
        policy_np = np.concatenate(all_policies, axis=0)
        value_np = np.concatenate(all_values, axis=0)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_npz = os.path.join(args.output_dir, f"self_play_{timestamp}_{len(features_np)}pos.npz")
        np.savez_compressed(
            out_npz,
            features=features_np,
            policy_labels=policy_np,
            value_labels=value_np
        )
        print(f"\nSaved {len(features_np)} training positions to: {out_npz}")

        if args.save_pgn:
            out_pgn = os.path.join(args.output_dir, f"self_play_{timestamp}.pgn")
            with open(out_pgn, "w") as f_pgn:
                f_pgn.writelines(pgn_lines)
            print(f"Saved PGN games to: {out_pgn}")

    total_time = time.time() - t0
    print(f"\nSelf-play finished in {total_time:.1f}s | Results: {results}")


if __name__ == "__main__":
    main()
