#!/usr/bin/env python3
"""
Model-vs-Model Tournament & Benchmarking Script
Pit two AlphaZero / ChessResNet checkpoints against each other
to measure win rates, draw rates, and relative Elo strength.
"""

import math
import argparse
import chess

from src.models import load_model
from src.mcts import choose_move_with_mcts


def play_one_game(
    white_model,
    black_model,
    device,
    simulations=100,
    max_moves=200,
    temperature=0.0
):
    board = chess.Board()
    move_count = 0

    while not board.is_game_over(claim_draw=True) and move_count < max_moves:
        curr_model = white_model if board.turn == chess.WHITE else black_model
        move, root = choose_move_with_mcts(
            board=board,
            model=curr_model,
            device=device,
            num_simulations=simulations,
            temperature=temperature
        )
        if move is None:
            break
        board.push(move)
        move_count += 1

    res = board.result(claim_draw=True)
    return res, move_count


def calculate_elo(score_ratio):
    if score_ratio <= 0.0:
        return -800.0
    elif score_ratio >= 1.0:
        return 800.0
    return -400.0 * math.log10((1.0 / score_ratio) - 1.0)


def main():
    parser = argparse.ArgumentParser(description="AlphaZero Chess Tournament Evaluator")
    parser.add_argument("--model1", type=str, required=True, help="Path to checkpoint 1")
    parser.add_argument("--model2", type=str, required=True, help="Path to checkpoint 2")
    parser.add_argument("--name1", type=str, default="Model_A", help="Name for Model 1")
    parser.add_argument("--name2", type=str, default="Model_B", help="Name for Model 2")
    parser.add_argument("--games", type=int, default=10, help="Number of games to play")
    parser.add_argument("--sims", type=int, default=60, help="MCTS simulations per move")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    print(f"[*] Loading {args.name1}: {args.model1}")
    m1 = load_model(args.model1, device=args.device)

    print(f"[*] Loading {args.name2}: {args.model2}")
    m2 = load_model(args.model2, device=args.device)

    m1_score = 0.0
    m2_score = 0.0
    draws = 0
    total_moves = []

    print(f"\n[*] Starting {args.games}-game tournament ({args.sims} sims/move)...\n" + "-" * 55)

    for g in range(1, args.games + 1):
        # Alternate colors: Odd games M1 is White, Even games M2 is White
        if g % 2 == 1:
            white_m, black_m = m1, m2
            white_name, black_name = args.name1, args.name2
        else:
            white_m, black_m = m2, m1
            white_name, black_name = args.name2, args.name1

        print(f"Game {g}/{args.games}: {white_name} (White) vs {black_name} (Black)...", end="", flush=True)
        res, plies = play_one_game(white_m, black_m, device=args.device, simulations=args.sims)
        total_moves.append(plies)

        if res == "1-0":
            winner = white_name
            if white_m is m1:
                m1_score += 1.0
            else:
                m2_score += 1.0
        elif res == "0-1":
            winner = black_name
            if black_m is m1:
                m1_score += 1.0
            else:
                m2_score += 1.0
        else:
            winner = "Draw"
            m1_score += 0.5
            m2_score += 0.5
            draws += 1

        print(f" Result: {res} ({winner}) in {plies} plies")

    m1_ratio = m1_score / args.games
    elo_diff = calculate_elo(m1_ratio)

    print("\n" + "=" * 55)
    print("TOURNAMENT RESULTS")
    print("=" * 55)
    print(f"{args.name1}: {m1_score:.1f} / {args.games} (Win Rate: {m1_ratio*100:.1f}%)")
    print(f"{args.name2}: {m2_score:.1f} / {args.games} (Win Rate: {(1.0-m1_ratio)*100:.1f}%)")
    print(f"Draws: {draws} ({(draws/args.games)*100:.1f}%)")
    print(f"Estimated Elo Advantage ({args.name1} vs {args.name2}): {elo_diff:+.1f}")
    print(f"Average Game Length: {sum(total_moves)/len(total_moves):.1f} plies")
    print("=" * 55)


if __name__ == "__main__":
    main()

