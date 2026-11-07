#!/usr/bin/env python3
"""
Universal Chess Interface (UCI) Protocol Wrapper
Allows playing with the AlphaZero model in standard GUIs:
Arena, Cutechess, ChessBase, BanksiaGUI, Lichess-bot, etc.
"""

import sys
import os
import torch
import chess

from src.models import load_model
from src.mcts import get_best_move


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    default_model_path = "models/chess_v1.4.onnx"
    if not os.path.exists(default_model_path):
        default_model_path = "models/chess_v1.4_deploy.pth"
    if not os.path.exists(default_model_path):
        default_model_path = "ver1.4/chess ver1.4.pth"

    model = None
    board = chess.Board()
    simulations = 100
    temperature = 0.0
    cache = {}

    def log(msg):
        sys.stdout.write(msg + "\n")
        sys.stdout.flush()

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            cmd = parts[0]

            if cmd == "uci":
                log("id name AlphaZero-Chess")
                log("id author Deep Residual Engine")
                log("option name Simulations type spin default 100 min 10 max 2000")
                log("option name Temperature type spin default 0 min 0 max 100")
                log(f"option name ModelPath type string default {default_model_path}")
                log("uciok")

            elif cmd == "isready":
                if model is None and os.path.exists(default_model_path):
                    try:
                        model = load_model(default_model_path, device=device)
                    except Exception as e:
                        sys.stderr.write(f"Failed to load default model: {e}\n")
                log("readyok")

            elif cmd == "setoption":
                # setoption name <Name> value <Value>
                if len(parts) >= 5 and parts[1].lower() == "name":
                    opt_name = parts[2].lower()
                    opt_val = parts[4]
                    if opt_name == "simulations":
                        simulations = int(opt_val)
                    elif opt_name == "temperature":
                        temperature = float(opt_val) / 100.0
                    elif opt_name == "modelpath":
                        default_model_path = opt_val
                        if os.path.exists(default_model_path):
                            model = load_model(default_model_path, device=device)

            elif cmd == "ucinewgame":
                board = chess.Board()
                cache.clear()

            elif cmd == "position":
                # position [startpos | fen <fen>] [moves <m1> ...]
                if len(parts) > 1:
                    if parts[1] == "startpos":
                        board = chess.Board()
                        moves_idx = 2
                    elif parts[1] == "fen":
                        fen_parts = []
                        moves_idx = 2
                        while moves_idx < len(parts) and parts[moves_idx] != "moves":
                            fen_parts.append(parts[moves_idx])
                            moves_idx += 1
                        board = chess.Board(" ".join(fen_parts))

                    if moves_idx < len(parts) and parts[moves_idx] == "moves":
                        for mv_str in parts[moves_idx + 1:]:
                            try:
                                mv = chess.Move.from_uci(mv_str)
                                if mv in board.legal_moves:
                                    board.push(mv)
                            except Exception:
                                pass

            elif cmd == "go":
                if model is None:
                    if os.path.exists(default_model_path):
                        model = load_model(default_model_path, device=device)
                    else:
                        legal = list(board.legal_moves)
                        if legal:
                            log(f"bestmove {legal[0].uci()}")
                        continue

                # Parse search limits if given
                search_sims = simulations
                if "depth" in parts:
                    d_idx = parts.index("depth")
                    if d_idx + 1 < len(parts):
                        search_sims = max(30, int(parts[d_idx + 1]) * 30)

                best_move, best_eval, scored_moves, resigned = get_best_move(
                    board=board,
                    model=model,
                    device=device,
                    num_simulations=search_sims,
                    temperature=temperature,
                    transposition_cache=cache
                )

                if best_move is None:
                    legal = list(board.legal_moves)
                    if legal:
                        best_move = legal[0]

                # info score in centipawns
                cp_score = int(best_eval * 300)
                if best_move:
                    log(f"info depth 10 score cp {cp_score} pv {best_move.uci()}")
                    log(f"bestmove {best_move.uci()}")
                else:
                    log("bestmove 0000")

            elif cmd == "quit":
                break

        except Exception as e:
            sys.stderr.write(f"Error in UCI loop: {e}\n")


if __name__ == "__main__":
    main()

