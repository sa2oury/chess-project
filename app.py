import os
import time
import chess
import chess.pgn
import streamlit as st

from src.models import load_model
from src.mcts import get_best_move
from src.component import interactive_chessboard

# =========================================================
# STREAMLIT CONFIG & CUSTOM CHESS.COM STYLING
# =========================================================
st.set_page_config(
    page_title="AlphaZero Chess - Play Online",
    page_icon="♟️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Dark Theme / Chess.com styling */
    .stApp {
        background: #161512;
        color: #bababa;
    }
    .main-title {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 2px;
        letter-spacing: -0.5px;
    }
    .sub-title {
        color: #81b64c;
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }
    .eval-card {
        background: #262421;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 15px;
        border: 1px solid #363431;
    }
    .player-card {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #21201d;
        padding: 8px 14px;
        border-radius: 6px;
        margin-bottom: 8px;
        border: 1px solid #312e2b;
    }
    .player-name {
        font-weight: 700;
        color: #ffffff;
        font-size: 0.95rem;
    }
    .captured-pieces {
        font-size: 1.15rem;
        letter-spacing: 2px;
    }
    .advantage-badge {
        background: #363431;
        color: #ffffff;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-left: 6px;
    }
    .turn-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 700;
    }
    .turn-human {
        background: rgba(129, 182, 76, 0.2);
        color: #81b64c;
        border: 1px solid #81b64c;
    }
    .turn-engine {
        background: rgba(235, 97, 80, 0.2);
        color: #eb6150;
        border: 1px solid #eb6150;
    }
    .moves-table-container {
        max-height: 220px;
        overflow-y: auto;
        background: #21201d;
        border-radius: 6px;
        padding: 8px;
        border: 1px solid #312e2b;
    }
    .move-row {
        display: flex;
        padding: 3px 6px;
        font-size: 0.88rem;
        font-family: monospace;
    }
    .move-row:nth-child(even) {
        background: #262421;
    }
    .move-num {
        width: 38px;
        color: #6d6b67;
        font-weight: 600;
    }
    .move-white, .move-black {
        width: 80px;
        color: #dedede;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL CACHING (Prioritizes ONNX SIMD Runtime)
# =========================================================
@st.cache_resource(show_spinner="Booting High-Speed ONNX Engine...")
def get_cached_model():
    candidate_paths = [
        "models/chess_v1.4.onnx",
        "models/chess_v1.4_deploy.pth",
        "ver1.4/chess ver1.4.pth",
        "ver1.3/chess ver1.3.pth",
    ]
    selected_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            selected_path = p
            break

    if selected_path is None:
        raise FileNotFoundError("No checkpoint found in models/ or ver1.X/")

    model = load_model(selected_path, device="cpu")
    is_onnx = getattr(model, "is_onnx", False)
    return model, os.path.basename(selected_path), is_onnx


# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================
if "board" not in st.session_state:
    st.session_state.board = chess.Board()
if "move_history" not in st.session_state:
    st.session_state.move_history = []
if "user_color" not in st.session_state:
    st.session_state.user_color = "White"
if "eval_white" not in st.session_state:
    st.session_state.eval_white = 0.0
if "top_moves" not in st.session_state:
    st.session_state.top_moves = []
if "cache" not in st.session_state:
    st.session_state.cache = {}
if "last_move_time" not in st.session_state:
    st.session_state.last_move_time = 0.0
if "last_processed_move" not in st.session_state:
    st.session_state.last_processed_move = None
if "hint" not in st.session_state:
    st.session_state.hint = None
if "resigned" not in st.session_state:
    st.session_state.resigned = False


# =========================================================
# MATERIAL & CAPTURED PIECES CALCULATOR
# =========================================================
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}
PIECE_SYMBOLS_WHITE = {
    chess.PAWN: "♙",
    chess.KNIGHT: "♘",
    chess.BISHOP: "♗",
    chess.ROOK: "♖",
    chess.QUEEN: "♕",
}
PIECE_SYMBOLS_BLACK = {
    chess.PAWN: "♟",
    chess.KNIGHT: "♞",
    chess.BISHOP: "♝",
    chess.ROOK: "♜",
    chess.QUEEN: "♛",
}

def get_material_info(board: chess.Board):
    initial_pieces = {chess.PAWN: 8, chess.KNIGHT: 2, chess.BISHOP: 2, chess.ROOK: 2, chess.QUEEN: 1}

    current_white = {p: 0 for p in initial_pieces}
    current_black = {p: 0 for p in initial_pieces}

    for sq, piece in board.piece_map().items():
        if piece.piece_type in initial_pieces:
            if piece.color == chess.WHITE:
                current_white[piece.piece_type] += 1
            else:
                current_black[piece.piece_type] += 1

    white_captured = []
    black_captured = []

    white_mat = 0
    black_mat = 0

    for pt in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]:
        lost_by_black = initial_pieces[pt] - current_black[pt]
        if lost_by_black > 0:
            white_captured.extend([PIECE_SYMBOLS_BLACK[pt]] * lost_by_black)

        lost_by_white = initial_pieces[pt] - current_white[pt]
        if lost_by_white > 0:
            black_captured.extend([PIECE_SYMBOLS_WHITE[pt]] * lost_by_white)

        white_mat += current_white[pt] * PIECE_VALUES[pt]
        black_mat += current_black[pt] * PIECE_VALUES[pt]

    diff = white_mat - black_mat
    return {
        "white_captured": "".join(white_captured),
        "black_captured": "".join(black_captured),
        "white_adv": f"+{diff}" if diff > 0 else "",
        "black_adv": f"+{-diff}" if diff < 0 else "",
    }


# =========================================================
# SIDEBAR CONTROLS
# =========================================================
st.sidebar.title("⚙️ Game Controls")

try:
    model, model_name, is_onnx = get_cached_model()
    engine_badge = "⚡ ONNX SIMD" if is_onnx else "PyTorch CPU"
    st.sidebar.success(f"Model: `{model_name}`\nBackend: **{engine_badge}**")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()

st.sidebar.subheader("Game Options")
selected_color = st.sidebar.radio(
    "Play as:",
    options=["White", "Black"],
    index=0 if st.session_state.user_color == "White" else 1,
    horizontal=True
)
if selected_color != st.session_state.user_color:
    st.session_state.user_color = selected_color
    st.session_state.board = chess.Board()
    st.session_state.move_history = []
    st.session_state.eval_white = 0.05
    st.session_state.top_moves = []
    st.session_state.last_processed_move = None
    st.session_state.hint = None
    st.session_state.resigned = False
    st.rerun()

DIFFICULTY_PRESETS = {
    "🥈 Intermediate (Club Player)": {"sims": 75, "temp": 0.1},
    "🥉 Beginner (Casual)": {"sims": 25, "temp": 0.3},
    "🥇 Master (Tactical)": {"sims": 140, "temp": 0.0},
    "⚡ Grandmaster (Max Depth)": {"sims": 240, "temp": 0.0},
    "🛠️ Custom Settings": None,
}

diff_choice = st.sidebar.selectbox(
    "Difficulty Level:",
    options=list(DIFFICULTY_PRESETS.keys()),
    index=0
)

if DIFFICULTY_PRESETS[diff_choice] is not None:
    simulations = DIFFICULTY_PRESETS[diff_choice]["sims"]
    temperature = DIFFICULTY_PRESETS[diff_choice]["temp"]
    st.sidebar.caption(f"⚡ MCTS Depth: **{simulations} sims** | Randomness: **{temperature}**")
else:
    simulations = st.sidebar.slider(
        "MCTS Simulations:",
        min_value=20,
        max_value=300,
        value=80,
        step=20,
        help="Higher values give stronger tactical play. With ONNX, 80 simulations takes ~0.4s."
    )
    temperature = st.sidebar.slider(
        "Temperature (Randomness):",
        min_value=0.0,
        max_value=0.8,
        value=0.0,
        step=0.1,
        help="0.0 plays the best move. >0 introduces variety."
    )

st.sidebar.markdown("---")
col_b1, col_b2 = st.sidebar.columns(2)
with col_b1:
    if st.button("🔄 New Game", use_container_width=True):
        st.session_state.board = chess.Board()
        st.session_state.move_history = []
        st.session_state.eval_white = 0.05
        st.session_state.top_moves = []
        st.session_state.last_processed_move = None
        st.session_state.hint = None
        st.session_state.resigned = False
        st.session_state.cache.clear()
        st.rerun()

with col_b2:
    if st.button("↩️ Takeback", use_container_width=True):
        if len(st.session_state.move_history) >= 2:
            st.session_state.board.pop()
            st.session_state.board.pop()
            st.session_state.move_history.pop()
            st.session_state.move_history.pop()
        elif len(st.session_state.move_history) == 1:
            st.session_state.board.pop()
            st.session_state.move_history.pop()
        st.session_state.last_processed_move = None
        st.session_state.hint = None
        st.session_state.resigned = False
        st.session_state.eval_white = 0.05 if len(st.session_state.move_history) == 0 else st.session_state.eval_white
        st.rerun()

col_act1, col_act2 = st.sidebar.columns(2)
with col_act1:
    if st.button("💡 Get Hint", use_container_width=True):
        if not st.session_state.board.is_game_over(claim_draw=True) and not st.session_state.resigned:
            with st.spinner("AlphaZero finding best move..."):
                h_move, h_eval, _, _ = get_best_move(
                    board=st.session_state.board,
                    model=model,
                    device="cpu",
                    num_simulations=min(simulations, 60),
                    temperature=0.0,
                    transposition_cache=st.session_state.cache
                )
                if h_move:
                    san_m = st.session_state.board.san(h_move)
                    st.session_state.hint = f"Recommended: **{san_m}** (`{h_move.uci()}`)"
            st.rerun()

with col_act2:
    if st.button("🏳️ Resign", use_container_width=True):
        if not st.session_state.board.is_game_over(claim_draw=True) and not st.session_state.resigned:
            st.session_state.resigned = True
            st.session_state.eval_white = -1.0 if st.session_state.user_color == "White" else 1.0
            st.session_state.hint = None
            st.rerun()

with st.sidebar.expander("Custom FEN Setup"):
    fen_input = st.text_input("Paste FEN", value=st.session_state.board.fen())
    if st.button("Load FEN"):
        try:
            st.session_state.board = chess.Board(fen_input)
            st.session_state.move_history = []
            st.session_state.last_processed_move = None
            st.session_state.hint = None
            st.session_state.resigned = False
            st.session_state.eval_white = 0.0
            st.success("Board loaded from FEN!")
            st.rerun()
        except Exception as e:
            st.error(f"Invalid FEN: {e}")


# =========================================================
# HEADER
# =========================================================
st.markdown("<h1 class='main-title'>♟️ AlphaZero Chess</h1>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Live, fluid Chess.com-style board with real-time moves, hints, and sound effects</div>", unsafe_allow_html=True)


# =========================================================
# LIVE GAMEPLAY ARENA (FRAGMENT: NO FULL PAGE RELOADS)
# =========================================================
@st.fragment
def render_game_arena():
    board = st.session_state.board
    user_is_white = (st.session_state.user_color == "White")
    is_human_turn = (board.turn == chess.WHITE and user_is_white) or (board.turn == chess.BLACK and not user_is_white)
    game_over = board.is_game_over(claim_draw=True) or st.session_state.resigned
    mat = get_material_info(board)

    # If user selected Black and game just started: AI plays White move 1 immediately
    if len(st.session_state.move_history) == 0 and not is_human_turn and not game_over:
        with st.spinner("AlphaZero opening move..."):
            best_move, best_eval, scored_moves, _ = get_best_move(
                board=board,
                model=model,
                device="cpu",
                num_simulations=simulations,
                temperature=temperature,
                transposition_cache=st.session_state.cache,
            )
            if best_move:
                board.push(best_move)
                st.session_state.move_history.append(best_move)
                st.session_state.eval_white = best_eval
                st.session_state.top_moves = scored_moves[:5]
                st.session_state.last_processed_move = best_move.uci()
                st.rerun(scope="fragment")

    col_board, col_panel = st.columns([1.18, 1.0])

    with col_board:
        # Opponent Top Bar
        opp_name = "AlphaZero AI (Engine)" if user_is_white else "You (Human)"
        opp_captured = mat["white_captured"] if not user_is_white else mat["black_captured"]
        opp_adv = mat["white_adv"] if not user_is_white else mat["black_adv"]

        st.markdown(f"""
        <div class="player-card">
            <div class="player-name">🤖 {opp_name}</div>
            <div>
                <span class="captured-pieces">{opp_captured}</span>
                {f'<span class="advantage-badge">{opp_adv}</span>' if opp_adv else ''}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Permanent static key so the board iframe is NEVER destroyed or reloaded
        last_played_uci = st.session_state.move_history[-1].uci() if st.session_state.move_history else None

        component_move = interactive_chessboard(
            fen=board.fen(),
            orientation="white" if user_is_white else "black",
            is_interactive=(is_human_turn and not game_over),
            last_move=last_played_uci,
            key="chessboard_interactive_main"
        )

        # Player Bottom Bar
        user_name = "You (Human)" if user_is_white else "AlphaZero AI (Engine)"
        user_captured = mat["white_captured"] if user_is_white else mat["black_captured"]
        user_adv = mat["white_adv"] if user_is_white else mat["black_adv"]

        st.markdown(f"""
        <div class="player-card">
            <div class="player-name">👤 {user_name}</div>
            <div>
                <span class="captured-pieces">{user_captured}</span>
                {f'<span class="advantage-badge">{user_adv}</span>' if user_adv else ''}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Process Move Instantaneously in the same execution cycle (Zero Reloads!)
        if component_move and "move" in component_move:
            incoming_uci = component_move["move"]
            if incoming_uci != st.session_state.last_processed_move:
                try:
                    mv = chess.Move.from_uci(incoming_uci)
                    if mv in board.legal_moves:
                        st.session_state.last_processed_move = incoming_uci
                        st.session_state.hint = None
                        board.push(mv)
                        st.session_state.move_history.append(mv)

                        # Immediately calculate Engine reply in the same fragment cycle!
                        if not board.is_game_over(claim_draw=True):
                            t0 = time.perf_counter()
                            search_turn = board.turn
                            best_move, best_eval, scored_moves, resigned = get_best_move(
                                board=board,
                                model=model,
                                device="cpu",
                                num_simulations=simulations,
                                temperature=temperature,
                                transposition_cache=st.session_state.cache,
                            )
                            st.session_state.last_move_time = time.perf_counter() - t0
                            # Convert best_eval (search_turn perspective) to absolute White perspective
                            eval_white = best_eval if search_turn == chess.WHITE else -best_eval
                            st.session_state.eval_white = eval_white
                            if best_move and not resigned:
                                st.session_state.top_moves = scored_moves[:5]
                                st.session_state.last_processed_move = best_move.uci()
                                board.push(best_move)
                                st.session_state.move_history.append(best_move)
                            elif resigned:
                                st.session_state.eval_white = 1.0 if search_turn == chess.BLACK else -1.0
                        else:
                            outcome = board.outcome(claim_draw=True)
                            if outcome and outcome.winner is not None:
                                st.session_state.eval_white = 1.0 if outcome.winner == chess.WHITE else -1.0
                            else:
                                st.session_state.eval_white = 0.0

                        # Smooth in-place fragment update: No page reload!
                        st.rerun(scope="fragment")
                except Exception:
                    pass

    with col_panel:
        # Evaluation Bar (Absolute White Perspective: +1.0 = White win, -1.0 = Black win)
        eval_for_white = max(-1.0, min(1.0, float(st.session_state.eval_white)))
        white_win_pct = max(0, min(100, int(round((eval_for_white + 1.0) / 2.0 * 100))))
        black_win_pct = 100 - white_win_pct

        st.markdown("<div class='eval-card'>", unsafe_allow_html=True)
        turn_label = "Your Turn" if is_human_turn else "AlphaZero Thinking..."
        turn_class = "turn-human" if is_human_turn else "turn-engine"
        if eval_for_white > 0.08:
            eval_text = f"White Advantage ({eval_for_white:+.2f})"
        elif eval_for_white < -0.08:
            eval_text = f"Black Advantage ({-eval_for_white:+.2f})"
        else:
            eval_text = f"Equal Position ({eval_for_white:+.2f})"

        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span class="turn-badge {turn_class}">● {turn_label}</span>
            <span style="font-weight: 700; color: #ffffff;">{eval_text}</span>
        </div>
        """, unsafe_allow_html=True)
        st.progress(white_win_pct / 100.0, text=f"White: {white_win_pct}% | Black: {black_win_pct}%")
        st.markdown("</div>", unsafe_allow_html=True)

        # Game Over Banner
        if st.session_state.resigned:
            winner = "AlphaZero AI"
            st.warning(f"🏳️ Game Over: You resigned. **{winner} wins!**")
        elif game_over:
            outcome = board.outcome(claim_draw=True)
            res = board.result(claim_draw=True)
            if outcome and outcome.winner is not None:
                winner = "White" if outcome.winner == chess.WHITE else "Black"
                st.success(f"🏆 Game Over: **{winner} won by checkmate!** ({res})")
            else:
                st.info(f"🤝 Game Over: **Draw ({res})**")

        # Coach Hint Banner
        if st.session_state.hint and not game_over and is_human_turn:
            st.info(f"💡 **Coach Advice:** {st.session_state.hint}")

        # Move History Table (Chess.com Style)
        st.subheader("📜 Move History")
        if st.session_state.move_history:
            st.markdown("<div class='moves-table-container'>", unsafe_allow_html=True)
            temp_b = chess.Board()
            table_html = ""
            for i in range(0, len(st.session_state.move_history), 2):
                move_num = (i // 2) + 1
                w_move = temp_b.san(st.session_state.move_history[i])
                temp_b.push(st.session_state.move_history[i])

                b_move = ""
                if i + 1 < len(st.session_state.move_history):
                    b_move = temp_b.san(st.session_state.move_history[i + 1])
                    temp_b.push(st.session_state.move_history[i + 1])

                table_html += f"""
                <div class="move-row">
                    <span class="move-num">{move_num}.</span>
                    <span class="move-white">{w_move}</span>
                    <span class="move-black">{b_move}</span>
                </div>
                """
            st.markdown(table_html + "</div>", unsafe_allow_html=True)

            # Generate and download standard PGN
            pgn_game = chess.pgn.Game()
            pgn_game.headers["Event"] = "AlphaZero Exhibition"
            pgn_game.headers["Site"] = "Streamlit Live Arena"
            pgn_game.headers["Date"] = time.strftime("%Y.%m.%d")
            pgn_game.headers["White"] = "Human" if user_is_white else "AlphaZero AI"
            pgn_game.headers["Black"] = "AlphaZero AI" if user_is_white else "Human"
            if st.session_state.resigned:
                pgn_game.headers["Result"] = "0-1" if user_is_white else "1-0"
            elif game_over:
                pgn_game.headers["Result"] = board.result(claim_draw=True)
            else:
                pgn_game.headers["Result"] = "*"

            node = pgn_game
            for mv in st.session_state.move_history:
                node = node.add_variation(mv)

            st.download_button(
                label="📥 Download Game PGN",
                data=str(pgn_game),
                file_name=f"alphazero_{time.strftime('%Y%m%d_%H%M%S')}.pgn",
                mime="application/x-chess-pgn",
                use_container_width=True
            )
        else:
            st.caption("No moves played yet. Click a piece to begin.")

        # Engine Candidate Analysis Tab
        if st.session_state.top_moves:
            st.subheader("🔍 Engine Thought Process")
            analysis_data = []
            for mv, q, prob in st.session_state.top_moves:
                analysis_data.append({
                    "Candidate Move": mv.uci(),
                    "Evaluation (Q)": f"{q:+.3f}",
                    "Prior Prob": f"{prob*100:.1f}%"
                })
            st.table(analysis_data)
            if st.session_state.last_move_time > 0:
                st.caption(f"⚡ Last move searched in {st.session_state.last_move_time:.2f}s ({simulations} sims) via {engine_badge}")

        # Fallback Keyboard / SAN Input in an Expander
        with st.expander("⌨️ Keyboard / Manual SAN Move Input", expanded=False):
            manual_txt = st.text_input("Enter Move (e.g. e4, Nf3, e2e4):", key="manual_move_txt")
            if st.button("Submit Move", key="manual_submit_btn"):
                if manual_txt.strip():
                    try:
                        parsed = board.parse_san(manual_txt.strip())
                    except Exception:
                        try:
                            parsed = chess.Move.from_uci(manual_txt.strip().lower())
                            if parsed not in board.legal_moves:
                                parsed = None
                        except Exception:
                            parsed = None

                    if parsed and parsed in board.legal_moves:
                        st.session_state.last_processed_move = parsed.uci()
                        st.session_state.hint = None
                        board.push(parsed)
                        st.session_state.move_history.append(parsed)

                        if not board.is_game_over(claim_draw=True):
                            t0 = time.perf_counter()
                            search_turn = board.turn
                            best_move, best_eval, scored_moves, resigned = get_best_move(
                                board=board,
                                model=model,
                                device="cpu",
                                num_simulations=simulations,
                                temperature=temperature,
                                transposition_cache=st.session_state.cache,
                            )
                            st.session_state.last_move_time = time.perf_counter() - t0
                            eval_white = best_eval if search_turn == chess.WHITE else -best_eval
                            st.session_state.eval_white = eval_white
                            if best_move and not resigned:
                                st.session_state.top_moves = scored_moves[:5]
                                st.session_state.last_processed_move = best_move.uci()
                                board.push(best_move)
                                st.session_state.move_history.append(best_move)
                            elif resigned:
                                st.session_state.eval_white = 1.0 if search_turn == chess.BLACK else -1.0
                        else:
                            outcome = board.outcome(claim_draw=True)
                            if outcome and outcome.winner is not None:
                                st.session_state.eval_white = 1.0 if outcome.winner == chess.WHITE else -1.0
                            else:
                                st.session_state.eval_white = 0.0

                        st.rerun(scope="fragment")
                    else:
                        st.error("Invalid chess move.")


# Render the live arena
render_game_arena()
