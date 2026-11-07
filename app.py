import os
import base64
import chess
import chess.svg
import streamlit as st

from src.models import load_model
from src.mcts import get_best_move

# =========================================================
# STREAMLIT CONFIG & STYLES
# =========================================================
st.set_page_config(
    page_title="AlphaZero Chess Engine",
    page_icon="♟️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .main-title {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-title {
        color: #888888;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .eval-container {
        margin: 15px 0;
        background: #262730;
        border-radius: 8px;
        padding: 12px;
    }
    .stButton>button {
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL CACHING (Loads once per instance)
# =========================================================
@st.cache_resource(show_spinner="Loading AlphaZero Neural Network...")
def get_cached_model():
    # Priority: lightweight deployment model -> full checkpoint
    candidate_paths = [
        "models/chess_v1.4_deploy.pth",
        "ver1.4/chess ver1.4.pth",
        "ver1.3/chess ver1.3.pth",
        "ver1.2/last_model.pth"
    ]
    selected_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            selected_path = p
            break

    if selected_path is None:
        raise FileNotFoundError("No checkpoint file found in models/ or ver1.X/")

    model = load_model(selected_path, device="cpu")
    return model, os.path.basename(selected_path)


# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================
if "board" not in st.session_state:
    st.session_state.board = chess.Board()
if "move_history" not in st.session_state:
    st.session_state.move_history = []
if "user_color" not in st.session_state:
    st.session_state.user_color = "White"
if "last_eval" not in st.session_state:
    st.session_state.last_eval = 0.0
if "top_moves" not in st.session_state:
    st.session_state.top_moves = []
if "cache" not in st.session_state:
    st.session_state.cache = {}
if "game_over_msg" not in st.session_state:
    st.session_state.game_over_msg = None


# =========================================================
# SIDEBAR CONTROLS
# =========================================================
st.sidebar.title("⚙️ Engine Controls")

try:
    model, model_name = get_cached_model()
    st.sidebar.success(f"Loaded: `{model_name}`")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()

# Game Options
st.sidebar.subheader("Game Setup")
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
    st.session_state.last_eval = 0.0
    st.session_state.top_moves = []
    st.session_state.game_over_msg = None
    st.rerun()

simulations = st.sidebar.slider(
    "MCTS Simulations per move:",
    min_value=20,
    max_value=400,
    value=80,
    step=20,
    help="Higher values result in stronger play but take longer per move."
)

temperature = st.sidebar.slider(
    "Move Temperature:",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.1,
    help="0.0 plays strictly the best move. Higher values add variety."
)

st.sidebar.markdown("---")
col_b1, col_b2 = st.sidebar.columns(2)
with col_b1:
    if st.button("🔄 New Game", use_container_width=True):
        st.session_state.board = chess.Board()
        st.session_state.move_history = []
        st.session_state.last_eval = 0.0
        st.session_state.top_moves = []
        st.session_state.game_over_msg = None
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
        st.session_state.game_over_msg = None
        st.rerun()

# FEN Custom Setup
with st.sidebar.expander("Custom FEN Setup"):
    fen_input = st.text_input("Paste FEN", value=st.session_state.board.fen())
    if st.button("Load FEN"):
        try:
            st.session_state.board = chess.Board(fen_input)
            st.session_state.move_history = []
            st.session_state.game_over_msg = None
            st.success("Board loaded from FEN!")
            st.rerun()
        except Exception as e:
            st.error(f"Invalid FEN: {e}")


# =========================================================
# MAIN INTERFACE
# =========================================================
st.markdown("<h1 class='main-title'>♟️ AlphaZero Chess Engine</h1>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Deep Residual Policy & Value Network powered by Monte Carlo Tree Search</div>", unsafe_allow_html=True)

board = st.session_state.board
user_is_white = (st.session_state.user_color == "White")
is_human_turn = (board.turn == chess.WHITE and user_is_white) or (board.turn == chess.BLACK and not user_is_white)

layout_col1, layout_col2 = st.columns([1.1, 1.0])

# =========================================================
# 1. BOARD DISPLAY
# =========================================================
with layout_col1:
    last_mv = st.session_state.move_history[-1] if st.session_state.move_history else None
    check_sq = board.king(board.turn) if board.is_check() else None

    # Render SVG
    svg_board = chess.svg.board(
        board=board,
        orientation=chess.WHITE if user_is_white else chess.BLACK,
        lastmove=last_mv,
        check=check_sq,
        size=460,
    )

    # Encode SVG into HTML image for crisp rendering
    b64_svg = base64.b64encode(svg_board.encode("utf-8")).decode("utf-8")
    st.markdown(
        f'<div style="text-align: center;"><img src="data:image/svg+xml;base64,{b64_svg}" width="460" /></div>',
        unsafe_allow_html=True
    )

    # Status Message
    if board.is_game_over(claim_draw=True):
        outcome = board.outcome(claim_draw=True)
        res = board.result(claim_draw=True)
        if outcome and outcome.winner is not None:
            winner = "White" if outcome.winner == chess.WHITE else "Black"
            st.success(f"🏆 Game Over: {winner} won by checkmate! ({res})")
        else:
            st.info(f"🤝 Game Over: Draw ({res})")
    elif board.is_check():
        st.warning("⚠️ Check!")

# =========================================================
# 2. MOVE INPUT & ENGINE SEARCH
# =========================================================
with layout_col2:
    # Evaluation Bar
    eval_val = st.session_state.last_eval  # in [-1.0, 1.0] from current player perspective
    # Convert to White win percentage
    eval_for_white = eval_val if board.turn == chess.WHITE else -eval_val
    white_win_pct = max(0, min(100, int((eval_for_white + 1.0) / 2.0 * 100)))

    st.markdown("<div class='eval-container'>", unsafe_allow_html=True)
    st.markdown(f"**Evaluation**: `{'White Advantage' if eval_for_white > 0.05 else ('Black Advantage' if eval_for_white < -0.05 else 'Equal')}` (Eval Score: `{eval_for_white:+.3f}`)")
    st.progress(white_win_pct / 100.0, text=f"White: {white_win_pct}% | Black: {100 - white_win_pct}%")
    st.markdown("</div>", unsafe_allow_html=True)

    if not board.is_game_over(claim_draw=True):
        if is_human_turn:
            st.subheader("Your Turn")
            legal_moves_list = list(board.legal_moves)
            legal_sans = sorted([board.san(m) for m in legal_moves_list])

            col_in1, col_in2 = st.columns([2, 1])
            with col_in1:
                selected_san = st.selectbox(
                    "Choose Legal Move:",
                    options=["-- select a move --"] + legal_sans,
                    key="move_dropdown"
                )
            with col_in2:
                text_move = st.text_input("Or enter SAN/UCI:", placeholder="e4 or e2e4", key="text_move_input")

            submit_move = st.button("Play Move", type="primary", use_container_width=True)

            if submit_move:
                move_to_play = None
                if text_move.strip():
                    try:
                        move_to_play = board.parse_san(text_move.strip())
                    except Exception:
                        try:
                            m = chess.Move.from_uci(text_move.strip().lower())
                            if m in board.legal_moves:
                                move_to_play = m
                        except Exception:
                            pass
                elif selected_san != "-- select a move --":
                    move_to_play = board.parse_san(selected_san)

                if move_to_play is not None:
                    board.push(move_to_play)
                    st.session_state.move_history.append(move_to_play)
                    st.rerun()
                else:
                    st.error("Invalid move. Please select or type a legal chess move.")
        else:
            # Engine Turn
            st.subheader("AlphaZero Thinking...")
            with st.spinner(f"Searching tree ({simulations} simulations)..."):
                best_move, best_eval, scored_moves, resigned = get_best_move(
                    board=board,
                    model=model,
                    device="cpu",
                    num_simulations=simulations,
                    temperature=temperature,
                    transposition_cache=st.session_state.cache,
                )

                if resigned or best_move is None:
                    st.session_state.game_over_msg = "AlphaZero Resigned."
                else:
                    st.session_state.last_eval = best_eval
                    st.session_state.top_moves = scored_moves[:5]
                    board.push(best_move)
                    st.session_state.move_history.append(best_move)

            st.rerun()

    # Engine Candidate Moves Tab
    if st.session_state.top_moves:
        st.subheader("🔍 Engine Analysis")
        data_rows = []
        for mv, q, prob in st.session_state.top_moves:
            try:
                # Board turn is currently the other player, so we display UCI or SAN
                data_rows.append({
                    "Move": mv.uci(),
                    "Q-Score": f"{q:+.3f}",
                    "Policy Prob": f"{prob*100:.1f}%"
                })
            except Exception:
                pass
        if data_rows:
            st.table(data_rows)

    # Move History Tab
    if st.session_state.move_history:
        with st.expander("📜 Move History (PGN)", expanded=False):
            temp_b = chess.Board()
            pgn_str = ""
            for i, mv in enumerate(st.session_state.move_history):
                if i % 2 == 0:
                    pgn_str += f"{i // 2 + 1}. "
                pgn_str += f"{temp_b.san(mv)} "
                temp_b.push(mv)
            st.text_area("PGN", value=pgn_str, height=100)

