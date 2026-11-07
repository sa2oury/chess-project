import os
import streamlit.components.v1 as components

# Path to the frontend component folder
_component_path = os.path.join(os.path.dirname(__file__), "frontend")
_interactive_board = components.declare_component("interactive_chessboard", path=_component_path)


def interactive_chessboard(
    fen: str,
    orientation: str = "white",
    is_interactive: bool = True,
    last_move: str = None,
    key: str = None
):
    """
    Renders a Chess.com-style interactive board component with:
      - Drag-and-drop & Click-to-move
      - Valid move hint dots (circles for moves, rings for captures)
      - Selected square & last move highlights
      - Synthesized sound effects (move, capture, check)
    """
    return _interactive_board(
        fen=fen,
        orientation=orientation,
        is_interactive=is_interactive,
        last_move=last_move,
        key=key,
        default=None
    )
