# ♟️ AlphaZero Chess AI Engine

A deep reinforcement and supervised learning chess engine inspired by DeepMind's **AlphaZero** and **Leela Chess Zero (Lc0)**. Built with PyTorch, ONNX Runtime, `python-chess`, and Streamlit.

---

## 🚀 Play Live in the Browser (Streamlit App)

The project includes an interactive web interface built with **Streamlit** that lets anyone play against the trained neural network directly in their browser.

### Key Web Features
- **Chess.com Gameplay**: Smooth drag-and-drop & click-to-move piece controls.
- **Valid Move Dots**: Displays circular hints on empty legal squares and target rings on capture squares.
- **Web Audio Effects**: Synthesized move clicks, capture thuds, and check alerts.
- **Difficulty Presets**: Play against 🥉 *Beginner*, 🥈 *Intermediate*, 🥇 *Master*, ⚡ *Grandmaster*, or customize simulations directly.
- **💡 Coach Hints**: One-click advice suggesting the best engine move with evaluation.
- **📥 PGN Export**: One-click standard PGN download to import your game into Chess.com or Lichess.
- **Opening Theory Book**: Instant 0ms book moves in the opening (Spanish, Italian, Sicilian, French, Queen's Gambit, etc.).
- **Tactical Sanity Filter**: Instant forced mate-in-1 detector and blunder-prevention shield.
- **Engine Analytics**: Real-time evaluation bar, win probability gauge, candidate move breakdown, and material difference tracker.
- **High-Speed Inference**: Powered by an **ONNX Runtime SIMD engine**, searching 80 MCTS simulations in **<0.45 seconds** on CPU.

### Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sa2oury/chess-project.git
   cd chess-project
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the web app:**
   ```bash
   streamlit run app.py
   ```
   Open `(https://chess-project.streamlit.app/)` in your browser.

---

## 🧠 Architecture Overview

### 1. Neural Network Architecture (`src/models.py`)
- **Backbone**: 12-block Residual Convolutional Neural Network (128 channels, $3 \times 3$ kernels, BatchNorm, ReLU).
- **Policy Head**: Predicts move distribution across **4,864 action planes** ($76 \text{ planes} \times 64 \text{ squares}$).
- **Value Head**: Evaluates position winning probability ($V \in [-1.0, 1.0]$) via convolutional reduction and fully connected layers.
- **ONNX Acceleration**: Standalone model weights (`models/chess_v1.4.onnx`) with fused SIMD operators for 2.5x speedup on CPU.

### 2. Move & Board Encoding (`src/encoding.py`)
- **4,864 Action Space**:
  - Planes 0–55: 8 ray directions $\times$ 7 distances = 56 queen-like moves.
  - Planes 56–63: 8 knight moves.
  - Planes 64–75: 3 directions (straight, capture left, capture right) $\times$ 4 promotion pieces (Knight, Bishop, Rook, Queen) = 12 promotion planes.
- **Dual Board Representation**:
  - `board_to_tensor_legacy`: 12-plane bitboards matching existing trained checkpoints.
  - `board_to_tensor_v2`: Enhanced 18-plane tensor incorporating Kingside/Queenside castling rights, en passant target squares, and normalized half-move clock.

### 3. Monte Carlo Tree Search (`src/mcts.py`)
- **PUCT Selection**: Balances exploitation $Q(s, a)$ and exploration $U(s, a)$ using network policy priors.
- **Zero-Copy NumPy Pathway**: Passes bitboards straight into ONNX C++ runtime without intermediate tensor allocation.
- **Transposition Caching**: Fast FEN-indexed cache avoiding redundant neural network passes on transposed move orders.

---

## 🛠️ CLI Tools & Protocols

### Universal Chess Interface (`uci_engine.py`)
Run the engine in standard chess GUIs (Arena, Cutechess, ChessBase, Banksia, or Lichess bots):
```bash
python uci_engine.py
```
Supports standard UCI commands: `uci`, `isready`, `position startpos moves e2e4`, `go depth 10`, `quit`.

### Production Training Pipeline (`train.py`)
Train on `.npz` shards with automatic mixed precision and holdout validation:
```bash
python train.py --data-dir archive --epochs 3 --batch-size 256 --num-val-shards 2
```

### Tournament Evaluator (`evaluate.py`)
Benchmark any two checkpoints head-to-head and compute Elo differences:
```bash
python evaluate.py --model1 models/chess_v1.4.onnx --model2 ver1.2/last_model.pth --games 20 --sims 80
```

---

## 📁 Repository Structure

```
chess-project/
├── app.py                      # Interactive Streamlit Chess Web App
├── requirements.txt            # Python dependencies for deployment
├── README.md                   # Project documentation
├── .gitignore                  # Git exclusions for large training datasets
├── uci_engine.py               # Standard Universal Chess Interface wrapper
├── train.py                    # Modular CLI training & fine-tuning pipeline
├── evaluate.py                 # Headless tournament benchmarking script
├── src/                        # Modular core engine package
│   ├── __init__.py
│   ├── models.py               # AlphaZero ResNet, SE-ResNet & ONNX engine
│   ├── encoding.py             # Board tensors & 4,864-move encoder/decoder
│   ├── mcts.py                 # MCTS engine & get_best_move search
│   ├── component.py            # Streamlit interactive board wrapper
│   └── frontend/               # Chessboard.js + Web Audio UI component
├── models/
│   ├── chess_v1.4.onnx         # High-speed ONNX model (51.8 MB)
│   └── chess_v1.4_deploy.pth   # Lightweight PyTorch model (51.9 MB)
├── training.ipynb              # Base supervised training pipeline
├── retrain.ipynb               # Curriculum fine-tuning pipeline
└── self-play.ipynb             # MCTS self-play generation & evaluation
```

---

## ☁️ Deploying to Streamlit Community Cloud

1. Push this repository to your GitHub account:
   ```bash
   git add .
   git commit -m "Deploy Streamlit Chess Engine"
   git push origin main
   ```
2. Navigate to [share.streamlit.io](https://share.streamlit.io/).
3. Connect your GitHub repository and select `app.py` as the main entry point.
4. Click **Deploy** to get a public link to share!
