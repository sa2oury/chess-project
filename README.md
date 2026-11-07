# ♟️ AlphaZero Chess AI Engine

A deep reinforcement and supervised learning chess engine inspired by DeepMind's **AlphaZero** and **Leela Chess Zero (Lc0)**. Built with PyTorch, `python-chess`, and Streamlit.

---

## 🚀 Play Live in the Browser (Streamlit App)

The project includes an interactive web interface built with **Streamlit** that lets anyone play against the trained neural network directly in their browser.

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
   Open `http://localhost:8501` in your browser.

---

## 🧠 Architecture Overview

### 1. Neural Network Architecture (`src/models.py`)
- **Backbone**: 12-block Residual Convolutional Neural Network (128 channels, $3 \times 3$ kernels, BatchNorm, ReLU).
- **Policy Head**: Predicts move distribution across **4,864 action planes** ($76 \text{ planes} \times 64 \text{ squares}$).
- **Value Head**: Evaluates position winning probability ($V \in [-1.0, 1.0]$) via convolutional reduction and fully connected layers.

### 2. Move & Board Encoding (`src/encoding.py`)
- **4,864 Action Space**:
  - Planes 0–55: 8 ray directions $\times$ 7 distances = 56 queen-like moves.
  - Planes 56–63: 8 knight moves.
  - Planes 64–75: 3 directions (straight, capture left, capture right) $\times$ 4 promotion pieces (Knight, Bishop, Rook, Queen) = 12 promotion planes.
- **Enhanced 18-Plane Input**: Includes piece positions, castling rights (White/Black, Kingside/Queenside), en passant target squares, and normalized half-move clock.

### 3. Monte Carlo Tree Search (`src/mcts.py`)
- **PUCT Selection**: Balances exploitation $Q(s, a)$ and exploration $U(s, a)$ using network policy priors.
- **Transposition Caching**: Fast FEN-indexed cache avoiding redundant neural network passes.
- **Temperature Annealing**: Allows deterministic play ($\tau = 0.0$) or creative exploratory play ($\tau > 0$).

---

## 📁 Repository Structure

```
chess-project/
├── app.py                      # Interactive Streamlit Chess Web App
├── requirements.txt            # Python dependencies for deployment
├── README.md                   # Project documentation
├── .gitignore                  # Git exclusions for large training datasets
├── src/                        # Modular core engine package
│   ├── __init__.py
│   ├── models.py               # AlphaZero ResNet & SE-ResNet definitions
│   ├── encoding.py             # Board tensors & 4,864-move encoder/decoder
│   └── mcts.py                 # MCTS engine & get_best_move search
├── models/
│   └── chess_v1.4_deploy.pth   # Lightweight deployment model (~51.9 MB)
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

