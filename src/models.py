import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# STANDARD RESIDUAL BLOCKS
# =========================================================
class ResidualBlock(nn.Module):
    """Standard pre/post-activation 2-conv residual block."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual
        out = F.relu(out, inplace=True)
        return out


# =========================================================
# SQUEEZE-AND-EXCITATION (SE) RESIDUAL BLOCKS
# =========================================================
class SEBlock(nn.Module):
    """Channel attention squeeze-and-excitation block."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        s = F.adaptive_avg_pool2d(x, 1).view(b, c)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s)).view(b, c, 1, 1)
        return x * s


class SEResidualBlock(nn.Module):
    """Residual block with channel attention."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        out = out + residual
        out = F.relu(out, inplace=True)
        return out


# =========================================================
# 1. ALPHAZERO RESNET (v1.3 & v1.4 default architecture)
# =========================================================
class AlphaZeroResNet(nn.Module):
    """
    Standard AlphaZero-style ResNet with dual policy and value heads.
    Compatible with: ver1.4/chess ver1.4.pth and ver1.3/chess ver1.3.pth
    """
    def __init__(
        self,
        input_channels: int = 12,
        num_blocks: int = 12,
        channels: int = 128,
        policy_size: int = 4864,
        value_hidden: int = 256,
        value_dropout: float = 0.15,
        value_channels: int = 3,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.channels = channels
        self.policy_size = policy_size

        # Stem
        self.input_conv = nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(channels)

        # Residual Tower
        self.res_tower = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])

        # Policy Head (32 planes x 8 x 8 -> 4864)
        self.policy_conv = nn.Conv2d(channels, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, policy_size)

        # Value Head (value_channels x 8 x 8 -> value_hidden -> 1)
        self.value_conv = nn.Conv2d(channels, value_channels, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(value_channels)
        self.value_fc1 = nn.Linear(value_channels * 8 * 8, value_hidden)
        self.value_dropout = nn.Dropout(value_dropout)
        self.value_fc2 = nn.Linear(value_hidden, 1)

    def forward(self, x: torch.Tensor):
        x = self.input_conv(x)
        x = self.input_bn(x)
        x = F.relu(x, inplace=True)

        x = self.res_tower(x)

        # Policy
        p = self.policy_conv(x)
        p = self.policy_bn(p)
        p = F.relu(p, inplace=True)
        p = p.reshape(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        # Value
        v = self.value_conv(x)
        v = self.value_bn(v)
        v = F.relu(v, inplace=True)
        v = v.reshape(v.size(0), -1)
        v = self.value_fc1(v)
        v = F.relu(v, inplace=True)
        v = self.value_dropout(v)
        value = torch.tanh(self.value_fc2(v)).squeeze(1)

        return policy_logits, value


# =========================================================
# 2. ALPHAZERO SE-RESNET (v1.2 architecture)
# =========================================================
class AlphaZeroSEResNet(nn.Module):
    """
    AlphaZero-style ResNet equipped with Squeeze-and-Excitation blocks.
    Compatible with: ver1.2/last_model.pth
    """
    def __init__(
        self,
        input_channels: int = 12,
        num_blocks: int = 12,
        channels: int = 128,
        policy_size: int = 4864,
        value_hidden: int = 256,
        value_dropout: float = 0.15,
        value_channels: int = 8,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.channels = channels
        self.policy_size = policy_size

        self.input_conv = nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(channels)

        self.res_tower = nn.Sequential(
            *[SEResidualBlock(channels, reduction=16) for _ in range(num_blocks)]
        )

        self.policy_conv = nn.Conv2d(channels, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, policy_size)

        self.value_conv = nn.Conv2d(channels, value_channels, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(value_channels)
        self.value_fc1 = nn.Linear(value_channels * 8 * 8, value_hidden)
        self.value_dropout = nn.Dropout(value_dropout)
        self.value_fc2 = nn.Linear(value_hidden, 1)

    def forward(self, x: torch.Tensor):
        x = self.input_conv(x)
        x = self.input_bn(x)
        x = F.relu(x, inplace=True)

        x = self.res_tower(x)

        p = self.policy_conv(x)
        p = self.policy_bn(p)
        p = F.relu(p, inplace=True)
        p = p.reshape(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        v = self.value_conv(x)
        v = self.value_bn(v)
        v = F.relu(v, inplace=True)
        v = v.reshape(v.size(0), -1)
        v = self.value_fc1(v)
        v = F.relu(v, inplace=True)
        v = self.value_dropout(v)
        value = torch.tanh(self.value_fc2(v)).squeeze(1)

        return policy_logits, value


# =========================================================
# 3. CHESS RESNET (blackie architecture)
# =========================================================
class ResBlockBlackie(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = F.relu(out)
        return out


class ChessResNet(nn.Module):
    """
    10-block baseline architecture used in blackie/final_chess_resnet.pth.
    """
    def __init__(self, num_blocks: int = 10, hidden_channels: int = 128):
        super().__init__()
        self.start_conv = nn.Conv2d(12, hidden_channels, kernel_size=3, padding=1)
        self.start_bn = nn.BatchNorm2d(hidden_channels)
        self.res_blocks = nn.ModuleList([ResBlockBlackie(hidden_channels) for _ in range(num_blocks)])

        self.policy_conv = nn.Conv2d(hidden_channels, 76, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(76)
        self.policy_fc = nn.Linear(76 * 8 * 8, 4864)

        self.eval_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.eval_bn = nn.BatchNorm2d(1)
        self.eval_fc1 = nn.Linear(8 * 8, 64)
        self.eval_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.start_bn(self.start_conv(x)))
        for block in self.res_blocks:
            x = block(x)

        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.reshape(p.size(0), -1)
        policy_out = self.policy_fc(p)

        v = F.relu(self.eval_bn(self.eval_conv(x)))
        v = v.reshape(v.size(0), -1)
        v = F.relu(self.eval_fc1(v))
        v = self.eval_fc2(v)
        value_out = torch.tanh(v).squeeze(1)

        return policy_out, value_out


# =========================================================
# ONNX RUNTIME ENGINE WRAPPER
# =========================================================
class ONNXModel:
    """
    High-performance inference engine using ONNX Runtime.
    Provides 2x-4x speedup over PyTorch on CPU via fused SIMD kernels.
    """
    def __init__(self, onnx_path: str, num_threads: int = None):
        import onnxruntime as ort
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"ONNX model file not found: {onnx_path}")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if num_threads:
            opts.intra_op_num_threads = num_threads
            opts.inter_op_num_threads = 2

        self.session = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.is_onnx = True

    def __call__(self, x):
        """
        Runs inference on either a PyTorch Tensor or NumPy ndarray.
        Returns (policy_logits, value) as NumPy arrays.
        """
        import numpy as np
        if hasattr(x, "cpu"):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x

        if x_np.ndim == 3:
            x_np = np.expand_dims(x_np, 0)
        if not x_np.flags["C_CONTIGUOUS"]:
            x_np = np.ascontiguousarray(x_np, dtype=np.float32)

        p, v = self.session.run(self.output_names, {self.input_name: x_np})
        return p, v


# =========================================================
# UNIVERSAL MODEL LOADER & EXPORTER
# =========================================================
def load_model(
    checkpoint_path: str,
    device: str = "cpu",
    model_type: str = "auto"
):
    """
    Universally loads any checkpoint from the project:
      - .onnx files via ONNXModel (Ultra-fast CPU inference)
      - .pth files (v1.2, v1.3, v1.4, blackie) via PyTorch
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # 1. Fast ONNX model loading
    if checkpoint_path.endswith(".onnx"):
        return ONNXModel(checkpoint_path)

    # 2. PyTorch checkpoint loading
    raw_data = torch.load(checkpoint_path, map_location=device)
    if isinstance(raw_data, dict) and "model_state_dict" in raw_data:
        state_dict = raw_data["model_state_dict"]
    elif isinstance(raw_data, dict):
        state_dict = raw_data
    else:
        raise ValueError(f"Unrecognized checkpoint format in: {checkpoint_path}")

    keys = list(state_dict.keys())
    has_se = any(".se." in k for k in keys)
    has_blackie = any(k.startswith("start_conv") for k in keys)

    if model_type == "auto":
        if has_blackie:
            model = ChessResNet()
        elif has_se:
            value_c = state_dict.get("value_conv.weight", torch.zeros(8)).shape[0]
            model = AlphaZeroSEResNet(value_channels=value_c)
        else:
            value_c = state_dict.get("value_conv.weight", torch.zeros(3)).shape[0]
            model = AlphaZeroResNet(value_channels=value_c)
    elif model_type == "v1.4" or model_type == "v1.3" or model_type == "resnet":
        model = AlphaZeroResNet(value_channels=3)
    elif model_type == "v1.2" or model_type == "se":
        model = AlphaZeroSEResNet(value_channels=8)
    elif model_type == "blackie":
        model = ChessResNet()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def export_to_onnx(
    model_or_checkpoint,
    output_path: str = "models/chess_v1.4.onnx",
    device: str = "cpu"
) -> str:
    """Exports any PyTorch model to a single, standalone optimized ONNX model file."""
    if isinstance(model_or_checkpoint, str):
        model = load_model(model_or_checkpoint, device=device)
    else:
        model = model_or_checkpoint

    model.eval()
    dummy_input = torch.randn(1, 12, 8, 8, dtype=torch.float32, device=device)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["board_input"],
        output_names=["policy_logits", "value"],
        dynamic_axes={
            "board_input": {0: "batch_size"},
            "policy_logits": {0: "batch_size"},
            "value": {0: "batch_size"}
        },
        dynamo=False
    )
    print(f"Exported standalone ONNX model to {output_path} ({os.path.getsize(output_path) / 1e6:.2f} MB)")
    return output_path


def export_lightweight_model(
    checkpoint_path: str,
    output_path: str,
    device: str = "cpu"
) -> str:
    """
    Strips optimizer states and metadata, saving only pure model weights (~54 MB),
    making it easily deployable to GitHub and Streamlit Cloud under the 100 MB limit.
    """
    model = load_model(checkpoint_path, device=device)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(model.state_dict(), output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Exported lightweight model to {output_path} ({size_mb:.1f} MB)")
    return output_path

