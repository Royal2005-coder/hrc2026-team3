"""
Định nghĩa các kiến trúc neural network cho Behavioral Cloning.

1. BCMlpPolicy   — MLP đơn giản, input: state_t → output: action_t
2. BCLstmPolicy  — LSTM với sliding window, input: (state_t-W..t,) → action_t
3. BCChunkPolicy — Action-chunking (kiểu ACT, không ảnh), input: state_t
                   → output: chunk K hành động liên tiếp [action_t..action_t+K-1]
"""

import torch
import torch.nn as nn
import config


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _build_mlp(
    in_dim:       int,
    hidden_dims:  list[int],
    out_dim:      int,
    dropout:      float = 0.1,
    activation:   type  = nn.ReLU,
    use_bn:       bool  = True,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        if use_bn:
            layers.append(nn.BatchNorm1d(h))
        layers += [activation(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


# ─────────────────────────────────────────────────────────────────────────────
# 1. MLP Policy
# ─────────────────────────────────────────────────────────────────────────────

class BCMlpPolicy(nn.Module):
    """
    Behavioral Cloning với MLP.
    Input:  state (B, STATE_DIM)
    Output: action (B, ACTION_DIM)
    """

    def __init__(
        self,
        state_dim:   int       = config.STATE_DIM,
        action_dim:  int       = config.ACTION_DIM,
        hidden_dims: list[int] = config.MLP_HIDDEN_DIMS,
        dropout:     float     = config.MLP_DROPOUT,
    ):
        super().__init__()
        self.net = _build_mlp(state_dim, hidden_dims, action_dim, dropout)

        # Khởi tạo trọng số
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

    def predict(self, state: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(state)


# ─────────────────────────────────────────────────────────────────────────────
# 2. LSTM Policy
# ─────────────────────────────────────────────────────────────────────────────

class BCLstmPolicy(nn.Module):
    """
    Behavioral Cloning với LSTM (encode lịch sử trạng thái).
    Input:  window (B, W, STATE_DIM)
    Output: action (B, ACTION_DIM)

    Kiến trúc:
      state → Linear encoder → LSTM → take last hidden → MLP head → action
    """

    def __init__(
        self,
        state_dim:   int   = config.STATE_DIM,
        action_dim:  int   = config.ACTION_DIM,
        hidden_dim:  int   = config.LSTM_HIDDEN_DIM,
        num_layers:  int   = config.LSTM_NUM_LAYERS,
        dropout:     float = config.LSTM_DROPOUT,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Encode state trước khi đưa vào LSTM
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.action_head = _build_mlp(
            in_dim=hidden_dim,
            hidden_dims=[hidden_dim // 2],
            out_dim=action_dim,
            dropout=dropout,
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # forget gate bias = 1 (giúp gradient flow ban đầu)
                n = param.size(0)
                param.data[n // 4: n // 2].fill_(1.0)

        for m in self.action_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        window: torch.Tensor,                         # (B, W, STATE_DIM)
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple]:
        B, W, _ = window.shape
        # Encode mỗi step
        enc = self.state_encoder(window.view(B * W, -1)).view(B, W, -1)
        out, hidden = self.lstm(enc, hidden)
        # Lấy hidden state của step cuối
        last = out[:, -1, :]                          # (B, hidden_dim)
        action = self.action_head(last)               # (B, ACTION_DIM)
        return action, hidden

    def predict(
        self,
        window: torch.Tensor,
        hidden: tuple | None = None,
    ) -> tuple[torch.Tensor, tuple]:
        self.eval()
        with torch.no_grad():
            return self.forward(window, hidden)

    def init_hidden(self, batch_size: int, device: torch.device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return h, c


# ─────────────────────────────────────────────────────────────────────────────
# 3. Action-Chunking Policy (kiểu ACT, không dùng ảnh)
# ─────────────────────────────────────────────────────────────────────────────

class BCChunkPolicy(nn.Module):
    """
    Behavioral Cloning với action-chunking (rút gọn theo kiến trúc ACT).

    Thay vì dự đoán từng action_t một (dễ tích lũy sai số — compounding error,
    và dễ bị "trung bình hoá" giữa các cách demo khác nhau → giật/khựng),
    model dự đoán MỘT LÚC một chuỗi K hành động liên tiếp [a_t, a_t+1, ..., a_t+K-1]
    từ state hiện tại. Khi suy luận, robot thực thi vài bước đầu của chuỗi này
    (hoặc trung bình trọng số các chuỗi chồng lấp — temporal ensembling) trước khi
    re-plan, giúp chuyển động mượt và nhất quán hơn, giống quỹ đạo demo gốc.

    Input:  state (B, STATE_DIM)
    Output: action_chunk (B, CHUNK_SIZE, ACTION_DIM)

    Kiến trúc:
      state → Linear encoder → context token (B, 1, H)
      learned query embeddings (K, H) → TransformerDecoder cross-attend vào context
        → action head cho từng vị trí trong chunk
    """

    def __init__(
        self,
        state_dim:   int   = config.STATE_DIM,
        action_dim:  int   = config.ACTION_DIM,
        chunk_size:  int   = config.CHUNK_SIZE,
        hidden_dim:  int   = config.CHUNK_HIDDEN_DIM,
        num_layers:  int   = config.CHUNK_NUM_LAYERS,
        num_heads:   int   = config.CHUNK_NUM_HEADS,
        dropout:     float = config.CHUNK_DROPOUT,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        # Encode state hiện tại thành 1 "context token"
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Mỗi vị trí trong chunk có 1 learned query embedding riêng
        # (giống object queries trong DETR / action queries trong ACT)
        self.query_embed = nn.Parameter(torch.randn(chunk_size, hidden_dim) * 0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.action_head = nn.Linear(hidden_dim, action_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        B = state.shape[0]
        memory  = self.state_encoder(state).unsqueeze(1)               # (B, 1, H) — "memory" cho cross-attention
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)      # (B, K, H)
        decoded = self.decoder(tgt=queries, memory=memory)             # (B, K, H)
        return self.action_head(decoded)                               # (B, K, A)

    def predict(self, state: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(state)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(model_type: str) -> nn.Module:
    if model_type == "mlp":
        return BCMlpPolicy()
    elif model_type == "lstm":
        return BCLstmPolicy()
    elif model_type == "chunk":
        return BCChunkPolicy()
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'mlp', 'lstm' or 'chunk'.")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    mlp   = BCMlpPolicy()
    lstm  = BCLstmPolicy()
    chunk = BCChunkPolicy()
    print(f"MLP   params: {count_parameters(mlp):,}")
    print(f"LSTM  params: {count_parameters(lstm):,}")
    print(f"Chunk params: {count_parameters(chunk):,}")

    # Dummy forward pass
    B, W = 4, config.LSTM_WINDOW_SIZE
    s  = torch.randn(B, config.STATE_DIM)
    sw = torch.randn(B, W, config.STATE_DIM)

    a_mlp     = mlp(s)
    a_lstm, _ = lstm(sw)
    a_chunk   = chunk(s)
    print(f"MLP   output: {a_mlp.shape}")
    print(f"LSTM  output: {a_lstm.shape}")
    print(f"Chunk output: {a_chunk.shape}")
