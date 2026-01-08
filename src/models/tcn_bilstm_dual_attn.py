import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Temporal Convolution Block (TCN)
# ============================================================
class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )

        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.res_connection = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.res_connection(x)

        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)

        return out + residual


# ============================================================
# 2. Dual Attention Module (Temporal + Feature Attention)
# ============================================================
class DualAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        # Temporal attention
        self.temp_attn = nn.Linear(hidden_dim, hidden_dim)

        # Feature attention
        self.feat_attn = nn.Linear(hidden_dim, hidden_dim)

        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        x = [B, T, H]
        """

        # Temporal projection
        t_proj = torch.tanh(self.temp_attn(x))  # [B, T, H]

        # Feature projection (mean over time)
        f_context = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H]
        f_proj = torch.tanh(self.feat_attn(f_context))  # [B, 1, H]

        # Combine
        combined = t_proj + f_proj  # broadcasting

        # Attention weights
        score = self.score(combined).squeeze(-1)  # [B, T]
        attn_weights = torch.softmax(score, dim=1).unsqueeze(-1)  # [B, T, 1]

        # Weighted output
        out = torch.sum(x * attn_weights, dim=1)  # [B, H]

        return out, attn_weights


# ============================================================
# 3. Full SOTA Model: TCN + BiLSTM + Dual Attention
# ============================================================
class TCN_BiLSTM_DualAttn(nn.Module):
    def __init__(
        self,
        input_dim=14,
        conv_channels=64,
        lstm_hidden=64,
        lstm_layers=2,
        dropout=0.2,
    ):
        super().__init__()

        # ---------------------------
        # TCN feature extractor
        # ---------------------------
        self.tcn = nn.Sequential(
            TemporalConvBlock(input_dim, conv_channels, dilation=1, dropout=dropout),
            TemporalConvBlock(conv_channels, conv_channels, dilation=2, dropout=dropout),
            TemporalConvBlock(conv_channels, conv_channels, dilation=4, dropout=dropout),
        )

        # ---------------------------
        # BiLSTM
        # ---------------------------
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        lstm_out_dim = lstm_hidden * 2  # bidirectional

        # ---------------------------
        # Dual Attention
        # ---------------------------
        self.attn = DualAttention(lstm_out_dim)

        # ---------------------------
        # Prediction Output Heads
        # ---------------------------
        self.fc_rul = nn.Linear(lstm_out_dim, 1)
        self.fc_hi = nn.Linear(lstm_out_dim, 1)

    def forward(self, x):
        """
        x: [B, T, D]
        """

        # 1) TCN expects: [B, D, T]
        x = x.permute(0, 2, 1)
        tcn_out = self.tcn(x)

        # 2) LSTM expects: [B, T, C]
        lstm_in = tcn_out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(lstm_in)

        # 3) Dual attention
        attn_out, attn_weights = self.attn(lstm_out)

        # 4) Predictions
        rul_pred = self.fc_rul(attn_out).squeeze(-1)
        hi_pred = self.fc_hi(attn_out).squeeze(-1)

        return rul_pred, hi_pred, attn_weights