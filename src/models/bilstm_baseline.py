import torch
import torch.nn as nn


class BiLSTM_RUL_Base(nn.Module):
    """
    Baseline model for comparison against the SOTA multitask model.
    Now returns (rul_pred, hi_pred, None) for compatibility.
    """

    def __init__(
        self,
        input_dim=14,
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        lstm_out_dim = hidden_dim * 2  # because bidirectional

        self.fc_rul = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Dummy HI head (baseline does not estimate HI, but returns something)
        self.fc_hi = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        """
        x: [B, T, D]
        """
        lstm_out, _ = self.lstm(x)      # [B, T, 2H]
        last = lstm_out[:, -1, :]       # last time step: [B, 2H]

        rul_pred = self.fc_rul(last).squeeze(-1)   # [B]
        hi_pred = self.fc_hi(last).squeeze(-1)     # [B]

        return rul_pred, hi_pred, None
