# src/config.py
"""
Configuration file for:
LLM-Assisted Engine Health Index Estimation and
Remaining Useful Life Prediction using NASA C-MAPSS.
"""

from pathlib import Path
import torch


# ================================
# DEVICE SELECTION
# ================================
# Automatically uses GPU if available (Colab), CPU otherwise.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ================================
# PATHS (relative to project root)
# ================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
PREPROCESSED_ROOT = OUTPUT_ROOT / "preprocessed"


# ================================
# RAW DATA FORMAT (NASA C-MAPSS)
# ================================
# Each row = 1 cycle
RAW_COLUMNS = [
    "unit",
    "cycle",
    "setting1",
    "setting2",
    "setting3",
] + [f"s{i}" for i in range(1, 22)]


# ================================
# INFORMATIVE SENSORS (SOTA choice)
# ================================
# Based on PHM 2008 Winner + multiple SOTA papers.
INFORMATIVE_SENSOR_IDX = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
INFORMATIVE_SENSORS = [f"s{i}" for i in INFORMATIVE_SENSOR_IDX]


# ================================
# BASE CONFIG (applies to all subsets)
# ================================
BASE_CONFIG = {
    # ----- Preprocessing -----
    "seq_len": 30,                  # sequence window length
    "seq_stride": 1,                # stride for windowing
    "rul_max": 125,                 # default RUL cap
    "val_split": 0.2,               # 20% validation, 80% train
    "scaler_type": "standard",      # "standard", "minmax", "robust"
    "use_informative_sensors_only": True,
    "random_seed": 42,

    # ----- Model Architecture -----
    # TCN (feature extraction)
    "conv_channels": 64,

    # BiLSTM (temporal modeling)
    "lstm_hidden": 64,
    "lstm_layers": 2,

    # Dual Attention
    "dropout": 0.2,

    # ----- Optimization -----
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "batch_size": 128,
    "num_epochs": 25,
}


# ================================
# SUBSET-SPECIFIC CONFIG
# ================================
TRAIN_CONFIGS = {
    # FD001: Simple operating condition
    "FD001": {**BASE_CONFIG, "subset": "FD001"},

    # FD002: Complex + multiple fault modes
    "FD002": {
        **BASE_CONFIG,
        "subset": "FD002",
        "rul_max": 130,             # recommended for FD002
    },

    # FD003: Single operating condition but multiple failure types
    "FD003": {**BASE_CONFIG, "subset": "FD003"},

    # FD004: Hardest — multi-conditions + multi-fault modes
    "FD004": {
        **BASE_CONFIG,
        "subset": "FD004",
        "rul_max": 130,             # SOTA recommendation
    },
}
