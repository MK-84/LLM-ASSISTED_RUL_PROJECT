# src/training/eval_rul.py

"""
Final Evaluation Script
- Computes RUL metrics
- Generates professional plots
- Runs LLM reasoning (structured)
- Saves report as clean DOCX
"""

from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, Union
from rich.console import Console

from src.config import OUTPUT_ROOT, TRAIN_CONFIGS, INFORMATIVE_SENSORS
from src.data.datasets import load_preprocessed_subset
from src.models.bilstm_baseline import BiLSTM_RUL_Base
from src.models.tcn_bilstm_dual_attn import TCN_BiLSTM_DualAttn
from src.utils.metrics import compute_mae, compute_phm_score, compute_r2, compute_rmse

# Professional plots
from src.utils.plots import (
    plot_pred_vs_true,
    plot_error_histogram,
    plot_rul_sequence,
    plot_hi_sequence,
    plot_attention_curve,
    plot_sensor_degradation,
    plot_best_worst_rul,
    plot_full_rul_curve
)

# LLM + DOCX
from src.llm.llm_reasoning_ollama import llm_engine_explanation
from src.utils.docx_export import save_docx_report

console = Console()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------
# MODEL LOADER
# ---------------------------------------------------------
def load_model(model_type: str, input_dim: int, cfg: Dict):
    if model_type == "baseline":
        return BiLSTM_RUL_Base(
            input_dim=input_dim,
            hidden_dim=cfg["lstm_hidden"],
            num_layers=cfg["lstm_layers"],
            dropout=cfg["dropout"],
        )
    else:
        return TCN_BiLSTM_DualAttn(
            input_dim=input_dim,
            conv_channels=cfg["conv_channels"],
            lstm_hidden=cfg["lstm_hidden"],
            lstm_layers=cfg["lstm_layers"],
            dropout=cfg["dropout"],
        )


# ---------------------------------------------------------
# SENSOR STATISTICS (for LLM)
# ---------------------------------------------------------
def build_sensor_stats(x_window: Union[np.ndarray, list]):
    x_window = np.asarray(x_window)
    last_step = x_window[-1]

    num_feats = last_step.shape[-1]
    names = (
        INFORMATIVE_SENSORS
        if num_feats == len(INFORMATIVE_SENSORS)
        else [f"feat_{i}" for i in range(num_feats)]
    )

    return {n: float(v) for n, v in zip(names, last_step)}


# ---------------------------------------------------------
# MAIN EVALUATION
# ---------------------------------------------------------
@torch.no_grad()
def evaluate(subset: str, model_type: str):

    console.rule(f"[bold yellow]EVALUATING — {subset} ({model_type})[/bold yellow]")
    cfg = TRAIN_CONFIGS[subset]

    # Load data
    data = load_preprocessed_subset(subset)
    X_test = np.asarray(data["X_test"])
    y_true = np.asarray(data["y_rul_test"])
    input_dim = X_test.shape[-1]

    # Load model weights
    model = load_model(model_type, input_dim, cfg).to(DEVICE)
    ckpt = OUTPUT_ROOT / "checkpoints" / subset / f"{model_type}_best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    console.print(f"[green]✔ Loaded checkpoint: {ckpt.name}[/green]")

    # Forward pass
    X_tensor = torch.from_numpy(X_test).float().to(DEVICE)
    outputs = model(X_tensor)

    hi_pred = None
    attn = None

    if isinstance(outputs, tuple):
        rul_out, hi_out, attn_out = outputs
        rul_pred = rul_out.cpu().numpy().reshape(-1)
        hi_pred = hi_out.cpu().numpy().reshape(-1)
        attn = attn_out.cpu().numpy() if attn_out is not None else None
    else:
        rul_pred = outputs.cpu().numpy().reshape(-1)

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------
    rmse = compute_rmse(torch.tensor(rul_pred), torch.tensor(y_true))
    mae = compute_mae(torch.tensor(rul_pred), torch.tensor(y_true))
    phm = compute_phm_score(torch.tensor(rul_pred), torch.tensor(y_true))
    r2 = compute_r2(torch.tensor(rul_pred), torch.tensor(y_true))

    console.print(f"[cyan]RMSE:[/cyan] {rmse:.3f}")
    console.print(f"[cyan]MAE:[/cyan]  {mae:.3f}")
    console.print(f"[cyan]PHM Score:[/cyan] {phm:.2f}")
    console.print(f"[cyan]R² Score:[/cyan] {r2:.3f}")

    # Save metrics
    save_dir = OUTPUT_ROOT / "evaluation" / subset / model_type
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "metrics.txt", "w") as f:
        f.write(f"RMSE: {rmse:.3f}\n")
        f.write(f"MAE: {mae:.3f}\n")
        f.write(f"PHM Score: {phm:.2f}\n")
        f.write(f"R2: {r2:.3f}\n")

    # -----------------------------------------------------
    # PLOTS
    # -----------------------------------------------------
    plot_pred_vs_true(y_true, rul_pred, save_path=save_dir / "scatter_pred_vs_true.png", rmse=rmse, mae=mae, r2=r2)
    plot_error_histogram(y_true, rul_pred, save_path=save_dir / "error_histogram.png")
    plot_rul_sequence(y_true, rul_pred, save_path=save_dir / "rul_sequence_sample.png")
    plot_full_rul_curve(y_true, rul_pred, save_path=save_dir / "rul_full_curve.png")
    plot_best_worst_rul(y_true, rul_pred, save_path=save_dir / "best_worst_samples.png")
    plot_sensor_degradation(X_test[0], save_path=save_dir / "sensor_degradation_sample.png")

    if hi_pred is not None:
        plot_hi_sequence(hi_pred, save_path=save_dir / "hi_sequence_sample.png")

    if attn is not None:
        plot_attention_curve(attn, save_path=save_dir / "attention_curve.png")

    console.print(f"[green]✔ Saved plots to {save_dir}[/green]")

    # -----------------------------------------------------
    # LLM REASONING (DOCX OUTPUT)
    # -----------------------------------------------------
    
    if model_type == "baseline" or hi_pred is None or attn is None:
        console.print("[yellow]LLM reasoning skipped (baseline model).[/yellow]")
        return

    console.print("[bold magenta]Running LLM-engine health reasoning (DOCX)...[/bold magenta]")

    # Use worst-case sample as the main diagnostic focus
    abs_err = np.abs(rul_pred - y_true)
    worst_idx = int(np.argmax(abs_err))

    sections = llm_engine_explanation(
        subset=subset,
        rul_pred=float(rul_pred[worst_idx]),
        hi_pred=float(hi_pred[worst_idx]),
        sensor_stats=build_sensor_stats(X_test[worst_idx]),
        attention_weights=attn[worst_idx],
        model_name="deepseek-r1",
    )

    docx_path = save_dir / "llm_report.docx"
    save_docx_report(str(docx_path), sections, subset)
    console.print(f"[green]✔ LLM DOCX report saved to {docx_path}[/green]")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", required=True)
    parser.add_argument("--model", required=True, choices=["baseline", "multitask"])
    args = parser.parse_args()
    evaluate(args.subset, args.model)


if __name__ == "__main__":
    main()
    
"""   
- runnung the code for all subsets with multitask model
python -m src.training.eval_rul --subset FD001 --model multitask
python -m src.training.eval_rul --subset FD002 --model multitask
python -m src.training.eval_rul --subset FD003 --model multitask
python -m src.training.eval_rul --subset FD004 --model multitask
"""