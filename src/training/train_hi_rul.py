# src/training/train_hi_rul.py

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, BarColumn, TimeRemainingColumn, TextColumn

from src.config import TRAIN_CONFIGS, OUTPUT_ROOT, DEVICE
from src.data.datasets import build_dataloaders
from src.models.bilstm_baseline import BiLSTM_RUL_Base
from src.models.tcn_bilstm_dual_attn import TCN_BiLSTM_DualAttn
from src.utils.metrics import compute_rmse, compute_phm_score

console = Console()

# ------------------------------------------------------------------
# GLOBAL DEVICE
# ------------------------------------------------------------------
DEVICE_TORCH = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
console.print(f"[bold cyan]Using device: {DEVICE_TORCH}[/bold cyan]")


# ============================================================
# Build the model
# ============================================================
def build_model(model_type: str, input_dim: int, config: dict):
    """Creates either the baseline BiLSTM or the multitask TCN-BiLSTM."""

    if model_type == "baseline":
        model = BiLSTM_RUL_Base(
            input_dim=input_dim,
            hidden_dim=config["lstm_hidden"],
            num_layers=config["lstm_layers"],
            dropout=config["dropout"],
        )
        multitask = False

    elif model_type == "multitask":
        model = TCN_BiLSTM_DualAttn(
            input_dim=input_dim,
            conv_channels=config["conv_channels"],
            lstm_hidden=config["lstm_hidden"],
            lstm_layers=config["lstm_layers"],
            dropout=config["dropout"],
        )
        multitask = True

    else:
        raise ValueError("model_type must be 'baseline' or 'multitask'")

    # Move model to GPU / CPU
    model = model.to(DEVICE_TORCH)
    return model, multitask


# ============================================================
# Safe model output handling
# ============================================================
def unpack_outputs(outputs):
    """
    Ensures compatibility with:
    - baseline: (rul, hi, None)
    - multitask: (rul, hi, attn)
    - single-output models (rare)
    """

    # Multitask or baseline models → tuple length 3
    if isinstance(outputs, tuple) and len(outputs) == 3:
        pred_rul, pred_hi, _ = outputs
        return pred_rul, pred_hi

    # Single output model → Tensor
    if isinstance(outputs, torch.Tensor):
        return outputs, None

    raise ValueError(f"Unexpected model output structure: {outputs}")


# ============================================================
# One training epoch
# ============================================================
def train_one_epoch(model, loader, optimizer, loss_rul_fn, loss_hi_fn, multitask: bool):
    model.train()
    total_loss = 0.0

    for batch in loader:
        optimizer.zero_grad()

        # Handle datasets with or without HI
        if multitask:
            X, y_rul, y_hi = batch
            y_hi = y_hi.to(DEVICE_TORCH)
        else:
            X, y_rul = batch
            y_hi = None

        X = X.to(DEVICE_TORCH)
        y_rul = y_rul.to(DEVICE_TORCH)

        pred_rul, pred_hi = unpack_outputs(model(X))

        # RUL loss
        loss_rul = loss_rul_fn(pred_rul, y_rul)

        # HI loss only if multitask and HI head exists
        if multitask and (pred_hi is not None) and (y_hi is not None):
            loss_hi = loss_hi_fn(pred_hi, y_hi)
            loss = loss_rul + 0.5 * loss_hi
        else:
            loss = loss_rul

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


# ============================================================
# Validation epoch
# ============================================================
@torch.no_grad()
def validate(model, loader, loss_fn, multitask: bool):
    model.eval()
    total_loss = 0.0
    preds, trues = [], []

    for batch in loader:
        if multitask:
            X, y_rul, _ = batch
        else:
            X, y_rul = batch

        X = X.to(DEVICE_TORCH)
        y_rul = y_rul.to(DEVICE_TORCH)

        pred_rul, _ = unpack_outputs(model(X))

        loss = loss_fn(pred_rul, y_rul)
        total_loss += loss.item()

        preds.append(pred_rul.detach().cpu())
        trues.append(y_rul.detach().cpu())

    preds = torch.cat(preds)
    trues = torch.cat(trues)

    rmse = compute_rmse(preds, trues)
    return total_loss / len(loader), rmse


# ============================================================
# Train on a single FD subset
# ============================================================
def train_fd_subset(subset: str, model_type: str):
    console.rule(f"[bold magenta] Training {subset} — Model: {model_type} [/bold magenta]")

    config = TRAIN_CONFIGS[subset]

    # Load dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(
        subset=subset,
        batch_size=config["batch_size"],
        multitask=(model_type == "multitask"),
    )

    # Infer input dimension
    sample_batch = next(iter(train_loader))[0]
    input_dim = sample_batch.shape[-1]

    # Build model
    model, multitask = build_model(model_type, input_dim, config)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    loss_rul_fn = nn.MSELoss()
    loss_hi_fn = nn.MSELoss()

    with Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        console=console
    ) as progress:

        task = progress.add_task(f"Training {subset}", total=config["num_epochs"])
        best_rmse = float("inf")

        ckpt_dir = OUTPUT_ROOT / "checkpoints" / subset
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(config["num_epochs"]):

            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                loss_rul_fn=loss_rul_fn,
                loss_hi_fn=loss_hi_fn,
                multitask=multitask,
            )

            val_loss, val_rmse = validate(
                model=model,
                loader=val_loader,
                loss_fn=loss_rul_fn,
                multitask=multitask,
            )

            progress.update(task, advance=1)

            console.print(
                f"[cyan]Epoch {epoch+1:02d}/{config['num_epochs']}[/cyan]  "
                f"│  [green]Train:[/green] {train_loss:10.4f}  "
                f"│  [magenta]Val RMSE:[/magenta] {val_rmse:7.3f}"
            )


            # Save best checkpoint
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                torch.save(model.state_dict(), ckpt_dir / f"{model_type}_best.pt")

    console.print(
        f"[bold green]✔ Finished {subset} — Best Val RMSE: {best_rmse:.3f}[/bold green]\n"
    )


# ============================================================
# Train all subsets FD001–FD004
# ============================================================
def train_all(model_type: str = "multitask"):
    for subset in ["FD001", "FD002", "FD003", "FD004"]:
        train_fd_subset(subset, model_type=model_type)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    console.print("[bold yellow]🚀 Starting training for all subsets[/bold yellow]\n")
    train_all(model_type="multitask")
    console.print("[bold green]✔ Training complete![/bold green]")
