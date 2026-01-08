# src/training/train_baseline.py
"""
Thin wrapper to train the BiLSTM baseline RUL model on FD001–FD004.

It reuses the existing training pipeline in train_hi_rul.py, but
switches model_type="baseline" so that:
- build_model(...) constructs BiLSTM_RUL_Base
- build_dataloaders(...) is configured with multitask=False
- TRAIN_CONFIGS, DEVICE, metrics, etc. stay identical
"""

from rich.console import Console

from src.training.train_hi_rul import train_all  # uses TRAIN_CONFIGS, build_dataloaders, etc.

console = Console()


def main():
    console.print(
        "[bold yellow]🚀 Starting BASELINE training for all subsets "
        "(FD001–FD004) — Model: BiLSTM_RUL_Base[/bold yellow]\n"
    )
    # This calls train_fd_subset(...) for each subset with model_type="baseline"
    train_all(model_type="baseline")
    console.print("[bold green]✔ Baseline training complete![/bold green]")


if __name__ == "__main__":
    main()