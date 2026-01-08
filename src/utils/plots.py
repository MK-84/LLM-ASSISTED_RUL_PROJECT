# src/utils/plots.py
# ------------------------------------------------------------------
# Used by eval_rul.py to generate high-quality evaluation figures
# ------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Global styling -------------------------------------------------------------
sns.set_theme(style="whitegrid")
AERO_BLUE = "#4A708B"     # steel blue
AERO_TEAL = "#5FB3B3"     # teal
AERO_ORANGE = "#E69F00"   # orange for true RUL
AERO_RED = "#D55E00"      # worst point
AERO_GREEN = "#009E73"    # best point

plt.rcParams["figure.dpi"] = 130
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["font.size"] = 12
plt.rcParams["legend.fontsize"] = 12
plt.rcParams["figure.facecolor"] = "white"


# ======================================================================
# 1. SCATTER PLOT: Predicted vs True RUL
# ======================================================================
def plot_pred_vs_true(y_true, y_pred, save_path, rmse=None, mae=None, r2=None):
    plt.figure(figsize=(9, 9))

    # Scatter Points
    plt.scatter(
        y_true,
        y_pred,
        s=12,
        alpha=0.25,
        color=AERO_BLUE,
        label="Predicted Samples"
    )

    # Diagonal Line (Ideal)
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot(
        [0, max_val],
        [0, max_val],
        color="black",
        linestyle="-",
        linewidth=1.3,
        label="Ideal Prediction Line"
    )

    plt.xlabel("True RUL")
    plt.ylabel("Predicted RUL")
    plt.title("Predicted vs True RUL")

    # Metrics Panel (Bottom-Right)
    if rmse is not None:
        text = f"RMSE = {rmse:.2f}\nMAE = {mae:.2f}\nR² = {r2:.3f}"
        plt.text(
            0.98, 0.02, text,
            transform=plt.gca().transAxes,
            ha="right",
            va="bottom",
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="gray")
        )

    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


# ======================================================================
# 2. HISTOGRAM: Prediction Error Distribution
# ======================================================================
def plot_error_histogram(y_true, y_pred, save_path):
    errors = y_pred - y_true

    plt.figure(figsize=(10, 6))

    sns.histplot(
        errors,
        bins=50,
        kde=True,
        color=AERO_RED,
        alpha=0.55,
        edgecolor="black",
        label="Prediction Error"
    )

    # Zero Error Line
    plt.axvline(0, color="black", linestyle="--", linewidth=1.4, label="Zero Error")

    plt.xlabel("Prediction Error (Predicted - True)")
    plt.ylabel("Frequency")
    plt.title("RUL Prediction Error Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


# ======================================================================
# 3. RUL Sequence Plot (First 300 Samples)
# ======================================================================
def plot_rul_sequence(y_true, y_pred, save_path, window=300):
    plt.figure(figsize=(12, 5))

    plt.plot(
        y_true[:window],
        label="True RUL",
        color=AERO_ORANGE,
        linewidth=2
    )
    plt.plot(
        y_pred[:window],
        label="Predicted RUL",
        color=AERO_TEAL,
        linewidth=2
    )

    plt.xlabel("Time Step")
    plt.ylabel("RUL")
    plt.title("RUL Prediction Sequence (First 300 Samples)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


# ======================================================================
# 4. FULL TEST SET RUL CURVE
# ======================================================================
def plot_full_rul_curve(y_true, y_pred, save_path):
    plt.figure(figsize=(14, 5))

    plt.plot(
        y_true,
        label="True RUL",
        color=AERO_ORANGE,
        linewidth=1.4
    )
    plt.plot(
        y_pred,
        label="Predicted RUL",
        color=AERO_TEAL,
        linewidth=1.2,
        alpha=0.8
    )

    plt.xlabel("Sample Index")
    plt.ylabel("RUL")
    plt.title("RUL Prediction Over Entire Test Set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


# ======================================================================
# 5. HEALTH INDEX SEQUENCE (Multitask Only)
# ======================================================================
def plot_hi_sequence(hi_pred, save_path):
    plt.figure(figsize=(14, 5))

    plt.plot(
        hi_pred,
        color=AERO_GREEN,
        linewidth=1,
        alpha=0.8,
        label="Health Index (HI)"
    )

    plt.xlabel("Time Step")
    plt.ylabel("Health Index (HI)")
    plt.title("Engine Health Index Sequence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


# ======================================================================
# 6. SENSOR DEGRADATION OVER TIME
# ======================================================================
def plot_sensor_degradation(sensor_data, save_path):
    plt.figure(figsize=(14, 6))

    num_sensors = sensor_data.shape[-1]
    timesteps = np.arange(sensor_data.shape[0])

    # Use seaborn color palette for sensor lines
    palette = sns.color_palette("husl", num_sensors)

    for i in range(num_sensors):
        plt.plot(
            timesteps,
            sensor_data[:, i],
            color=palette[i],
            linewidth=1.3,
            label=f"Sensor {i+1}"
        )

    plt.xlabel("Time Step")
    plt.ylabel("Sensor Reading")
    plt.title("Sensor Degradation Over Time")

    # Legend outside right
    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()


# ======================================================================
# 7. BEST vs WORST CASE COMPARISON
# ======================================================================
def plot_best_worst_rul(y_true, y_pred, save_path):
    errors = np.abs(y_pred - y_true)
    best = np.argmin(errors)
    worst = np.argmax(errors)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ------------------------- BEST CASE ------------------------------
    ax = axes[0]
    ax.scatter(
        y_true[best],
        y_pred[best],
        s=100,
        color=AERO_GREEN,
        zorder=5,
        label=f"Predicted (Best)"
    )
    max_val = max(y_true[best], y_pred[best])
    ax.plot([0, max_val], [0, max_val], "k-", label="Ideal Line")
    ax.set_title(f"Best Prediction\nError={errors[best]:.2f}")
    ax.set_xlabel("True RUL")
    ax.set_ylabel("Predicted RUL")
    ax.legend()

    # ------------------------- WORST CASE ------------------------------
    ax = axes[1]
    ax.scatter(
        y_true[worst],
        y_pred[worst],
        s=100,
        color=AERO_RED,
        zorder=5,
        label=f"Predicted (Worst)"
    )
    max_val = max(y_true[worst], y_pred[worst])
    ax.plot([0, max_val], [0, max_val], "k-", label="Ideal Line")
    ax.set_title(f"Worst Prediction\nError={errors[worst]:.2f}")
    ax.set_xlabel("True RUL")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


# ======================================================================
# 8. ATTENTION WEIGHTS OVER TIME (Temporal Attention Curve)
# ======================================================================
def plot_attention_curve(attention, save_path):
    """
    Handles attention arrays safely.
    If attention is (T, 1) or (T,), plot line.
    If attention is (T, F), plot heatmap.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    att = np.asarray(attention)

    # Case 1: 3D input (batch, T, F) → pick first sample
    if att.ndim == 3:
        att = att[0]

    # Case 2: (T, 1) → line plot
    if att.ndim == 2 and att.shape[1] == 1:
        att = att.reshape(-1)

        plt.figure(figsize=(12, 4))
        plt.plot(att, color="#006699", linewidth=2)
        plt.title("Attention Weights Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Attention Weight")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        return

    # Case 3: (T,) → line plot
    if att.ndim == 1:
        plt.figure(figsize=(12, 4))
        plt.plot(att, color="#006699", linewidth=2)
        plt.title("Attention Weights Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Attention Weight")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        return

    # Case 4: (T, F) → proper heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(att, cmap="YlGnBu", linewidths=0.2, cbar_kws={"label": "Attention Weight"})
    plt.title("Attention Heatmap")
    plt.xlabel("Feature Index")
    plt.ylabel("Time Step")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ======================================================================