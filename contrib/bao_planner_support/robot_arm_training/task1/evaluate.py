"""
Evaluation và inference cho model đã train.

Tính các metrics:
  - MSE (Mean Squared Error)
  - MAE (Mean Absolute Error)
  - Per-joint MAE để xem joint nào khó predict nhất

Cách dùng:
    python evaluate.py --model mlp
    python evaluate.py --model lstm
    python evaluate.py --model mlp --plot    # vẽ biểu đồ (cần matplotlib)
"""

import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "shared"))
sys.path.insert(0, _HERE)

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config
from data_loader import (
    Normalizer, build_mlp_loaders, build_lstm_loaders, load_csv, episode_split,
    df_to_arrays,
)
from model import build_model, count_parameters


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    """Tính MSE, RMSE, MAE toàn bộ và per-joint."""
    err      = preds - targets
    mse      = float(np.mean(err ** 2))
    rmse     = float(np.sqrt(mse))
    mae      = float(np.mean(np.abs(err)))
    per_joint_mae = np.mean(np.abs(err), axis=0)   # (ACTION_DIM,)

    return {
        "MSE":  mse,
        "RMSE": rmse,
        "MAE":  mae,
        "per_joint_mae": per_joint_mae.tolist(),
    }


def print_report(metrics: dict, action_norm: Normalizer | None = None):
    """In báo cáo metrics ra console.
       Nếu truyền action_norm thì hiển thị thêm lỗi ở đơn vị gốc (radian).
    """
    print(f"\n{'='*55}")
    print(f"  {'Metric':<15} {'Normalized':>15} {'Original (rad)':>15}")
    print(f"{'='*55}")

    # Các scalar metrics
    for key in ["MSE", "RMSE", "MAE"]:
        val_norm = metrics[key]
        if action_norm is not None:
            # Ước tính: nhân với mean std để có đơn vị gốc
            mean_std = float(np.mean(action_norm.std))
            if key == "MSE":
                val_orig = val_norm * (mean_std ** 2)
            else:
                val_orig = val_norm * mean_std
            print(f"  {key:<15} {val_norm:>15.6f} {val_orig:>15.6f}")
        else:
            print(f"  {key:<15} {val_norm:>15.6f}")

    print(f"\n  Per-joint MAE (normalized):")
    print(f"  {'Joint':<45} {'MAE':>10}")
    print(f"  {'-'*55}")
    for j, (name, val) in enumerate(zip(config.ACTION_COLS, metrics["per_joint_mae"])):
        short = name.replace("action.", "").replace("_joint.pos", "")
        print(f"  {short:<45} {val:>10.6f}")
    print(f"{'='*55}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Collect predictions on entire loader
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions_mlp(
    model: nn.Module,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_targets = [], []
    for states, actions in loader:
        preds = model(states.to(device)).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(actions.numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


@torch.no_grad()
def collect_predictions_lstm(
    model: nn.Module,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_targets = [], []
    for windows, actions in loader:
        preds, _ = model(windows.to(device))
        all_preds.append(preds.cpu().numpy())
        all_targets.append(actions.numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


# ─────────────────────────────────────────────────────────────────────────────
# Rollout simulation: chạy model qua một episode và so sánh trajectory
# ─────────────────────────────────────────────────────────────────────────────

def rollout_episode(
    model:      nn.Module,
    df:         pd.DataFrame,
    episode_id: int,
    s_norm:     Normalizer,
    a_norm:     Normalizer,
    model_type: str,
    device:     torch.device,
    window_size: int = config.LSTM_WINDOW_SIZE,
) -> dict:
    """
    Chạy model trên một episode và trả về predicted vs. true actions.
    """
    ep_df   = df[df["episode_index"] == episode_id].sort_values("frame_index")
    states  = ep_df[config.STATE_COLS].values.astype(np.float32)
    actions = ep_df[config.ACTION_COLS].values.astype(np.float32)
    T       = len(states)

    states_norm  = s_norm.transform(states)
    actions_norm = a_norm.transform(actions)

    model.eval()
    pred_list = []

    with torch.no_grad():
        if model_type == "mlp":
            s_tensor = torch.from_numpy(states_norm).float().to(device)
            preds    = model(s_tensor).cpu().numpy()
            pred_list = preds
        else:
            hidden = None
            for t in range(T):
                start  = max(0, t - window_size + 1)
                window = states_norm[start: t + 1]
                # Pad nếu chưa đủ window
                if len(window) < window_size:
                    pad    = np.repeat(window[:1], window_size - len(window), axis=0)
                    window = np.concatenate([pad, window], axis=0)
                window_t = torch.from_numpy(window).float().unsqueeze(0).to(device)
                pred, hidden = model(window_t, hidden)
                pred_list.append(pred.squeeze(0).cpu().numpy())
            pred_list = np.stack(pred_list, axis=0)

    # Inverse normalize để ra đơn vị radian
    pred_orig   = a_norm.inverse_transform(pred_list)
    target_orig = actions

    mae_per_step = np.mean(np.abs(pred_orig - target_orig), axis=1)  # (T,)

    return {
        "episode_id":    episode_id,
        "T":             T,
        "pred_actions":  pred_orig,
        "true_actions":  target_orig,
        "mae_per_step":  mae_per_step,
        "mean_mae":      float(mae_per_step.mean()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Optional: plot (chỉ dùng nếu có matplotlib)
# ─────────────────────────────────────────────────────────────────────────────

def plot_rollout(rollout: dict, save_path: str | None = None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[eval] matplotlib không có, bỏ qua plot.")
        return

    T    = rollout["T"]
    pred = rollout["pred_actions"]
    true = rollout["true_actions"]
    t    = np.arange(T)

    fig, axes = plt.subplots(4, 5, figsize=(20, 12))
    axes_flat = axes.flatten()
    for j, (ax, name) in enumerate(zip(axes_flat, config.ACTION_COLS)):
        short = name.replace("action.", "").replace("_joint.pos", "")
        ax.plot(t, true[:, j], label="GT",   color="steelblue",  linewidth=1.5)
        ax.plot(t, pred[:, j], label="Pred", color="darkorange", linewidth=1.5,
                linestyle="--")
        ax.set_title(short, fontsize=8)
        ax.set_xlabel("Frame", fontsize=7)
        ax.set_ylabel("rad", fontsize=7)
        ax.tick_params(labelsize=6)
        if j == 0:
            ax.legend(fontsize=7)

    fig.suptitle(f"Episode {rollout['episode_id']} — Mean MAE = "
                 f"{rollout['mean_mae']:.4f} rad", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100)
        print(f"[eval] Plot saved → {save_path}")
    else:
        plt.show()
    plt.close()


def plot_training_history(history_path: str, save_path: str | None = None):
    try:
        import json
        import matplotlib.pyplot as plt
    except ImportError:
        print("[eval] matplotlib không có, bỏ qua plot.")
        return

    with open(history_path) as f:
        h = json.load(f)

    epochs = range(1, len(h["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, h["train_loss"], label="Train MSE")
    axes[0].plot(epochs, h["val_loss"],   label="Val MSE")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(epochs, h["lr"])
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_title("Learning Rate Schedule")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100)
        print(f"[eval] History plot saved → {save_path}")
    else:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(args):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir  = os.path.join(config.CHECKPOINT_DIR, args.model)
    best_ckpt = os.path.join(ckpt_dir, "best.pt")

    if not os.path.exists(best_ckpt):
        print(f"[eval] Không tìm thấy checkpoint: {best_ckpt}")
        print(f"       Hãy train trước: python train.py --model {args.model}")
        return

    # Load normalizer
    s_norm = Normalizer.load(os.path.join(ckpt_dir, "state_norm.npz"))
    a_norm = Normalizer.load(os.path.join(ckpt_dir, "action_norm.npz"))

    # Load model
    model = build_model(args.model).to(device)
    ckpt  = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"[eval] Loaded {args.model.upper()} from epoch {ckpt['epoch']} "
          f"| params: {count_parameters(model):,}")

    # Test loader
    if args.model == "mlp":
        _, _, test_loader, _, _ = build_mlp_loaders(batch_size=512)
        preds, targets = collect_predictions_mlp(model, test_loader, device)
    else:
        _, _, test_loader, _, _ = build_lstm_loaders(batch_size=512)
        preds, targets = collect_predictions_lstm(model, test_loader, device)

    metrics = compute_metrics(preds, targets)
    print_report(metrics, action_norm=a_norm)

    # Rollout một episode
    df  = load_csv()
    _, _, test_df = episode_split(df)
    ep  = test_df["episode_index"].iloc[0]
    print(f"[eval] Rollout episode {ep}...")
    rollout = rollout_episode(model, df, ep, s_norm, a_norm,
                              args.model, device)
    print(f"[eval] Episode {ep}: T={rollout['T']} frames, "
          f"Mean MAE = {rollout['mean_mae']:.4f} rad")

    if args.plot:
        plot_rollout(
            rollout,
            save_path=os.path.join(ckpt_dir, f"rollout_ep{ep}.png"),
        )
        history_path = os.path.join(ckpt_dir, "history.json")
        if os.path.exists(history_path):
            plot_training_history(
                history_path,
                save_path=os.path.join(ckpt_dir, "training_history.png"),
            )


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate robot arm BC policy")
    p.add_argument("--model", choices=["mlp", "lstm"], default="mlp")
    p.add_argument("--plot",  action="store_true",
                   help="Vẽ biểu đồ rollout và training history")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
