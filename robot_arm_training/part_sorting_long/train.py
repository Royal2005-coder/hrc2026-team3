"""
Training Behavioral Cloning cho Part_Sorting Long dataset (756 episodes, 4 grasps/ep).

Cách dùng:
    python train.py                    # MLP, default hyperparams
    python train.py --model lstm       # LSTM với sliding window
    python train.py --model mlp --epochs 200 --lr 5e-4 --batch 256
    python train.py --model lstm --window 30 --patience 40
"""

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "shared"))  # model.py
sys.path.insert(0, _HERE)                                 # config.py, data_loader.py

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

import config
from data_loader import build_mlp_loaders, build_lstm_loaders
from model import build_model, count_parameters


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = config.SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum   += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Train / eval một epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch_mlp(model, loader, optimizer, criterion, device, grad_clip: float) -> float:
    model.train()
    meter = AverageMeter()
    for states, actions in loader:
        states  = states.to(device)
        actions = actions.to(device)
        optimizer.zero_grad()
        loss = criterion(model(states), actions)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        meter.update(loss.item(), states.size(0))
    return meter.avg


@torch.no_grad()
def eval_epoch_mlp(model, loader, criterion, device) -> float:
    model.eval()
    meter = AverageMeter()
    for states, actions in loader:
        states  = states.to(device)
        actions = actions.to(device)
        meter.update(criterion(model(states), actions).item(), states.size(0))
    return meter.avg


def train_epoch_lstm(model, loader, optimizer, criterion, device, grad_clip: float) -> float:
    model.train()
    meter = AverageMeter()
    for windows, actions in loader:
        windows = windows.to(device)
        actions = actions.to(device)
        optimizer.zero_grad()
        preds, _ = model(windows)
        loss = criterion(preds, actions)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        meter.update(loss.item(), windows.size(0))
    return meter.avg


@torch.no_grad()
def eval_epoch_lstm(model, loader, criterion, device) -> float:
    model.eval()
    meter = AverageMeter()
    for windows, actions in loader:
        windows = windows.to(device)
        actions = actions.to(device)
        preds, _ = model(windows)
        meter.update(criterion(preds, actions).item(), windows.size(0))
    return meter.avg


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model: nn.Module):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    return ckpt.get("epoch", 0), ckpt.get("best_val_loss", float("inf"))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    set_seed(config.SEED)
    device = get_device()
    print(f"[train] Device: {device} | Model: {args.model.upper()}")

    # ── Data ──────────────────────────────────────────────────────────────────
    if args.model == "mlp":
        train_loader, val_loader, test_loader, s_norm, a_norm = \
            build_mlp_loaders(batch_size=args.batch)
        train_fn = train_epoch_mlp
        eval_fn  = eval_epoch_mlp
    else:
        train_loader, val_loader, test_loader, s_norm, a_norm = \
            build_lstm_loaders(batch_size=args.batch, window_size=args.window)
        train_fn = train_epoch_lstm
        eval_fn  = eval_epoch_lstm

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(args.model).to(device)
    print(f"[train] Parameters: {count_parameters(model):,}")

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr / 100)

    # ── Checkpoint dir ────────────────────────────────────────────────────────
    ckpt_dir  = os.path.join(config.CHECKPOINT_DIR, args.model)
    best_ckpt = os.path.join(ckpt_dir, "best.pt")
    last_ckpt = os.path.join(ckpt_dir, "last.pt")
    os.makedirs(ckpt_dir, exist_ok=True)

    s_norm.save(os.path.join(ckpt_dir, "state_norm.npz"))
    a_norm.save(os.path.join(ckpt_dir, "action_norm.npz"))

    # ── Loop ──────────────────────────────────────────────────────────────────
    best_val_loss  = float("inf")
    ema_val_loss   = float("inf")
    best_ema_val   = float("inf")
    patience_count = 0
    history        = {"train_loss": [], "val_loss": [], "lr": []}

    print(f"\n{'Epoch':>6} {'Train MSE':>12} {'Val MSE':>12} {'LR':>10} {'Time':>8}")
    print("-" * 55)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_fn(model, train_loader, optimizer, criterion, device, config.GRAD_CLIP)
        val_loss   = eval_fn(model, val_loader, criterion, device)
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        elapsed    = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        log_this = (epoch % config.LOG_EVERY_N_EPOCHS == 0 or epoch == 1)
        if log_this:
            print(f"{epoch:>6d} {train_loss:>12.6f} {val_loss:>12.6f} "
                  f"{current_lr:>10.2e} {elapsed:>7.1f}s")

        # Checkpoint
        ckpt_data = {
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss":   best_val_loss,
            "args":            vars(args),
        }
        save_checkpoint(ckpt_data, last_ckpt)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(ckpt_data, best_ckpt)
            if log_this:
                print(f"       ↳ New best val MSE: {best_val_loss:.6f}")

        # Early stopping theo EMA (tránh dừng vì 1 epoch nhiễu)
        alpha        = config.VAL_EMA_ALPHA
        ema_val_loss = val_loss if epoch == 1 else alpha * val_loss + (1 - alpha) * ema_val_loss
        if ema_val_loss < best_ema_val:
            best_ema_val   = ema_val_loss
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"\n[train] Early stopping tại epoch {epoch} "
                      f"(EMA val không cải thiện trong {args.patience} epochs)")
                break

    # ── Test evaluation ───────────────────────────────────────────────────────
    load_checkpoint(best_ckpt, model)
    model.to(device)
    test_loss = eval_fn(model, test_loader, criterion, device)
    print(f"\n[train] Test MSE (best checkpoint): {test_loss:.6f}")

    history_path = os.path.join(ckpt_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[train] History → {history_path}")
    print(f"[train] Best checkpoint → {best_ckpt}")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="BC training — Part Sorting Long dataset")
    p.add_argument("--model",    choices=["mlp", "lstm"], default="mlp")
    p.add_argument("--epochs",   type=int,   default=config.NUM_EPOCHS)
    p.add_argument("--lr",       type=float, default=config.LEARNING_RATE)
    p.add_argument("--batch",    type=int,   default=config.BATCH_SIZE)
    p.add_argument("--window",   type=int,   default=config.LSTM_WINDOW_SIZE,
                   help="LSTM sliding window size")
    p.add_argument("--patience", type=int,   default=config.PATIENCE)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
