"""
Data loading, preprocessing và tạo PyTorch Dataset cho robot arm.

- Split theo episode (không random row) để tránh data leakage.
- Normalize bằng mean/std tính trên train set.
- Hỗ trợ hai chế độ: MLP (single-step) và LSTM (sliding window).
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import config


class Normalizer:
    """Z-score normalization, lưu mean/std để dùng cho inference."""

    def __init__(self):
        self.mean: np.ndarray | None = None
        self.std:  np.ndarray | None = None

    def fit(self, data: np.ndarray) -> "Normalizer":
        self.mean = data.mean(axis=0)
        self.std  = data.std(axis=0)
        # Tránh chia cho 0 với các cột constant (ví dụ gripper luôn = -1)
        self.std = np.where(self.std < 1e-8, 1.0, self.std)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean

    def save(self, path: str):
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: str) -> "Normalizer":
        obj = cls()
        data = np.load(path)
        obj.mean = data["mean"]
        obj.std  = data["std"]
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# MLP Dataset: mỗi sample = (state_t, action_t)
# ─────────────────────────────────────────────────────────────────────────────

class RobotMLPDataset(Dataset):
    """Dataset cho Behavioral Cloning với MLP (single-step)."""

    def __init__(
        self,
        states:     np.ndarray,   # (N, STATE_DIM)  đã normalize
        actions:    np.ndarray,   # (N, ACTION_DIM) đã normalize
        noise_std:  float = 0.0,  # Gaussian noise augmentation trên state
    ):
        self.states    = torch.from_numpy(states).float()
        self.actions   = torch.from_numpy(actions).float()
        self.noise_std = noise_std

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        s = self.states[idx]
        if self.noise_std > 0.0:
            s = s + torch.randn_like(s) * self.noise_std
        return s, self.actions[idx]


# ─────────────────────────────────────────────────────────────────────────────
# LSTM Dataset: mỗi sample = (state_window_t-W..t, action_t)
# ─────────────────────────────────────────────────────────────────────────────

class RobotLSTMDataset(Dataset):
    """Dataset cho Behavioral Cloning với LSTM (sliding window)."""

    def __init__(
        self,
        states:          np.ndarray,
        actions:         np.ndarray,
        window_size:     int = config.LSTM_WINDOW_SIZE,
        episode_lengths: list[int] | None = None,
        noise_std:       float = 0.0,
    ):
        self.window_size = window_size
        self.noise_std   = noise_std

        all_state_windows = []
        all_actions       = []

        if episode_lengths is None:
            episode_lengths = [len(states)]

        idx = 0
        for ep_len in episode_lengths:
            ep_states  = states[idx: idx + ep_len]
            ep_actions = actions[idx: idx + ep_len]
            for t in range(window_size - 1, ep_len):
                window = ep_states[t - window_size + 1: t + 1]   # (W, S)
                all_state_windows.append(window)
                all_actions.append(ep_actions[t])
            idx += ep_len

        self.state_windows = torch.from_numpy(
            np.stack(all_state_windows, axis=0)
        ).float()   # (N', W, STATE_DIM)
        self.actions = torch.from_numpy(
            np.stack(all_actions, axis=0)
        ).float()   # (N', ACTION_DIM)

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int):
        w = self.state_windows[idx]
        if self.noise_std > 0.0:
            w = w + torch.randn_like(w) * self.noise_std
        return w, self.actions[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Hàm tiện ích
# ─────────────────────────────────────────────────────────────────────────────

_OBJ_COLS = [f"state.obj{i}_{a}" for i in range(4) for a in ["x", "y", "z", "qx", "qy", "qz", "qw"]]
_OBJ_VALID_RANGE = (-3.0, 3.0)  # values outside this range are lost-track sentinels


def _impute_obj_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill object pose columns that fall outside the valid spatial range."""
    obj_xyz = [c for c in _OBJ_COLS if c.endswith(("_x", "_y", "_z")) and c in df.columns]
    lo, hi = _OBJ_VALID_RANGE
    mask = pd.DataFrame(False, index=df.index, columns=obj_xyz)
    for c in obj_xyz:
        mask[c] = (df[c] < lo) | (df[c] > hi)

    n_bad = mask.any(axis=1).sum()
    if n_bad:
        # Replace outlier positions with NaN, then forward-fill within each episode
        df = df.copy()
        for c in obj_xyz:
            df.loc[mask[c], c] = np.nan
        # Also NaN the corresponding quaternion cols so orientation stays consistent
        for i in range(4):
            xyz_cols = [f"state.obj{i}_{a}" for a in ["x", "y", "z"]]
            quat_cols = [f"state.obj{i}_{a}" for a in ["qx", "qy", "qz", "qw"]]
            bad_rows = mask[[c for c in xyz_cols if c in mask.columns]].any(axis=1)
            for c in quat_cols:
                if c in df.columns:
                    df.loc[bad_rows, c] = np.nan
        all_obj_cols = [c for c in _OBJ_COLS if c in df.columns]
        for c in all_obj_cols:
            df[c] = df.groupby("episode_index")[c].transform(lambda x: x.ffill().bfill())
        print(f"[data] Imputed {n_bad} frames with out-of-range object poses (forward-fill)")
    return df


def _load_one_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _impute_obj_outliers(df)
    before = len(df)
    df = df.dropna(subset=config.ACTION_COLS).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"[data]   Dropped {dropped} rows with NaN actions")
    return df


def load_csv(
    path: str | None = None,
    extra_paths: list[str] | None = None,
) -> pd.DataFrame:
    """Load one or more CSV files and concatenate them.

    episode_index is re-numbered globally so episodes from different files
    never collide.  All other columns are preserved as-is.

    Args:
        path:        Primary CSV path. Defaults to config.DATA_PATH.
        extra_paths: Additional CSVs to merge. Defaults to config.EXTRA_DATA_PATHS.
    """
    if path is None:
        path = config.DATA_PATH
    if extra_paths is None:
        extra_paths = getattr(config, "EXTRA_DATA_PATHS", [])

    all_paths = [path] + [p for p in extra_paths if p and os.path.exists(p)]

    dfs = []
    ep_offset = 0
    for p in all_paths:
        print(f"[data] Loading {p}")
        df = _load_one_csv(p)
        if df.empty:
            continue
        # Re-number episodes to avoid index collision across files
        ep_map = {old: new + ep_offset
                  for new, old in enumerate(sorted(df["episode_index"].unique()))}
        df["episode_index"] = df["episode_index"].map(ep_map)
        ep_offset += len(ep_map)
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No valid CSV found in {all_paths}")

    result = pd.concat(dfs, ignore_index=True)
    print(f"[data] Total: {len(result):,} rows, {result['episode_index'].nunique()} episodes "
          f"(from {len(dfs)} file(s))")
    return result


def episode_split(
    df: pd.DataFrame,
    val_ratio:  float = config.VAL_RATIO,
    test_ratio: float = config.TEST_RATIO,
    seed:       int   = config.SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chia train/val/test theo episode, không theo row."""
    rng = np.random.default_rng(seed)
    eps = df["episode_index"].unique()
    rng.shuffle(eps)

    n_test  = max(1, int(len(eps) * test_ratio))
    n_val   = max(1, int(len(eps) * val_ratio))

    test_eps  = eps[:n_test]
    val_eps   = eps[n_test: n_test + n_val]
    train_eps = eps[n_test + n_val:]

    train_df = df[df["episode_index"].isin(train_eps)].reset_index(drop=True)
    val_df   = df[df["episode_index"].isin(val_eps)].reset_index(drop=True)
    test_df  = df[df["episode_index"].isin(test_eps)].reset_index(drop=True)

    print(f"[data] Split — train: {len(train_eps)} eps ({len(train_df):,} rows) | "
          f"val: {len(val_eps)} eps ({len(val_df):,} rows) | "
          f"test: {len(test_eps)} eps ({len(test_df):,} rows)")
    return train_df, val_df, test_df


def df_to_arrays(df: pd.DataFrame):
    """Trả về (states, actions, episode_lengths) dưới dạng numpy."""
    states  = df[config.STATE_COLS].values.astype(np.float32)
    actions = df[config.ACTION_COLS].values.astype(np.float32)
    ep_lengths = df.groupby("episode_index", sort=False).size().tolist()
    return states, actions, ep_lengths


def build_mlp_loaders(
    batch_size:  int   = config.BATCH_SIZE,
    noise_std:   float = config.STATE_NOISE_STD,
    num_workers: int   = 2,
) -> tuple[DataLoader, DataLoader, DataLoader, Normalizer, Normalizer]:
    """
    Trả về (train_loader, val_loader, test_loader, state_norm, action_norm).
    noise_std chỉ áp dụng cho train loader.
    """
    df = load_csv()
    train_df, val_df, test_df = episode_split(df)

    train_s, train_a, _ = df_to_arrays(train_df)
    val_s,   val_a,   _ = df_to_arrays(val_df)
    test_s,  test_a,  _ = df_to_arrays(test_df)

    s_norm = Normalizer().fit(train_s)
    a_norm = Normalizer().fit(train_a)

    def make_loader(s, a, shuffle, noise=0.0):
        ds = RobotMLPDataset(s_norm.transform(s), a_norm.transform(a),
                             noise_std=noise)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True)

    return (
        make_loader(train_s, train_a, shuffle=True,  noise=noise_std),
        make_loader(val_s,   val_a,   shuffle=False, noise=0.0),
        make_loader(test_s,  test_a,  shuffle=False, noise=0.0),
        s_norm,
        a_norm,
    )


def build_lstm_loaders(
    batch_size:  int   = config.BATCH_SIZE,
    window_size: int   = config.LSTM_WINDOW_SIZE,
    noise_std:   float = config.STATE_NOISE_STD,
    num_workers: int   = 2,
) -> tuple[DataLoader, DataLoader, DataLoader, Normalizer, Normalizer]:
    """
    Trả về (train_loader, val_loader, test_loader, state_norm, action_norm).
    noise_std chỉ áp dụng cho train loader.
    """
    df = load_csv()
    train_df, val_df, test_df = episode_split(df)

    train_s, train_a, train_ep = df_to_arrays(train_df)
    val_s,   val_a,   val_ep   = df_to_arrays(val_df)
    test_s,  test_a,  test_ep  = df_to_arrays(test_df)

    s_norm = Normalizer().fit(train_s)
    a_norm = Normalizer().fit(train_a)

    def make_loader(s, a, eps, shuffle, noise=0.0):
        ds = RobotLSTMDataset(s_norm.transform(s), a_norm.transform(a),
                              window_size=window_size, episode_lengths=eps,
                              noise_std=noise)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True)

    return (
        make_loader(train_s, train_a, train_ep, shuffle=True,  noise=noise_std),
        make_loader(val_s,   val_a,   val_ep,   shuffle=False, noise=0.0),
        make_loader(test_s,  test_a,  test_ep,  shuffle=False, noise=0.0),
        s_norm,
        a_norm,
    )


if __name__ == "__main__":
    # Kiểm tra nhanh
    train_l, val_l, test_l, sn, an = build_mlp_loaders()
    for s, a in train_l:
        print(f"MLP batch — state: {s.shape}, action: {a.shape}")
        break

    train_l, val_l, test_l, sn, an = build_lstm_loaders()
    for s, a in train_l:
        print(f"LSTM batch — state: {s.shape}, action: {a.shape}")
        break
