"""
Data loading từ LeRobot parquet format cho Part_Sorting Long dataset.

- Đọc trực tiếp file parquet (không qua CSV).
- Split theo episode để tránh data leakage.
- Normalize bằng mean/std tính trên train set.
- Hỗ trợ MLP (single-step) và LSTM (sliding window).
"""

import glob
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import config


# ─────────────────────────────────────────────────────────────────────────────
# Normalizer
# ─────────────────────────────────────────────────────────────────────────────

class Normalizer:
    def __init__(self):
        self.mean: np.ndarray | None = None
        self.std:  np.ndarray | None = None

    def fit(self, data: np.ndarray) -> "Normalizer":
        self.mean = data.mean(axis=0)
        self.std  = data.std(axis=0)
        # Tránh chia cho 0 với cột constant (vd: gripper luôn = -1)
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
        d = np.load(path)
        obj.mean = d["mean"]
        obj.std  = d["std"]
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# Load parquet
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(
    dataset_dir: str = config.DATASET_DIR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Đọc tất cả parquet chunk, trả về:
      states      (N, STATE_DIM)  float32
      actions     (N, ACTION_DIM) float32
      episode_ids (N,)            int64  — để dùng cho episode_split
    """
    data_dir = os.path.join(dataset_dir, "data")
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True))

    if not files:
        raise FileNotFoundError(f"Không tìm thấy parquet trong {data_dir}")

    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    # Sắp xếp theo episode rồi frame để đảm bảo thứ tự time-series đúng
    df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

    states      = np.stack(df["observation.state"].values).astype(np.float32)
    actions     = np.stack(df["action"].values).astype(np.float32)
    episode_ids = df["episode_index"].values.astype(np.int64)

    # Chọn feature subset
    states  = states[:, config.STATE_IDX]
    actions = actions[:, config.ACTION_IDX]

    n_eps = np.unique(episode_ids).size
    print(f"[data] Loaded {len(df):,} frames | {n_eps} episodes | "
          f"state={states.shape[1]}d action={actions.shape[1]}d "
          f"(từ {len(files)} file(s))")
    return states, actions, episode_ids


# ─────────────────────────────────────────────────────────────────────────────
# Train / val / test split theo episode
# ─────────────────────────────────────────────────────────────────────────────

def episode_split(
    states:      np.ndarray,
    actions:     np.ndarray,
    episode_ids: np.ndarray,
    val_ratio:   float = config.VAL_RATIO,
    test_ratio:  float = config.TEST_RATIO,
    seed:        int   = config.SEED,
) -> tuple:
    """
    Chia dữ liệu theo episode (không theo row) để tránh leakage.

    Trả về 3 tuple (states, actions, ep_lengths) cho train/val/test.
    ep_lengths là list số frame mỗi episode, cần cho LSTM Dataset.
    """
    rng = np.random.default_rng(seed)
    unique_eps = np.unique(episode_ids)
    rng.shuffle(unique_eps)

    n_test = max(1, int(len(unique_eps) * test_ratio))
    n_val  = max(1, int(len(unique_eps) * val_ratio))

    test_eps  = set(unique_eps[:n_test])
    val_eps   = set(unique_eps[n_test: n_test + n_val])
    train_eps = set(unique_eps[n_test + n_val:])

    def _select(ep_set):
        mask = np.isin(episode_ids, list(ep_set))
        s  = states[mask]
        a  = actions[mask]
        ep = episode_ids[mask]
        # episode_lengths: số frame mỗi episode (theo thứ tự sorted)
        ep_lengths = [int((ep == e).sum()) for e in sorted(ep_set)]
        return s, a, ep_lengths

    tr = _select(train_eps)
    va = _select(val_eps)
    te = _select(test_eps)

    print(f"[data] Split — train: {len(train_eps)} eps ({len(tr[0]):,} rows) | "
          f"val: {len(val_eps)} eps ({len(va[0]):,} rows) | "
          f"test: {len(test_eps)} eps ({len(te[0]):,} rows)")
    return tr, va, te


# ─────────────────────────────────────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────────────────────────────────────

class RobotMLPDataset(Dataset):
    """Mỗi sample = (state_t, action_t)."""

    def __init__(self, states: np.ndarray, actions: np.ndarray, noise_std: float = 0.0):
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


class RobotLSTMDataset(Dataset):
    """Mỗi sample = (state_window[t-W:t], action_t). Window không vượt qua ranh giới episode."""

    def __init__(
        self,
        states:          np.ndarray,
        actions:         np.ndarray,
        episode_lengths: list[int],
        window_size:     int   = config.LSTM_WINDOW_SIZE,
        noise_std:       float = 0.0,
    ):
        self.window_size = window_size
        self.noise_std   = noise_std

        all_windows = []
        all_actions = []

        idx = 0
        for ep_len in episode_lengths:
            ep_s = states[idx: idx + ep_len]
            ep_a = actions[idx: idx + ep_len]
            for t in range(window_size - 1, ep_len):
                all_windows.append(ep_s[t - window_size + 1: t + 1])
                all_actions.append(ep_a[t])
            idx += ep_len

        self.windows = torch.from_numpy(np.stack(all_windows)).float()  # (N, W, S)
        self.actions = torch.from_numpy(np.stack(all_actions)).float()  # (N, A)

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int):
        w = self.windows[idx]
        if self.noise_std > 0.0:
            w = w + torch.randn_like(w) * self.noise_std
        return w, self.actions[idx]


class RobotChunkDataset(Dataset):
    """
    Mỗi sample = (state_t, action_chunk[t : t+K]).
    Chunk không vượt ranh giới episode — nếu thiếu (gần cuối episode),
    pad bằng cách lặp lại action cuối cùng (giống ACT: model học "đứng yên"
    ở cuối chuỗi thay vì học một hành động bịa đặt không có thật).
    """

    def __init__(
        self,
        states:          np.ndarray,
        actions:         np.ndarray,
        episode_lengths: list[int],
        chunk_size:      int   = config.CHUNK_SIZE,
        noise_std:       float = 0.0,
    ):
        self.chunk_size = chunk_size
        self.noise_std  = noise_std

        all_states = []
        all_chunks = []

        idx = 0
        for ep_len in episode_lengths:
            ep_s = states[idx: idx + ep_len]
            ep_a = actions[idx: idx + ep_len]
            for t in range(ep_len):
                end = min(t + chunk_size, ep_len)
                chunk = ep_a[t:end]
                if len(chunk) < chunk_size:
                    pad   = np.repeat(chunk[-1:], chunk_size - len(chunk), axis=0)
                    chunk = np.concatenate([chunk, pad], axis=0)
                all_states.append(ep_s[t])
                all_chunks.append(chunk)
            idx += ep_len

        self.states = torch.from_numpy(np.stack(all_states)).float()       # (N, S)
        self.chunks = torch.from_numpy(np.stack(all_chunks)).float()       # (N, K, A)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        s = self.states[idx]
        if self.noise_std > 0.0:
            s = s + torch.randn_like(s) * self.noise_std
        return s, self.chunks[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Builder functions
# ─────────────────────────────────────────────────────────────────────────────

def _make_loader(dataset: Dataset, shuffle: bool, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )


def build_mlp_loaders(
    batch_size:  int   = config.BATCH_SIZE,
    noise_std:   float = config.STATE_NOISE_STD,
    num_workers: int   = 2,
) -> tuple[DataLoader, DataLoader, DataLoader, Normalizer, Normalizer]:
    """Trả về (train_loader, val_loader, test_loader, state_norm, action_norm)."""
    states, actions, episode_ids = load_dataset()
    (tr_s, tr_a, _), (va_s, va_a, _), (te_s, te_a, _) = episode_split(states, actions, episode_ids)

    s_norm = Normalizer().fit(tr_s)
    a_norm = Normalizer().fit(tr_a)

    tr_ds = RobotMLPDataset(s_norm.transform(tr_s), a_norm.transform(tr_a), noise_std)
    va_ds = RobotMLPDataset(s_norm.transform(va_s), a_norm.transform(va_a))
    te_ds = RobotMLPDataset(s_norm.transform(te_s), a_norm.transform(te_a))

    return (
        _make_loader(tr_ds, shuffle=True,  batch_size=batch_size, num_workers=num_workers),
        _make_loader(va_ds, shuffle=False, batch_size=batch_size, num_workers=num_workers),
        _make_loader(te_ds, shuffle=False, batch_size=batch_size, num_workers=num_workers),
        s_norm,
        a_norm,
    )


def build_lstm_loaders(
    batch_size:  int   = config.BATCH_SIZE,
    window_size: int   = config.LSTM_WINDOW_SIZE,
    noise_std:   float = config.STATE_NOISE_STD,
    num_workers: int   = 2,
) -> tuple[DataLoader, DataLoader, DataLoader, Normalizer, Normalizer]:
    """Trả về (train_loader, val_loader, test_loader, state_norm, action_norm)."""
    states, actions, episode_ids = load_dataset()
    (tr_s, tr_a, tr_ep), (va_s, va_a, va_ep), (te_s, te_a, te_ep) = \
        episode_split(states, actions, episode_ids)

    s_norm = Normalizer().fit(tr_s)
    a_norm = Normalizer().fit(tr_a)

    tr_ds = RobotLSTMDataset(s_norm.transform(tr_s), a_norm.transform(tr_a), tr_ep, window_size, noise_std)
    va_ds = RobotLSTMDataset(s_norm.transform(va_s), a_norm.transform(va_a), va_ep, window_size)
    te_ds = RobotLSTMDataset(s_norm.transform(te_s), a_norm.transform(te_a), te_ep, window_size)

    return (
        _make_loader(tr_ds, shuffle=True,  batch_size=batch_size, num_workers=num_workers),
        _make_loader(va_ds, shuffle=False, batch_size=batch_size, num_workers=num_workers),
        _make_loader(te_ds, shuffle=False, batch_size=batch_size, num_workers=num_workers),
        s_norm,
        a_norm,
    )


def build_chunk_loaders(
    batch_size:  int   = config.BATCH_SIZE,
    chunk_size:  int   = config.CHUNK_SIZE,
    noise_std:   float = config.STATE_NOISE_STD,
    num_workers: int   = 2,
) -> tuple[DataLoader, DataLoader, DataLoader, Normalizer, Normalizer]:
    """Trả về (train_loader, val_loader, test_loader, state_norm, action_norm)."""
    states, actions, episode_ids = load_dataset()
    (tr_s, tr_a, tr_ep), (va_s, va_a, va_ep), (te_s, te_a, te_ep) = \
        episode_split(states, actions, episode_ids)

    s_norm = Normalizer().fit(tr_s)
    a_norm = Normalizer().fit(tr_a)

    tr_ds = RobotChunkDataset(s_norm.transform(tr_s), a_norm.transform(tr_a), tr_ep, chunk_size, noise_std)
    va_ds = RobotChunkDataset(s_norm.transform(va_s), a_norm.transform(va_a), va_ep, chunk_size)
    te_ds = RobotChunkDataset(s_norm.transform(te_s), a_norm.transform(te_a), te_ep, chunk_size)

    return (
        _make_loader(tr_ds, shuffle=True,  batch_size=batch_size, num_workers=num_workers),
        _make_loader(va_ds, shuffle=False, batch_size=batch_size, num_workers=num_workers),
        _make_loader(te_ds, shuffle=False, batch_size=batch_size, num_workers=num_workers),
        s_norm,
        a_norm,
    )


if __name__ == "__main__":
    print("=== MLP loaders ===")
    tl, vl, tel, sn, an = build_mlp_loaders()
    s, a = next(iter(tl))
    print(f"  train batch: state={s.shape}, action={a.shape}")

    print("=== LSTM loaders ===")
    tl, vl, tel, sn, an = build_lstm_loaders()
    w, a = next(iter(tl))
    print(f"  train batch: window={w.shape}, action={a.shape}")

    print("=== Chunk loaders ===")
    tl, vl, tel, sn, an = build_chunk_loaders()
    s, c = next(iter(tl))
    print(f"  train batch: state={s.shape}, action_chunk={c.shape}")
