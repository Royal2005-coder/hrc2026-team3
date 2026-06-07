"""
Kiểm tra lệch pose vật thể (obj0..3) giữa dữ liệu train (parquet) và lúc inference (Isaac Sim).

State 38 chiều có 28 chiều cuối (index 10..37) là obj0..3 pose, mỗi obj 7 giá trị
[x, y, z, qx, qy, qz, qw] — xem config.STATE_IDX / meta/info.json.

Cách dùng:
  1) Tính thống kê pose vật từ tập train, lưu ra JSON tham chiếu:
       python check_obj_pose.py --mode train

  2) Trong lúc chạy run_il_policy.py, bật cờ --debug_obj_pose để ghi log pose vật
     lúc inference ra JSON (xem patch trong run_il_policy.py), ví dụ:
       python run_il_policy.py --model chunk --debug_obj_pose --episodes 1

  3) So sánh hai bên:
       python check_obj_pose.py --mode compare --inference_log logs/inference_obj_pose.json
"""

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np

import config
from data_loader import load_dataset

OBJ_AXES = ["x", "y", "z", "qx", "qy", "qz", "qw"]
N_OBJS   = 4

# Trong state đã chọn theo STATE_IDX, obj0..3 nằm ở 28 chiều cuối (index 10..37)
OBJ_START_IDX = config.STATE_DIM - N_OBJS * 7   # = 10

DEFAULT_TRAIN_STATS_PATH = os.path.join(_HERE, "checkpoints", "obj_pose_train_stats.json")


# ─────────────────────────────────────────────────────────────────────────────
# Mode "train": thống kê pose vật từ dataset
# ─────────────────────────────────────────────────────────────────────────────

def _stats_block(values: np.ndarray) -> dict:
    """values: (N,) → {min, max, mean, std}"""
    return {
        "min":  float(np.min(values)),
        "max":  float(np.max(values)),
        "mean": float(np.mean(values)),
        "std":  float(np.std(values)),
    }


def compute_obj_pose_stats(obj_block: np.ndarray) -> dict:
    """
    obj_block: (N, 4, 7) → {"obj0": {"x": {...}, "y": {...}, ...}, "obj1": {...}, ...}
    """
    out = {}
    for i in range(N_OBJS):
        out[f"obj{i}"] = {
            axis: _stats_block(obj_block[:, i, a])
            for a, axis in enumerate(OBJ_AXES)
        }
    return out


def run_train_mode(out_path: str):
    states, _, episode_ids = load_dataset()
    obj_block = states[:, OBJ_START_IDX:].reshape(-1, N_OBJS, 7)   # (N, 4, 7)

    # Frame đầu tiên của mỗi episode — đại diện cho vị trí scatter ban đầu
    _, first_idx = np.unique(episode_ids, return_index=True)
    start_block = obj_block[first_idx]                              # (n_eps, 4, 7)

    stats = {
        "episode_start": compute_obj_pose_stats(start_block),
        "all_frames":    compute_obj_pose_stats(obj_block),
        "n_episodes":    int(len(first_idx)),
        "n_frames":      int(len(obj_block)),
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n[check_obj_pose] Đã lưu thống kê train → {out_path}\n")
    _print_stats_table("Episode-start (vị trí scatter ban đầu)", stats["episode_start"])
    _print_stats_table("Toàn bộ frame (bao gồm cả lúc bị cầm/di chuyển)", stats["all_frames"])

    # Sanity check: norm quaternion phải ~ 1
    quat = start_block[:, :, 3:7]                                    # (n_eps, 4, 4) [qx,qy,qz,qw]
    norms = np.linalg.norm(quat, axis=-1)
    print(f"\n[check_obj_pose] Quaternion norm (episode-start): "
          f"mean={norms.mean():.4f}  min={norms.min():.4f}  max={norms.max():.4f}  "
          f"(kỳ vọng ≈ 1.0 — nếu lệch nhiều, có thể đã đọc sai thứ tự qx,qy,qz,qw)")


def _print_stats_table(title: str, block: dict):
    print(f"── {title} " + "─" * max(0, 60 - len(title)))
    header = f"{'obj':>6} {'axis':>5} {'min':>10} {'max':>10} {'mean':>10} {'std':>10}"
    print(header)
    print("-" * len(header))
    for obj_name, axes in block.items():
        for axis_name, s in axes.items():
            print(f"{obj_name:>6} {axis_name:>5} "
                  f"{s['min']:>10.4f} {s['max']:>10.4f} {s['mean']:>10.4f} {s['std']:>10.4f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Mode "compare": đối chiếu log inference với thống kê train
# ─────────────────────────────────────────────────────────────────────────────

def run_compare_mode(train_stats_path: str, inference_log_path: str, tol: float):
    with open(train_stats_path) as f:
        train_stats = json.load(f)
    with open(inference_log_path) as f:
        inference_log = json.load(f)

    # Dùng "all_frames" làm khoảng tham chiếu — bao trùm cả lúc vật bị cầm/di chuyển
    ref = train_stats["all_frames"]

    # Gom toàn bộ sample inference lại theo obj/axis
    samples = {f"obj{i}": {axis: [] for axis in OBJ_AXES} for i in range(N_OBJS)}
    for _, frames in inference_log.items():
        for frame in frames:
            for obj_name, axes in frame.items():
                if obj_name == "step":
                    continue
                for axis_name, value in axes.items():
                    samples[obj_name][axis_name].append(value)

    print(f"\n[check_obj_pose] So sánh inference ({inference_log_path}) "
          f"với train stats ({train_stats_path}), tolerance=±{tol}\n")
    header = (f"{'obj':>6} {'axis':>5} {'train_min':>11} {'train_max':>11} "
              f"{'infer_min':>11} {'infer_max':>11} {'%_in_range':>11}  flag")
    print(header)
    print("-" * len(header))

    n_flagged = 0
    for obj_name in samples:
        for axis_name in OBJ_AXES:
            vals = np.array(samples[obj_name][axis_name], dtype=np.float64)
            if len(vals) == 0:
                continue
            lo = ref[obj_name][axis_name]["min"] - tol
            hi = ref[obj_name][axis_name]["max"] + tol
            in_range = np.mean((vals >= lo) & (vals <= hi)) * 100.0
            flag = ""
            if in_range < 50.0:
                flag = "<-- LỆCH MẠNH (khả năng sai hệ tọa độ / đơn vị / thứ tự obj)"
                n_flagged += 1
            elif in_range < 90.0:
                flag = "<-- lệch một phần (kiểm tra thêm)"
                n_flagged += 1
            print(f"{obj_name:>6} {axis_name:>5} "
                  f"{ref[obj_name][axis_name]['min']:>11.4f} {ref[obj_name][axis_name]['max']:>11.4f} "
                  f"{vals.min():>11.4f} {vals.max():>11.4f} {in_range:>10.1f}%  {flag}")

    print()
    if n_flagged == 0:
        print("[check_obj_pose] Không phát hiện lệch đáng kể — pose vật lúc inference "
              "nằm trong khoảng quan sát được lúc train. Vấn đề gắp sai vị trí có lẽ "
              "không phải do lệch hệ tọa độ/đơn vị.")
    else:
        print(f"[check_obj_pose] Phát hiện {n_flagged} trục lệch đáng kể — rất có thể "
              f"obj_vec lúc inference KHÔNG cùng hệ quy chiếu / đơn vị / thứ tự với lúc "
              f"ghi dataset (vd. world frame vs robot-relative frame, scale m vs cm, "
              f"hoặc thứ tự obj0..3 không khớp). Đây là nghi phạm hàng đầu cho việc "
              f"robot gắp sai vị trí dù MAE bám lệnh thấp.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Kiểm tra lệch pose vật train vs inference")
    p.add_argument("--mode", choices=["train", "compare"], required=True)
    p.add_argument("--train_stats", default=DEFAULT_TRAIN_STATS_PATH,
                   help="Đường dẫn file JSON thống kê train (đọc/ghi tùy mode)")
    p.add_argument("--inference_log", default=None,
                   help="[compare] Đường dẫn JSON log pose vật ghi lúc inference")
    p.add_argument("--tol", type=float, default=0.02,
                   help="[compare] Sai số cho phép khi so khoảng giá trị (mặc định 0.02)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        run_train_mode(args.train_stats)
    else:
        if not args.inference_log:
            sys.exit("--inference_log là bắt buộc ở mode compare")
        run_compare_mode(args.train_stats, args.inference_log, args.tol)
