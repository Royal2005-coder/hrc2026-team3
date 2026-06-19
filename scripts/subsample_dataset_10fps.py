#!/usr/bin/env python3
"""
Subsample LeRobot dataset from 30fps to 10fps (keep every 3rd frame).

Strategy:
- Data parquets: keep rows where original frame_index % 3 == 0
- Videos: symlink to original (video files unchanged, lerobot seeks by timestamp)
- Episodes meta: update lengths, dataset indices
- info.json: update fps, total_frames
- stats.json: recompute from subsampled data
"""

import json
import os
import sys
import shutil
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

STRIDE = 3  # keep every 3rd frame: 30fps → 10fps

SRC = Path("Part_Sorting/part_sorting_long_756_episode")
DST = Path("Part_Sorting/part_sorting_long_756_episode_10fps")


def subsample_data_parquets():
    """Subsample data parquets, reindex frame_index and global index."""
    src_data = SRC / "data"
    dst_data = DST / "data"

    global_index = 0

    for chunk_dir in sorted(src_data.iterdir()):
        dst_chunk = dst_data / chunk_dir.name
        dst_chunk.mkdir(parents=True, exist_ok=True)

        for parquet_file in sorted(chunk_dir.glob("*.parquet")):
            print(f"  Processing {parquet_file.relative_to(SRC)} ...")
            table = pq.read_table(parquet_file)
            d = table.to_pydict()

            n = len(d["frame_index"])
            keep_mask = [i for i in range(n) if d["frame_index"][i] % STRIDE == 0]

            new_d = {}
            for col in d:
                new_d[col] = [d[col][i] for i in keep_mask]

            # Re-index frame_index within each episode (reset at episode boundaries)
            new_frame_indices = []
            per_ep_counter = {}
            for old_frame_idx, ep_idx in zip(new_d["frame_index"], new_d["episode_index"]):
                if ep_idx not in per_ep_counter:
                    per_ep_counter[ep_idx] = 0
                new_frame_indices.append(per_ep_counter[ep_idx])
                per_ep_counter[ep_idx] += 1
            new_d["frame_index"] = new_frame_indices

            # Re-index global index
            new_d["index"] = list(range(global_index, global_index + len(keep_mask)))
            global_index += len(keep_mask)

            # Rebuild pyarrow table preserving original schema types
            schema = table.schema
            arrays = []
            for field in schema:
                col_data = new_d[field.name]
                arrays.append(pa.array(col_data, type=field.type))
            new_table = pa.table(
                {field.name: arrays[i] for i, field in enumerate(schema)},
                schema=schema,
            )

            dst_file = dst_chunk / parquet_file.name
            pq.write_table(new_table, dst_file, compression="snappy")
            print(f"    {n} → {len(keep_mask)} rows (global_index up to {global_index})")

    return global_index


def symlink_videos():
    """Create symlinks for video directories (no re-encoding needed)."""
    src_videos = SRC / "videos"
    dst_videos = DST / "videos"
    dst_videos.mkdir(parents=True, exist_ok=True)

    for cam_dir in sorted(src_videos.iterdir()):
        dst_cam = dst_videos / cam_dir.name
        if dst_cam.exists() or dst_cam.is_symlink():
            dst_cam.unlink() if dst_cam.is_symlink() else shutil.rmtree(dst_cam)
        # Use absolute path for symlink target
        dst_cam.symlink_to(cam_dir.resolve())
        print(f"  Symlinked {cam_dir.name}")


def subsample_episodes_meta(new_frames_per_episode: int):
    """Update episode metadata with new lengths and dataset indices."""
    src_eps = SRC / "meta" / "episodes"
    dst_eps = DST / "meta" / "episodes"

    dataset_index = 0

    for chunk_dir in sorted(src_eps.iterdir()):
        dst_chunk = dst_eps / chunk_dir.name
        dst_chunk.mkdir(parents=True, exist_ok=True)

        for parquet_file in sorted(chunk_dir.glob("*.parquet")):
            print(f"  Processing {parquet_file.relative_to(SRC / 'meta')} ...")
            table = pq.read_table(parquet_file)
            d = table.to_pydict()
            n_eps = len(d["episode_index"])

            # Update length and dataset indices
            new_d = {k: list(v) for k, v in d.items()}
            for i in range(n_eps):
                new_d["length"][i] = new_frames_per_episode
                new_d["dataset_from_index"][i] = dataset_index
                new_d["dataset_to_index"][i] = dataset_index + new_frames_per_episode
                dataset_index += new_frames_per_episode

            # Rebuild table
            schema = table.schema
            arrays = []
            for field in schema:
                arrays.append(pa.array(new_d[field.name], type=field.type))
            new_table = pa.table(
                {field.name: arrays[i] for i, field in enumerate(schema)},
                schema=schema,
            )

            dst_file = dst_chunk / parquet_file.name
            pq.write_table(new_table, dst_file, compression="snappy")

    return dataset_index


def update_info_json(total_frames: int):
    """Update info.json: fps=10, total_frames, keep video codec info as-is."""
    src_info = SRC / "meta" / "info.json"
    with open(src_info) as f:
        info = json.load(f)

    info["fps"] = 10
    info["total_frames"] = total_frames

    # Update non-video feature fps fields
    for key, feat in info["features"].items():
        if feat.get("dtype") != "video" and "fps" in feat:
            feat["fps"] = 10
        # video codec fps stays at 30 — actual video files unchanged

    dst_info = DST / "meta" / "info.json"
    with open(dst_info, "w") as f:
        json.dump(info, f, indent=4)
    print(f"  fps=10, total_frames={total_frames}")


def compute_stats():
    """Recompute stats.json from subsampled data parquets."""
    print("  Computing stats from subsampled data ...")
    import numpy as np

    data_dir = DST / "data"
    all_states = []
    all_actions = []

    for chunk_dir in sorted(data_dir.iterdir()):
        for parquet_file in sorted(chunk_dir.glob("*.parquet")):
            table = pq.read_table(parquet_file, columns=["observation.state", "action"])
            d = table.to_pydict()
            all_states.extend(d["observation.state"])
            all_actions.extend(d["action"])

    states = np.array(all_states, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.float32)

    def compute_feature_stats(arr):
        return {
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "count": [int(arr.shape[0])],  # must be [N] not N (lerobot expects 1-D)
        }

    # Load original stats to preserve visual stats (we don't load video pixels)
    src_stats_path = SRC / "meta" / "stats.json"
    with open(src_stats_path) as f:
        original_stats = json.load(f)

    stats = dict(original_stats)  # copy visual stats as-is
    stats["observation.state"] = compute_feature_stats(states)
    stats["action"] = compute_feature_stats(actions)

    dst_stats = DST / "meta" / "stats.json"
    with open(dst_stats, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats computed: {states.shape[0]} frames")


def copy_tasks_parquet():
    src = SRC / "meta" / "tasks.parquet"
    dst = DST / "meta" / "tasks.parquet"
    shutil.copy2(src, dst)
    print("  Copied tasks.parquet")


def main():
    print(f"Subsampling {SRC} → {DST} (stride={STRIDE}, 30fps → 10fps)")

    # Check source exists
    if not SRC.exists():
        print(f"ERROR: Source not found: {SRC}")
        sys.exit(1)

    # Create destination structure
    DST.mkdir(parents=True, exist_ok=True)
    (DST / "meta").mkdir(exist_ok=True)

    # Original stats: 756 episodes × 768 frames = 580608; after stride=3: 256 frames/ep
    orig_frames_per_ep = 768
    assert orig_frames_per_ep % STRIDE == 0, "frames per episode must be divisible by stride"
    new_frames_per_ep = orig_frames_per_ep // STRIDE  # 256

    print(f"\n[1/5] Subsampling data parquets (every {STRIDE} frames) ...")
    total_frames = subsample_data_parquets()
    print(f"  Total frames after subsample: {total_frames}")

    print("\n[2/5] Symlinking videos ...")
    symlink_videos()

    print("\n[3/5] Updating episodes meta ...")
    subsample_episodes_meta(new_frames_per_ep)

    print("\n[4/5] Updating info.json ...")
    (DST / "meta").mkdir(exist_ok=True)
    update_info_json(total_frames)

    print("\n[5/5] Copying tasks.parquet and computing stats ...")
    copy_tasks_parquet()
    compute_stats()

    print(f"\nDone! Dataset at: {DST}")
    print(f"  Original: 756 ep × 768 frames @ 30fps = 580608 frames")
    print(f"  New:      756 ep × {new_frames_per_ep} frames @ 10fps = {total_frames} frames")


if __name__ == "__main__":
    main()
