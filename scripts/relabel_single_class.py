"""Re-label dataset to single class (0=part) — ignores part_A/part_B distinction.
Color classification happens at inference time, not in YOLO training."""
import os
import glob

DATASET_DIR = os.path.expanduser("~/tai/dataset")

for split in ("train", "val"):
    label_dir = f"{DATASET_DIR}/labels/{split}"
    files = glob.glob(f"{label_dir}/*.txt")
    changed = 0
    for fpath in files:
        with open(fpath) as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                parts[0] = "0"
                new_lines.append(" ".join(parts) + "\n")
        with open(fpath, "w") as f:
            f.writelines(new_lines)
        changed += 1
    print(f"[{split}] re-labeled {changed} files → all class 0")

print("Done. Now update task1.yaml nc=1 and retrain.")
