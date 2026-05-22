"""Train YOLOv8n — single class 'part'. Color classifies part_A/B at inference."""
import os
import glob

DATASET_DIR = "/home/ubuntu/tai/dataset"
YAML_PATH   = f"{DATASET_DIR}/task1_single.yaml"

# ── Step 1: force all labels to class 0 ──────────────────────────────────────
print("Re-labeling all label files to class 0 ...")
for split in ("train", "val"):
    label_dir = f"{DATASET_DIR}/labels/{split}"
    files = glob.glob(f"{label_dir}/*.txt")
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
    print(f"  [{split}] {len(files)} files re-labeled")

# ── Step 2: verify no stray class IDs ────────────────────────────────────────
bad = []
for split in ("train", "val"):
    for fpath in glob.glob(f"{DATASET_DIR}/labels/{split}/*.txt"):
        for line in open(fpath):
            parts = line.strip().split()
            if len(parts) == 5 and int(parts[0]) != 0:
                bad.append(fpath)
                break
if bad:
    raise RuntimeError(f"Found {len(bad)} files with class != 0 after relabeling: {bad[:5]}")
print("Verification OK — all labels are class 0")

# ── Step 3: delete YOLO label cache (stale cache causes class OOB crash) ──────
for split in ("train", "val"):
    cache = f"{DATASET_DIR}/labels/{split}.cache"
    if os.path.exists(cache):
        os.remove(cache)
        print(f"  deleted cache: {cache}")

# ── Step 4: dataset stats ─────────────────────────────────────────────────────
train_imgs = glob.glob(f"{DATASET_DIR}/images/train/*.jpg") + glob.glob(f"{DATASET_DIR}/images/train/*.png")
val_imgs   = glob.glob(f"{DATASET_DIR}/images/val/*.jpg")   + glob.glob(f"{DATASET_DIR}/images/val/*.png")
train_lbls = glob.glob(f"{DATASET_DIR}/labels/train/*.txt")
print(f"train: {len(train_imgs)} images, {len(train_lbls)} labels")
print(f"val  : {len(val_imgs)} images")

# ── Step 5: train ─────────────────────────────────────────────────────────────
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"   # clearer CUDA errors if any

from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data=YAML_PATH,
    epochs=100,
    imgsz=640,
    batch=16,
    project="/home/ubuntu/tai/runs/detect",
    name="task1_single",
    exist_ok=False,
    patience=30,
    lr0=0.01,
    mosaic=1.0,
)
print("Training done.")
print("Best weights: /home/ubuntu/tai/runs/detect/task1_single/weights/best.pt")
