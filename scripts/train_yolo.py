"""Train YOLOv8n on collected synthetic dataset."""
import os
import glob

DATASET_DIR = "/home/ubuntu/tai/dataset"
YAML_PATH   = f"{DATASET_DIR}/task1.yaml"

# Quick sanity check before training
train_imgs  = glob.glob(f"{DATASET_DIR}/images/train/*.jpg") + glob.glob(f"{DATASET_DIR}/images/train/*.png")
train_lbls  = glob.glob(f"{DATASET_DIR}/labels/train/*.txt")
val_imgs    = glob.glob(f"{DATASET_DIR}/images/val/*.jpg")   + glob.glob(f"{DATASET_DIR}/images/val/*.png")

print(f"Dataset check:")
print(f"  train images : {len(train_imgs)}")
print(f"  train labels : {len(train_lbls)}")
print(f"  val   images : {len(val_imgs)}")

if len(train_imgs) == 0:
    raise RuntimeError("No training images found — check DATASET_DIR path")
if len(train_lbls) == 0:
    raise RuntimeError("No training labels found")

# Check a few labels are non-empty
empty = sum(1 for f in train_lbls[:20] if os.path.getsize(f) == 0)
print(f"  empty labels (first 20 checked): {empty}")

from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data=YAML_PATH,
    epochs=100,
    imgsz=640,
    batch=16,
    project="/home/ubuntu/tai/runs/detect",
    name="task1_v2",
    exist_ok=False,
    patience=20,
)
print("Training done. Best weights: /home/ubuntu/tai/runs/detect/task1_v2/weights/best.pt")
