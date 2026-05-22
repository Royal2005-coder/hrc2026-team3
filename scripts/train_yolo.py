"""Train YOLOv8n on collected synthetic dataset — single class 'part'.
Color classification (part_A vs part_B) is done at inference, not here."""
import os
import glob

DATASET_DIR = "/home/ubuntu/tai/dataset"
YAML_PATH   = f"{DATASET_DIR}/task1_single.yaml"

# Sanity check
train_imgs = glob.glob(f"{DATASET_DIR}/images/train/*.jpg") + glob.glob(f"{DATASET_DIR}/images/train/*.png")
train_lbls = glob.glob(f"{DATASET_DIR}/labels/train/*.txt")
val_imgs   = glob.glob(f"{DATASET_DIR}/images/val/*.jpg")   + glob.glob(f"{DATASET_DIR}/images/val/*.png")

print("Dataset check:")
print(f"  train images : {len(train_imgs)}")
print(f"  train labels : {len(train_lbls)}")
print(f"  val   images : {len(val_imgs)}")

empty = sum(1 for f in train_lbls[:20] if os.path.getsize(f) == 0)
print(f"  empty labels (first 20): {empty}")

# Check class distribution in a few labels
counts = {0: 0, 1: 0}
for f in train_lbls[:50]:
    for line in open(f):
        cls = int(line.split()[0])
        counts[cls] = counts.get(cls, 0) + 1
print(f"  class dist (first 50 files): {counts}")

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
