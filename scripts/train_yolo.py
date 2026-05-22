"""
train_yolo.py — Train YOLOv8n để detect Part A / Part B.

Chạy:
    python3 scripts/train_yolo.py
    python3 scripts/train_yolo.py --epochs 100 --data /path/to/task1.yaml
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data",   default="/home/ubuntu/tai/dataset/task1.yaml")
parser.add_argument("--model",  default="yolov8n.pt")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--imgsz",  type=int, default=640)
parser.add_argument("--batch",  type=int, default=16)
parser.add_argument("--device", default="0")
parser.add_argument("--project", default="/home/ubuntu/tai/runs/detect")
parser.add_argument("--name",   default="task1")
args = parser.parse_args()

from ultralytics import YOLO

model = YOLO(args.model)
results = model.train(
    data=args.data,
    epochs=args.epochs,
    imgsz=args.imgsz,
    batch=args.batch,
    device=args.device,
    amp=False,
    project=args.project,
    name=args.name,
    exist_ok=True,
)

print(f"\n[Done] Best weights: {args.project}/{args.name}/weights/best.pt")
