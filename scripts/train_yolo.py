"""Train YOLOv8n on collected synthetic dataset."""
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="/home/ubuntu/tai/dataset/task1.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    project="/home/ubuntu/tai/runs/detect",
    name="task1_yolo",
    exist_ok=True,
)
print("Training done. Best weights: runs/detect/task1_yolo/weights/best.pt")
