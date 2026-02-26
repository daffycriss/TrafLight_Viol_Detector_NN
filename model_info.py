from ultralytics import YOLO

model = YOLO("yolov8m.pt")  # Load YOLOv8m model
model.info()  # Display model architecture details
