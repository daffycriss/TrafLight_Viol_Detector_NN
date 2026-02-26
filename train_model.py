from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(data="mappings.yaml", epochs=10, device="cpu", batch=2)
