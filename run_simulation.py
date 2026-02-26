import os
import cv2
import time
import numpy as np
from flask import Flask, Response
from ultralytics import YOLO

# Configuration
IMAGE_DIR = "Infrastructure/images"  # <- change this
CHECK_RECT = (436, 171, 800, 500)
MODEL_PATH = "runs/detect/train6/weights/best.pt"
IMAGE_WIDTH, IMAGE_HEIGHT = 1024, 640
VIOLATION_CLASSES = {"car", "truck", "bus", "motorbike"}
TRAFFIC_SEQ = [("red", 8), ("green", 12), ("yellow", 2)]

# Load YOLO model
model = YOLO(MODEL_PATH)

app = Flask(__name__)

def get_traffic_light_state(elapsed_time):
    cycle = sum(d for _, d in TRAFFIC_SEQ)
    t = elapsed_time % cycle
    acc = 0
    for color, duration in TRAFFIC_SEQ:
        acc += duration
        if t < acc:
            return color
    return "red"

def is_inside_check_area(box, area):
    x1, y1, x2, y2 = area
    bx1, by1, bx2, by2 = box
    cx = (bx1 + bx2) / 2
    cy = (by1 + by2) / 2
    return x1 <= cx <= x2 and y1 <= cy <= y2

def draw_annotations(img, detections, traffic_light):
    violation = False
    for box in detections.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        if class_name in VIOLATION_CLASSES:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            if is_inside_check_area(xyxy, CHECK_RECT) and traffic_light == "red":
                violation = True
                x1, y1, x2, y2 = xyxy
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, f"{class_name} (Violation)", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return violation

def stream_images():
    image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png'))])
    start_time = time.time()

    frame_index = 0
    total_images = len(image_files)

    while True:
        elapsed = time.time() - start_time
        traffic_light = get_traffic_light_state(elapsed)

        # Cycle through images like a video stream
        img_path = os.path.join(IMAGE_DIR, image_files[frame_index % total_images])
        frame_index += 1

        frame = cv2.imread(img_path)
        if frame is None:
            continue

        frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
        results = model(frame)[0]

        draw_annotations(frame, results, traffic_light)

        # Draw check area
        x1, y1, x2, y2 = CHECK_RECT
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, f"Traffic Light: {traffic_light.upper()}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255) if traffic_light == 'red'
                    else (0, 255, 0) if traffic_light == 'green' else (0, 255, 255), 3)

        # Encode frame as JPEG
        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        #time.sleep(1 / 100)  # Simulate 30 FPS stream (one image every 33ms)

@app.route('/')
def index():
    return """
    <html>
        <head>
            <title>Traffic Violation Stream</title>
        </head>
        <body>
            <h1>Traffic Monitoring Stream</h1>
            <img src="/video_feed" width="1024" height="640">
        </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(stream_images(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)

