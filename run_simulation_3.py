import os
import cv2
import time
import numpy as np
from flask import Flask, Response
from ultralytics import YOLO

# Configuration
IMAGE_DIR = "Infrastructure/images"

# Define the polygon check area by 4 corner points (in order)
CHECK_AREA_POINTS = np.array([
    [392, 201],  # top-left
    [1017, 553],  # top-right
    [626, 636],  # bottom-right
    [234, 304]   # bottom-left
], dtype=np.int32)

MODEL_PATH = "runs/detect/train7/weights/best.pt"
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

def is_inside_check_area(box):
    bx1, by1, bx2, by2 = box
    cx = (bx1 + bx2) / 2
    cy = (by1 + by2) / 2
    # Check if the center point is inside the polygon defined by CHECK_AREA_POINTS
    return cv2.pointPolygonTest(CHECK_AREA_POINTS, (cx, cy), False) >= 0

def draw_annotations(img, detections, traffic_light, violation_ids):
    for box in detections.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        if hasattr(box, 'id') and box.id is not None:
            object_id = int(box.id[0])
        else:
            object_id = None

        if class_name in VIOLATION_CLASSES:
            if object_id is not None and object_id in violation_ids:
                x1, y1, x2, y2 = xyxy
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, f"{class_name} (Violation)", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                x1, y1, x2, y2 = xyxy
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(img, f"{class_name}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

def stream_images():
    violation_ids = set()
    image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png'))])
    start_time = time.time()
    frame_index = 0
    total_images = len(image_files)

    while True:
        elapsed = time.time() - start_time
        traffic_light = get_traffic_light_state(elapsed)

        img_path = os.path.join(IMAGE_DIR, image_files[frame_index % total_images])
        frame_index += 3

        frame = cv2.imread(img_path)
        if frame is None:
            continue

        # Draw black rectangle from (0,0) to (140,26)
        cv2.rectangle(frame, (0, 0), (140, 26), (0, 0, 0), thickness=-1)  # thickness=-1 fills the rectangle

        frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
        results = model.track(frame, persist=True)[0]

        # Update violation tracking
        current_ids = set()
        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            xyxy = box.xyxy[0].cpu().numpy().astype(int)

            if hasattr(box, 'id') and box.id is not None:
                object_id = int(box.id[0])
            else:
                object_id = None

            if class_name in VIOLATION_CLASSES and object_id is not None:
                current_ids.add(object_id)
                if is_inside_check_area(xyxy) and traffic_light == "red":
                    violation_ids.add(object_id)

        # Remove objects no longer detected
        violation_ids.intersection_update(current_ids)

        draw_annotations(frame, results, traffic_light, violation_ids)

        # Draw polygon check area
        cv2.polylines(frame, [CHECK_AREA_POINTS], isClosed=True, color=(0, 255, 255), thickness=2)

        cv2.putText(frame, f"Traffic Light: {traffic_light.upper()}", (35, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 255) if traffic_light == 'red' else
                    (0, 255, 0) if traffic_light == 'green' else
                    (0, 255, 255), 3)

        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # time.sleep(1 / 30)  # simulate 30 FPS

@app.route('/')
def index():
    return """
    <html>
        <head><title>Traffic Violation Stream</title></head>
        <body>
            <h1>Traffic Monitoring Stream</h1>
            <img src="/video_feed" width="1024" height="640" />
        </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(stream_images(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, threaded=True)
