import os
import cv2
import time
import numpy as np
from flask import Flask, Response
from ultralytics import YOLO

# Configuration
IMAGE_DIR = "Infrastructure/images"
CHECK_RECT_ROTATED = ((520, 381), (500, 220), 40)  # center (x, y), size (w, h), angle in degrees
MODEL_PATH = "runs/detect/train6/weights/best.pt"
IMAGE_WIDTH, IMAGE_HEIGHT = 1024, 640
VIOLATION_CLASSES = {"car", "truck", "bus", "motorbike"}
TRAFFIC_SEQ = [("red", 8), ("green", 12), ("yellow", 2)]

# Precompute the rotated check area polygon
rotated_box_pts = cv2.boxPoints(CHECK_RECT_ROTATED)
rotated_box_pts = np.int32(rotated_box_pts)

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
    # Check if the center point is inside the rotated polygon
    return cv2.pointPolygonTest(rotated_box_pts, (cx, cy), False) >= 0

def draw_annotations(img, detections, traffic_light):
    violation = False
    for box in detections.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        if class_name in VIOLATION_CLASSES:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            if is_inside_check_area(xyxy) and traffic_light == "red":
                violation = True
                x1, y1, x2, y2 = xyxy
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, f"{class_name} (Violation)", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return violation

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
        frame_index += 9
        if frame_index >= total_images:
            frame_index = 0

        frame = cv2.imread(img_path)
        if frame is None:
            continue

        frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
        results = model.track(frame, persist=True)[0]
        
        # Determine traffic light state
        traffic_light = get_traffic_light_state(elapsed)

        # Draw and track violations here
        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            xyxy = box.xyxy[0].cpu().numpy().astype(int)

            # Get tracking ID
            if hasattr(box, 'id'):
                object_id = int(box.id[0])
            else:
                continue  # Skip untrackable object

            # Check if it's a violating vehicle
            if class_name in VIOLATION_CLASSES:
                if is_inside_check_area(xyxy):
                    if traffic_light == 'red':
                        violation_ids.add(object_id)

                if object_id in violation_ids:
                    x1, y1, x2, y2 = xyxy
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"{class_name} (Violation)", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        draw_annotations(frame, results, traffic_light)

        # Draw rotated check area
        cv2.drawContours(frame, [rotated_box_pts], 0, (0, 255, 255), 2)
        cv2.putText(frame, f"Traffic Light: {traffic_light.upper()}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 255) if traffic_light == 'red'
                    else (0, 255, 0) if traffic_light == 'green' else (0, 255, 255), 3)

        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        time.sleep(1 / 30)  # simulate 30 FPS

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

