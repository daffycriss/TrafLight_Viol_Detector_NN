import os
import random
from PIL import Image, ImageDraw

# Configuration
INPUT_DIR = "0"
OUTPUT_DIR = "0_new"
TRAFFIC_LIGHT_SIZE = (30, 80)  # Width x Height
LIGHT_RADIUS = 10
CORNER_GAP = 10  # Gap from image edge

# Traffic light durations (in image counts)
GREEN_MIN, GREEN_MAX = 250, 350
YELLOW_FRAMES = 60
RED_FRAMES = 150

COLORS = {
    "red": (255, 0, 0),
    "yellow": (255, 255, 0),
    "green": (0, 255, 0),
    "off": (50, 50, 50)
}

def draw_traffic_light(base_image, active_color):
    img = base_image.copy()
    draw = ImageDraw.Draw(img)
    width, height = img.size
    light_w, light_h = TRAFFIC_LIGHT_SIZE

    x0 = width - light_w - CORNER_GAP
    y0 = CORNER_GAP
    x1 = x0 + light_w
    y1 = y0 + light_h

    # Draw outer box
    draw.rectangle([x0, y0, x1, y1], fill=(30, 30, 30), outline=(255, 255, 255))

    # Light positions
    centers = [
        (x0 + light_w // 2, y0 + LIGHT_RADIUS + 5),            # red
        (x0 + light_w // 2, y0 + light_h // 2),                 # yellow
        (x0 + light_w // 2, y1 - LIGHT_RADIUS - 5)             # green
    ]
    light_order = ["red", "yellow", "green"]

    for i, light in enumerate(light_order):
        color = COLORS[light] if light == active_color else COLORS["off"]
        cx, cy = centers[i]
        draw.ellipse([cx - LIGHT_RADIUS, cy - LIGHT_RADIUS,
                      cx + LIGHT_RADIUS, cy + LIGHT_RADIUS], fill=color)

    return img

def process_images():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Get and sort image filenames
    filenames = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    # Initialize cycle
    cycle = []
    while len(cycle) < len(filenames):
        green_duration = random.randint(GREEN_MIN, GREEN_MAX)
        cycle += ["green"] * green_duration
        cycle += ["yellow"] * YELLOW_FRAMES
        cycle += ["red"] * RED_FRAMES

    # Only keep as many as needed
    cycle = cycle[:len(filenames)]

    for idx, filename in enumerate(filenames):
        path = os.path.join(INPUT_DIR, filename)
        image = Image.open(path).convert("RGB")
        color = cycle[idx]

        updated = draw_traffic_light(image, color)
        out_path = os.path.join(OUTPUT_DIR, filename)
        updated.save(out_path)

if __name__ == "__main__":
    process_images()
