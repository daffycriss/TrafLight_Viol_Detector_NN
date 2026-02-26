import json
import os

# Load the JSON file
with open('infrastructure-mscoco.json', 'r') as file:
    data = json.load(file)

# Extract data from "annotations"
print("Annotations:")
for ann in data.get('annotations', []):
    print(f"  ID: {ann['id']}, Category ID: {ann['category_id']}, Area: {ann['area']}, BBox: {ann['bbox']}, Object ID: {ann['object_id']}")

# Extract data from "categories"
print("\nCategories:")
for cat in data.get('categories', []):
    print(f"  ID: {cat['id']}, Name: {cat['name']}, Supercategory: {cat.get('supercategory')}, Other Names: {cat.get('other_names', [])}")

# Extract data from "images"
print("\nImages:")
for img in data.get('images', []):
    print(f"  ID: {img['id']}, File Name: {img['file_name']}, Width: {img['width']}, Height: {img['height']}")

# Extract custom field
print("\nCustom Filename Formatting Flags:")
print(data.get('custom', {}).get('filenameformatting', []))

json_file_path = "infrastructure-mscoco.json"

import json

def fetch_json_data(json_file_path: str, target_file_name: str) -> str:
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        matches = []
        for img in data.get('images', []):
            if img.get('file_name') == target_file_name:
                image_identity = img.get('id')
                for ann in data.get('annotations', []):
                    if ann.get('image_id') == image_identity:
                        bbox = ann.get('bbox', [])
                        object_class = ann.get('category_id')
                        if len(bbox) >= 4:
                            x_init = bbox[0]
                            y_init = bbox[1]
                            x_dist = bbox[2]
                            y_dist = bbox[3]
                            x_center = (x_dist / 2) + x_init
                            y_center = (y_dist / 2) + y_init
                            x_dist_normalized = x_dist / 1024
                            y_dist_normalized = y_dist / 640
                            x_center_normalized = x_center / 1024
                            y_center_normalized = y_center / 640
                        matches.append(f"{object_class} {x_center_normalized} {y_center_normalized} {x_dist_normalized} {y_dist_normalized}")

        if matches:
            return "\n".join(matches) + "\n"
        else:
            return f"No image records found with file_name = '{target_file_name}'."

    except FileNotFoundError:
        return f"File '{json_file_path}' not found."
    except json.JSONDecodeError:
        return f"File '{json_file_path}' is not a valid JSON file."
    except Exception as e:
        return f"An error occurred: {e}"


def create_annotation_files(image_dir, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}

    for filename in os.listdir(image_dir):
        name, ext = os.path.splitext(filename)
        if ext.lower() in image_extensions:
            txt_filename = f"{name}.txt"
            txt_filepath = os.path.join(output_dir, txt_filename)
            message = fetch_json_data(json_file_path, filename)

            # Write the annotation line
            with open(txt_filepath, 'w') as f:
                f.write(f"{message}\n")

    print(f"Annotation files created in '{output_dir}'.")

# Run association to create .txts
create_annotation_files("Infrastructure/images", "Infrastructure/labels")
