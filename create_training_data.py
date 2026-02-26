import os
import random
import shutil

def split_subset_dataset(images_dir, labels_dir, subset_ratio=0.3, train_ratio=0.8, seed=42):
    # Set up output directories
    output_dirs = {
        "train_images": "data/train/images",
        "train_labels": "data/train/labels",
        "val_images": "data/val/images",
        "val_labels": "data/val/labels"
    }
    
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    # List all .jpg images
    all_image_files = [f for f in os.listdir(images_dir) if f.lower().endswith('.jpg')]
    
    # Shuffle and select subset
    random.seed(seed)
    random.shuffle(all_image_files)

    subset_size = int(len(all_image_files) * subset_ratio)
    subset_files = all_image_files[:subset_size]

    # Split subset into train and val sets
    split_index = int(len(subset_files) * train_ratio)
    train_files = subset_files[:split_index]
    val_files = subset_files[split_index:]

    def copy_files(file_list, target_image_dir, target_label_dir):
        for img_file in file_list:
            label_file = os.path.splitext(img_file)[0] + '.txt'
            img_src = os.path.join(images_dir, img_file)
            label_src = os.path.join(labels_dir, label_file)

            img_dst = os.path.join(target_image_dir, img_file)
            label_dst = os.path.join(target_label_dir, label_file)

            if os.path.exists(label_src):
                shutil.copyfile(img_src, img_dst)
                shutil.copyfile(label_src, label_dst)
            else:
                print(f"Warning: Label file not found for image {img_file}")

    # Copy files to train and val directories
    copy_files(train_files, output_dirs["train_images"], output_dirs["train_labels"])
    copy_files(val_files, output_dirs["val_images"], output_dirs["val_labels"])

    print(f"Subset size: {subset_size}")
    print(f"Train images: {len(train_files)}, Validation images: {len(val_files)}")
    print("Dataset split and copy completed.")

# Run code to create training and validation data
split_subset_dataset("Infrastructure/images", "Infrastructure/labels")
