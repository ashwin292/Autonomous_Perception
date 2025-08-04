import os
import pandas as pd
import random
from collections import defaultdict
import shutil
from tqdm import tqdm

# --- 1. CONFIGURATION ---
# Point these paths to your VALIDATION data folders
image_dir = './datasets/bdd100k/images/100k/val'
yolo_label_dir = './datasets/bdd100k/labels/100k/val'
output_dir = './datasets/bdd100k_balanced'

# --- 2. REVERSE CLASS MAP ---
# This must match your conversion script's map
class_map_reverse = {
    0: 'pedestrian', 1: 'rider', 2: 'car', 3: 'truck',
    4: 'bus', 5: 'train', 6: 'motorcycle', 7: 'bicycle',
    8: 'traffic light', 9: 'traffic sign'
}

# --- 3. VALIDATION SAMPLING TARGETS ---
# These numbers are smaller than the training set (e.g., ~20% of the training targets)
# The goal is to match the *proportion* of classes in your balanced training set.
sampling_targets = {
    'car': 2000,
    'truck': 1000,
    'bus': 1000,
    'pedestrian': 1600,
    'rider': 800,
    'bicycle': 800,
    'motorcycle': 800,
    'train': 400,
    'traffic light': 1400,
    'traffic sign': 1400,
}

# --- 4. STEP 1: CATALOG YOUR VALIDATION DATA ---
print("🔍 Starting to catalog all validation .txt label data...")
image_to_classes = defaultdict(set)
all_label_files = [f for f in os.listdir(yolo_label_dir) if f.endswith('.txt')]

for filename in tqdm(all_label_files, desc="Cataloging validation labels"):
    image_name = filename.replace('.txt', '.jpg')
    with open(os.path.join(yolo_label_dir, filename), 'r') as f:
        lines = f.readlines()
        for line in lines:
            try:
                class_id = int(line.split()[0])
                class_name = class_map_reverse.get(class_id)
                if class_name:
                    image_to_classes[image_name].add(class_name)
            except (ValueError, IndexError):
                continue

print(f"✅ Catalog complete. Found {len(image_to_classes)} validation images with labels.")

# --- 5. STEP 2: SELECT IMAGES ---
print("🧠 Selecting a balanced set of validation images...")
final_image_set = set()

for class_name, target_count in sampling_targets.items():
    images_with_class = [img for img, classes in image_to_classes.items() if class_name in classes]
    
    if len(images_with_class) > target_count:
        selected = random.sample(images_with_class, target_count)
    else:
        selected = images_with_class
        
    final_image_set.update(selected)
    print(f"  - Class '{class_name}': Found {len(images_with_class)} images, selected {len(selected)}.")

print(f"✅ Image selection complete. Total unique validation images selected: {len(final_image_set)}")

# --- 6. STEP 3: COPY FILES TO THE VALIDATION FOLDER ---
print("🚚 Copying files to new balanced validation directory...")
# Create a 'val' subdirectory inside the 'images' and 'labels' folders
balanced_img_dir = os.path.join(output_dir, 'images/val')
balanced_lbl_dir = os.path.join(output_dir, 'labels/val')

os.makedirs(balanced_img_dir, exist_ok=True)
os.makedirs(balanced_lbl_dir, exist_ok=True)

for image_name in tqdm(final_image_set, desc="Copying validation files"):
    # Source paths
    src_img_path = os.path.join(image_dir, image_name)
    src_lbl_path = os.path.join(yolo_label_dir, image_name.replace('.jpg', '.txt'))
    
    # Destination paths
    dst_img_path = os.path.join(balanced_img_dir, image_name)
    dst_lbl_path = os.path.join(balanced_lbl_dir, image_name.replace('.jpg', '.txt'))
    
    if os.path.exists(src_img_path) and os.path.exists(src_lbl_path):
        shutil.copyfile(src_img_path, dst_img_path)
        shutil.copyfile(src_lbl_path, dst_lbl_path)

print(f"🎉 Success! Your balanced validation set is ready at '{output_dir}'.")