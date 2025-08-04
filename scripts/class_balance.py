import os
import pandas as pd
import random
from collections import defaultdict
import shutil
from tqdm import tqdm

# --- 1. CONFIGURATION ---
# IMPORTANT: Point these paths to your folders containing the YOLO .txt files
image_dir = './datasets/bdd100k/images/100k/train'
yolo_label_dir = './datasets/bdd100k/labels/100k/train' # Your output_dir from the conversion script
output_dir = './datasets/bdd100k_balanced'

# --- 2. REVERSE CLASS MAP ---
# This map is the reverse of the one in your conversion script.
# It's crucial for understanding the numeric IDs in your .txt files.
class_map_reverse = {
    0: 'pedestrian', 1: 'rider', 2: 'car', 3: 'truck',
    4: 'bus', 5: 'train', 6: 'motorcycle', 7: 'bicycle',
    8: 'traffic light', 9: 'traffic sign'
}

# --- 3. SAMPLING TARGETS ---
# Define how many images you want that contain at least one instance of each class.
# We're still over-sampling rare classes and under-sampling common ones.
sampling_targets = {
    'car': 10000,
    'truck': 5000,
    'bus': 5000,
    'pedestrian': 8000,
    'rider': 4000,
    'bicycle': 4000,
    'motorcycle': 4000,
    'train': 2000,
    'traffic light': 7000,
    'traffic sign': 7000,
}

# --- 4. STEP 1: CATALOG YOUR YOLO DATA ---
print("🔍 Starting to catalog all YOLO .txt label data...")
image_to_classes = defaultdict(set)
all_label_files = [f for f in os.listdir(yolo_label_dir) if f.endswith('.txt')]

for filename in tqdm(all_label_files, desc="Cataloging labels"):
    image_name = filename.replace('.txt', '.jpg')
    with open(os.path.join(yolo_label_dir, filename), 'r') as f:
        lines = f.readlines()
        for line in lines:
            class_id = int(line.split()[0])
            class_name = class_map_reverse.get(class_id)
            if class_name:
                image_to_classes[image_name].add(class_name)

print(f"✅ Catalog complete. Found {len(image_to_classes)} images with labels.")

# --- 5. STEP 2: SELECT IMAGES (This logic is the same) ---
print("🧠 Selecting a balanced set of images...")
final_image_set = set()

for class_name, target_count in sampling_targets.items():
    images_with_class = [img for img, classes in image_to_classes.items() if class_name in classes]
    
    if len(images_with_class) > target_count:
        selected = random.sample(images_with_class, target_count)
    else:
        selected = images_with_class
        
    final_image_set.update(selected)
    print(f"  - Class '{class_name}': Found {len(images_with_class)} images, selected {len(selected)}.")

print(f"✅ Image selection complete. Total unique images selected: {len(final_image_set)}")

# --- 6. STEP 3: COPY FILES (Now copies .txt instead of .json) ---
print("🚚 Copying files to new balanced directory...")
balanced_img_dir = os.path.join(output_dir, 'images/train')
balanced_lbl_dir = os.path.join(output_dir, 'labels/train')

os.makedirs(balanced_img_dir, exist_ok=True)
os.makedirs(balanced_lbl_dir, exist_ok=True)

for image_name in tqdm(final_image_set, desc="Copying files"):
    # Source paths
    src_img_path = os.path.join(image_dir, image_name)
    src_lbl_path = os.path.join(yolo_label_dir, image_name.replace('.jpg', '.txt'))
    
    # Destination paths
    dst_img_path = os.path.join(balanced_img_dir, image_name)
    dst_lbl_path = os.path.join(balanced_lbl_dir, image_name.replace('.jpg', '.txt'))
    
    if os.path.exists(src_img_path) and os.path.exists(src_lbl_path):
        shutil.copyfile(src_img_path, dst_img_path)
        shutil.copyfile(src_lbl_path, dst_lbl_path)

print(f"🎉 Success! Your new balanced dataset is ready at '{output_dir}'.")
