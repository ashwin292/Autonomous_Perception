import os
from collections import defaultdict
import pandas as pd
import random
import shutil
from tqdm import tqdm

# --- 1. CONFIGURATION ---
CLASS_MAP_REVERSE = {
    0: 'person', 1: 'rider', 2: 'car', 3: 'truck',
    4: 'bus', 5: 'train', 6: 'motor', 7: 'bike',
    8: 'traffic light', 9: 'traffic sign'
}

# Define different sampling targets for train and val sets
# You typically need fewer images for validation.
TRAIN_TARGETS = {
    'car': 10000, 'truck': 5000, 'bus': 5000, 'person': 8000,
    'rider': 4000, 'bike': 4000, 'motor': 4000, 'train': 2000,
    'traffic light': 7000, 'traffic sign': 7000
}
VAL_TARGETS = {
    'car': 2000, 'truck': 1000, 'bus': 1000, 'person': 1500,
    'rider': 800, 'bike': 800, 'motor': 800, 'train': 400,
    'traffic light': 1200, 'traffic sign': 1200
}

def balance_dataset(split_name, sampling_targets):
    """
    Scans, catalogs, and creates a balanced dataset for a given split ('train' or 'val').
    """
    print(f"\n{'='*20} PROCESSING '{split_name.upper()}' SET {'='*20}")
    
    # --- DYNAMIC PATHS ---
    image_dir = f'./datasets/bdd100k/images/100k/{split_name}'
    yolo_label_dir = f'./datasets/bdd100k/labels/100k/{split_name}'
    output_dir = './datasets/bdd100k_balanced'

    # --- Cataloging ---
    print(f"🔍 Scanning all .txt files in: {yolo_label_dir}")
    image_to_classes = defaultdict(set)
    all_label_files = [f for f in os.listdir(yolo_label_dir) if f.endswith('.txt')]

    for filename in tqdm(all_label_files, desc=f"Cataloging {split_name} labels"):
        with open(os.path.join(yolo_label_dir, filename), 'r') as f:
            for line in f.readlines():
                try:
                    class_id = int(line.split()[0])
                    class_name = CLASS_MAP_REVERSE.get(class_id)
                    if class_name:
                        image_name = filename.replace('.txt', '.jpg')
                        image_to_classes[image_name].add(class_name)
                except (ValueError, IndexError):
                    continue
    
    # --- Image Selection ---
    print("🧠 Selecting a balanced set of images...")
    final_image_set = set()
    for class_name, target_count in sampling_targets.items():
        images_with_class = [img for img, classes in image_to_classes.items() if class_name in classes]
        if not images_with_class:
            print(f"⚠️ Warning: No images found for class '{class_name}' in the {split_name} set.")
            continue
        selected = random.sample(images_with_class, min(len(images_with_class), target_count))
        final_image_set.update(selected)

    print(f"✅ Image selection complete. Total unique images for balanced {split_name} set: {len(final_image_set)}")

    # --- File Copy ---
    print("🚚 Copying files to new balanced directory...")
    balanced_img_dir = os.path.join(output_dir, f'images/{split_name}')
    balanced_lbl_dir = os.path.join(output_dir, f'labels/{split_name}')
    os.makedirs(balanced_img_dir, exist_ok=True)
    os.makedirs(balanced_lbl_dir, exist_ok=True)

    for image_name in tqdm(final_image_set, desc=f"Copying {split_name} files"):
        shutil.copyfile(os.path.join(image_dir, image_name), os.path.join(balanced_img_dir, image_name))
        shutil.copyfile(os.path.join(yolo_label_dir, image_name.replace('.jpg', '.txt')), os.path.join(balanced_lbl_dir, image_name.replace('.jpg', '.txt')))

if __name__ == '__main__':
    # Make sure to delete the old balanced folder first for a clean run
    output_dir = './datasets/bdd100k_balanced'
    if os.path.exists(output_dir):
        print(f"Deleting old balanced dataset folder: {output_dir}")
        shutil.rmtree(output_dir)

    # Run the function for both the training and validation sets
    balance_dataset('train', TRAIN_TARGETS)
    balance_dataset('val', VAL_TARGETS)
    
    print("\n🎉 Success! Your new balanced training and validation datasets are ready.")
