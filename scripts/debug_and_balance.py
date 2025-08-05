"""
Create a balanced subset of the BDD100k dataset from its YOLO format.

This script addresses class imbalance by performing stratified sampling. It allows
you to define a target number of images for each class, then creates a new
dataset by copying the required image and label files. This is useful for
preventing a model from becoming biased towards over-represented classes like 'car'.

The script processes both 'train' and 'val' splits independently.
"""
import os
import random
import shutil
from collections import defaultdict

from tqdm import tqdm

# --- Dataset Configuration ---
# This dictionary maps the integer class IDs from the YOLO .txt files to their
# human-readable string names. Ensure this matches your dataset's convention.
CLASS_NAMES = {
    0: 'person', 1: 'rider', 2: 'car', 3: 'truck',
    4: 'bus', 5: 'train', 6: 'motor', 7: 'bike',
    8: 'traffic light', 9: 'traffic sign'
}

# Define the target number of images for each class in the final balanced dataset.
# The strategy is to over-sample rare classes (e.g., 'train') and under-sample
# common ones (e.g., 'car') to create a more balanced distribution.
TRAIN_TARGETS = {
    'car': 10000, 'truck': 5000, 'bus': 5000, 'person': 8000,
    'rider': 4000, 'bike': 4000, 'motor': 4000, 'train': 2000,
    'traffic light': 7000, 'traffic sign': 7000
}
# Validation sets typically require fewer images than training sets.
VAL_TARGETS = {
    'car': 2000, 'truck': 1000, 'bus': 1000, 'person': 1500,
    'rider': 800, 'bike': 800, 'motor': 800, 'train': 400,
    'traffic light': 1200, 'traffic sign': 1200
}

def balance_dataset(split_name: str, sampling_targets: dict):
    """
    Scans, samples, and copies files for a given data split ('train' or 'val').

    This function performs the entire balancing pipeline:
    1. Catalogs all labels to map images to the classes they contain.
    2. Selects a unique set of images based on the sampling targets.
    3. Copies the selected images and their corresponding labels to a new
       directory structure.

    Args:
        split_name: The name of the dataset split (e.g., 'train', 'val').
        sampling_targets: A dictionary defining the target image count per class.
    """
    print(f"\n{'='*20} PROCESSING '{split_name.upper()}' SET {'='*20}")

    # Define source and destination paths based on the current split
    image_dir = f'./datasets/bdd100k/images/100k/{split_name}'
    yolo_label_dir = f'./datasets/bdd100k/labels/100k/{split_name}'
    output_dir = './datasets/bdd100k_balanced'

    # First, build a catalog of which classes appear in each image.
    print(f"Scanning all .txt files in: {yolo_label_dir}")
    image_to_classes = defaultdict(set)
    all_label_files = [f for f in os.listdir(yolo_label_dir) if f.endswith('.txt')]

    for filename in tqdm(all_label_files, desc=f"Cataloging {split_name} labels"):
        with open(os.path.join(yolo_label_dir, filename), 'r') as f:
            for line in f.readlines():
                # This try-except block handles potentially empty or malformed
                # lines in the label files, preventing the script from crashing.
                try:
                    class_id = int(line.split()[0])
                    class_name = CLASS_NAMES.get(class_id)
                    if class_name:
                        image_name = filename.replace('.txt', '.jpg')
                        image_to_classes[image_name].add(class_name)
                except (ValueError, IndexError):
                    continue  # Ignore malformed lines and proceed

    # Next, select a unique set of images that satisfies the sampling targets.
    print("Selecting a balanced set of images...")
    final_image_set = set()
    for class_name, target_count in sampling_targets.items():
        images_with_class = [img for img, classes in image_to_classes.items() if class_name in classes]
        if not images_with_class:
            print(f"Warning: No images found for class '{class_name}' in the {split_name} set.")
            continue
        
        # Take the smaller of the two values to avoid errors if a class has
        # fewer images available than the desired target.
        num_to_sample = min(len(images_with_class), target_count)
        selected = random.sample(images_with_class, num_to_sample)
        final_image_set.update(selected)

    print(f"Image selection complete. Total unique images for balanced {split_name} set: {len(final_image_set)}")

    # Finally, copy the selected image and label files to the new directory.
    print("Copying files to new balanced directory...")
    balanced_img_dir = os.path.join(output_dir, f'images/{split_name}')
    balanced_lbl_dir = os.path.join(output_dir, f'labels/{split_name}')
    os.makedirs(balanced_img_dir, exist_ok=True)
    os.makedirs(balanced_lbl_dir, exist_ok=True)

    for image_name in tqdm(final_image_set, desc=f"Copying {split_name} files"):
        label_name = image_name.replace('.jpg', '.txt')
        shutil.copyfile(os.path.join(image_dir, image_name), os.path.join(balanced_img_dir, image_name))
        shutil.copyfile(os.path.join(yolo_label_dir, label_name), os.path.join(balanced_lbl_dir, label_name))

if __name__ == '__main__':
    # For a consistent result, it's best to start with a clean slate.
    # This removes the output directory to prevent mixing files from previous runs.
    output_dir = './datasets/bdd100k_balanced'
    if os.path.exists(output_dir):
        print(f"Deleting old balanced dataset folder: {output_dir}")
        shutil.rmtree(output_dir)

    # Process the training and validation sets independently using their respective targets.
    balance_dataset('train', TRAIN_TARGETS)
    balance_dataset('val', VAL_TARGETS)

    print("\nSuccess! Your new balanced training and validation datasets are ready.")
