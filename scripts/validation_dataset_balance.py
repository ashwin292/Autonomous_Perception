import os
import random
import shutil
from collections import defaultdict
from tqdm import tqdm

# Path to the original BDD100K *validation* JPEG images
image_dir = "./datasets/bdd100k/images/100k/val"

# Path to the matching YOLO-style *.txt* label files
yolo_label_dir = "./datasets/bdd100k/labels/100k/val"

# Where to place the new, balanced validation subset
output_dir = "./datasets/bdd100k_balanced"

# YOLO stores classes as integers.  This dictionary converts them back
# to human-readable names.  Make sure it matches whatever mapping you
# used when you *created* the YOLO labels.
class_map_reverse = {
    0: "pedestrian",
    1: "rider",
    2: "car",
    3: "truck",
    4: "bus",
    5: "train",
    6: "motorcycle",
    7: "bicycle",
    8: "traffic light",
    9: "traffic sign",
}

# These targets keep the validation set about one-fifth the size of
# the balanced training set while preserving the same proportions.
sampling_targets = {
    "car": 2000,
    "truck": 1000,
    "bus": 1000,
    "pedestrian": 1600,
    "rider": 800,
    "bicycle": 800,
    "motorcycle": 800,
    "train": 400,
    "traffic light": 1400,
    "traffic sign": 1400,
}

print("Scanning validation labels and building an image→classes map...")

image_to_classes = defaultdict(set)
all_label_files = [f for f in os.listdir(yolo_label_dir) if f.endswith(".txt")]

for filename in tqdm(all_label_files, desc="Reading *.txt files"):
    image_name = filename.replace(".txt", ".jpg")

    # Each line in a YOLO label file looks like:
    # <class_id> <x_center> <y_center> <width> <height>
    with open(os.path.join(yolo_label_dir, filename), "r") as f:
        for line in f:
            try:
                class_id = int(line.split()[0])
                class_name = class_map_reverse.get(class_id)
                if class_name:
                    image_to_classes[image_name].add(class_name)
            except (ValueError, IndexError):
                # Ignore malformed lines instead of crashing
                continue

print(f"Found labels for {len(image_to_classes):,} validation images.")

print("Selecting a balanced set of validation images...")

final_image_set: set[str] = set()

for class_name, target_count in sampling_targets.items():
    # Grab every image containing *at least one* instance of this class
    images_with_class = [
        img for img, classes in image_to_classes.items()
        if class_name in classes
    ]

    # If we have more than we need, sample; otherwise keep them all
    if len(images_with_class) > target_count:
        selected = random.sample(images_with_class, target_count)
    else:
        selected = images_with_class

    final_image_set.update(selected)
    print(
        f"  • {class_name:<13} "
        f"(need ≤{target_count}, found {len(images_with_class)}) → "
        f"using {len(selected)}"
    )

print(
    f"Done. Total unique validation images selected: "
    f"{len(final_image_set):,}"
)

print("Copying the chosen images and labels...")

# Destination directories:
balanced_img_dir = os.path.join(output_dir, "images/val")
balanced_lbl_dir = os.path.join(output_dir, "labels/val")
os.makedirs(balanced_img_dir, exist_ok=True)
os.makedirs(balanced_lbl_dir, exist_ok=True)

for image_name in tqdm(final_image_set, desc="Copying"):
    # Source paths
    src_img_path = os.path.join(image_dir, image_name)
    src_lbl_path = os.path.join(
        yolo_label_dir, image_name.replace(".jpg", ".txt")
    )

    # Destination paths
    dst_img_path = os.path.join(balanced_img_dir, image_name)
    dst_lbl_path = os.path.join(
        balanced_lbl_dir, image_name.replace(".jpg", ".txt")
    )

    # Only copy if both image *and* its label exist
    if os.path.exists(src_img_path) and os.path.exists(src_lbl_path):
        shutil.copyfile(src_img_path, dst_img_path)
        shutil.copyfile(src_lbl_path, dst_lbl_path)

print(
    f"All done! Your balanced validation split lives in "
    f"'{output_dir}'. Point your training script at that folder."
)
