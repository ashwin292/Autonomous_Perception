import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

# --- Configuration Constants ---
# This map defines the meaning of the class IDs in the YOLO .txt files.
CLASS_MAP = {
    0: 'person', 1: 'rider', 2: 'car', 3: 'truck',
    4: 'bus', 5: 'train', 6: 'motor', 7: 'bike',
    8: 'traffic light', 9: 'traffic sign'
}

# Define the target number of images for each class in the balanced dataset.
SAMPLING_TARGETS = {
    'car': 10000,
    'truck': 5000,
    'bus': 5000,
    'person': 8000,
    'rider': 4000,
    'bike': 4000,
    'motor': 4000,
    'train': 2000,
    'traffic light': 7000,
    'traffic sign': 7000,
}


def catalog_dataset(label_dir: Path) -> dict[str, set[str]]:
    """
    Scans a directory of YOLO .txt files and catalogs which classes appear in each image.

    Args:
        label_dir: The path to the directory containing YOLO label files.

    Returns:
        A dictionary mapping each image basename (e.g., 'image01.jpg') to a set
        of class names present in that image.
    """
    print("Cataloging all YOLO label data...")
    image_to_classes = defaultdict(set)
    label_files = list(label_dir.glob('*.txt'))

    for label_file in tqdm(label_files, desc="Parsing labels"):
        image_basename = label_file.with_suffix('.jpg').name
        with open(label_file, 'r') as f:
            for line in f:
                try:
                    class_id = int(line.split()[0])
                    class_name = CLASS_MAP.get(class_id)
                    if class_name:
                        image_to_classes[image_basename].add(class_name)
                except (ValueError, IndexError):
                    print(f"Warning: Skipping malformed line in {label_file.name}")
    
    print(f"Catalog complete. Found {len(image_to_classes)} images with labels.")
    return image_to_classes


def select_balanced_subset(image_catalog: dict[str, set[str]]) -> set[str]:
    """
    Selects a balanced subset of images based on the sampling targets.

    Args:
        image_catalog: A dictionary mapping image names to the classes they contain.

    Returns:
        A set of unique image basenames that form the balanced dataset.
    """
    print("Selecting a balanced set of images...")
    final_image_set = set()

    for class_name, target_count in SAMPLING_TARGETS.items():
        # Find all images that contain the current class
        images_with_class = [
            img for img, classes in image_catalog.items() if class_name in classes
        ]
        
        # Sample from the list to meet the target count
        if len(images_with_class) > target_count:
            selected = random.sample(images_with_class, target_count)
        else:
            selected = images_with_class  # Take all if not enough are available
            
        final_image_set.update(selected)
        print(f"  - Class '{class_name}': Found {len(images_with_class)} images, using {len(selected)}.")
        
    print(f"Image selection complete. Total unique images selected: {len(final_image_set)}")
    return final_image_set


def create_balanced_dataset(
    image_files_to_copy: set[str],
    src_img_dir: Path,
    src_lbl_dir: Path,
    output_dir: Path
):
    """
    Copies the selected image and label files to a new directory structure.
    
    Args:
        image_files_to_copy: A set of image basenames to include in the new dataset.
        src_img_dir: The original directory of images.
        src_lbl_dir: The original directory of labels.
        output_dir: The root directory for the new balanced dataset.
    """
    print("Copying files to the new balanced directory...")
    balanced_img_dir = output_dir / 'images' / 'train'
    balanced_lbl_dir = output_dir / 'labels' / 'train'

    balanced_img_dir.mkdir(parents=True, exist_ok=True)
    balanced_lbl_dir.mkdir(parents=True, exist_ok=True)

    for image_name in tqdm(image_files_to_copy, desc="Copying files"):
        label_name = Path(image_name).with_suffix('.txt').name

        src_img = src_img_dir / image_name
        src_lbl = src_lbl_dir / label_name
        
        dest_img = balanced_img_dir / image_name
        dest_lbl = balanced_lbl_dir / label_name
        
        if src_img.exists() and src_lbl.exists():
            shutil.copyfile(src_img, dest_img)
            shutil.copyfile(src_lbl, dest_lbl)

    print(f"Success! Your new balanced dataset is ready at '{output_dir}'.")


def main():
    """Main function to run the dataset balancing process."""
    parser = argparse.ArgumentParser(
        description="Create a balanced subset of a YOLO-formatted dataset."
    )
    parser.add_argument(
        "--image_dir", type=Path, required=True,
        help="Path to the directory containing the original images."
    )
    parser.add_argument(
        "--label_dir", type=Path, required=True,
        help="Path to the directory containing the original YOLO .txt label files."
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True,
        help="Path to the directory where the balanced dataset will be saved."
    )
    args = parser.parse_args()

    # Step 1: Understand the content of the dataset
    image_catalog = catalog_dataset(args.label_dir)

    # Step 2: Choose which images to use for the balanced set
    selected_images = select_balanced_subset(image_catalog)

    # Step 3: Copy the chosen files to the new location
    create_balanced_dataset(
        selected_images, args.image_dir, args.label_dir, args.output_dir
    )


if __name__ == "__main__":
    main()
