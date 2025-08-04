import json
import os
import shutil
from tqdm import tqdm

def convert_to_yolo(source_dir, output_dir):
    """
    Converts BDD100K JSON files to YOLOv8 TXT format using the
    correct class names discovered from forensic analysis.
    """
    # --- FINAL, CORRECTED CLASS MAP ---
    # This now correctly uses 'person', 'motor', and 'bike'.
    class_map = {
        'person': 0, 'rider': 1, 'car': 2, 'truck': 3,
        'bus': 4, 'train': 5, 'motor': 6, 'bike': 7,
        'traffic light': 8, 'traffic sign': 9
    }

    json_files = [f for f in os.listdir(source_dir) if f.endswith('.json')]
    print(f"\nFound {len(json_files)} json files. Converting to YOLO format...")

    for json_file in tqdm(json_files, desc=f"Converting {os.path.basename(source_dir)}"):
        json_path = os.path.join(source_dir, json_file)
        with open(json_path) as f:
            data = json.load(f)

        txt_filename = data['name'] + '.txt'
        txt_filepath = os.path.join(output_dir, txt_filename)

        all_yolo_labels = []
        if 'frames' in data:
            for frame in data['frames']:
                if 'objects' in frame:
                    for label in frame['objects']:
                        category = label.get('category')
                        if category in class_map and 'box2d' in label:
                            class_id = class_map[category]
                            x1, y1, x2, y2 = label['box2d'].values()

                            img_width, img_height = 1280, 720
                            box_width = x2 - x1
                            box_height = y2 - y1
                            center_x = x1 + box_width / 2
                            center_y = y1 + box_height / 2

                            norm_center_x = center_x / img_width
                            norm_center_y = center_y / img_height
                            norm_width = box_width / img_width
                            norm_height = box_height / img_height

                            all_yolo_labels.append(f"{class_id} {norm_center_x:.6f} {norm_center_y:.6f} {norm_width:.6f} {norm_height:.6f}")

        unique_labels = sorted(list(set(all_yolo_labels)))
        if unique_labels:
            with open(txt_filepath, 'w') as f_out:
                f_out.write('\n'.join(unique_labels))

if __name__ == '__main__':
    source_root = './datasets/bdd100k/labels_json/100k'
    output_root = './datasets/bdd100k/labels/100k'

    if os.path.exists(output_root):
        print(f"Deleting old incorrect labels in '{output_root}'...")
        shutil.rmtree(output_root)

    for split in ['train', 'val']:
        source_dir = os.path.join(source_root, split)
        output_dir = os.path.join(output_root, split)
        if os.path.exists(source_dir):
            os.makedirs(output_dir, exist_ok=True)
            convert_to_yolo(source_dir, output_dir)
        else:
            print(f"Source directory for '{split}' not found, skipping: {source_dir}")

    print("\nConversion complete!")
