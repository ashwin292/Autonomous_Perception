import json
import os
from tqdm import tqdm

def convert_to_yolo(source_dir, output_dir):
    """
    Converts individual BDD100K JSON files (with 'frames' structure) to YOLOv8 TXT format.
    """
    class_map = {
        'pedestrian': 0, 'rider': 1, 'car': 2, 'truck': 3,
        'bus': 4, 'train': 5, 'motorcycle': 6, 'bicycle': 7,
        'traffic light': 8, 'traffic sign': 9
    }
    
    json_files = [f for f in os.listdir(source_dir) if f.endswith('.json')]
    
    print(f"\nFound {len(json_files)} json files in {source_dir}. Converting to YOLO format...")

    for json_file in tqdm(json_files, desc=f"Converting {os.path.basename(source_dir)}"):
        json_path = os.path.join(source_dir, json_file)

        with open(json_path) as f:
            data = json.load(f)

        img_width = 1280
        img_height = 720
        
        yolo_labels = []
        
        # --- START: CORRECTED LOGIC ---
        # Check for the 'frames' key and that it's not empty
        if 'frames' in data and data['frames']:
            # Get the list of objects from the first frame
            objects_list = data['frames'][0].get('objects', [])
            
            for label in objects_list:
                category = label.get('category')
                # We only care about objects with bounding boxes for detection
                if category in class_map and 'box2d' in label:
                    class_id = class_map[category]
                    
                    x1 = label['box2d']['x1']
                    y1 = label['box2d']['y1']
                    x2 = label['box2d']['x2']
                    y2 = label['box2d']['y2']
                    
                    box_width = x2 - x1
                    box_height = y2 - y1
                    center_x = x1 + box_width / 2
                    center_y = y1 + box_height / 2
                    
                    norm_center_x = center_x / img_width
                    norm_center_y = center_y / img_height
                    norm_width = box_width / img_width
                    norm_height = box_height / img_height
                    
                    yolo_labels.append(f"{class_id} {norm_center_x:.6f} {norm_center_y:.6f} {norm_width:.6f} {norm_height:.6f}")
        # --- END: CORRECTED LOGIC ---

        if yolo_labels:
            txt_filename = os.path.splitext(json_file)[0] + '.txt'
            txt_filepath = os.path.join(output_dir, txt_filename)
            with open(txt_filepath, 'w') as f_out:
                f_out.write('\n'.join(yolo_labels))

if __name__ == '__main__':
    base_dir = './datasets/bdd100k'
    source_root = os.path.join(base_dir, 'labels_json', '100k')
    output_root = os.path.join(base_dir, 'labels', '100k')

    for split in ['train', 'val', 'test']:
        source_dir = os.path.join(source_root, split)
        output_dir = os.path.join(output_root, split)
        if os.path.exists(source_dir):
            os.makedirs(output_dir, exist_ok=True)
            convert_to_yolo(source_dir, output_dir)
        else:
            print(f"Source directory for '{split}' not found, skipping: {source_dir}")

    print("\nConversion complete! Check your 'labels' folder.")
