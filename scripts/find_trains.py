import os
import json
from tqdm import tqdm

# --- 1. CONFIGURATION ---
# The category name we are looking for
TARGET_CATEGORY = 'train'

# This is the correct path you provided
test_label_dir = './datasets/bdd100k/labels_json/100k/test'


def find_videos_with_category(label_dir, target_category):
    """
    Scans a directory of BDD100K JSON label files and finds all files
    containing a specific object category.
    """
    print(f"\n🔍 Scanning for category '{target_category}' in directory: {label_dir}")

    if not os.path.exists(label_dir):
        print(f"Error: Directory not found -> {label_dir}")
        return []

    files_with_target = []
    all_json_files = [f for f in os.listdir(label_dir) if f.endswith('.json')]

    if not all_json_files:
        print(f"Error: No .json files found in '{label_dir}'.")
        return []

    for filename in tqdm(all_json_files, desc="Scanning JSON labels"):
        json_path = os.path.join(label_dir, filename)
        with open(json_path, 'r') as f:
            data = json.load(f)

        found = False
        if 'frames' in data:
            for frame in data['frames']:
                # The 'objects' key in the JSON is what we need to check
                if 'objects' in frame:
                    for label in frame['objects']:
                        if label.get('category') == target_category:
                            # Found it! Record the video name and stop checking this file.
                            video_name = data.get('name')
                            files_with_target.append(video_name)
                            found = True
                            break # Stop checking objects in this frame
                if found:
                    break # Stop checking frames in this file

    return files_with_target

if __name__ == '__main__':
    videos_containing_trains = find_videos_with_category(test_label_dir, TARGET_CATEGORY)

    if videos_containing_trains:
        print(f"\n✅ Found {len(videos_containing_trains)} test videos containing at least one '{TARGET_CATEGORY}'.")
        print("Here is the list:")
        # Print a clean, sorted list
        for video_name in sorted(list(set(videos_containing_trains))):
            print(f"  - {video_name}")
    else:
        print(f"\n❌ No videos containing a '{TARGET_CATEGORY}' were found in the specified directory.")
