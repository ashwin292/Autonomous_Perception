"""
Scans a directory of BDD100K JSON label files to find and list all video
files that contain a specific object category. This is useful for creating
a targeted subset of a larger dataset for analysis or training.
"""
import os
import json
from tqdm import tqdm

# --- Script Configuration ---
# Define the object category to search for within the label files.
TARGET_CATEGORY = 'train'

# Set the path to the directory containing the BDD100K test set JSON labels.
TEST_LABEL_DIR = './datasets/bdd100k/labels_json/100k/test'


def find_videos_with_category(label_dir: str, target_category: str) -> list[str]:
    """
    Scans a directory of BDD100K JSON files and identifies all videos
    containing at least one instance of a specific object category.

    Args:
        label_dir: The path to the directory containing the .json label files.
        target_category: The name of the category to search for (e.g., 'train').

    Returns:
        A list of video names (e.g., 'b1c66a42-6f7d68ca.mov') that contain the
        target category. The list may contain duplicates if a video is processed
        multiple times, which should be handled by the caller.
    """
    print(f"\nScanning for category '{target_category}' in directory: {label_dir}")

    if not os.path.exists(label_dir):
        print(f"Error: Directory not found -> {label_dir}")
        return []

    files_with_target = []
    all_json_files = [f for f in os.listdir(label_dir) if f.endswith('.json')]

    if not all_json_files:
        print(f"Error: No .json files found in '{label_dir}'.")
        return []

    # Iterate through each JSON file with a progress bar.
    for filename in tqdm(all_json_files, desc="Scanning JSON labels"):
        json_path = os.path.join(label_dir, filename)
        with open(json_path, 'r') as f:
            data = json.load(f)

        found_in_file = False
        if 'frames' in data:
            for frame in data['frames']:
                if 'objects' in frame:
                    for label in frame['objects']:
                        # Use .get() to safely access the 'category' key, preventing KeyErrors.
                        if label.get('category') == target_category:
                            # Once the category is found, record the video name and
                            # break out of the inner loops to improve efficiency.
                            video_name = data.get('name')
                            if video_name:
                                files_with_target.append(video_name)
                            found_in_file = True
                            break  # Exit the 'objects' loop
                if found_in_file:
                    break  # Exit the 'frames' loop

    return files_with_target


# --- Main execution block ---
if __name__ == '__main__':
    # Find all videos that match the target category.
    videos_containing_trains = find_videos_with_category(TEST_LABEL_DIR, TARGET_CATEGORY)

    if videos_containing_trains:
        # Use a set to get unique video names, then sort for a clean, deterministic output.
        unique_videos = sorted(list(set(videos_containing_trains)))
        
        print(f"\nFound {len(unique_videos)} test videos containing at least one '{TARGET_CATEGORY}'.")
        print("Here is the list:")
        for video_name in unique_videos:
            print(f"  - {video_name}")
    else:
        print(f"\nNo videos containing a '{TARGET_CATEGORY}' were found in the specified directory.")

