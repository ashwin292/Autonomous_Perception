import json
import os
from collections import defaultdict
from tqdm import tqdm

def run_forensic_analysis(label_dir):
    """
    Scans BDD100K JSON files to discover all unique object structures.
    """
    print(f"--- 🕵️‍♂️ Starting Forensic Analysis on: {label_dir} ---")
    
    # This set will store string representations of the keys for each unique object type
    found_structures = defaultdict(int)
    
    json_files = [f for f in os.listdir(label_dir) if f.endswith('.json')]
    
    # We'll just scan the first 5000 files to get a representative sample quickly
    sample_files = json_files[:5000]
    
    print(f"Analyzing a sample of {len(sample_files)} files...")

    for json_file in tqdm(sample_files, desc="Analyzing JSON structures"):
        json_path = os.path.join(label_dir, json_file)
        with open(json_path) as f:
            data = json.load(f)

        if 'frames' in data:
            for frame in data['frames']:
                # Check for both 'objects' and 'labels' keys
                object_list = frame.get('objects', frame.get('labels', []))
                
                for label_obj in object_list:
                    # Get the category to see if it's one we are missing
                    category = label_obj.get('category', 'N/A')
                    
                    # Create a sorted tuple of the keys in this object's dictionary
                    # This gives us a unique signature for its structure
                    structure_signature = tuple(sorted(label_obj.keys()))
                    
                    # We'll store the category along with the structure
                    found_structures[(category, structure_signature)] += 1

    print("\n--- 🕵️‍♂️ Forensic Report ---")
    print("Found the following unique object structures (Category, Keys):\n")
    
    if not found_structures:
        print("No objects or labels found in any of the sampled files.")
        return

    # Print out the findings in a readable format
    for (category, structure), count in sorted(found_structures.items()):
        print(f"Category: '{category}'")
        print(f"  Keys: {structure}")
        print(f"  Occurrences: {count}\n")
        
    print("--------------------------------------------------")
    print("Please share this report. It will tell us the exact keys needed to parse all classes.")


if __name__ == '__main__':
    # IMPORTANT: This must point to your ORIGINAL BDD100K JSON labels, not the .txt files
    source_label_dir = './datasets/bdd100k/labels_json/100k/train'
    
    if not os.path.exists(source_label_dir):
        print(f"ERROR: Source directory not found: {source_label_dir}")
        print("Please make sure this script is pointing to the original BDD100K JSON files.")
    else:
        run_forensic_analysis(source_label_dir)
