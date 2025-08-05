import json
import os
from collections import defaultdict
from tqdm import tqdm


def run_forensic_analysis(label_dir: str) -> None:
    """
    Look through a folder of BDD100K-style JSON label files and report every
    *different* object/label dictionary structure we encounter.

    Why this matters
    ----------------
    In BDD100K (and many other datasets) not every annotation has the same set
    of keys.  Some categories include extra fields; others omit common ones.
    Before you write a parser, it helps to know exactly which permutations
    of keys exist in the wild.  This little “forensic” pass gives you that
    inventory.
    """
    print(f"Starting forensic pass in: {label_dir}")

    # (category, key-tuple) → how many times we’ve seen it
    found_structures = defaultdict(int)

    # Limit ourselves to .json files only
    json_files = [f for f in os.listdir(label_dir) if f.endswith(".json")]

    # A quick scan of the first 5 000 files is usually plenty to get coverage
    sample_files = json_files[:5000]
    print(f"Scanning {len(sample_files)} file(s) for object structures…")

    for json_file in tqdm(sample_files, desc="Parsing JSON"):
        json_path = os.path.join(label_dir, json_file)

        # Load a single JSON label file
        with open(json_path, "r") as f:
            data = json.load(f)

        # Each file is a list of video frames; every frame contains objects/labels
        if "frames" in data:
            for frame in data["frames"]:
                # BDD100K sometimes uses the key 'objects', sometimes 'labels'
                object_list = frame.get("objects", frame.get("labels", []))

                for obj in object_list:
                    # We’ll group by category so we know which structures belong to what
                    category = obj.get("category", "N/A")

                    # A sorted tuple of keys uniquely identifies the object’s schema
                    structure_signature = tuple(sorted(obj.keys()))

                    found_structures[(category, structure_signature)] += 1

    # ---------- Report ---------- #
    print("\nForensic Report")
    if not found_structures:
        print("No objects or labels were found in the sampled files.")
        return

    print("Below are all unique (category, key-set) pairs and their frequencies:\n")
    for (category, keys), count in sorted(found_structures.items()):
        print(f"Category: {category!r}")
        print(f"  Keys: {keys}")
        print(f"  Seen: {count} time(s)\n")

if __name__ == "__main__":
    # Path to the *original* BDD100K JSON labels (NOT the YOLO-style .txt files)
    source_label_dir = "./datasets/bdd100k/labels_json/100k/train"

    if not os.path.exists(source_label_dir):
        print(f"ERROR: Cannot find directory: {source_label_dir}")
        print("Double-check that the path points to the unmodified JSON labels.")
    else:
        run_forensic_analysis(source_label_dir)
