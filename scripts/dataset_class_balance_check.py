import os
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

CLASS_MAP = {
    0: 'person', 1: 'rider', 2: 'car', 3: 'truck',
    4: 'bus', 5: 'train', 6: 'motor', 7: 'bike',
    8: 'traffic light', 9: 'traffic sign'
}

def analyze_dataset_balance(label_dir, dataset_name):
    """
    Analyzes the class distribution of a dataset in YOLO format.
    """
    print(f"\nAnalyzing '{dataset_name}' in directory: {label_dir}")
    
    if not os.path.exists(label_dir):
        print(f"Error: Directory not found -> {label_dir}")
        return

    class_counts = defaultdict(int)
    all_label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

    if not all_label_files:
        print(f"Error: No .txt files found in '{label_dir}'.")
        return

    for filename in tqdm(all_label_files, desc=f"Processing {dataset_name}"):
        with open(os.path.join(label_dir, filename), 'r') as f:
            for line in f.readlines():
                try:
                    class_id = int(line.split()[0])
                    class_name = CLASS_MAP.get(class_id)
                    if class_name:
                        class_counts[class_name] += 1
                except (ValueError, IndexError):
                    continue

    print(f"Analysis complete for '{dataset_name}'.")

    if not class_counts:
        print("No objects were found in any label files.")
        return

    df = pd.DataFrame(list(class_counts.items()), columns=['Class', 'Instance Count'])
    df = df.sort_values(by='Instance Count', ascending=False).reset_index(drop=True)
    
    print(f"\n--- {dataset_name} Class Distribution ---")
    print(df.to_string())

    plt.figure(figsize=(12, 8))
    bars = plt.bar(df['Class'], df['Instance Count'], color='green')
    plt.xlabel('Object Class', fontsize=12)
    plt.ylabel('Number of Instances', fontsize=12)
    plt.title(f'Class Distribution in {dataset_name}', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom')

    chart_filename = f"{dataset_name.lower().replace(' ', '_')}_distribution.png"
    plt.savefig(chart_filename)
    print(f"\nA bar chart has been saved to '{chart_filename}'")
    plt.close()


if __name__ == '__main__':

    balanced_train_labels = './datasets/bdd100k_balanced/labels/train'
    balanced_val_labels = './datasets/bdd100k_balanced/labels/val'

    analyze_dataset_balance(balanced_train_labels, "Balanced Training Set")
    analyze_dataset_balance(balanced_val_labels, "Balanced Validation Set")
