import os
from ultralytics import YOLO

def main():
    # --- START OF NEW, MORE ROBUST PATH LOGIC ---

    # Get the absolute path of the directory where this script is located (i.e., .../scripts/)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Get the path to the project's root directory (by going up one level from /scripts)
    project_root = os.path.dirname(script_dir)

    # Construct the full, absolute path to the model file
    input_model_path = os.path.join(project_root, 'models', 'best.pt')
    output_model_path = os.path.join(project_root, 'models', 'best.onnx')

    # --- END OF NEW LOGIC ---

    print(f"Attempting to load model from absolute path: {input_model_path}")

    # Load your custom-trained model
    model = YOLO(input_model_path)

    # Export the model to ONNX format
    model.export(format='onnx')

    print(f"Model successfully exported to: {output_model_path}")


if __name__ == '__main__':
    main()
