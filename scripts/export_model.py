"""
This script converts a PyTorch-based YOLOv8 model (.pt) to the ONNX format.

ONNX (Open Neural Network Exchange) is a standard format for machine learning
models that allows them to be used across different frameworks and inference engines,
like OpenCV's DNN module.

The script is designed to be run from anywhere within the project by dynamically
calculating the absolute paths to the model and output files.
"""

from pathlib import Paths
from ultralytics import YOLO

def main():
    # Use pathlib to dynamically construct absolute paths. This is the modern,
    # preferred way to handle file paths in Python.
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    input_model_path = project_root / "models" / "best.pt"
    output_model_path = project_root / "models" / "best.onnx"

    print(f"Loading model from: {input_model_path}")

    # Load the YOLOv8 model from its PyTorch checkpoint.
    model = YOLO(input_model_path)

    # Export the model to ONNX format for deployment.
    # This creates a portable model for various inference engines.
    model.export(format='onnx')

    print(f"Model successfully exported to: {output_model_path}")


if __name__ == '__main__':
    main()
