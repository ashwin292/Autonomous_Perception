# Autonomous Perception Simulation

This project is a C++ application that uses a custom-trained YOLOv8 model to perform real-time object detection on driving videos.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone [your-repo-url]
    cd autonomous_perception
    ```
2.  **Set up the Python environment and download data:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt 
    # Add instructions here for where to download BDD100K
    ```
3.  **Train the model and export to ONNX:**
    ```bash
    python train.py
    python export_model.py
    ```
4.  **Build the C++ Application:**
    ```bash
    mkdir build
    cd build
    cmake ..
    make
    ```

## Usage

Run the application from the project's root directory:
```bash
./build/perception_app
```