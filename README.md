# Real-Time Driving Perception System

This project is a C++ application that performs real-time object detection on driving videos using a custom-trained YOLOv8 model. It processes an input video and generates an output video with bounding boxes drawn around detected objects like cars, pedestrians, and traffic signs.

---

## 🎥 Demo

System demo in various driving conditions.

<table align="center">
  <tr>
    <td align="center">
      <img src="demo/bike.gif" width="400">
    </td>
    <td align="center">
      <img src="demo/night.gif" width="400">
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="demo/person.gif" width="400">
    </td>
    <td align="center">
      <img src="demo/truck.gif" width="400">
    </td>
  </tr>
</table>

<a href="https://www.youtube.com/watch?v=PZyQY9SmM7c&list=PLC8cnhLHv7nnzu-YifiSjzikhgsCITUx9" target="_blank"><strong>Watch the Full-Quality Video on YouTube</strong></a>

---

## ✨ Key Features

* **High-Performance C++ Core**: The detection logic is built in C++ using OpenCV's DNN module for efficient video processing.
* **Custom YOLOv8 Model**: The system uses a `best.onnx` model custom-trained on the BDD100k dataset.
* **Data Processing Pipeline**: Includes a suite of Python scripts in the `/scripts` directory for transforming the BDD100k dataset into YOLO `.txt` format and analyzing class distribution.

---

## 🛠️ Tech Stack

* **C++17**
* **OpenCV**
* **ONNX Runtime** (or OpenCV DNN)
* **YOLOv8** (trained with PyTorch)
* **CMake** for building the project
* **Python** for data processing scripts

---

## 🚀 Setup and Run

### Prerequisites

* A C++ compiler (g++)
* CMake (version 3.10 or higher)
* OpenCV library installed

### Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ashwin292/Autonomous_Perception.git
    cd autonomous_perception
    ```
2.  **Build the project:**
    ```bash
    mkdir build
    cd build
    cmake ..
    make
    ```

3.  **Run the application:**
    After building, the executable will be inside the `build/` directory. Run it from the root directory of the project like this:
    ```bash
    ./build/your_executable_name
    ```
    *Note: Ensure the paths to the model (`models/best.onnx`) and input video (`data/driving.mov`) are correctly specified inside the `src/main.cpp` file.*