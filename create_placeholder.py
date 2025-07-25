from ultralytics import YOLO

def main():
    # Load the base pre-trained model (not your custom one)
    model = YOLO('yolov8s.pt')

    # Export it directly to ONNX format
    model.export(format='onnx')

    print("Placeholder model 'yolov8s.onnx' created successfully!")
    print("You can now use this file for your C++ development.")

if __name__ == '__main__':
    main()
