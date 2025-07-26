from ultralytics import YOLO

def main():
    # Load your custom-trained model
    model = YOLO('best.pt')

    # Export the model to ONNX format
    model.export(format='onnx')
    print("Model successfully exported to best.onnx")

if __name__ == '__main__':
    main()
