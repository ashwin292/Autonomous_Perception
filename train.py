from ultralytics import YOLO

def main():
    # Load a pre-trained YOLOv8s model
    model = YOLO('best.pt')

    # Train the model on our BDD100K dataset
    results = model.train(
        data='bdd100k.yaml',
        epochs=50,
        imgsz=640,
        # 'mps' tells PyTorch to use Apple's Metal GPU.
        # If you have an older Intel Mac, use 'cpu'.
        device='mps', 
        batch=8  # Start with 8 for Mac, increase if you don't get memory errors.
    )
    print("Training complete. Results saved in the 'runs' directory.")

if __name__ == '__main__':
    main()