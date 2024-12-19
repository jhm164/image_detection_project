from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # Nano model (small and fast)

# Train the model
model.train(data='data.yaml', epochs=50, imgsz=640)
