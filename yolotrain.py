from ultralytics import YOLO

# model = YOLO("yolo8n.yaml") 
# # Load a pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # Nano model (small and fast)
# # Load a model
# model = YOLO("yolo8n.yaml")  # build a new model from YAML
# model = YOLO("yolo8n.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolo8n.yaml").load("yolo8n.pt")  # build from YAML and transfer weights
# Train the model
model.train(data='data.yaml')
