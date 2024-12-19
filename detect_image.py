import torch
from PIL import Image
from ultralytics import YOLO

# 1. Load the trained YOLOv5 model using the ultralytics package
model = YOLO('runs/detect/train3/weights/best.pt')
# model = YOLO('yolov8n.pt')
# 2. Load and preprocess the random image
image_path = 'test_images/IMG20241218113802.jpg'
img = Image.open(image_path)

# 3. Perform inference on the image
results = model(img)
# Print the class names the model can detect
# print("Class names the model can detect:", results.names)

# Assuming only one image is passed (results will be a list of one result)
result = results[0]

# 4. Check if there are any detections
if len(result.boxes) > 0:
    print(f"Detected {len(result.boxes)} objects:")
    for box in result.boxes:
        print(f"Class: {box.cls}, Coordinates: {box.xywh}, Confidence: {box.conf}")
else:
    print("No objects detected.")

# 5. Show the annotated image
result.show()  # This will display the image with bounding boxes if detections exist

# 6. Save the annotated image
result.save(filename="result.jpg")  # Save annotated image to 'result.jpg'
