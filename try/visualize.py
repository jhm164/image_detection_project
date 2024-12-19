import albumentations as A
import cv2
import matplotlib.pyplot as plt

# Define augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.Resize(512, 512)  # Resizes the image to 512x512
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

# Input image and labels
image = cv2.imread("C:\Users\saura\OneDrive\Documents\AI_project\ML\code\dataset\original_images\L.jpeg")  # Make sure the path is correct
bboxes = [[50, 60, 200, 300]]  # Example bounding box (xmin, ymin, xmax, ymax)
class_labels = ['laptop']  # Labels for bounding boxes

# Apply augmentation
augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
augmented_image = augmented['image']
augmented_bboxes = augmented['bboxes']
augmented_labels = augmented['class_labels']

# Print the augmented bounding boxes
print("Augmented Bounding Boxes:", augmented_bboxes)

# Display the augmented image and bounding boxes
for bbox in augmented_bboxes:
    x_min, y_min, x_max, y_max = bbox
    # Draw bounding box on the image
    cv2.rectangle(augmented_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

# Convert the image from BGR (OpenCV format) to RGB (for matplotlib display)
augmented_image_rgb = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB)

# Display the image with bounding boxes
plt.imshow(augmented_image_rgb)
plt.axis('off')
plt.show()

# Optionally, save the augmented image
cv2.imwrite("C:\Users\saura\OneDrive\Documents\AI_project\ML\code\dataset\original_images\L1.jpeg", augmented_image)
