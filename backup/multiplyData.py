import albumentations as A
import cv2
import os
from tqdm import tqdm

# Define the augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
    A.GaussianBlur(p=0.3),
])

# Paths
input_dir = "dataset/original_images"  # Folder with original images
output_dir = "dataset/augmented_images"  # Folder to save augmented images
os.makedirs(output_dir, exist_ok=True)

# Multiply dataset
n_augmentations = 5  # Number of augmented images per original image

# Iterate through images
for filename in tqdm(os.listdir(input_dir)):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(input_dir, filename)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        
        # Apply augmentations
        for i in range(n_augmentations):
            augmented = transform(image=image)["image"]
            output_filename = f"{os.path.splitext(filename)[0]}_aug_{i}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save augmented image
            augmented_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, augmented_bgr)

print(f"Augmented dataset saved to {output_dir}")
