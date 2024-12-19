import os
import cv2
import albumentations as A
from albumentations.augmentations import transforms
from albumentations.core.composition import OneOf
from tqdm import tqdm

# Paths
images_dir = "datasets/images/val"  # Path to original images
labels_dir = "datasets/labels/val"  # Path to corresponding YOLO labels
augmented_images_dir = "datasets/augmented/images/val"  # Path for augmented images
augmented_labels_dir = "datasets/augmented/labels/val"  # Path for augmented labels
print(os.listdir(images_dir))
print(os.listdir(labels_dir))
# Create directories for augmented data
os.makedirs(augmented_images_dir, exist_ok=True)
os.makedirs(augmented_labels_dir, exist_ok=True)

# Augmentation Pipeline
augmentations = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
    A.GaussNoise(p=0.2),
])

def augment_image(image_path, label_path, output_img_path, output_label_path, count=5):
    """
    Apply augmentations to an image and its label.
    
    Args:
        image_path (str): Path to the image.
        label_path (str): Path to the YOLO label.
        output_img_path (str): Output path for augmented image.
        output_label_path (str): Output path for augmented label.
        count (int): Number of augmentations to generate.
    """
    image = cv2.imread(image_path)
    with open(label_path, "r") as file:
        annotations = file.readlines()

    for i in range(count):
        augmented = augmentations(image=image)
        augmented_image = augmented["image"]

        # Save the augmented image
        aug_img_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_aug_{i}.jpg"
        aug_img_path = os.path.join(output_img_path, aug_img_name)
        cv2.imwrite(aug_img_path, augmented_image)

        # Copy the original annotations (no change needed for augmentations that don't affect bounding boxes)
        aug_label_name = f"{os.path.splitext(os.path.basename(label_path))[0]}_aug_{i}.txt"
        aug_label_path = os.path.join(output_label_path, aug_label_name)
        with open(aug_label_path, "w") as output_file:
            output_file.writelines(annotations)


if __name__ == "__main__":
    # Iterate through all images and labels
    for filename in tqdm(os.listdir(images_dir)):
        if filename.endswith(".jpg"):
            image_path = os.path.join(images_dir, filename)
            label_path = os.path.join(labels_dir, os.path.splitext(filename)[0] + ".txt")
            print(f"label_path {label_path} {os.getcwd()}")
            label_path = f"{os.getcwd()}/{label_path}"
            converted_path = label_path.replace("\\", "/")
            print(f"converted_path {converted_path}")
            # Skip if label file doesn't exist
            if not os.path.exists(converted_path):
                print(f"Label file missing for {filename}")
                continue

            # Apply augmentations
            augment_image(
                image_path=image_path,
                label_path=converted_path,
                output_img_path=augmented_images_dir,
                output_label_path=augmented_labels_dir
            )
