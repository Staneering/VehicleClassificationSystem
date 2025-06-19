import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import shutil

from PIL import Image
import os
import glob

# Define correct paths for Lightning AI environment
RAW_DIR = "/teamspace/studios/this_studio/car_model_detection/data/raw"
PROCESSED_DIR = "/teamspace/studios/this_studio/car_model_detection/data/processed"
IMAGE_SIZE = (224, 224)


def convert_webp_to_jpg(root_dir):
    webp_files = glob.glob(os.path.join(root_dir, '**', '*.webp'), recursive=True)

    for webp_path in webp_files:
        try:
            img = Image.open(webp_path).convert('RGB')
            jpg_path = webp_path.rsplit('.', 1)[0] + ".jpg"
            img.save(jpg_path, 'JPEG')
            os.remove(webp_path)  # Remove the original .webp
        except Exception as e:
            print(f"❌ Error converting {webp_path}: {e}")

    print(f"✅ Converted {len(webp_files)} .webp images to .jpg")

# Example usage
convert_webp_to_jpg("/teamspace/studios/this_studio/car_model_detection/data/raw")


def split_and_save_images():
    class_names = os.listdir(RAW_DIR)
    for class_name in class_names:
        class_path = os.path.join(RAW_DIR, class_name)
        if not os.path.isdir(class_path):
            continue

        images = [os.path.join(class_path, img) for img in os.listdir(class_path)
                  if img.lower().endswith((".jpg", ".jpeg", ".png"))]

        if len(images) == 0:
            print(f"⚠️ Skipping '{class_name}' — no images found.")
            continue

        # Split into train (70%), val (20%), test (10%)
        train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=1/3, random_state=42)

        for split_name, split_imgs in zip(["train", "val", "test"], [train_imgs, val_imgs, test_imgs]):
            split_dir = os.path.join(PROCESSED_DIR, split_name, class_name)
            os.makedirs(split_dir, exist_ok=True)
            for img_path in split_imgs:
                shutil.copy(img_path, os.path.join(split_dir, os.path.basename(img_path)))

    print("✅ Images successfully split into train/val/test and copied to processed directory.")