import os

# Determine the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Paths relative to the project root
BASE_DIR = os.path.join(PROJECT_ROOT, "Data")
VEHICLE_DIR = os.path.join(BASE_DIR, "vehicle")
NON_VEHICLE_DIR = os.path.join(BASE_DIR, "non-vehicle")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "processed_images")
SPLIT_DIR = os.path.join(OUTPUT_DIR, "data-split")

# HOG features and labels
HOG_FEATURE_PATH = os.path.join(PROJECT_ROOT, "Data", "hog_features.npy")
HOG_LABEL_PATH = os.path.join(PROJECT_ROOT, "Data", "hog_labels.npy")


# HSV paths defined here manually
HSV_FEATURE_PATH = os.path.join(PROJECT_ROOT, "Data", "splits", "hsv_features.npy")
HSV_LABEL_PATH   = os.path.join(PROJECT_ROOT, "Data", "splits", "hsv_labels.npy")


# Ensure the output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SPLIT_DIR, exist_ok=True)