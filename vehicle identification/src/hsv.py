import os
import cv2
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
from skimage.feature import local_binary_pattern

# --- Feature Extraction Functions --- #

def extract_hsv_histogram(image: np.ndarray, bins: List[int] = [8, 8, 8]) -> np.ndarray:
    """Extract a flattened and normalized HSV histogram from an image."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv_image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256]
    )
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extract_normalized_hsv_histogram(image: np.ndarray, bins: List[int] = [8, 8, 8]) -> np.ndarray:
    """Normalized HSV histogram so its sum = 1 (scale-invariant)."""
    hist = extract_hsv_histogram(image, bins)
    hist /= (hist.sum() + 1e-6)
    return hist

def extract_lbp_histogram(image: np.ndarray, P: int = 8, R: float = 1.0, bins: int = 256) -> np.ndarray:
    """Extract a normalized LBP histogram from a grayscale image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P, R, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

# --- Folder Processing Functions --- #

def load_images_from_folder(folder_path: str, label: int, bins=(8, 8, 8)) -> Tuple[List[np.ndarray], List[int]]:
    """Load images, extract HSV histograms, and assign labels."""
    features, labels = [], []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(root, filename)
                image = cv2.imread(path)
                if image is not None:
                    hist = extract_normalized_hsv_histogram(image, bins)
                    features.append(hist)
                    labels.append(label)
    return features, labels

def extract_batch_features(data_root: str = "../Data", bins=(8, 8, 8)) -> Tuple[np.ndarray, np.ndarray]:
    """Extract HSV histograms from vehicle and non-vehicle images."""
    vehicle_dir     = os.path.join(data_root, "vehicle")
    non_vehicle_dir = os.path.join(data_root, "non-vehicle")

    vf, vl = load_images_from_folder(vehicle_dir,     label=1, bins=bins)
    nf, nl = load_images_from_folder(non_vehicle_dir, label=0, bins=bins)

    all_features = np.array(vf + nf)
    all_labels   = np.array(vl + nl)
    return all_features, all_labels

def extract_batch_hsv_lbp(data_root: str = "../Data") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract HSV and LBP histograms separately, then combine."""
    vehicle_dir     = os.path.join(data_root, "vehicle")
    non_vehicle_dir = os.path.join(data_root, "non-vehicle")

    X_hsv, X_lbp, y = [], [], []

    def process_folder(folder, label):
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    path = os.path.join(root, f)
                    image = cv2.imread(path)
                    if image is not None:
                        X_hsv.append(extract_normalized_hsv_histogram(image))
                        X_lbp.append(extract_lbp_histogram(image))
                        y.append(label)

    process_folder(vehicle_dir, label=1)
    process_folder(non_vehicle_dir, label=0)

    return np.array(X_hsv), np.array(X_lbp), np.array(y)

# --- Main Execution --- #

if __name__ == "__main__":
    os.makedirs("../Data/splits", exist_ok=True)

    # Option 1: Only HSV features
    feats, labels = extract_batch_features("../Data")
    np.save("../Data/splits/hsv_features.npy", feats)
    np.save("../Data/splits/hsv_labels.npy", labels)
    print(f"[HSV] Saved {len(feats)} feature vectors.")

    # Option 2: Combined HSV + LBP features
    X_hsv, X_lbp, y = extract_batch_hsv_lbp("../Data")
    X_combined = np.hstack([X_hsv, X_lbp])
    np.save("../Data/splits/combined_features.npy", X_combined)
    np.save("../Data/splits/labels.npy", y)
    print(f"[Combined] Saved {X_combined.shape[0]} combined feature vectors.")