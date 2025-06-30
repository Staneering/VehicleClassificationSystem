import cv2
import numpy as np
from src.model import load_model
from src.preprocessing import extract_hog_features

def slide_window(image_shape, window_size=(128, 128), stride=16):
    """
    Generate a list of (x1, y1, x2, y2) tuples for sliding windows.
    """
    H, W = image_shape[:2]
    win_h, win_w = window_size
    windows = []
    for y in range(0, H - win_h + 1, stride):
        for x in range(0, W - win_w + 1, stride):
            windows.append((x, y, x + win_w, y + win_h))
    return windows

def non_max_suppression(detections, iou_threshold=0.3):
    """
    Detections: list of (x1, y1, x2, y2, label, score)
    Returns filtered list after NMS (highest-score wins).
    """
    if not detections:
        return []

    boxes = np.array([d[:4] for d in detections])
    labels = [d[4] for d in detections]
    scores = np.array([d[5] for d in detections])

    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  # from highest score

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return [detections[idx] for idx in keep]


import joblib
import os

# Modify this path based on where your detector model and scaler are saved
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")

def load_model():
    model_path = os.path.join(MODEL_DIR, "vehicle_svm_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "hog_scaler.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Detector model not found at {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def detect_vehicles(
    image: np.ndarray,
    model,
    scaler,
    window_size=(128, 128),
    stride=16,
    threshold: float = 0.0
):
    """
    Sliding‐window + HOG + SVM → returns list of
    (x1, y1, x2, y2, label, score), already NMS‐filtered.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detections = []

    for (x1, y1, x2, y2) in slide_window(gray.shape, window_size, stride):
        patch = gray[y1:y2, x1:x2]
        if patch.shape != (window_size[1], window_size[0]):
            continue

        feat = extract_hog_features(patch)
        scaled = scaler.transform([feat])

        label = model.predict(scaled)[0]            # string label, e.g. 'cars'
        score = float(model.decision_function(scaled).ravel()[0])

        if score > threshold:
            detections.append((x1, y1, x2, y2, label, score))

    # apply NMS
    return non_max_suppression(detections, iou_threshold=0.3)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    model, scaler = load_model()

    # demo image path
    demo_path = os.path.abspath(
        os.path.join(os.getcwd(), "..", "Data", "vehicle", "cars", "car177.jpg")
    )
    img = cv2.imread(demo_path)
    if img is None:
        raise FileNotFoundError(demo_path)

    boxes = detect_vehicles(img, model, scaler, threshold=1.0)
    print(f"Detections after NMS: {len(boxes)}")

    class_names = {
        "cars": "Car",
        "trucks": "Truck",
        "motorcycle": "Motorbike",
        "threewheel": "ThreeWheel",
        "non-vehicle": "Background"
    }

    vis = img.copy()
    for (x1, y1, x2, y2, lbl, scr) in boxes:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0,255,0), 2)
        text = f"{class_names.get(lbl, lbl)} {scr*100:.1f}%"
        cv2.putText(vis, text, (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    plt.figure(figsize=(10,6))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title("Detections with Labels + Scores")
    plt.axis("off")
    plt.show()




from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def load_test_data(
    feature_path: str,
    label_path: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load test features and labels from .npy files.
    """
    X_test = np.load(feature_path)
    y_test = np.load(label_path)
    return X_test, y_test

def evaluate_classification(
    model,
    scaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
    average: str = "weighted"
) -> dict:
    """
    Scale X_test, predict, and compute metrics.
    Returns a dict with accuracy, precision, recall, f1, and confusion matrix.
    """
    X_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_scaled)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average=average, zero_division=0)
    rec  = recall_score(y_test, y_pred, average=average, zero_division=0)
    f1   = f1_score(y_test, y_pred, average=average, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "classification_report": classification_report(y_test, y_pred)
    }



import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str]
):
    """
    Plot a confusion matrix using seaborn heatmap.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
