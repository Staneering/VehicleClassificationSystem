from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import joblib
import cv2
from skimage.feature import hog

app = FastAPI()

# Load SVM model and scaler (update path if needed)
general_model = joblib.load("vehicle_svm_model.pkl")
hog_scaler = joblib.load("hog_scaler.pkl")
# Load brand classifier (Keras)
brand_model = tf.keras.models.load_model("models/efficientnetb0_car_model.keras", compile=False)

general_classes = ["non vehicle", "three wheel", "motorcycle", "truck", "cars"]
brand_classes = ["honda_accord", "peugeot", "toyota_camry", "toyota_corolla"]

IMG_SIZE = (224, 224)  # For CNN

def preprocess_image_for_cnn(contents):
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def preprocess_image_for_svm(contents):
    '''
    Preprocess the image for SVM model:
    '''
    # Convert to grayscale and resize to 128x128
    image = Image.open(io.BytesIO(contents)).convert("L")
    image = image.resize((128, 128))
    img_array = np.array(image)
    # Extract HOG features
    feats = hog(
        img_array,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True
    )
    # Scale features
    feats_scaled = hog_scaler.transform([feats])
    return feats_scaled

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    # Stage 1: General vehicle type (SVM)
    svm_input = preprocess_image_for_svm(contents)
    general_label = general_model.predict(svm_input)[0]
    #general_label = general_classes[general_idx]
    result = {"type": general_label}

    # Stage 2: Brand classification if it's a car
    if general_label == "cars":
        cnn_input = preprocess_image_for_cnn(contents)
        brand_pred = brand_model.predict(cnn_input)[0]
        brand_label = brand_classes[np.argmax(brand_pred)]
        brand_confidence = float(np.max(brand_pred))
        result["brand"] = brand_label
        result["brand_confidence"] = brand_confidence

    return result