from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
from tensorflow.keras import layers, backend as K
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
brand_model = tf.keras.models.load_model(
    "models/efficientnetv2s_car_model.keras",
    compile=False,
    custom_objects={"SpatialAttention": SpatialAttention}
)
# Define class names
# General vehicle types for SVM

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

@tf.keras.utils.register_keras_serializable()
class SpatialAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        self.conv = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')
        super().build(input_shape)
        
    def call(self, x):
        avg_pool = K.expand_dims(K.mean(x, axis=-1), axis=-1)
        max_pool = K.expand_dims(K.max(x, axis=-1), axis=-1)
        concat = K.concatenate([avg_pool, max_pool], axis=-1)
        attention = self.conv(concat)
        return layers.multiply([x, attention])

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