from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import tensorflow as tf
from tensorflow.keras import layers, backend as K
from PIL import Image
import numpy as np
import io
import joblib
from skimage.feature import hog
import base64


import os

print("Current working directory:", os.getcwd())
print("Files in cwd:", os.listdir("."))
print("Files in models folder:", os.listdir("models") if os.path.exists("models") else "No models dir")


# ==========================
# Custom Layer (still here if you ever load .keras)
# ==========================
@tf.keras.utils.register_keras_serializable()
class SpatialAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.conv = layers.Conv2D(1, kernel_size=7, padding="same", activation="sigmoid")
        super().build(input_shape)

    def call(self, x):
        avg_pool = K.expand_dims(K.mean(x, axis=-1), axis=-1)
        max_pool = K.expand_dims(K.max(x, axis=-1), axis=-1)
        concat = K.concatenate([avg_pool, max_pool], axis=-1)
        attention = self.conv(concat)
        return layers.multiply([x, attention])


# ==========================
# Globals
# ==========================
filter_interpreter = None
filter_input_details = None
filter_output_details = None

brand_interpreter = None
brand_input_details = None
brand_output_details = None

general_model = None
hog_scaler = None

general_classes = ["non vehicle", "three wheel", "motorcycle", "truck", "cars"]
brand_classes = ["honda_accord", "peugeot", "toyota_camry", "toyota_corolla"]

IMG_SIZE_filter = (224, 224)
IMG_SIZE = (384, 384)


# ==========================
# Lifespan (startup/shutdown)
# ==========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global filter_interpreter, filter_input_details, filter_output_details
    global brand_interpreter, brand_input_details, brand_output_details
    global general_model, hog_scaler

    # Load TFLite models
    filter_interpreter = tf.lite.Interpreter(
        model_path="vehicle_filter_efficientnetv2.tflite"
    )
    filter_interpreter.allocate_tensors()
    filter_input_details = filter_interpreter.get_input_details()
    filter_output_details = filter_interpreter.get_output_details()

    brand_interpreter = tf.lite.Interpreter(
        model_path="efficientnetv2s_car_model.tflite"
    )
    brand_interpreter.allocate_tensors()
    brand_input_details = brand_interpreter.get_input_details()
    brand_output_details = brand_interpreter.get_output_details()

    # Load sklearn models
    general_model = joblib.load("vehicle_svm_model.pkl")
    hog_scaler = joblib.load("hog_scaler.pkl")

    yield  # <-- app runs here

    # Cleanup
    filter_interpreter = None
    brand_interpreter = None
    general_model = None
    hog_scaler = None


# ==========================
# App + Templates
# ==========================
app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")


# ==========================
# Helpers
# ==========================
def preprocess_image_for_cnn(contents, IMG_SIZE=IMG_SIZE):
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array


def preprocess_image_for_svm(contents):
    image = Image.open(io.BytesIO(contents)).convert("L")
    image = image.resize((128, 128))
    img_array = np.array(image)
    feats = hog(
        img_array,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    )
    feats_scaled = hog_scaler.transform([feats])
    return feats_scaled


def run_tflite_inference(interpreter, input_details, output_details, input_data):
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])
    return output_data


# ==========================
# Routes
# ==========================
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    # Convert image to base64 for frontend
    image_base64 = base64.b64encode(contents).decode("utf-8")
    image_data_url = f"data:{file.content_type};base64,{image_base64}"

    # Stage 1: Vehicle vs Non Vehicle
    filter_input = preprocess_image_for_cnn(contents, IMG_SIZE_filter)
    filter_pred = run_tflite_inference(
        filter_interpreter, filter_input_details, filter_output_details, filter_input
    )[0]
    filter_class = general_classes[int(np.argmax(filter_pred))]

    if filter_class == "non vehicle":
        return {"filter": filter_class, "image": image_data_url}

    # Stage 2: General vehicle type
    svm_input = preprocess_image_for_svm(contents)
    general_label = general_model.predict(svm_input)[0]
    confidence = float(max(general_model.decision_function(svm_input)[0]))

    result = {
        "type": general_label,
        "image": image_data_url,
        "type_confidence": confidence,
    }

    # Stage 3: Brand classification if it's a car
    if general_label == "cars":
        cnn_input = preprocess_image_for_cnn(contents)
        brand_pred = run_tflite_inference(
            brand_interpreter, brand_input_details, brand_output_details, cnn_input
        )[0]
        brand_label = brand_classes[np.argmax(brand_pred)]
        brand_confidence = float(np.max(brand_pred))
        result["brand"] = brand_label
        result["brand_confidence"] = brand_confidence
    else:
        result["brand"] = "N/A"
        result["brand_confidence"] = 0.0

    return result
