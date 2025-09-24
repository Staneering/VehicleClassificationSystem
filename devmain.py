"""from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
from tensorflow.keras import layers, backend as K
from PIL import Image
import numpy as np
import io
import joblib
import cv2
from skimage.feature import hog
import base64
from tensorflow.keras.models import load_model


app = FastAPI()

templates = Jinja2Templates(directory="templates")

@tf.keras.utils.register_keras_serializable()
class SpatialAttention(layers.Layer):
    def _init_(self, **kwargs):
        super()._init_(**kwargs)
        
    def build(self, input_shape):
        self.conv = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')
        super().build(input_shape)
        
    def call(self, x):
        avg_pool = K.expand_dims(K.mean(x, axis=-1), axis=-1)
        max_pool = K.expand_dims(K.max(x, axis=-1), axis=-1)
        concat = K.concatenate([avg_pool, max_pool], axis=-1)
        attention = self.conv(concat)
        return layers.multiply([x, attention])

from fastapi import FastAPI

app = FastAPI()



# Load filter model (vehicle vs non vehicle)
filter_model = load_model("models/vehicle_filter_efficientnetv2.keras")  # Update filename as needed


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

IMG_SIZE_filter = (224, 224)  # For CNN
IMG_SIZE = (384, 384)  # For CNN, adjust as needed

def preprocess_image_for_cnn(contents, IMG_SIZE=IMG_SIZE):
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
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
        feature_vector=True
    )
    feats_scaled = hog_scaler.transform([feats])
    return feats_scaled


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()


    # Convert image to base64 for frontend display
    image_base64 = base64.b64encode(contents).decode("utf-8")
    image_data_url = f"data:{file.content_type};base64,{image_base64}"

    # Stage 1: Vehicle vs Non Vehicle
    filter_input = preprocess_image_for_cnn(contents,IMG_SIZE_filter) 
    filter_label = filter_model.predict(filter_input)[0]
    result = {"filter": filter_label}

    if filter_label == "non vehicle":
        return result
#implement type confidence showing on index.html
    # Stage 2: General vehicle type
    svm_input = preprocess_image_for_svm(contents)
    general_label = general_model.predict(svm_input)[0]
    confidence = max(general_model.decision_function(svm_input)[0])
    result = {"type": general_label, "image": image_data_url, "type_confidence": confidence} 

    #result = {"type": general_label, "image": image_data_url}
    

    # Stage 3: Brand classification if it's a car
    if general_label == "cars":
        cnn_input = preprocess_image_for_cnn(contents)
        brand_pred = brand_model.predict(cnn_input)[0]
        brand_label = brand_classes[np.argmax(brand_pred)]
        brand_confidence = float(np.max(brand_pred))
        result["brand"] = brand_label
        result["brand_confidence"] = brand_confidence
    else:
        result["brand"] = "N/A"
        result["brand_confidence"] = 0.0
    return result"""


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
from tensorflow.keras.models import load_model


# ==========================
# Custom Layer
# ==========================
@tf.keras.utils.register_keras_serializable()
class SpatialAttention(layers.Layer):
    def __init__(self, **kwargs):   # <-- fix: use __init__ not _init_
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
# Globals (populated in lifespan)
# ==========================
filter_model = None
general_model = None
hog_scaler = None
brand_model = None

general_classes = ["non vehicle", "three wheel", "motorcycle", "truck", "cars"]
brand_classes = ["honda_accord", "peugeot", "toyota_camry", "toyota_corolla"]

IMG_SIZE_filter = (224, 224)
IMG_SIZE = (384, 384)


# ==========================
# Lifespan (startup/shutdown)
# ==========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio

    # Start binding immediately
    await asyncio.sleep(0)  

    # Then load heavy models in the background
    global filter_model, general_model, hog_scaler, brand_model
    filter_model = load_model("models/vehicle_filter_efficientnetv2.keras")
    general_model = joblib.load("vehicle_svm_model.pkl")
    hog_scaler = joblib.load("hog_scaler.pkl")
    brand_model = tf.keras.models.load_model(
        "models/efficientnetv2s_car_model.keras",
        compile=False,
        custom_objects={"SpatialAttention": SpatialAttention},
    )
    yield

    # Cleanup on shutdown
    filter_model = None
    general_model = None
    hog_scaler = None
    brand_model = None


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
    img_array = np.expand_dims(img_array, axis=0)
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
    filter_pred = filter_model.predict(filter_input)[0]
    filter_class = general_classes[int(np.argmax(filter_pred))]  # fix: map to label

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
        brand_pred = brand_model.predict(cnn_input)[0]
        brand_label = brand_classes[np.argmax(brand_pred)]
        brand_confidence = float(np.max(brand_pred))
        result["brand"] = brand_label
        result["brand_confidence"] = brand_confidence
    else:
        result["brand"] = "N/A"
        result["brand_confidence"] = 0.0

    return result

if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 8000))  # default to 8000 for local dev
    uvicorn.run("devmain:app", host="0.0.0.0", port=port, reload=False)

