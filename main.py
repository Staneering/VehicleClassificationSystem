from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Load your model (update path as needed)
model = tf.keras.models.load_model("models/efficientnetb0_car_model.keras", compile=False)
IMG_SIZE = (224, 224)  # Use the size your model expects

# Class names (update as needed)
class_names = ["honda_accord", "peugeot", "toyota_camry", "toyota_corolla"]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    pred_class = class_names[np.argmax(preds[0])]
    confidence = float(np.max(preds[0]))
    return {"class": pred_class, "confidence": confidence}

