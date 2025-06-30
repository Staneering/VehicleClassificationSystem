# Vehicle Brand Classification API

This project provides a FastAPI-based web API for hierarchical vehicle image classification. It first detects the general vehicle type (car, truck, motorcycle, etc.) using an SVM, and if the type is "car", it further classifies the car brand using a deep learning model.

## Features

- **Two-stage classification:**  
  1. SVM for general vehicle type  
  2. CNN (Keras) for car brand (if applicable)
- REST API endpoint for image prediction
- Ready for deployment on [Render](https://render.com/)

## Project Structure

```
.
├── data/                # Image data (ignored in git)
├── models/              # Model files (.keras, tracked with Git LFS)
├── vehicle identification/model/ # SVM and scaler (.pkl, tracked with Git LFS)
├── notebooks/           # Jupyter notebooks for EDA and training
├── src/                 # Source code for preprocessing and training
├── devmain.py           # Main FastAPI app
├── requirements.txt     # Python dependencies
├── render.yaml          # Render deployment config
├── .gitignore           # Git ignore rules
```

## API Usage

### `/predict` (POST)

Upload an image file to get predictions.

**Example (using `curl`):**
```bash
curl -X POST "https://<your-render-url>/predict" -F "file=@path_to_image.jpg"
```

**Response:**
```json
{
  "type": "cars",
  "brand": "honda_accord",
  "brand_confidence": 0.98
}
```

## Deployment

This project is ready for deployment on Render.  
The `render.yaml` file configures the build and start commands.

## Setup (Local)

1. Clone the repo:
   ```bash
   git clone https://github.com/YourUsername/VehicleBrandClassification.git
   cd VehicleBrandClassification
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Run the API locally:
   ```bash
   uvicorn devmain:app --reload
   ```

## Notes

- Large model files are tracked with Git LFS.
- Data folders are ignored in git for efficiency.
- For deployment, ensure all model files are present in the correct directories.

---
