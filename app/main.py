from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the model
model = joblib.load("model.pkl")

# Define input schema
class InputData(BaseModel):
    features: list[float]  # assuming a list of floats for input features

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Model is up and running 🚀"}

@app.post("/predict")
def predict(data: InputData):
    try:
        prediction = model.predict([data.features])
        return {"prediction": int(prediction[0])}
    except Exception as e:
        return {"error": str(e)}


