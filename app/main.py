import pickle
import pandas as pd
import shap
import xgboost as xgb
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

app = FastAPI()

# Load model on startup
model = xgb.Booster()
model.load_model("model.json")

@app.get("/")
def root():
    return {"message": "SHAP XGBoost Service is live!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    prediction = model.predict(df)
    return {"predictions": prediction.tolist()}

@app.post("/shap")
async def shap_values(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    explainer = shap.Explainer(model)
    shap_vals = explainer(df)
    shap_summary = shap_vals.values.mean(0).tolist()
    return {
        "feature_names": df.columns.tolist(),
        "shap_mean_values": shap_summary
    }
