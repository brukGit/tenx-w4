from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import joblib
import pandas as pd
import sys
import os
from typing import List
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessor import Preprocessor

app = FastAPI(title="Rossmann Sales Forecasting API", version="1.0.0")

# Initialize the preprocessor
preprocessor = Preprocessor()

class StoreData(BaseModel):
    Store: int
    DayOfWeek: int
    Date: str
    Open: int
    Promo: int
    StateHoliday: str
    SchoolHoliday: int
    StoreType: str
    Assortment: str
    CompetitionDistance: float
    CompetitionOpenSinceMonth: float
    CompetitionOpenSinceYear: float
    Promo2: int
    Promo2SinceWeek: float
    Promo2SinceYear: float
    PromoInterval: str

class PredictionResponse(BaseModel):
    store_id: int
    predicted_sales: float
    prediction_date: str

def load_model():
    model_dir = '../models'
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
    if not model_files:
        raise HTTPException(status_code=500, detail="No model file found")
    latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(model_dir, x)))
    model_path = os.path.join(model_dir, latest_model)
    return joblib.load(model_path)

@app.on_event("startup")
async def startup_event():
    app.state.model = load_model()

@app.get("/")
async def root():
    return {"message": "Welcome to the Rossmann Sales Forecasting API"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_sales(store_data: StoreData):
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([store_data.dict()])
        
        # Preprocess the input data
        processed_data = preprocessor.preprocess(input_df)
        
        # Make prediction
        prediction = app.state.model.predict(processed_data)
        
        # Inverse transform the scaled prediction
        sales_prediction = preprocessor.inverse_transform_sales(prediction)[0]
        
        return PredictionResponse(
            store_id=store_data.Store,
            predicted_sales=sales_prediction,
            prediction_date=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch", response_model=List[PredictionResponse])
async def predict_sales_batch(store_data_list: List[StoreData]):
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([data.dict() for data in store_data_list])
        
        # Preprocess the input data
        processed_data = preprocessor.preprocess(input_df)
        
        # Make predictions
        predictions = app.state.model.predict(processed_data)
        
        # Inverse transform the scaled predictions
        sales_predictions = preprocessor.inverse_transform_sales(predictions)
        
        return [
            PredictionResponse(
                store_id=store_data.Store,
                predicted_sales=sales_prediction,
                prediction_date=datetime.now().isoformat()
            )
            for store_data, sales_prediction in zip(store_data_list, sales_predictions)
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model_info")
async def get_model_info():
    model_dir = '../models'
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
    if not model_files:
        raise HTTPException(status_code=500, detail="No model file found")
    latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(model_dir, x)))
    model_path = os.path.join(model_dir, latest_model)
    model_stats = os.stat(model_path)
    return {
        "model_name": latest_model,
        "last_modified": datetime.fromtimestamp(model_stats.st_mtime).isoformat(),
        "size_bytes": model_stats.st_size
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)