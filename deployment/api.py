from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import pickle
import logging
import pandas as pd
from typing import List, Dict, Any, Optional
from enum import Enum
import io
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Logging konfigürasyonu
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Task tiplerini tanımla
class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"

# FastAPI uygulamasını oluştur
app = FastAPI(
    title="Dynamic ML Pipeline API",
    description="Dinamik makine öğrenmesi pipeline'ı için REST API",
    version="1.0.0"
)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Veri setini ön işler ve kategorik değişkenleri dönüştürür"""
    try:
        # Kategorik değişkenleri tespit et ve dönüştür
        for column in df.columns:
            if df[column].dtype == 'object':  # Kategorik değişken
                le = LabelEncoder()
                df[column] = le.fit_transform(df[column].astype(str))
                logger.info(f"Kategorik değişken dönüştürüldü: {column}")
        
        return df
    except Exception as e:
        logger.error(f"Veri ön işleme sırasında hata oluştu: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Veri ön işleme hatası: {str(e)}")

def detect_task_type(df: pd.DataFrame) -> TaskType:
    """Veri setinin tipini otomatik tespit eder"""
    try:
        # Son sütunu kontrol et
        last_column = df.columns[-1]
        unique_values = df[last_column].nunique()
        
        # Eğer son sütundaki benzersiz değer sayısı az ise classification
        if unique_values < 10:
            return TaskType.CLASSIFICATION
        return TaskType.REGRESSION
    except Exception as e:
        logger.error(f"Task tipi tespit edilirken hata oluştu: {str(e)}")
        raise HTTPException(status_code=400, detail="Veri seti formatı uygun değil")

def train_model(df: pd.DataFrame, task_type: TaskType):
    """Veri setini kullanarak model eğitir"""
    try:
        # Veriyi ön işle
        df = preprocess_data(df)
        
        # Özellikler ve hedef değişkeni ayır
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        # Veriyi böl
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Model seç ve eğit
        if task_type == TaskType.CLASSIFICATION:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "task_type": "classification"
            }
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = {
                "mse": float(mean_squared_error(y_test, y_pred)),
                "r2": float(r2_score(y_test, y_pred)),
                "task_type": "regression"
            }
        
        # Modeli kaydet
        model_path = f"models/{task_type}_model.pkl"
        os.makedirs("models", exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        logger.info(f"Model eğitildi ve kaydedildi. Metrikler: {metrics}")
        return model, metrics
        
    except Exception as e:
        logger.error(f"Model eğitimi sırasında hata oluştu: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_data")
async def process_data(
    file: UploadFile = File(...),
    mode: str = Form("predict")  # "train" veya "predict"
):
    """
    CSV dosyası yükler ve işler
    
    - **file**: CSV dosyası
    - **mode**: "train" (model eğitimi) veya "predict" (tahmin)
    """
    try:
        # CSV dosyasını oku
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Task tipini tespit et
        task_type = detect_task_type(df)
        logger.info(f"Tespit edilen task tipi: {task_type}")
        
        if mode == "train":
            # Model eğit
            model, metrics = train_model(df, task_type)
            return JSONResponse(content={
                "status": "success",
                "message": "Model başarıyla eğitildi",
                "task_type": task_type,
                "metrics": metrics
            })
        else:
            # Model yükle veya eğit
            model_path = f"models/{task_type}_model.pkl"
            if not os.path.exists(model_path):
                model, _ = train_model(df, task_type)
            else:
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
            
            # Veriyi ön işle
            df = preprocess_data(df)
            
            # Tahmin yap
            X = df.iloc[:, :-1].values
            predictions = model.predict(X)
            
            # Classification için olasılıkları hesapla
            confidence = None
            if task_type == TaskType.CLASSIFICATION:
                confidence = model.predict_proba(X).max(axis=1).tolist()
            
            return JSONResponse(content={
                "status": "success",
                "task_type": task_type,
                "predictions": predictions.tolist(),
                "confidence": confidence
            })
            
    except Exception as e:
        logger.error(f"İşlem sırasında hata oluştu: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 