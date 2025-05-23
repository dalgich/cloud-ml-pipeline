from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pydantic import BaseModel
import numpy as np
import joblib
import logging
import pandas as pd
from typing import List, Dict, Any, Optional
from enum import Enum
import io
import os
import sys
from datetime import datetime

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from utils.model_utils import MODEL_MAP, create_model, evaluate_model, save_model, load_model

# Logging konfigürasyonu
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Task tiplerini tanımla
class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"

# Model tiplerini tanımla
class ModelType(str, Enum):
    LOGISTIC_REGRESSION = "logistic_regression"
    LINEAR_REGRESSION = "linear_regression"
    DECISION_TREE = "decision_tree"
    KNN = "knn"
    SVM = "svm"
    XGB = "xgb"
    RANDOM_FOREST = "random_forest"

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

def find_best_model(df: pd.DataFrame, task_type: TaskType) -> Dict[str, Any]:
    """Tüm modelleri test eder ve en iyi performans gösteren modeli seçer"""
    try:
        # Veriyi ön işle
        df = preprocess_data(df)
        
        # Özellikler ve hedef değişkeni ayır
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        # Veriyi böl
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        best_model_name = None
        best_metrics = None
        best_score = float('-inf')
        all_metrics = {}
        
        # Her model için
        for model_name in ModelType:
            try:
                logger.info(f"Testing model: {model_name}")
                
                # Model oluştur ve eğit
                model = create_model(task_type, model_name)
                model.fit(X_train, y_train)
                
                # Modeli değerlendir
                metrics = evaluate_model(model, X_test, y_test, task_type)
                all_metrics[model_name] = metrics
                
                # En iyi modeli güncelle
                current_score = metrics.get('accuracy', metrics.get('r2_score', 0))
                if current_score > best_score:
                    best_score = current_score
                    best_model_name = model_name
                    best_metrics = metrics
                
                logger.info(f"Model {model_name} metrics: {metrics}")
                
            except Exception as e:
                logger.error(f"Model {model_name} test edilirken hata oluştu: {str(e)}")
                continue
        
        if best_model_name is None:
            raise HTTPException(status_code=500, detail="Hiçbir model başarıyla test edilemedi")
        
        # En iyi modeli kaydet
        best_model = create_model(task_type, best_model_name)
        best_model.fit(X_train, y_train)
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = save_model(best_model, task_type, best_model_name, version)
        
        return {
            "best_model": best_model_name,
            "best_metrics": best_metrics,
            "all_metrics": all_metrics,
            "model_path": model_path,
            "version": version,
            "model": best_model
        }
        
    except Exception as e:
        logger.error(f"Model seçimi sırasında hata oluştu: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# FastAPI uygulamasını oluştur
app = FastAPI(
    title="Dynamic ML Pipeline API",
    description="Dinamik makine öğrenmesi pipeline'ı için REST API",
    version="1.0.0"
)

@app.post("/process_data")
async def process_data(
    file: UploadFile = File(...),
    mode: str = Form("predict")
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
        
        # En iyi modeli bul
        result = find_best_model(df, task_type)
        
        if mode == "train":
            return JSONResponse(content={
                "status": "success",
                "message": "En iyi model seçildi ve eğitildi",
                "task_type": task_type,
                "best_model": result["best_model"],
                "version": result["version"],
                "best_metrics": result["best_metrics"],
                "all_metrics": result["all_metrics"]
            })
        else:
            # Tahmin yap
            X = df.iloc[:, :-1].values
            predictions = result["model"].predict(X)
            
            # Classification için olasılıkları hesapla
            confidence = None
            if task_type == TaskType.CLASSIFICATION and hasattr(result["model"], "predict_proba"):
                confidence = result["model"].predict_proba(X).max(axis=1).tolist()
            
            return JSONResponse(content={
                "status": "success",
                "task_type": task_type,
                "model_name": result["best_model"],
                "version": result["version"],
                "predictions": predictions.tolist(),
                "confidence": confidence,
                "model_metrics": result["best_metrics"]
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
    uvicorn.run(app, host="0.0.0.0", port=8003) 