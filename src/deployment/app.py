from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import joblib
import logging
import pandas as pd
from typing import List, Dict, Any, Optional
from enum import Enum
import io
import os
from datetime import datetime
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

def train_model(
    df: pd.DataFrame,
    task_type: TaskType,
    model_name: str
) -> Dict[str, Any]:
    """Veri setini kullanarak model eğitir"""
    try:
        # Veriyi ön işle
        df = preprocess_data(df)
        
        # Özellikler ve hedef değişkeni ayır
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        # Veriyi böl
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Model oluştur ve eğit
        model = create_model(task_type, model_name)
        model.fit(X_train, y_train)
        
        # Modeli değerlendir
        metrics = evaluate_model(model, X_test, y_test, task_type)
        
        # Modeli kaydet
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = save_model(model, task_type, model_name, version)
        
        logger.info(f"Model eğitildi ve kaydedildi. Metrikler: {metrics}")
        return {
            "model_path": model_path,
            "metrics": metrics,
            "version": version
        }
        
    except Exception as e:
        logger.error(f"Model eğitimi sırasında hata oluştu: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_data")
async def process_data(
    file: UploadFile = File(...),
    mode: str = Form("predict"),
    model_name: str = Form(None)
):
    """
    CSV dosyası yükler ve işler
    
    - **file**: CSV dosyası
    - **mode**: "train" (model eğitimi) veya "predict" (tahmin)
    - **model_name**: Kullanılacak model adı (opsiyonel)
    """
    try:
        # CSV dosyasını oku
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Task tipini tespit et
        task_type = detect_task_type(df)
        logger.info(f"Tespit edilen task tipi: {task_type}")
        
        # Model adını belirle
        if model_name is None:
            model_name = "random_forest"  # Varsayılan model
        
        if mode == "train":
            # Model eğit
            result = train_model(df, task_type, model_name)
            return JSONResponse(content={
                "status": "success",
                "message": "Model başarıyla eğitildi",
                "task_type": task_type,
                "model_name": model_name,
                "version": result["version"],
                "metrics": result["metrics"]
            })
        else:
            # En son modeli bul
            model_dir = os.path.join("models", task_type, model_name)
            if not os.path.exists(model_dir):
                # Model yoksa eğit
                result = train_model(df, task_type, model_name)
                version = result["version"]
            else:
                # En son versiyonu bul
                versions = [f.split(".")[0] for f in os.listdir(model_dir) if f.endswith(".joblib")]
                version = max(versions)
            
            # Modeli yükle
            model = load_model(task_type, model_name, version)
            
            # Veriyi ön işle
            df = preprocess_data(df)
            
            # Tahmin yap
            X = df.iloc[:, :-1].values
            predictions = model.predict(X)
            
            # Classification için olasılıkları hesapla
            confidence = None
            if task_type == TaskType.CLASSIFICATION and hasattr(model, "predict_proba"):
                confidence = model.predict_proba(X).max(axis=1).tolist()
            
            return JSONResponse(content={
                "status": "success",
                "task_type": task_type,
                "model_name": model_name,
                "version": version,
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