from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle
import logging
from typing import List, Dict, Any

# Logging konfigürasyonu
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI uygulamasını oluştur
app = FastAPI(
    title="Iris Classification API",
    description="Iris çiçeği sınıflandırma modeli için REST API",
    version="1.0.0"
)

# Giriş verisi için Pydantic modeli
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

    class Config:
        json_schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }

# Tahmin sonucu için Pydantic modeli
class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    class_name: str

# Model yükleme fonksiyonu
def load_model():
    try:
        with open("models/model.pkl", "rb") as f:
            model = pickle.load(f)
        logger.info("Model başarıyla yüklendi")
        return model
    except Exception as e:
        logger.error(f"Model yüklenirken hata oluştu: {str(e)}")
        raise HTTPException(status_code=500, detail="Model yüklenemedi")

# Sınıf isimlerini tanımla
class_names = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

# Modeli yükle
model = load_model()

@app.get("/")
async def root():
    """API'nin çalışıp çalışmadığını kontrol etmek için basit bir endpoint"""
    return {"message": "Iris Classification API çalışıyor"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: IrisInput):
    """
    Iris çiçeği özelliklerine göre sınıflandırma yapar
    
    - **sepal_length**: Çanak yaprak uzunluğu (cm)
    - **sepal_width**: Çanak yaprak genişliği (cm)
    - **petal_length**: Taç yaprak uzunluğu (cm)
    - **petal_width**: Taç yaprak genişliği (cm)
    
    Returns:
        - prediction: Tahmin edilen sınıf (0, 1, 2)
        - probability: Tahmin olasılığı
        - class_name: Sınıf ismi (setosa, versicolor, virginica)
    """
    try:
        # Giriş verisini numpy dizisine dönüştür
        features = np.array([[
            input_data.sepal_length,
            input_data.sepal_width,
            input_data.petal_length,
            input_data.petal_width
        ]])
        
        # Tahmin yap
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        probability = float(probabilities[prediction])
        
        # Sonucu hazırla
        result = {
            "prediction": int(prediction),
            "probability": probability,
            "class_name": class_names[prediction]
        }
        
        logger.info(f"Tahmin başarılı: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Tahmin yapılırken hata oluştu: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 