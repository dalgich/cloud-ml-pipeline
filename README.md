## Proje Zaman Çizelgesi (Güncel)

| Hafta   | Görev                                    | Durum      |
|---------|-------------------------------------------|------------|
| 1-2     | Ortam kurulumu, mimari tasarım            | TAMAMLANDI |
| 3-4     | Veri ingestion, model eğitimi, API kurulumu | SONRAKİ ADIM |
| 5-6     | Dockerization, Cloud deploy, MLflow entegrasyon | - |
| 7       | İlk prototip demo ve rapor hazırlığı      | - |

# ML Pipeline API

## API Endpoint'leri

### POST /predict
- **Amaç:** Gönderilen veriden tahmin döndürür.
- **Input:** JSON formatında veri ([feature1, feature2, ...])
- **Output:** Tahmin sonucu (JSON)

### POST /retrain (Gelişmiş aşama)
- **Amaç:** Modelin performansı düştüğünde retrain tetikler.
- **Input:** (Opsiyonel parametreler)
- **Output:** Başarılı retrain mesajı

## Performans Metrikleri

- **Model Doğruluk (Accuracy)**
- **F1-Score**
- **Recall & Precision**
- **API Yanıt Süresi (ms)**
- **CPU, Memory Kullanımı (CloudWatch)**

## İzleme

- **MLflow Tracking Server:** Model geçmişini, parametreleri, metrikleri ve versiyonları loglar.
- **CloudWatch:** API ve sistem sağlık kontrollerini sağlar.