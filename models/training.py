# models/training.py

import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import mlflow

def train_and_save_model(X, y, model_path="iris_model.pkl"):
    """
    Decision Tree modeli eğitir, accuracy skorunu hesaplar,
    modeli hem pickle ile kaydeder hem de MLflow'a loglar.

    Returns:
        model: Eğitilmiş sklearn model nesnesi
        accuracy: Başarı skoru
    """

    mlflow.set_experiment("ml_pipeline_experiment")

    with mlflow.start_run():
        model = DecisionTreeClassifier()
        model.fit(X, y)

        preds = model.predict(X)
        accuracy = accuracy_score(y, preds)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "decision_tree_model")

        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        print(f"[INFO] Model accuracy: {accuracy:.2f}")
        print(f"[INFO] Model kaydedildi: {model_path}")

        return model, accuracy
