import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from data.ingestion import load_data, get_data_info
from data.preprocessing import preprocess_data, handle_missing_values
from models.training import train_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 1. Load Iris dataset
    logger.info("Loading Iris dataset...")
    iris = load_iris()
    data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    data['target'] = iris.target
    
    # Save the data to CSV for demonstration
    data.to_csv('iris_data.csv', index=False)
    logger.info("Data saved to iris_data.csv")
    
    # 2. Load data using our ingestion module
    df = load_data('iris_data.csv')
    get_data_info(df)
    
    # 3. Preprocess the data
    logger.info("\nPreprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(
        data=df,
        target_column='target',
        test_size=0.2,
        random_state=42
    )
    
    # 4. Train the model
    logger.info("\nTraining model...")
    model, metrics = train_model(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_params={
            'n_estimators': 100,
            'max_depth': 5,
            'random_state': 42
        }
    )
    
    # Print results
    logger.info("\nModel Performance:")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info("\nClassification Report:")
    logger.info(metrics['classification_report'])

if __name__ == "__main__":
    main() 