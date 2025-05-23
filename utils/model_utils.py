from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import numpy as np
import joblib
import os
from typing import Dict, Any, Union, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model mapping dictionary
MODEL_MAP = {
    'classification': {
        'logistic_regression': {
            'model': LogisticRegression,
            'params': {
                'C': 1.0,
                'max_iter': 1000,
                'random_state': 42
            }
        },
        'decision_tree': {
            'model': DecisionTreeClassifier,
            'params': {
                'max_depth': 5,
                'random_state': 42
            }
        },
        'knn': {
            'model': KNeighborsClassifier,
            'params': {
                'n_neighbors': 5
            }
        },
        'svm': {
            'model': SVC,
            'params': {
                'C': 1.0,
                'kernel': 'rbf',
                'random_state': 42
            }
        },
        'xgb': {
            'model': XGBClassifier,
            'params': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'random_state': 42
            }
        },
        'random_forest': {
            'model': RandomForestClassifier,
            'params': {
                'n_estimators': 100,
                'max_depth': 5,
                'random_state': 42
            }
        }
    },
    'regression': {
        'linear_regression': {
            'model': LinearRegression,
            'params': {}
        },
        'decision_tree': {
            'model': DecisionTreeRegressor,
            'params': {
                'max_depth': 5,
                'random_state': 42
            }
        },
        'knn': {
            'model': KNeighborsRegressor,
            'params': {
                'n_neighbors': 5
            }
        },
        'svm': {
            'model': SVR,
            'params': {
                'C': 1.0,
                'kernel': 'rbf'
            }
        },
        'xgb': {
            'model': XGBRegressor,
            'params': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'random_state': 42
            }
        },
        'random_forest': {
            'model': RandomForestRegressor,
            'params': {
                'n_estimators': 100,
                'max_depth': 5,
                'random_state': 42
            }
        }
    }
} 

def get_model(model_type: str, model_name: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Get model class and parameters from MODEL_MAP
    
    Args:
        model_type (str): Type of model ('classification' or 'regression')
        model_name (str): Name of the model
        
    Returns:
        Tuple[Any, Dict[str, Any]]: Model class and parameters
    """
    if model_type not in MODEL_MAP:
        raise ValueError(f"Invalid model type: {model_type}")
    
    if model_name not in MODEL_MAP[model_type]:
        raise ValueError(f"Invalid model name: {model_name} for type {model_type}")
    
    model_config = MODEL_MAP[model_type][model_name]
    return model_config['model'], model_config['params']

def create_model(model_type: str, model_name: str) -> Any:
    """
    Create a model instance
    
    Args:
        model_type (str): Type of model ('classification' or 'regression')
        model_name (str): Name of the model
        
    Returns:
        Any: Model instance
    """
    model_class, params = get_model(model_type, model_name)
    return model_class(**params)

def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray, model_type: str) -> Dict[str, float]:
    """
    Evaluate model performance
    
    Args:
        model: Trained model instance
        X_test: Test features
        y_test: Test labels
        model_type: Type of model ('classification' or 'regression')
        
    Returns:
        Dict[str, float]: Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    if model_type == 'classification':
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
    else:  # regression
        return {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }

def save_model(model: Any, model_type: str, model_name: str, version: str) -> str:
    """
    Save model to disk
    
    Args:
        model: Trained model instance
        model_type: Type of model ('classification' or 'regression')
        model_name: Name of the model
        version: Model version
        
    Returns:
        str: Path to saved model
    """
    save_dir = os.path.join('models', model_type, model_name)
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = os.path.join(save_dir, f'{version}.joblib')
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    return model_path

def load_model(model_type: str, model_name: str, version: str) -> Any:
    """
    Load model from disk
    
    Args:
        model_type: Type of model ('classification' or 'regression')
        model_name: Name of the model
        version: Model version
        
    Returns:
        Any: Loaded model instance
    """
    model_path = os.path.join('models', model_type, model_name, f'{version}.joblib')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = joblib.load(model_path)
    logger.info(f"Model loaded from {model_path}")
    
    return model 