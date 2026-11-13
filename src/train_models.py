import os
import time
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data_processing import load_and_prepare_data

MODELS_DIR = 'models'

def train_and_evaluate_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    _, X_train, X_test, y_train, y_test, _ = load_and_prepare_data()
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'HistGradientBoosting': HistGradientBoostingRegressor(random_state=42)
    }
    
    results = {}
    best_model_name = None
    best_rmse = float('inf')
    
    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = round(time.time() - start_time, 2)
        
        predictions = model.predict(X_test)
        
        mae = round(mean_absolute_error(y_test, predictions), 2)
        rmse = round(np.sqrt(mean_squared_error(y_test, predictions)), 2)
        r2 = round(r2_score(y_test, predictions), 2)
        
        results[name] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'time': train_time
        }
        
        joblib.dump(model, os.path.join(MODELS_DIR, f"{name.replace(' ', '_').lower()}.joblib"))
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            
    return results, best_model_name
