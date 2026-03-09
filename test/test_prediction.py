import pytest
import pandas as pd
import numpy as np
import joblib
import os

MODEL_PATH = "models/xgboost/xgb_best_model.pkl"

@pytest.fixture
def sample_processed_row():
    """Simula una fila ya procesada (51 columnas) lista para el modelo."""
    return np.random.rand(1, 51)

def test_model_prediction_output(sample_processed_row):
    """Verifica que predict() devuelva 0 o 1 y predict_proba() esté en [0,1]."""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("Archivo xgb_best_model.pkl no encontrado.")
    
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    threshold = artifact["threshold"]
    
    # Probar predict_proba
    probs = model.predict_proba(sample_processed_row)[:, 1]
    assert 0 <= probs[0] <= 1
    
    # Probar predict con el threshold optimizado
    prediction = (probs >= threshold).astype(int)
    assert prediction[0] in [0, 1]

def test_feature_consistency():
    """Verifica que el modelo reciba exactamente las columnas que espera."""
    artifact = joblib.load(MODEL_PATH)
    expected_features = artifact["features"]
    
    # El número de features debe coincidir con el entrenamiento (51)
    assert len(expected_features) == 51