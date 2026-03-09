import pytest
import pandas as pd
import joblib
from sklearn.metrics import f1_score
import os

MODEL_PATH = "models/xgboost/xgb_best_model.pkl"
X_TEST_PATH = "data/processed/X_test.csv"
Y_TEST_PATH = "data/processed/y_test.csv"

def test_minimum_f1_score():
    """Valida que el modelo mantenga un F1-score mínimo de 0.40."""
    if not (os.path.exists(MODEL_PATH) and os.path.exists(X_TEST_PATH)):
        pytest.skip("Archivos necesarios para el test de calidad no encontrados.")
    
    # Cargar modelo y datos
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    threshold = artifact["threshold"]
    
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).values.ravel()
    
    # Generar predicciones con threshold optimizado
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)
    
    current_f1 = f1_score(y_test, y_pred)
    
    # Definimos 0.40 como el mínimo aceptable basado en el notebook (que dio ~0.44)
    min_f1_threshold = 0.40
    
    assert current_f1 >= min_f1_threshold, f"El F1-score ({current_f1:.4f}) es inferior al mínimo ({min_f1_threshold})"