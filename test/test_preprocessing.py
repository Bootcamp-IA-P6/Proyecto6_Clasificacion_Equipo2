import pytest
import pandas as pd
import numpy as np
import joblib
import os

# Cargar el preprocesador guardado
PREPROCESSOR_PATH = "models/preprocessor.pkl"

@pytest.fixture
def sample_raw_data():
    """Genera un dato de entrada raw para probar el pipeline."""
    return pd.DataFrame({
        "age": [30],
        "job": ["management"],
        "marital": ["married"],
        "education": ["tertiary"],
        "default": ["no"],
        "balance": [1500],
        "housing": ["yes"],
        "loan": ["no"],
        "contact": ["cellular"],
        "day": [15],
        "month": ["may"],
        "campaign": [1],
        "pdays": [-1],
        "previous": [0],
        "poutcome": ["unknown"]
    })

def test_feature_engineering_logic(sample_raw_data):
    """Valida que la lógica de nuevas variables sea correcta antes del pipeline."""
    df = sample_raw_data.copy()
    
    # Aplicar la misma lógica del notebook
    df["es_cliente_nuevo"] = (df["pdays"] == -1).astype(int)
    df["tuvo_contacto_previo"] = (df["previous"] > 0).astype(int)
    alta_conversion = ["mar", "sep", "oct", "dec"]
    df["temporada_alta"] = df["month"].isin(alta_conversion).astype(int)
    
    assert df["es_cliente_nuevo"].iloc[0] == 1
    assert df["tuvo_contacto_previo"].iloc[0] == 0
    assert df["temporada_alta"].iloc[0] == 0

def test_pipeline_transform_output_shape(sample_raw_data):
    """Verifica que el pipeline devuelva el número correcto de columnas (51)."""
    if not os.path.exists(PREPROCESSOR_PATH):
        pytest.skip("Archivo preprocessor.pkl no encontrado.")
    
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    
    # Preparar el dato (el pipeline espera las columnas originales tras el drop de duration, pdays y previous)
    # Nota: El pipeline en el notebook se ajustó a NUM_COLS y CAT_COLS definidos tras el feature engineering
    df_test = sample_raw_data.copy()
    df_test["es_cliente_nuevo"] = (df_test["pdays"] == -1).astype(int)
    df_test["tuvo_contacto_previo"] = (df_test["previous"] > 0).astype(int)
    df_test["temporada_alta"] = df_test["month"].isin(["mar", "sep", "oct", "dec"]).astype(int)
    
    # Eliminar las que el modelo eliminó
    df_input = df_test.drop(columns=["pdays", "previous"])
    
    X_transformed = preprocessor.transform(df_input)
    
    # Según X_test.csv proporcionado, debe tener 51 columnas
    assert X_transformed.shape[1] == 51
    assert isinstance(X_transformed, np.ndarray)