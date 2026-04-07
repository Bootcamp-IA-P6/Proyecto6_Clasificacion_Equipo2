import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# =====================================
# Configuración general
# =====================================
st.set_page_config(
    page_title="Predicción de Suscripción Bancaria",
    page_icon="💳",
    layout="wide"
)

# =====================================
# Rutas
# =====================================
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "xgboost" / "xgb_best_model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"
LOGO_PATH = BASE_DIR / "assets" / "logo.png"

# =====================================
# Estilos personalizados
# =====================================
st.markdown(
    """
    <style>
        .main-title { font-size: 2.6rem; font-weight: 800; margin-bottom: 0.2rem; color: #F5F7FA; }
        .subtitle { font-size: 1rem; color: #B8C1CC; margin-bottom: 1rem; }
        .result-title { font-size: 1.4rem; font-weight: 700; margin-bottom: 0.6rem; }
        .big-prob { font-size: 2.8rem; font-weight: 800; line-height: 1; margin-top: 0.5rem; margin-bottom: 0.5rem; }
        .small-muted { color: #B8C1CC; font-size: 0.95rem; }
        .risk-badge { display: inline-block; padding: 0.35rem 0.8rem; border-radius: 999px; font-size: 0.9rem; font-weight: 700; margin-top: 0.4rem; margin-bottom: 0.8rem; }
        .risk-low { background-color: rgba(220, 53, 69, 0.15); color: #ff8a9a; border: 1px solid rgba(220, 53, 69, 0.4); }
        .risk-medium { background-color: rgba(255, 193, 7, 0.15); color: #ffd666; border: 1px solid rgba(255, 193, 7, 0.4); }
        .risk-high { background-color: rgba(25, 135, 84, 0.15); color: #8ff0b3; border: 1px solid rgba(25, 135, 84, 0.4); }
        .kpi-box { border-radius: 14px; padding: 0.9rem 1rem; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); margin-bottom: 0.8rem; }
        .kpi-label { font-size: 0.9rem; color: #B8C1CC; margin-bottom: 0.2rem; }
        .kpi-value { font-size: 1.2rem; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True
)

# =====================================
# Cargar artefactos
# =====================================
@st.cache_resource
def load_artifacts():
    model_obj = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model_obj["model"], float(model_obj["threshold"]), model_obj["features"], preprocessor

model, threshold, feature_names, preprocessor = load_artifacts()

# =====================================
# default values
# =====================================
default_values = {
    "age": 35,
    "balance": 1000,
    "job": "Administrativo",
    "marital": "Casado",
    "education": "Secundaria",
    "default": False,
    "housing": True,
    "loan": False,
    "day": 15,
    "month": "Mayo",
    "campaign": 1,
    "poutcome_label": "Sin contacto anterior",
    "es_cliente_nuevo": True,
    "tuvo_contacto_previo": False,
}

example_values = {
    "age": 58,
    "balance": 5000,
    "job": "Jubilado",
    "marital": "Casado",
    "education": "Universitaria",
    "default": False,
    "housing": False,
    "loan": False,
    "day": 10,
    "month": "Octubre",
    "campaign": 1,
    "poutcome_label": "Campaña anterior exitosa",
    "es_cliente_nuevo": False,
    "tuvo_contacto_previo": True,
}

if "form_values" not in st.session_state:
    st.session_state.form_values = default_values.copy()


# =====================================
# Mapeos
# =====================================
job_map = {"Administrativo": "admin.", "Obrero": "blue-collar", "Emprendedor": "entrepreneur", "Empleado doméstico": "housemaid", "Gerencia / directivo": "management", "Jubilado": "retired", "Autónomo": "self-employed", "Servicios": "services", "Estudiante": "student", "Técnico": "technician", "Desempleado": "unemployed", "Desconocido": "unknown"}
marital_map = {"Divorciado": "divorced", "Casado": "married", "Soltero": "single"}
education_map = {"Primaria": "primary", "Secundaria": "secondary", "Universitaria": "tertiary", "Desconocido": "unknown"}
month_map = {"Enero": "jan", "Febrero": "feb", "Marzo": "mar", "Abril": "apr", "Mayo": "may", "Junio": "jun", "Julio": "jul", "Agosto": "aug", "Septiembre": "sep", "Octubre": "oct", "Noviembre": "nov", "Diciembre": "dec"}
poutcome_map = {"Sin contacto anterior": "unknown", "Campaña anterior fallida": "failure", "Campaña anterior exitosa": "success", "Otro resultado": "other"}

# =====================================
# Layout principal
# =====================================
header_col1, header_col2 = st.columns([1, 5])
with header_col1:
    if LOGO_PATH.exists(): st.image(str(LOGO_PATH), width=120)
with header_col2:
    st.markdown('<div class="main-title">Predicción de Suscripción de Depósito Bancario</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Estima si un cliente aceptará una oferta bancaria utilizando el modelo XGBoost.</div>', unsafe_allow_html=True)

st.markdown("---")


left_col, right_col = st.columns([1.35, 1], gap="large")

with left_col:
    st.subheader("📋 Datos del cliente")
    
    es_cliente_nuevo = st.checkbox("🆕 ¿Es un cliente nuevo? (Sin historial de campañas anteriores)", value=True)

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Edad", 18, 95, 35)
            balance = st.number_input("Saldo de la cuenta (€)", value=500, step=100)
            job_label = st.selectbox("Profesión", list(job_map.keys()))
            marital_label = st.radio("Estado civil", list(marital_map.keys()), horizontal=True)
            education_label = st.radio("Nivel educativo", list(education_map.keys()), horizontal=True)

        with col2:
            month_label = st.selectbox("Mes del contacto", list(month_map.keys()), index=4) # Mayo por defecto
            day = st.slider("Día del mes", 1, 31, 15)
            campaign = st.slider("Contactos en esta campaña", 1, 20, 1)
            
            st.write("**Situación Financiera (Check si es Sí):**")
            default = st.checkbox("¿Ha dejado de pagar algún crédito?")
            housing = st.checkbox("¿Tiene préstamo hipotecario?")
            loan = st.checkbox("¿Tiene préstamo personal?")
            
            if not es_cliente_nuevo:
                st.markdown("---")
                st.write("**Historial de Campañas:**")
                poutcome_label = st.selectbox("Resultado de campaña anterior", list(poutcome_map.keys()))
                tuvo_contacto_previo = st.checkbox("¿Tuvo contacto previo en otras campañas?")
            else:
                poutcome_label = "Sin contacto anterior"
                tuvo_contacto_previo = False

        submitted = st.form_submit_button("🔍 Predecir")



# =====================================
# Panel derecho: Resultado 
# =====================================
with right_col:
    st.markdown('<div class="result-title">📊 Resultado del modelo</div>', unsafe_allow_html=True)
    
    if submitted:
        input_data = pd.DataFrame({
            "age": [age],
            "job": [job_map[job_label]],
            "marital": [marital_map[marital_label]],
            "education": [education_map[education_label]],
            "default": ["yes" if default else "no"],
            "balance": [balance],
            "housing": ["yes" if housing else "no"],
            "loan": ["yes" if loan else "no"],
            "contact": ["cellular"],
            "day": [day],
            "month": [month_map[month_label]],
            "campaign": [campaign],
            "poutcome": [poutcome_map[poutcome_label]],
            "es_cliente_nuevo": [1 if es_cliente_nuevo else 0],
            "tuvo_contacto_previo": [1 if tuvo_contacto_previo else 0],
            "temporada_alta": [1 if month_map[month_label] in ["mar", "sep", "nov", "dec"] else 0]
        })

        try:
            processed_data = preprocessor.transform(input_data)
            processed_df = pd.DataFrame(processed_data, columns=feature_names)
            probability = model.predict_proba(processed_df)[0][1]
            prediction = int(probability >= threshold)

            if prediction == 1:
                st.success("✅ El cliente probablemente aceptará el depósito.")
            else:
                st.error("❌ El cliente probablemente no aceptará el depósito.")

            st.markdown(f'<div class="big-prob">{probability:.2%}</div>', unsafe_allow_html=True)
            st.markdown('<div class="small-muted">Probabilidad estimada de suscripción</div>', unsafe_allow_html=True)
            st.progress(float(probability))

            risk_text = "Alta" if probability >= 0.7 else "Media" if probability >= 0.4 else "Baja"
            risk_class = "risk-high" if probability >= 0.7 else "risk-medium" if probability >= 0.4 else "risk-low"
            st.markdown(f'<div class="risk-badge {risk_class}">Prioridad {risk_text} de aceptación</div>', unsafe_allow_html=True)

            st.markdown(f"""
                <div class="kpi-box">
                    <div class="kpi-label">Threshold del modelo</div>
                    <div class="kpi-value">{threshold:.2f}</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">Mes de alta conversión</div>
                    <div class="kpi-value">{"Sí" if month_map[month_label] in ["mar", "sep", "oct", "dec"] else "No"}</div>
                </div>
            """, unsafe_allow_html=True)

            with st.expander("🔍 Ver resumen de la solicitud"):
                resumen = pd.DataFrame({
                    "Campo": ["Edad", "Saldo", "Profesión", "Estado Civil", "Hipoteca", "Préstamo", "Mes", "Campaña", "Cliente Nuevo"],
                    "Valor": [age, f"{balance}€", job_label, marital_label, "Sí" if housing else "No", "Sí" if loan else "No", month_label, campaign, "Sí" if es_cliente_nuevo else "No"]
                })
                st.table(resumen)

        except Exception as e:
            st.error(f"Error al realizar la predicción: {e}")
    else:
        st.markdown('<div class="small-muted">Complete el formulario y pulse <b>Predecir</b> para ver el resultado.</div>', unsafe_allow_html=True)
        st.info("Esperando datos de entrada...")