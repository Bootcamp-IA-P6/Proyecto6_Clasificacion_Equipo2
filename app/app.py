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
LOGO_PATH = BASE_DIR / "app" / "assets" / "logo.png"

# =====================================
# Estilos personalizados
# =====================================
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.6rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
            color: #F5F7FA;
        }
        .subtitle {
            font-size: 1rem;
            color: #B8C1CC;
            margin-bottom: 1rem;
        }
        .card {
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 1.2rem;
            background: rgba(255,255,255,0.02);
            box-shadow: 0 0 0 1px rgba(255,255,255,0.02);
        }
        .result-card {
            border-radius: 18px;
            padding: 1.4rem;
            background: linear-gradient(135deg, rgba(20,30,48,0.95), rgba(36,59,85,0.9));
            border: 1px solid rgba(255,255,255,0.08);
        }
        }
        .result-title {
            font-size: 1.4rem;
            font-weight: 700;
            margin-bottom: 0.6rem;
        }
        .big-prob {
            font-size: 2.8rem;
            font-weight: 800;
            line-height: 1;
            margin-top: 0.5rem;
            margin-bottom: 0.5rem;
        }
        .small-muted {
            color: #B8C1CC;
            font-size: 0.95rem;
        }
        .risk-badge {
            display: inline-block;
            padding: 0.35rem 0.8rem;
            border-radius: 999px;
            font-size: 0.9rem;
            font-weight: 700;
            margin-top: 0.4rem;
            margin-bottom: 0.8rem;
        }
        .risk-low {
            background-color: rgba(220, 53, 69, 0.15);
            color: #ff8a9a;
            border: 1px solid rgba(220, 53, 69, 0.4);
        }
        .risk-medium {
            background-color: rgba(255, 193, 7, 0.15);
            color: #ffd666;
            border: 1px solid rgba(255, 193, 7, 0.4);
        }
        .risk-high {
            background-color: rgba(25, 135, 84, 0.15);
            color: #8ff0b3;
            border: 1px solid rgba(25, 135, 84, 0.4);
        }
        .kpi-box {
            border-radius: 14px;
            padding: 0.9rem 1rem;
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);
            margin-bottom: 0.8rem;
        }
        .kpi-label {
            font-size: 0.9rem;
            color: #B8C1CC;
            margin-bottom: 0.2rem;
        }
        .kpi-value {
            font-size: 1.2rem;
            font-weight: 700;
        }
        .stButton > button {
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.12);
            padding: 0.55rem 1rem;
            font-weight: 700;
        }
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

    model = model_obj["model"]
    threshold = float(model_obj["threshold"])
    feature_names = model_obj["features"]

    return model, threshold, feature_names, preprocessor

model, threshold, feature_names, preprocessor = load_artifacts()

# =====================================
# Funciones auxiliares
# =====================================
def classify_probability(prob):
    if prob >= 0.7:
        return "Alta probabilidad de aceptación", "risk-high"
    elif prob >= 0.4:
        return "Probabilidad media de aceptación", "risk-medium"
    else:
        return "Baja probabilidad de aceptación", "risk-low"

def build_input_df(
    age, job, marital, education, default, balance,
    housing, loan, contact, day, month, campaign,
    poutcome, es_cliente_nuevo, tuvo_contacto_previo
):
    alta_conversion = ["mar", "sep", "oct", "dec"]
    temporada_alta = 1 if month in alta_conversion else 0

    return pd.DataFrame({
        "age": [age],
        "job": [job],
        "marital": [marital],
        "education": [education],
        "default": [default],
        "balance": [balance],
        "housing": [housing],
        "loan": [loan],
        "contact": [contact],
        "day": [day],
        "month": [month],
        "campaign": [campaign],
        "poutcome": [poutcome],
        "es_cliente_nuevo": [es_cliente_nuevo],
        "tuvo_contacto_previo": [tuvo_contacto_previo],
        "temporada_alta": [temporada_alta]
    })

def predict_client(input_data: pd.DataFrame):
    input_processed = preprocessor.transform(input_data)
    input_processed_df = pd.DataFrame(input_processed, columns=feature_names)
    input_processed_df = input_processed_df[feature_names]

    probability = model.predict_proba(input_processed_df)[0][1]
    prediction = 1 if probability >= threshold else 0
    return prediction, probability

# =====================================
# Sidebar
# =====================================
with st.sidebar:
    st.header("ℹ️ Información del modelo")
    st.write("**Modelo:** XGBoost optimizado")
    st.write(f"**Threshold:** {threshold:.2f}")
    st.write("**Objetivo:** estimar si un cliente aceptará una oferta de depósito.")
    st.markdown("---")
    st.caption("Aplicación construida sobre el pipeline final del proyecto.")

# =====================================
# Estado inicial
# =====================================
default_values = {
    "age": 37,
    "balance": 500,
    "day": 11,
    "campaign": 2,
    "job": "admin.",
    "marital": "divorced",
    "education": "primary",
    "default": "no",
    "housing": "no",
    "loan": "no",
    "contact": "cellular",
    "month": "jan",
    "poutcome_label": "No hubo campaña anterior",
    "es_cliente_nuevo_label": "Sí",
    "tuvo_contacto_previo_label": "Sí",
}

example_values = {
    "age": 52,
    "balance": 3500,
    "day": 9,
    "campaign": 1,
    "job": "management",
    "marital": "married",
    "education": "tertiary",
    "default": "no",
    "housing": "no",
    "loan": "no",
    "contact": "cellular",
    "month": "oct",
    "poutcome_label": "Campaña anterior exitosa",
    "es_cliente_nuevo_label": "No",
    "tuvo_contacto_previo_label": "Sí",
}

if "form_values" not in st.session_state:
    st.session_state.form_values = default_values.copy()

# =====================================
# Encabezado con logo
# =====================================
LOGO_PATH = BASE_DIR / "assets" / "logo.png"

header_col1, header_col2 = st.columns([1, 5], gap="small")

with header_col1:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width= 160)
    else:
        st.error(f"Logo no encontrado: {LOGO_PATH}")

with header_col2:
    st.markdown(
        '<div class="main-title">Predicción de Suscripción de Depósito Bancario</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">Estima si un cliente aceptará una oferta bancaria utilizando el modelo final XGBoost optimizado.</div>',
        unsafe_allow_html=True
    )

st.markdown("---")
# =====================================
# Botón de ejemplo
# =====================================
top_btn_col1, top_btn_col2, top_btn_col3 = st.columns([1.2, 1.2, 6])

with top_btn_col1:
    if st.button("🧪 Cargar ejemplo"):
        st.session_state.form_values = example_values.copy()

with top_btn_col2:
    if st.button("🔄 Restaurar valores"):
        st.session_state.form_values = default_values.copy()

# =====================================
# Mapeos
# =====================================
poutcome_map = {
    "No hubo campaña anterior": "unknown",
    "Campaña anterior fallida": "failure",
    "Campaña anterior exitosa": "success",
    "Otro resultado": "other"
}

# =====================================
# Layout principal
# =====================================
left_col, right_col = st.columns([1.35, 1], gap="large")

with left_col:
    st.subheader("📋 Datos del cliente")

    job_map = {
    "Administrativo": "admin.",
    "Obrero": "blue-collar",
    "Emprendedor": "entrepreneur",
    "Empleado doméstico": "housemaid",
    "Gerencia / directivo": "management",
    "Jubilado": "retired",
    "Autónomo": "self-employed",
    "Servicios": "services",
    "Estudiante": "student",
    "Técnico": "technician",
    "Desempleado": "unemployed",
    "Desconocido": "unknown"
    }
    job_inverse_map = {v: k for k, v in job_map.items()}

    marital_map = {
    "Divorciado": "divorced",
    "Casado": "married",
    "Soltero": "single"
    }
    marital_inverse_map = {v: k for k, v in marital_map.items()}
    
    education_map = {
    "Primaria": "primary",
    "Secundaria": "secondary",
    "Universitaria": "tertiary",
    "Desconocido": "unknown"
    }
    education_inverse_map = {v: k for k, v in education_map.items()}

    yes_no_map = {
    "No": "no",
    "Sí": "yes"
    }

    yes_no_inverse_map = {v: k for k, v in yes_no_map.items()}

    with st.form("prediction_form"):
        form_col1, form_col2 = st.columns(2)

        with form_col1:
            age = st.slider("Edad", 18, 95, st.session_state.form_values["age"])
            balance = st.number_input("Saldo de la cuenta", value=st.session_state.form_values["balance"], step=100)
            day = st.slider("Día del mes", 1, 31, st.session_state.form_values["day"])
            campaign = st.slider("Veces que fue contactado en esta campaña", 1, 20, st.session_state.form_values["campaign"])

            job_label = st.selectbox(
                "Profesión",
                list(job_map.keys())
            )


            marital_label = st.selectbox(
                "Estado civil",
                options=list(marital_map.keys()),
                index=list(marital_map.keys()).index(
                marital_inverse_map[st.session_state.form_values["marital"]]
                )
            )

            education_label = st.selectbox(
                "Nivel educativo",
                options=list(education_map.keys()),
                index=list(education_map.keys()).index(
                education_inverse_map[st.session_state.form_values["education"]]
                )
            )

        with form_col2:
            default_label = st.selectbox(
                "¿Ha dejado de pagar algún crédito?",
                options=list(yes_no_map.keys()),
                index=list(yes_no_map.keys()).index(
                yes_no_inverse_map[st.session_state.form_values["default"]]
                )
            )

            housing_label = st.selectbox(
                "¿Tiene préstamo hipotecario?",
                options=list(yes_no_map.keys()),
                index=list(yes_no_map.keys()).index(
                yes_no_inverse_map[st.session_state.form_values["housing"]]
                )
            )

            loan_label = st.selectbox(
                "¿Tiene préstamo personal?",
                options=list(yes_no_map.keys()),
                index=list(yes_no_map.keys()).index(
                yes_no_inverse_map[st.session_state.form_values["loan"]]
                )
            )

            contact = "cellular"

            month = st.selectbox(
                "Mes del contacto",
                ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
                index=["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"].index(st.session_state.form_values["month"])
            )

            poutcome_label = st.selectbox(
                "Resultado de campaña anterior",
                list(poutcome_map.keys()),
                index=list(poutcome_map.keys()).index(st.session_state.form_values["poutcome_label"])
            )

            es_cliente_nuevo_label = st.selectbox("¿Es cliente nuevo?", ["Sí", "No"], index=["Sí", "No"].index(st.session_state.form_values["es_cliente_nuevo_label"]))
            tuvo_contacto_previo_label = st.selectbox("¿Tuvo contacto previo?", ["Sí", "No"], index=["Sí", "No"].index(st.session_state.form_values["tuvo_contacto_previo_label"]))

        submitted = st.form_submit_button("🔍 Predecir")

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================
# Panel derecho: resultado
# =====================================
with right_col:
    st.markdown('<div class="result-title">📊 Resultado del modelo</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-muted">Completa el formulario y pulsa <b>Predecir</b> para ver el resultado.</div>', unsafe_allow_html=True)

    if submitted:
        default = yes_no_map[default_label]
        housing = yes_no_map[housing_label]
        loan = yes_no_map[loan_label]
        job = job_map[job_label]
        marital = marital_map[marital_label]
        education = education_map[education_label]
        es_cliente_nuevo = 1 if es_cliente_nuevo_label == "Sí" else 0
        tuvo_contacto_previo = 1 if tuvo_contacto_previo_label == "Sí" else 0
        poutcome = poutcome_map[poutcome_label]

        input_data = build_input_df(
            age, job, marital, education, default, balance,
            housing, loan, contact, day, month, campaign,
            poutcome, es_cliente_nuevo, tuvo_contacto_previo
        )

        try:
            prediction, probability = predict_client(input_data)
            risk_text, risk_class = classify_probability(probability)

            if prediction == 1:
                st.success("✅ El cliente probablemente aceptará el depósito.")
            else:
                st.error("❌ El cliente probablemente no aceptará el depósito.")

            st.markdown(f'<div class="big-prob">{probability:.2%}</div>', unsafe_allow_html=True)
            st.markdown('<div class="small-muted">Probabilidad estimada de suscripción</div>', unsafe_allow_html=True)

            st.progress(float(probability))

            st.markdown(
                f'<div class="risk-badge {risk_class}">{risk_text}</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="kpi-box">
                    <div class="kpi-label">Threshold del modelo</div>
                    <div class="kpi-value">{threshold:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="kpi-box">
                    <div class="kpi-label">Mes de alta conversión</div>
                    <div class="kpi-value">{"Sí" if month in ["mar", "sep", "oct", "dec"] else "No"}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            with st.expander("Ver resumen de la solicitud"):
                resumen = pd.DataFrame({
                    "Campo": [
                        "Edad", "Saldo de la cuenta", "Día del mes", "Veces contactado en campaña",
                        "Profesión", "Estado civil", "Nivel educativo",
                        "¿Ha dejado de pagar algún crédito?", "¿Tiene préstamo hipotecario?", "¿Tiene préstamo personal?",
                        "Medio de contacto", "Mes", "Resultado de campaña anterior",
                        "¿Es cliente nuevo?", "¿Tuvo contacto previo?"
                    ],
                    "Valor": [
                        age, balance, day, campaign,
                        job_label, marital_label, education_label,
                        default_label, housing_label, loan_label,
                        "Celular", month, poutcome_label,
                        es_cliente_nuevo_label, tuvo_contacto_previo_label
                    ]
                })
                st.dataframe(resumen, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Error al realizar la predicción: {e}")

    st.markdown("</div>", unsafe_allow_html=True)