# Informe Técnico — Clasificación de Clientes Bancarios
### Predicción de Suscripción a Depósito a Plazo Fijo
*Proyecto 6 — Bootcamp IA | 2026*

---

## 1. Introducción y Contexto de Negocio

El objetivo de este proyecto es desarrollar un modelo de clasificación capaz de predecir si un cliente aceptará una oferta de depósito bancario basándose en información demográfica, financiera y de campañas de marketing anteriores.

El dataset utilizado corresponde a campañas de marketing directo de una institución bancaria y presenta un problema de clasificación binaria con **desbalance de clases**, donde aproximadamente el **12%** de los clientes aceptan la oferta.

El fin último es reducir los costes operativos de las campañas de marketing telefónico, concentrando los esfuerzos en los perfiles de mayor probabilidad de conversión.

| Característica | Detalle |
|---|---|
| Registros totales | 45,211 clientes |
| Variables de entrada | 17 (7 numéricas, 10 categóricas) |
| Variable objetivo | `y` — suscripción al depósito (Sí/No) |
| Desbalanceo de clases | 88% No suscribe / 12% Sí suscribe (ratio 7.5:1) |
| Valores nulos reales | 0 — pero `unknown` encubiertos en 4 variables |

---

## 2. Análisis Exploratorio de Datos (EDA)

El análisis exploratorio se estructuró en tres niveles: univariante, bivariante y multivariante, con el objetivo de identificar patrones, relaciones y variables con mayor poder predictivo.

### 2.1 Hallazgos Univariantes

- La variable objetivo presenta un desbalanceo severo de 7.5:1, lo que requiere técnicas específicas durante el modelado.
- **`pdays`:** el 81.8% de los clientes tienen valor -1, indicando que nunca fueron contactados anteriormente.
- **`balance`:** fuerte asimetría positiva con outliers extremos (máximo: 102,127 €).
- **`campaign`:** concentración en 1-3 contactos; más de 3 contactos reduce drásticamente la conversión.

### 2.2 Hallazgos Bivariantes y Multivariantes

- **`poutcome = success`:** tasa de conversión del 65% — el predictor más fuerte del dataset.
- **Estacionalidad:** los meses de marzo, septiembre, octubre y diciembre presentan tasas de éxito hasta 4x superiores a la media.
- **Perfil profesional:** estudiantes y jubilados tienen las tasas de conversión más altas, independientemente del nivel educativo.
- **Saldo bancario:** clientes con balance positivo muestran mayor propensión a suscribir; saldos negativos prácticamente no convierten.
- **Canal de contacto:** cellular duplica la efectividad respecto a telephone o unknown.

### 2.3 Decisión sobre Data Leakage

La variable `duration` (duración de la llamada en segundos) presentó una correlación muy alta con la variable objetivo. Sin embargo, esta información solo está disponible **después** de realizar la llamada, por lo que incluirla en el modelo constituiría data leakage. Fue eliminada del dataset para garantizar que el modelo prediga antes del contacto con el cliente, que es el escenario real de uso.

---

## 3. Preprocesamiento de Datos

El pipeline de preprocesamiento fue diseñado para respetar la integridad del proceso de evaluación, aplicando todas las transformaciones exclusivamente sobre el conjunto de entrenamiento y evitando cualquier forma de data leakage.

| Acción | Detalle | Motivo |
|---|---|---|
| **Eliminación de variable** | `duration` | Data leakage: solo disponible tras la llamada |
| **Transformación de variables** | `pdays` → `es_cliente_nuevo` | Capturar si el cliente tiene historial previo |
| **Transformación de variables** | `previous` → `tuvo_contacto_previo` | Binarizar número de contactos anteriores |
| **Feature Engineering** | `temporada_alta` (desde `month`) | Meses de alta conversión detectados en EDA |
| **Encoding categórico** | OneHotEncoder (`unknown` como categoría) | Variables nominales |
| **Escalado numérico** | RobustScaler | Resistente a outliers (especialmente `balance`) |
| **Balanceo de clases** | SMOTE solo sobre train set | Ratio 7.5:1 → 1:1 |
| **Split estratificado** | 80% train / 20% test | Preserva proporción de clases |

> Los valores `unknown` en variables categóricas fueron conservados como categoría propia, dado que el análisis bivariante demostró que son informativos: en `poutcome` identifican clientes nuevos sin historial previo, y en `contact` indican registros sin canal de comunicación conocido.

---

## 4. Modelos Evaluados y Optimización

Se entrenaron cuatro modelos de clasificación, seleccionados para cubrir diferentes enfoques algorítmicos y proporcionar una comparación robusta.

- **Regresión Logística:** modelo lineal de referencia (baseline interpretable).
- **Random Forest:** ensemble de árboles, robusto a outliers y captura relaciones no lineales.
- **SVM:** clasificador por hiperplanos, efectivo en espacios de alta dimensión.
- **XGBoost:** gradient boosting, estado del arte para datos tabulares.

### 4.1 Métrica de evaluación

Dado el desbalanceo de clases, la accuracy fue descartada como métrica principal. Un modelo que prediga siempre "No suscribe" obtendría 88% de accuracy sin ningún valor predictivo real.

| Métrica | Uso |
|---|---|
| **F1-Score** | Métrica principal de optimización — equilibrio Precision/Recall |
| **AUC-ROC** | Capacidad de separación global entre clases |
| **Recall** | Impacto de negocio — clientes potenciales detectados |
| **Precision** | Control de falsos positivos — coste de llamadas innecesarias |

### 4.2 Optimización de hiperparámetros

Se utilizó **Optuna** para la optimización bayesiana de hiperparámetros, con 50 trials por modelo y validación cruzada estratificada de 3 folds. Este enfoque supera al GridSearch tradicional al explorar de forma inteligente el espacio de hiperparámetros, aprendiendo de cada iteración anterior.

---

## 5. Comparación de Resultados

| Modelo | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| **XGBoost ✓** | **0.891** | **0.569** | 0.298 | **0.391** | **0.792** |
| Random Forest | 0.884 | 0.508 | **0.317** | 0.390 | 0.783 |
| Logistic Regression | 0.746 | 0.260 | 0.637 | 0.370 | 0.767 |
| SVM | 0.843 | 0.304 | 0.268 | 0.285 | 0.651 |

*(✓) Modelo seleccionado como solución final.*

---

## 6. Modelo Final: XGBoost

El modelo XGBoost optimizado con Optuna fue seleccionado como solución final por los siguientes motivos:

- **Mayor AUC-ROC (0.792):** mejor capacidad de separación global entre clases.
- **Mayor F1-Score (0.391):** mejor equilibrio entre Precision y Recall.
- **Mayor Precision (0.569):** menos falsos positivos que los modelos alternativos.
- **Robustez:** gradient boosting maneja nativamente outliers y relaciones no lineales.

Aunque la Regresión Logística obtuvo un Recall significativamente mayor (0.637 vs 0.298), su Precision fue muy inferior (0.260), lo que generaría un volumen excesivo de falsos positivos en una campaña real, incrementando el coste operativo sin incrementar las conversiones.

### 6.1 Ajuste de umbral de decisión

| Umbral | Recall | Precision | Clientes captados (de 1,058) |
|---|---|---|---|
| 0.50 (por defecto) | 0.298 | 0.569 | ~315 |
| **0.30 (recomendado)** | **~0.53** | ~0.37 | **~561** |

Con umbral 0.30, el modelo detecta aproximadamente **246 clientes adicionales** que sí suscribirían, representando un impacto directo y cuantificable en los ingresos de la campaña.

---

## 7. Variables Más Influyentes

| Pos. | Variable | Importancia | Interpretación |
|---|---|---|---|
| 1 | `balance` | 0.1072 | Solvencia financiera del cliente |
| 2 | `campaign` | 0.1034 | Saturación por contactos repetidos |
| 3 | `day` | 0.1011 | Día del mes del contacto |
| 4 | `age` | 0.0937 | Perfil de edad (jóvenes y jubilados) |
| 5 | `poutcome_success` | 0.0404 | Éxito en campaña anterior |

---

## 8. Limitaciones y Trabajo Futuro

### 8.1 Limitaciones identificadas

- **Desbalanceo estructural:** aunque SMOTE mitiga el problema, el modelo sigue teniendo dificultades para detectar la clase minoritaria con alta precisión simultáneamente.
- **Alta tasa de `unknown`:** `poutcome` (81.7%) y `contact` (28.8%) limitan la calidad de la información disponible para muchos clientes.
- **Ausencia de variables digitales:** no hay variables de comportamiento digital ni historial de productos contratados, que probablemente tendrían alto poder predictivo.

### 8.2 Líneas de mejora propuestas

- **Ajuste dinámico del umbral:** implementar un umbral adaptativo según el coste relativo de falsos positivos vs falsos negativos en cada campaña.
- **Data drift monitoring:** monitorizar la distribución de las variables de entrada en producción para detectar degradación del modelo.
- **Feature engineering avanzado:** explorar interacciones entre variables (ej. `balance × poutcome`) y variables temporales derivadas de `month` y `day`.
- **Reentrenamiento periódico:** el comportamiento de los clientes bancarios evoluciona; se recomienda reentrenar el modelo con datos de los últimos 12 meses.

---

## 9. Conclusiones

El proyecto ha demostrado que es posible predecir con una capacidad discriminativa sólida (AUC-ROC 0.792) qué clientes tienen mayor probabilidad de suscribir un depósito a plazo fijo, utilizando exclusivamente información disponible antes del contacto telefónico.

El modelo XGBoost optimizado con Optuna constituye la solución más equilibrada entre precisión y capacidad de captura. La decisión de eliminar `duration` del modelo, aunque reduce las métricas brutas, garantiza un sistema válido para su uso real en precampaña, que es el objetivo de negocio definido.

La recomendación final para el equipo de marketing es desplegar el modelo con **umbral 0.30**, priorizando la captura de clientes potenciales. Con este enfoque, el modelo identificaría correctamente a más de la mitad de los clientes que suscribirían, permitiendo reducir significativamente el volumen total de llamadas necesarias para alcanzar los objetivos de captación.

---

### Resumen Ejecutivo

| | |
|---|---|
| **Modelo final** | XGBoost optimizado con Optuna |
| **AUC-ROC** | 0.792 |
| **F1-Score** | 0.391 (umbral 0.50) → ~0.44 (umbral 0.30) |
| **Recall con umbral 0.30** | ~53% de clientes potenciales detectados |
| **Impacto estimado** | +246 clientes adicionales captados por campaña |
| **Variables clave** | balance, campaign, age, poutcome_success |
