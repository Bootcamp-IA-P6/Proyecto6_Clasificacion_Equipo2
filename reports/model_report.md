# Informe Técnico del Modelo de Clasificación

## 1. Introducción

El objetivo de este proyecto es desarrollar un modelo de clasificación capaz de predecir si un cliente aceptará una oferta de depósito bancario basándose en información demográfica, financiera y de campañas de marketing anteriores.

El dataset utilizado corresponde a campañas de marketing directo de una institución bancaria y presenta un problema de clasificación binaria con **desbalance de clases**, donde aproximadamente el **12%** de los clientes aceptan la oferta.

---

## 2. Dataset

**Descripción general del dataset:**

- **Número de registros:** 45,211  
- **Número de variables:** 17  
- **Variable objetivo:** `y`

### Valores de la variable objetivo

| Clase | Significado                     |
|-------|---------------------------------|
| 0     | Cliente no acepta el depósito   |
| 1     | Cliente acepta el depósito      |

**Distribución de clases:**

- **No:** ≈ 88%  
- **Yes:** ≈ 12%

Por lo tanto, se trata de un problema con **clases desbalanceadas**.

---

## 3. Análisis Exploratorio de Datos (EDA)

### Hallazgos importantes

#### Duration es *data leakage*
La variable `duration` representa la duración de la llamada, información que **no está disponible antes de realizar la llamada**, por lo que fue eliminada.

#### 🔹 Variables relevantes detectadas
El análisis bivariante mostró que variables como:

- `poutcome`
- `month`
- `housing`
- `job`

tienen influencia en la probabilidad de suscripción.

#### 🔹 Clientes nuevos
La variable `pdays` fue transformada en:

- `es_cliente_nuevo`

para distinguir clientes con historial previo.

---

## 4. Preprocesamiento de Datos

Se aplicaron varias técnicas de preparación:

### 🔹 Eliminación de variables
- `duration`  
- `pdays`  
- `previous`

### 🔹 Ingeniería de variables
Nuevas variables creadas:

- `es_cliente_nuevo`
- `tuvo_contacto_previo`
- `temporada_alta`

### 🔹 Codificación de variables categóricas
Se utilizó:

- **OneHotEncoder**

### 🔹 Escalado de variables numéricas
Se utilizó:

- **RobustScaler**, debido a la presencia de outliers (ej. `balance`).

### 🔹 Balanceo de clases
Se aplicó:

- **SMOTE** para balancear las clases en el conjunto de entrenamiento.

---

## 5. Modelos Evaluados

Se entrenaron cuatro modelos de clasificación:

| Modelo              | Tipo                           |
|---------------------|--------------------------------|
| Logistic Regression | Modelo lineal                  |
| Random Forest       | Ensemble basado en árboles     |
| SVM                 | Modelo basado en hiperplanos   |
| XGBoost             | Gradient Boosting              |

---

## 6. Optimización de hiperparámetros

Se utilizó **Optuna** para optimizar los hiperparámetros de los modelos más prometedores.

El objetivo de optimización fue **maximizar el F1-score**, debido al desbalance de clases.

---

## 7. Comparación de modelos

### Resultados finales

| Modelo              | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| **XGBoost**         | 0.891    | 0.568     | 0.298  | 0.391    | 0.792   |
| Random Forest       | 0.884    | 0.508     | 0.317  | 0.390    | 0.783   |
| Logistic Regression | 0.746    | 0.260     | 0.637  | 0.370    | 0.767   |
| SVM                 | 0.843    | 0.304     | 0.268  | 0.285    | 0.651   |

---

## 8. Selección del modelo final

El modelo seleccionado fue:

# **XGBoost**

### Razones:

- Mayor **F1-score**
- Mayor **ROC-AUC**
- Mayor **precision**
- Buen equilibrio entre precisión y recall

Aunque Logistic Regression obtuvo un recall mayor, su precision fue significativamente inferior, generando demasiados falsos positivos.

---

## 9. Análisis de errores

El análisis de la matriz de confusión muestra que:

- El modelo clasifica correctamente la mayoría de clientes que **no aceptan** la oferta.
- Algunos clientes que **sí aceptarían** el depósito son clasificados como negativos.

Esto es común en problemas con clases desbalanceadas.

---

## 10. Limitaciones

- Dataset desbalanceado  
- Información limitada sobre el comportamiento financiero de los clientes  
- Algunas variables categóricas contienen valores `"unknown"`

---

## 11. Trabajo futuro

Posibles mejoras:

- Ajustar el **threshold** de clasificación  
- Incorporar nuevas variables  
- Reentrenamiento con datos más recientes  
- Implementar monitoreo de **data drift**

---

## 12. Conclusión

El modelo **XGBoost optimizado con Optuna** demostró ser el más eficaz para predecir la suscripción de depósitos bancarios.  
Ofrece un buen equilibrio entre precisión y capacidad predictiva, convirtiéndose en la mejor opción para el caso de uso.
