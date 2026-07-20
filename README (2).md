# Detección de anomalías para la identificación del riesgo de podredumbre en frutas mediante datos hiperespectrales

**Trabajo de Fin de Grado — Grado en Ciencia de Datos, ETSE-UV**
**Autora:** Mireia Pérez García
**Tutor:** Fernando Mateo Jiménez

---

## Descripción

Este TFG explora la espectroscopía NIR (Near-Infrared) como alternativa no destructiva para
evaluar la calidad interna de limones. A partir de un conjunto de 978 limones de tres
variedades (Fino, Verna, Eureka), procedentes de dos campos agrícolas distintos, se trabaja
en dos frentes complementarios:

- **Detección de anomalías**, comparando métodos univariantes clásicos (percentil, 3-Sigma,
  IQR) con cinco algoritmos multivariantes de la librería [PyOD](https://github.com/yzhao062/pyod)
  (Isolation Forest, LOF, OCSVM, PCA, COPOD), aplicados sobre las 212 bandas del espectro NIR.
- **Predicción de parámetros de calidad interna**, como apoyo complementario al análisis de
  anomalías:
  - **TSS** (Sólidos Solubles Totales, °Brix)
  - **TA** (Acidez Titulable, g/100ml)
  - **TPC peel** (Polifenoles Totales en piel, mg/100g)
  - **TPC juice** (Polifenoles Totales en zumo, mg/100ml)

La memoria completa del TFG desarrolla la metodología y los resultados con más detalle que
este README.

---

## Estructura del proyecto

```
TFG-anomalias-NIR-limones/
├── src/                             # Módulos Python reutilizables
│   ├── data_io.py                   # Carga del dataset limpio y sus metadatos
│   ├── preprocessing.py             # Limpieza, splits, imputación, escalado y PCA
│   ├── outliers.py                  # Detección univariante (percentil, 3-Sigma, IQR)
│   ├── anomaly_detection.py         # Detección multivariante con PyOD y consenso
│   ├── models.py                    # Modelos baseline de regresión
│   ├── metrics.py                   # Métricas de regresión (MAE, RMSE, R²)
│   └── training.py                  # Entrenamiento, evaluación y guardado de modelos
│
├── TFG_1.ipynb                      # Notebook 1: limpieza, EDA y anomalías univariantes
├── deteccion_anomalias_pyod.ipynb   # Notebook 2: detección de anomalías con PyOD
├── modelado_final.ipynb             # Notebook 3: modelado y evaluación de calidad interna
│
├── requirements.txt
├── .gitignore
└── README.md
```

> **Nota sobre los datos:** [PENDIENTE — indica aquí si el dataset original
> (`MUESTRAS SELECCIONADAS.xlsx`/`.csv`, proporcionado por el IVIA) está incluido en el
> repositorio, o si hay que solicitarlo aparte y a quién. No lo veo en el listado del repo,
> así que no sé cuál es tu caso.]

---

## Metodología (resumen)

- **Validación externa por origen agrícola:** el conjunto de test lo forma íntegramente el
  campo de Cartagena (origen B), no visto durante el entrenamiento; Librilla (origen A) se
  divide 70/30 en entrenamiento y validación. Así se mide la capacidad real de generalización
  a un huerto nuevo, no solo a muestras nuevas del mismo campo.
- **Imputación por variedad:** los valores faltantes en variables morfológicas se imputan con
  la mediana de la misma variedad, calculada sobre el conjunto de entrenamiento.
- **Comparación univariante vs. multivariante:** se contrastan los tres métodos clásicos
  (percentil, 3-Sigma, IQR) con los cinco algoritmos de PyOD, y se define un criterio de
  consenso (≥3 de 5 modelos de acuerdo) para quedarse con las anomalías detectadas de forma
  robusta.
- **Validación de las anomalías frente a calidad química:** las muestras marcadas por consenso
  se contrastan contra TSS, TA y TPC (piel y zumo) mediante test de Mann-Whitney.

---

## Cómo ejecutar

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Ejecutar los notebooks, en este orden

Los tres notebooks son secuenciales: cada uno depende de los archivos que genera el anterior
en la carpeta `results/` (no incluida en el repositorio, se genera al ejecutar).

1. **`TFG_1.ipynb`** — limpieza inicial y EDA. Genera `results/df_clean_initial.csv` y
   `results/df_meta_initial.json`, que usan los otros dos notebooks.
2. **`deteccion_anomalias_pyod.ipynb`** — detección de anomalías multivariante. Depende de
   los archivos generados en el paso 1.
3. **`modelado_final.ipynb`** — modelado y evaluación de los 4 parámetros de calidad. También
   depende de los archivos generados en el paso 1.

---

## Resultados principales

### Predicción de calidad interna (test = campo de Cartagena)

| Target    | Mejor modelo  | RMSE test | R² test |
|-----------|---------------|-----------|---------|
| TSS       | Ridge         | 0.783     | 0.507   |
| TA        | Random Forest | 0.690     | 0.389   |
| TPC juice | Ridge         | 7.709     | -0.067  |
| TPC peel  | Random Forest | 87.87     | 0.269   |

### Detección de anomalías

De las 431 muestras de entrenamiento, PyOD identifica **21 anomalías de consenso** (≥3 de 5
modelos de acuerdo), con muy poco solapamiento respecto a los outliers detectados por métodos
univariantes clásicos. De los 4 parámetros de calidad, solo **TPC juice** muestra una
diferencia estadísticamente significativa entre las muestras anómalas y el resto
(p=0.0038, Mann-Whitney).

Los detalles completos (metodología, tablas y discusión de resultados) están en la memoria
del TFG.
