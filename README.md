# Detección de anomalías para la identificación del riesgo de podredumbre en frutas mediante datos hiperespectrales

**Trabajo de Fin de Grado — Grado en Ciencia de Datos, ETSE-UV**  
**Autora:** Mireia Pérez García  
**Tutor:** Fernando Mateo Jiménez  

---

## Descripción

Este TFG desarrolla un sistema de predicción y detección de anomalías sobre limones
utilizando datos hiperespectrales NIR (Near-Infrared). A partir de las 212 bandas
espectrales de cada limón y sus variables morfológicas (peso, longitud, diámetro),
se predicen 4 parámetros de calidad interna:

- **TSS** (Sólidos Solubles Totales, °Brix) — indicador de dulzor
- **TA** (Acidez Titulable, g/100ml)
- **TPC peel** (Polifenoles Totales en piel, mg/100g)
- **TPC juice** (Polifenoles Totales en zumo, mg/100ml)

---

## Estructura del proyecto

```
TFG/
├── src/                        # Módulos Python reutilizables
│   ├── __init__.py
│   ├── data_io.py              # Lectura de datos y metadatos
│   ├── preprocessing.py        # Preprocesado, splits y escalado
│   ├── models.py               # Definición de modelos baseline
│   ├── training.py             # Entrenamiento y evaluación
│   ├── metrics.py              # Métricas de regresión (MAE, RMSE, R²)
│   ├── outliers.py             # Detección univariante de anomalías
│   └── utils.py                # Utilidades de guardado/carga
│
├── TFG_1.ipynb                       # Notebook 1: EDA y detección de anomalías
├── modelado_final.ipynb              # Notebook 2: Modelado y evaluación
│
├── requirements.txt
└── README.md
```

---

## Metodología

### Cambios metodológicos aplicados

**1. Validación externa por origen agrícola**
El conjunto de test se forma íntegramente con los limones de un campo (origen)
no visto durante el entrenamiento, simulando la aplicación real en una cooperativa.

**2. Imputación inteligente por variedad**
Los valores faltantes en variables morfológicas se imputan con la mediana de la
misma variedad (Fino, Verna, Eureka), en lugar de la mediana global del dataset.

**3. Tabla comparativa de métodos de detección univariante**
Se comparan tres reglas clásicas sobre todas las variables de entrada:
Percentil (P1-P99), 3-Sigma y IQR (Tukey), identificando qué tipo de anomalía
detecta cada método.

---

## Cómo ejecutar

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Ejecutar los notebooks en orden

Primero el notebook de EDA y limpieza:
```
TFG_parte1.ipynb
```

Este genera los archivos en `results/` que necesita el segundo notebook.
A continuación el notebook de modelado:
```
modelado.ipynb
```

---

## Resultados principales

| Target | Mejor modelo | RMSE test | R² test |
|--------|-------------|-----------|---------|
| TSS    | Ridge       | 0.783     | 0.507   |
| TA     | —           | —         | —       |
| TPC peel | —         | —         | —       |
| TPC juice | —        | —         | —       |

> Los resultados completos para los 4 targets se encuentran en `modelado.ipynb`.

