# Academic Performance Analysis

Laboratorio de producción para construir un producto analítico reproducible a partir de datos académicos seudonimizados.

## Laboratorio 1

Ingesta, versionamiento y validación de datos.

Flujo inicial:

`data/incoming -> ingest -> metadata -> validate -> data/validated`

# Ingesta
uv run python -m academic_analysis.ingest --file data/incoming/academic_performance_ING-20260910-101728.csv  

# Validación
 uv run python -m academic_analysis.validate --dataset ING-20260910-101728

# Preparacion
uv run python -m academic_analysis.prepare --dataset ING-20260910-101728

# Analisis (prueba de hipotesis: asistencia vs. nota)
uv run python -m academic_analysis.analysis --dataset ING-20260910-101728

## Laboratorio 2

Primeros modelos predictivos, con seguimiento de experimentos en MLflow.

El análisis del Laboratorio 1 (prueba t) responde si existe una diferencia
estadística entre asistir y no asistir. Los modelos de este laboratorio van
un paso más allá y usan asistencia, curso y semestre para predecir:

- **Regresión** (`LinearRegression`): la nota exacta del curso.
- **Clasificación** (`LogisticRegression`): si el estudiante aprueba
  (`Nota Curso >= 3.0`).

Ambos son una primera aproximación (*baseline*), con desempeño modesto —ver
`notebooks/Unidad4_lab2.ipynb` para la interpretación completa, incluyendo
por qué un *accuracy* del 78% en el modelo de clasificación en realidad no
es tan bueno como parece.

Cada ejecución queda registrada en MLflow (parámetros, métricas y el modelo
como artefacto) y también resumida en `metadata/<dataset_id>.json`, bajo
`training.regression` y `training.classification`, junto al `run_id`
correspondiente de cada una.

```bash
# Entrenamiento (regresión)
uv run python -m academic_analysis.train regression --dataset ING-20260910-101728

# Entrenamiento (clasificación: ¿aprueba el curso?)
uv run python -m academic_analysis.train classification --dataset ING-20260910-101728

# Explorar los experimentos registrados
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

`mlruns/` y `mlflow.db` son estado local de MLflow: se reconstruyen
ejecutando `train.py` y no se versionan en Git. Lo que sí se versiona es
cada modelo entrenado (`models/`), la matriz de confusión (`results/`) y el
resumen de cada experimento en el manifiesto.

