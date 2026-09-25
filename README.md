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

Primer modelo predictivo, con seguimiento de experimentos en MLflow.

El análisis del Laboratorio 1 (prueba t) responde si existe una diferencia
estadística entre asistir y no asistir. El modelo de regresión va un paso
más allá: intenta predecir la nota a partir de la asistencia, el curso y el
semestre. Es una primera aproximación (`LinearRegression`, desempeño
modesto) que sirve como línea base para comparar modelos futuros.

Cada ejecución queda registrada en MLflow (parámetros, métricas y el modelo
como artefacto) y también resumida en `metadata/<dataset_id>.json`, bajo la
clave `training`, junto al `run_id` correspondiente.

```bash
# Entrenamiento (regresión)
uv run python -m academic_analysis.train --dataset ING-20260910-101728

# Explorar los experimentos registrados
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

`mlruns/` y `mlflow.db` son estado local de MLflow: se reconstruyen
ejecutando `train.py` y no se versionan en Git. Lo que sí se versiona es el
modelo entrenado (`models/`) y el resumen del experimento en el manifiesto.

