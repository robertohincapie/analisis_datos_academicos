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

- **Regresión**: la nota exacta del curso.
- **Clasificación**: si el estudiante aprueba (`Nota Curso >= 3.0`).

Para cada tarea hay dos algoritmos candidatos (ver `--model-type` abajo).
Todos son una primera aproximación, con desempeño modesto —ver
`notebooks/Unidad4_lab2.ipynb` para la interpretación completa, incluyendo
por qué un *accuracy* del 78% en clasificación en realidad no es tan bueno
como parece.

Cada ejecución queda registrada en MLflow (parámetros, métricas y el modelo
como artefacto) y también resumida en `metadata/<dataset_id>.json`, bajo
`training.regression.<model_type>` y `training.classification.<model_type>`,
junto al `run_id` correspondiente de cada una.

```bash
# Entrenamiento (regresión: linear o random_forest)
uv run python -m academic_analysis.train regression --model-type linear --dataset ING-20260910-101728
uv run python -m academic_analysis.train regression --model-type random_forest --dataset ING-20260910-101728

# Entrenamiento (clasificación: logistic o random_forest)
uv run python -m academic_analysis.train classification --model-type logistic --dataset ING-20260910-101728
uv run python -m academic_analysis.train classification --model-type random_forest --dataset ING-20260910-101728

# Explorar los experimentos registrados
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

`mlruns/` y `mlflow.db` son estado local de MLflow: se reconstruyen
ejecutando `train.py` y no se versionan en Git. Lo que sí se versiona es
cada modelo entrenado (`models/`), la matriz de confusión (`results/`) y el
resumen de cada experimento en el manifiesto.

## Laboratorio 3

Comparar los candidatos entrenados y seleccionar uno, con un criterio
explícito: superar un **baseline ingenuo** (predecir siempre el promedio, o
predecir siempre la clase mayoritaria). Ver
`notebooks/Unidad4_lab3_seleccion.ipynb` para la lectura completa.

```bash
uv run python -m academic_analysis.compare regression --dataset ING-20260910-101728
uv run python -m academic_analysis.compare classification --dataset ING-20260910-101728
```

El resultado (criterio, baseline, cada candidato y cuál —si alguno— fue
seleccionado) queda en `metadata/<dataset_id>.json`, bajo
`selection.regression` / `selection.classification`. En este dataset, la
regresión sí encuentra un candidato válido; en clasificación, **ningún
candidato supera el baseline todavía**, así que no se selecciona ninguno —
es el comportamiento esperado, no un error.

### Model Registry

Registrar en MLflow lo que decidió la comparación anterior, con un alias
por estado:

- `candidato`: superó el baseline (regresión).
- `archivado`: no lo superó, pero queda registrado con la razón exacta del
  rechazo (clasificación) — el registry conserva también el historial de
  lo descartado, no solo lo que funcionó.

```bash
uv run python -m academic_analysis.registry regression --dataset ING-20260910-101728
uv run python -m academic_analysis.registry classification --dataset ING-20260910-101728
```

Requiere haber corrido antes `academic_analysis.compare` para esa tarea.
El resultado queda en `metadata/<dataset_id>.json`, bajo
`registry.regression` / `registry.classification`, y es visible en la
pestaña *Models* de `mlflow ui`.

