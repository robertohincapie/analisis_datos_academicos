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

