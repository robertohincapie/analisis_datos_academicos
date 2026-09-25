# Servicio de inferencia — Unidad 4, Laboratorio 4.
#
# Imagen de una sola etapa, deliberadamente mínima: solo entra
# lo que el servicio necesita para responder /predict,
# /health y /model-info.
#
# Lo que NO entra a esta imagen (queda en el entorno de
# desarrollo, no en el de despliegue):
#   - el resto del código (ingest/validate/prepare/analysis/
#     train/compare/registry)
#   - el manifiesto completo del dataset (metadata/)
#   - MLflow (mlflow.db, mlruns/) y todo su historial de
#     experimentos
#   - los notebooks
#   - los datos (data/) y el modelo de clasificación rechazado
#   - dependencias de desarrollo (jupyter, pytest, ruff, mlflow)
#
# Ver academic_analysis.package: es el paso previo que produce
# deploy/, el único artefacto de modelo que esta imagen recibe.

FROM python:3.12-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

# --no-default-groups excluye tanto "dev" (jupyter, pytest,
# ruff...) como "pipeline" (mlflow, matplotlib, scipy,
# pyyaml...): la imagen solo instala lo declarado en
# [project.dependencies] -fastapi, uvicorn, pandas,
# scikit-learn-, que es lo único que usa academic_analysis.api.
COPY pyproject.toml uv.lock ./
RUN uv sync --no-default-groups --frozen --no-install-project

# Copiar solo el módulo de la API (no el resto de
# src/academic_analysis) y el paquete de modelo ya empaquetado.
COPY src/academic_analysis/__init__.py src/academic_analysis/api.py src/academic_analysis/
COPY deploy/ deploy/

ENV PYTHONPATH=/app/src

EXPOSE 8000

# Importante: NO usar "uv run" aquí. "uv run" resincroniza el
# entorno con los grupos por defecto (dev + pipeline) en cada
# arranque, lo que volvería a instalar mlflow/jupyter/pytest
# dentro del contenedor y anularía todo el trabajo de dejar la
# imagen mínima. Se invoca uvicorn directamente desde el venv
# que ya quedó armado en el paso de build.
CMD ["/app/.venv/bin/uvicorn", "academic_analysis.api:app", "--host", "0.0.0.0", "--port", "8000"]
