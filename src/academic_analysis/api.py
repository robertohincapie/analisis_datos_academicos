from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ============================================================
# Servicio de inferencia — Unidad 4, Capítulo 4
#
# Este módulo es deliberadamente independiente del resto del
# proyecto: NO importa nada de train.py / compare.py /
# registry.py, y por lo tanto no necesita MLflow en tiempo de
# ejecución. Solo sabe leer el paquete mínimo que dejó
# academic_analysis.package en deploy/ (model.joblib +
# model_info.json). Esa separación es la que permite construir
# una imagen de Docker liviana, sin el manifiesto completo del
# dataset ni el historial de experimentos.
# ============================================================

DEPLOY_DIR = Path(__file__).resolve().parents[2] / "deploy"

MODEL_FILE = DEPLOY_DIR / "model.joblib"
MODEL_INFO_FILE = DEPLOY_DIR / "model_info.json"

_state: dict = {}


def load_deployment_bundle() -> None:
    """Carga el modelo empaquetado y su ficha pública."""

    if not MODEL_FILE.exists() or not MODEL_INFO_FILE.exists():
        raise RuntimeError(
            "No se encontró el paquete de despliegue en "
            f"{DEPLOY_DIR}. Ejecute primero "
            "academic_analysis.package."
        )

    bundle = joblib.load(MODEL_FILE)

    with MODEL_INFO_FILE.open("r", encoding="utf-8") as f:
        info = json.load(f)

    _state["model"] = bundle["model"]
    _state["feature_columns"] = bundle["feature_columns"]
    _state["categories"] = bundle["categories"]
    _state["info"] = info


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_deployment_bundle()
    yield
    _state.clear()


app = FastAPI(
    title="Academic Performance — Servicio de inferencia",
    description=(
        "Predice la nota de un curso a partir de la "
        "asistencia, el curso y el semestre, usando el "
        "modelo de regresión registrado como 'candidato' "
        "(Unidad 4, Laboratorio 4)."
    ),
    lifespan=lifespan,
)


# ============================================================
# Contrato de la API
# ============================================================

class PredictionRequest(BaseModel):

    n_asistencias: int = Field(
        ge=0,
        description="Número de actividades a las que asistió.",
    )

    curso: str = Field(
        alias="Curso",
        description="Nombre exacto del curso.",
    )

    semestre: str = Field(
        alias="Semestre",
        description="Semestre, formato AAAA-N (p. ej. 2026-1).",
    )

    model_config = {"populate_by_name": True}


class PredictionResponse(BaseModel):

    predicted_grade: float
    model_name: str
    model_version: int
    model_alias: str


# ============================================================
# Utilidades de predicción
# ============================================================

def _validate_category(
    field_name: str,
    value: str,
) -> None:

    valid_values = _state["categories"][field_name]

    if value not in valid_values:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{field_name} desconocido: '{value}'. "
                f"Valores válidos: {valid_values}"
            ),
        )


def _build_model_input(
    request: PredictionRequest,
) -> pd.DataFrame:

    # No usamos pd.get_dummies aquí a propósito: sobre una
    # sola fila, get_dummies solo "ve" la categoría de esa
    # fila, y drop_first la descarta siempre -sea o no la
    # categoría de referencia real de entrenamiento-. Con un
    # único curso/semestre en la solicitud, eso codificaría
    # casi cualquier valor como si fuera la referencia
    # (todo en cero), sin importar cuál sea. Por eso
    # construimos las columnas manualmente a partir de
    # feature_columns, que sí refleja el esquema real
    # aprendido en entrenamiento.

    feature_columns = _state["feature_columns"]

    values: dict[str, float] = {
        "n_asistencias": request.n_asistencias
    }

    for column in feature_columns:

        if column == "n_asistencias":
            continue

        if column.startswith("Curso_"):
            category = column.removeprefix("Curso_")
            values[column] = int(request.curso == category)

        elif column.startswith("Semestre_"):
            category = column.removeprefix("Semestre_")
            values[column] = int(request.semestre == category)

    return pd.DataFrame([values])[feature_columns]


# ============================================================
# Endpoints
# ============================================================

@app.get("/health")
def health() -> dict:

    return {
        "status": "ok",
        "model_loaded": "model" in _state,
    }


@app.get("/model-info")
def model_info() -> dict:

    if "info" not in _state:
        raise HTTPException(
            status_code=503,
            detail="El modelo todavía no se ha cargado.",
        )

    return _state["info"]


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:

    if "model" not in _state:
        raise HTTPException(
            status_code=503,
            detail="El modelo todavía no se ha cargado.",
        )

    _validate_category("Curso", request.curso)
    _validate_category("Semestre", request.semestre)

    model_input = _build_model_input(request)

    prediction = float(
        _state["model"].predict(model_input)[0]
    )

    info = _state["info"]

    return PredictionResponse(
        predicted_grade=prediction,
        model_name=info["model_name"],
        model_version=info["model_version"],
        model_alias=info["model_alias"],
    )
