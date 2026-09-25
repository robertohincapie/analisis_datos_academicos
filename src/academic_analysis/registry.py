from __future__ import annotations

import argparse
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient

from academic_analysis.train import (
    CLASSIFICATION_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    REGRESSION_EXPERIMENT_NAME,
    load_manifest,
    save_manifest,
)

# Vocabulario de estados tomado de la Unidad 4, Capítulo 3
# ("Registrar versiones de modelos" / "Model Registry"):
# un modelo registrado puede ser candidato, estar en producción,
# o estar archivado. Aquí solo distinguimos "candidato" (superó
# el baseline) de "archivado" (no lo superó, pero queda
# registrado igual, con su historial completo de por qué se
# descartó) — no manejamos "producción" todavía porque eso
# implica un servicio real (etapa 5).

ALIAS_CANDIDATO = "candidato"
ALIAS_ARCHIVADO = "archivado"


def _client() -> MlflowClient:

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    return MlflowClient(
        tracking_uri=MLFLOW_TRACKING_URI
    )


def register_regression_model(
    dataset_id: str,
) -> dict:
    """
    Registra en el Model Registry el candidato de
    regresión seleccionado por compare.py (debe existir
    manifest["selection"]["regression"]["selected_run_id"]).

    Se registra bajo el alias "candidato": superó el
    baseline, pero eso no significa automáticamente que
    esté en producción — esa decisión queda para cuando
    exista un servicio real (etapa 5).
    """

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    manifest_path, manifest = load_manifest(dataset_id)

    selection = manifest.get("selection", {}).get(
        "regression"
    )

    if not selection or not selection.get(
        "selected_run_id"
    ):
        raise ValueError(
            "No hay un modelo de regresión seleccionado "
            "todavía. Ejecute primero "
            "academic_analysis.compare regression."
        )

    run_id = selection["selected_run_id"]
    model_type = selection["selected_model_type"]

    selected_candidate = next(
        c
        for c in selection["candidates"]
        if c["run_id"] == run_id
    )

    model_uri = f"runs:/{run_id}/model"

    registered = mlflow.register_model(
        model_uri=model_uri,
        name=REGRESSION_EXPERIMENT_NAME,
    )

    client = _client()

    client.set_registered_model_alias(
        name=REGRESSION_EXPERIMENT_NAME,
        alias=ALIAS_CANDIDATO,
        version=registered.version,
    )

    tags = {
        "dataset_id": dataset_id,
        "model_type": model_type,
        "passes_baseline": "true",
        "rmse": str(selected_candidate["rmse"]),
        "status": "candidato",
    }

    for key, value in tags.items():
        client.set_model_version_tag(
            name=REGRESSION_EXPERIMENT_NAME,
            version=registered.version,
            key=key,
            value=value,
        )

    entry = {
        "registered_model_name": REGRESSION_EXPERIMENT_NAME,
        "version": registered.version,
        "alias": ALIAS_CANDIDATO,
        "run_id": run_id,
        "model_type": model_type,
        "status": "candidato",
        "registered_at": (
            datetime.now().astimezone().isoformat()
        ),
    }

    manifest.setdefault("registry", {})
    manifest["registry"]["regression"] = entry

    save_manifest(manifest_path, manifest)

    return entry


def register_rejected_classification_model(
    dataset_id: str,
) -> dict:
    """
    Registra igualmente el mejor candidato de
    clasificación (por F1), aunque compare.py no lo haya
    seleccionado por no superar el baseline. Queda
    registrado bajo el alias "archivado": el Model
    Registry conserva también el historial de lo que se
    intentó y se descartó, no solo lo que funcionó.
    """

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    manifest_path, manifest = load_manifest(dataset_id)

    selection = manifest.get("selection", {}).get(
        "classification"
    )

    if not selection or not selection.get("candidates"):
        raise ValueError(
            "No hay candidatos de clasificación evaluados "
            "todavía. Ejecute primero "
            "academic_analysis.compare classification."
        )

    if selection.get("selected_run_id"):
        raise ValueError(
            "El modelo de clasificación sí superó el "
            "baseline: use register_regression_model-style "
            "flow (alias 'candidato'), no esta función "
            "pensada para el caso rechazado."
        )

    # Los candidatos ya vienen ordenados por F1 descendente
    # (ver compare.select_classification_model).

    best_rejected = selection["candidates"][0]

    run_id = best_rejected["run_id"]
    model_type = best_rejected["model_type"]

    model_uri = f"runs:/{run_id}/model"

    registered = mlflow.register_model(
        model_uri=model_uri,
        name=CLASSIFICATION_EXPERIMENT_NAME,
    )

    client = _client()

    client.set_registered_model_alias(
        name=CLASSIFICATION_EXPERIMENT_NAME,
        alias=ALIAS_ARCHIVADO,
        version=registered.version,
    )

    tags = {
        "dataset_id": dataset_id,
        "model_type": model_type,
        "passes_baseline": "false",
        "accuracy": str(best_rejected["accuracy"]),
        "f1": str(best_rejected["f1"]),
        "status": "archivado",
        "rejection_reason": (
            "No supera el baseline ingenuo "
            f"(accuracy {best_rejected['accuracy']:.4f} vs. "
            f"baseline {selection['baseline']['accuracy']:.4f})."
        ),
    }

    for key, value in tags.items():
        client.set_model_version_tag(
            name=CLASSIFICATION_EXPERIMENT_NAME,
            version=registered.version,
            key=key,
            value=value,
        )

    entry = {
        "registered_model_name": CLASSIFICATION_EXPERIMENT_NAME,
        "version": registered.version,
        "alias": ALIAS_ARCHIVADO,
        "run_id": run_id,
        "model_type": model_type,
        "status": "archivado",
        "rejection_reason": tags["rejection_reason"],
        "registered_at": (
            datetime.now().astimezone().isoformat()
        ),
    }

    manifest.setdefault("registry", {})
    manifest["registry"]["classification"] = entry

    save_manifest(manifest_path, manifest)

    return entry


# ============================================================
# CLI
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Registra en el MLflow Model Registry el "
            "resultado de la comparación de modelos."
        )
    )

    parser.add_argument(
        "task",
        choices=["regression", "classification"],
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help=(
            "Identificador del dataset, "
            "por ejemplo ING-20260910-101728."
        ),
    )

    args = parser.parse_args()

    try:

        if args.task == "regression":
            entry = register_regression_model(args.dataset)
        else:
            entry = register_rejected_classification_model(
                args.dataset
            )

    except Exception as exc:

        print("REGISTRO FALLIDO\n")
        print(exc)
        raise SystemExit(1)

    print("\nREGISTRO CORRECTO\n")

    for key, value in entry.items():
        print(f"{key:<22}: {value}")


if __name__ == "__main__":
    main()
