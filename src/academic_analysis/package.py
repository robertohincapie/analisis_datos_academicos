from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from academic_analysis.train import MODELS_DIR, load_manifest

ROOT = Path(__file__).resolve().parents[2]

DEPLOY_DIR = ROOT / "deploy"

# ============================================================
# Empaquetar el modelo candidato para despliegue
#
# Todo el repo (manifiestos completos, MLflow, notebooks,
# datos, resultados de los modelos rechazados) es información
# de DESARROLLO: útil para quien entrena y decide, pero no
# necesaria -ni conveniente- dentro del servicio que finalmente
# atiende solicitudes. Este módulo extrae de manifest["registry"]
# solo lo indispensable para servir el modelo candidato:
#
#   - el modelo entrenado (pesos + columnas + categorías)
#   - una ficha pública mínima (versión, métricas, alias)
#
# deploy/ -y solo deploy/- es lo que entra a la imagen Docker
# (ver Dockerfile). No se copian datos, ni el manifiesto
# completo, ni mlruns/mlflow.db, ni el modelo de clasificación
# rechazado.
# ============================================================


def export_regression_bundle(
    dataset_id: str,
    output_dir: Path = DEPLOY_DIR,
) -> Path:
    """
    Exporta a output_dir el modelo de regresión registrado
    con alias "candidato" (manifest["registry"]["regression"]),
    junto con una ficha pública mínima (model_info.json).

    Falla si todavía no hay un candidato registrado
    (academic_analysis.registry debe ejecutarse antes).
    """

    _, manifest = load_manifest(dataset_id)

    registry_entry = manifest.get("registry", {}).get(
        "regression"
    )

    if not registry_entry:
        raise ValueError(
            "No hay un modelo de regresión registrado "
            "todavía. Ejecute primero "
            "academic_analysis.registry regression."
        )

    model_type = registry_entry["model_type"]

    source_model = (
        MODELS_DIR
        / f"regression_{model_type}_{dataset_id}.joblib"
    )

    if not source_model.exists():
        raise FileNotFoundError(
            f"No existe el artefacto del modelo: {source_model}"
        )

    bundled = joblib.load(source_model)

    metrics = (
        manifest["training"]["regression"][model_type][
            "metrics"
        ]
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    model_output = output_dir / "model.joblib"

    joblib.dump(
        {
            "model": bundled["model"],
            "feature_columns": bundled["feature_columns"],
            "categories": bundled["categories"],
        },
        model_output,
    )

    # Ficha pública: solo lo necesario para responder
    # /model-info. Deliberadamente NO incluye run_id,
    # experiment_id, sha256 de los datos fuente, ni nada
    # sobre el modelo de clasificación rechazado.

    info = {
        "task": "regression",
        "target": "Nota Curso",
        "model_type": model_type,
        "model_name": registry_entry["registered_model_name"],
        "model_version": registry_entry["version"],
        "model_alias": registry_entry["alias"],
        "metrics": metrics,
        "registered_at": registry_entry["registered_at"],
    }

    info_output = output_dir / "model_info.json"

    with info_output.open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    return output_dir


# ============================================================
# CLI
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Empaqueta el modelo candidato registrado en un "
            "paquete mínimo para desplegar (deploy/)."
        )
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
        output_dir = export_regression_bundle(args.dataset)

    except Exception as exc:

        print("EMPAQUETADO FALLIDO\n")
        print(exc)
        raise SystemExit(1)

    print("\nEMPAQUETADO CORRECTO\n")
    print(f"Paquete de despliegue: {output_dir}")
    print(f"  - {output_dir / 'model.joblib'}")
    print(f"  - {output_dir / 'model_info.json'}")


if __name__ == "__main__":
    main()
