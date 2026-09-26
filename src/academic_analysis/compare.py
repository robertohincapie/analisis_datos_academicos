from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from academic_analysis.train import (
    PASSING_GRADE,
    build_classification_frame,
    build_training_frame,
    load_manifest,
    load_prepared_dataset,
    save_manifest,
)

# ============================================================
# Baselines "ingenuos"
#
# Un modelo solo es útil si supera a la respuesta trivial
# que no necesita aprender nada de los datos. Ver Unidad 4,
# Capítulo 3, "¿Cuál modelo debemos desplegar?".
# ============================================================


def regression_baseline_rmse(
    dataset_id: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> float:
    """
    RMSE de un modelo trivial que siempre predice el
    promedio de las notas de entrenamiento, evaluado
    sobre la misma partición de prueba que usaron los
    modelos candidatos.
    """

    _, _, df = load_prepared_dataset(dataset_id)

    x, y = build_training_frame(df)

    _, _, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    baseline_prediction = np.full(
        len(y_test),
        y_train.mean(),
    )

    return float(
        mean_squared_error(
            y_test,
            baseline_prediction,
        )
        ** 0.5
    )


def classification_baseline_accuracy(
    dataset_id: str,
    test_size: float = 0.2,
    random_state: int = 42,
    passing_grade: float = PASSING_GRADE,
) -> float:
    """
    Accuracy de un modelo trivial que siempre predice la
    clase mayoritaria de entrenamiento ("siempre aprueba",
    en este dataset), evaluado sobre la misma partición de
    prueba que usaron los modelos candidatos.
    """

    _, _, df = load_prepared_dataset(dataset_id)

    x, y = build_classification_frame(
        df,
        passing_grade=passing_grade,
    )

    _, _, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    majority_class = y_train.mode().iloc[0]

    return float((y_test == majority_class).mean())


# ============================================================
# Selección
# ============================================================


def select_regression_model(
    dataset_id: str,
) -> dict:
    """
    Compara los candidatos de regresión ya entrenados
    (manifest["training"]["regression"]) contra el
    baseline ingenuo y entre sí. Selecciona el de menor
    RMSE, siempre que supere al baseline.
    """

    manifest_path, manifest = load_manifest(dataset_id)

    candidates = manifest.get("training", {}).get("regression", {})

    if not candidates:
        raise ValueError(
            "No hay modelos de regresión entrenados todavía para este dataset."
        )

    baseline_rmse = regression_baseline_rmse(dataset_id)

    ranked = []

    for model_type, training in candidates.items():
        rmse = training["metrics"]["rmse"]

        ranked.append(
            {
                "model_type": model_type,
                "run_id": training["mlflow"]["run_id"],
                "rmse": rmse,
                "r2": training["metrics"]["r2"],
                "passes_gate": rmse < baseline_rmse,
            }
        )

    ranked.sort(key=lambda c: c["rmse"])

    passing = [c for c in ranked if c["passes_gate"]]

    selected = passing[0] if passing else None

    selection = {
        "criteria": (
            "Menor RMSE entre los candidatos que superan "
            "el baseline (predecir siempre el promedio de "
            "entrenamiento)."
        ),
        "baseline": {
            "description": "Predecir siempre la nota promedio.",
            "rmse": baseline_rmse,
        },
        "candidates": ranked,
        "selected_model_type": (selected["model_type"] if selected else None),
        "selected_run_id": (selected["run_id"] if selected else None),
        "evaluated_at": (datetime.now().astimezone().isoformat()),
    }

    manifest.setdefault("selection", {})
    manifest["selection"]["regression"] = selection

    save_manifest(manifest_path, manifest)

    return selection


def select_classification_model(
    dataset_id: str,
) -> dict:
    """
    Compara los candidatos de clasificación ya entrenados
    (manifest["training"]["classification"]) contra el
    baseline ingenuo y entre sí. Selecciona el de mayor
    F1 entre los que superan el baseline de accuracy.
    """

    manifest_path, manifest = load_manifest(dataset_id)

    candidates = manifest.get("training", {}).get("classification", {})

    if not candidates:
        raise ValueError(
            "No hay modelos de clasificación entrenados todavía para este dataset."
        )

    baseline_accuracy = classification_baseline_accuracy(dataset_id)

    ranked = []

    for model_type, training in candidates.items():
        accuracy = training["metrics"]["accuracy"]

        ranked.append(
            {
                "model_type": model_type,
                "run_id": training["mlflow"]["run_id"],
                "accuracy": accuracy,
                "f1": training["metrics"]["f1"],
                "passes_gate": accuracy > baseline_accuracy,
            }
        )

    ranked.sort(key=lambda c: c["f1"], reverse=True)

    passing = [c for c in ranked if c["passes_gate"]]

    selected = passing[0] if passing else None

    selection = {
        "criteria": (
            "Mayor F1 entre los candidatos que superan el "
            "baseline (predecir siempre la clase mayoritaria "
            "de entrenamiento)."
        ),
        "baseline": {
            "description": ("Predecir siempre la clase mayoritaria (aprueba)."),
            "accuracy": baseline_accuracy,
        },
        "candidates": ranked,
        "selected_model_type": (selected["model_type"] if selected else None),
        "selected_run_id": (selected["run_id"] if selected else None),
        "evaluated_at": (datetime.now().astimezone().isoformat()),
    }

    manifest.setdefault("selection", {})
    manifest["selection"]["classification"] = selection

    save_manifest(manifest_path, manifest)

    return selection


# ============================================================
# CLI
# ============================================================


def _print_selection(task: str, selection: dict) -> None:

    print(f"\nComparación — {task}\n")

    print(f"Criterio: {selection['criteria']}")

    baseline = selection["baseline"]

    print(f"Baseline: {baseline}\n")

    for candidate in selection["candidates"]:
        print(candidate)

    print()

    if selection["selected_model_type"]:
        print(
            "MODELO CANDIDATO SELECCIONADO: "
            f"{selection['selected_model_type']} "
            f"(run_id={selection['selected_run_id']})"
        )

    else:
        print(
            "NINGÚN CANDIDATO SUPERA EL BASELINE.\n"
            "No se selecciona ningún modelo. "
            "Antes de continuar, hay que revisar variables, "
            "datos o algoritmos — no promover un modelo que "
            "no aporta nada por encima de adivinar."
        )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Compara los modelos candidatos ya entrenados "
            "para una tarea y selecciona el mejor, si supera "
            "el baseline ingenuo."
        )
    )

    parser.add_argument(
        "task",
        choices=["regression", "classification"],
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help=("Identificador del dataset, por ejemplo ING-20260910-101728."),
    )

    args = parser.parse_args()

    try:
        if args.task == "regression":
            selection = select_regression_model(args.dataset)
        else:
            selection = select_classification_model(args.dataset)

    # Borde del comando: cualquier fallo se informa como un error controlado
    # (mensaje claro + código de salida 1) en vez de una traza de Python.
    except Exception as exc:  # noqa: BLE001
        print("COMPARACIÓN FALLIDA\n")
        print(exc)
        raise SystemExit(1)

    _print_selection(args.task, selection)


if __name__ == "__main__":
    main()
