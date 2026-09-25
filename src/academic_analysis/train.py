from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]

PREPARED_DIR = ROOT / "data" / "prepared"
MODELS_DIR = ROOT / "models"
METADATA_DIR = ROOT / "metadata"

MLFLOW_DB_FILE = ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_FILE}"
MLFLOW_EXPERIMENT_NAME = "nota_curso_regresion"

FEATURE_COLUMNS = [
    "n_asistencias",
    "Curso",
    "Semestre",
]

TARGET_COLUMN = "Nota Curso"


# ============================================================
# Utilidades
# ============================================================

def sha256_file(path: Path) -> str:
    """Calcula SHA-256 de un archivo."""

    sha256 = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            sha256.update(block)

    return sha256.hexdigest()


def load_manifest(
    dataset_id: str,
) -> tuple[Path, dict]:
    """Carga el manifiesto del dataset."""

    manifest_path = (
        METADATA_DIR
        / f"{dataset_id}.json"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No existe un manifiesto para {dataset_id}."
        )

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        manifest = json.load(f)

    return manifest_path, manifest


def save_manifest(
    manifest_path: Path,
    manifest: dict,
) -> None:
    """Actualiza el manifiesto."""

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# Datos de entrenamiento
# ============================================================

def build_training_frame(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Construye X (features) e y (target) para el modelo
    de regresión que predice la nota del curso.

    Solo se conservan los registros que tienen una nota
    válida. Las variables categóricas (Curso, Semestre)
    se codifican mediante one-hot encoding.
    """

    data = df[
        df[TARGET_COLUMN].notna()
    ].copy()

    y = data[TARGET_COLUMN].astype(float)

    x = pd.get_dummies(
        data[FEATURE_COLUMNS],
        columns=["Curso", "Semestre"],
        drop_first=True,
    )

    return x, y


# ============================================================
# Entrenamiento
# ============================================================

def train_regression_model(
    dataset_id: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Path, dict]:
    """
    Entrena un modelo de regresión lineal que predice
    'Nota Curso' a partir de la asistencia y el curso.

    El experimento se registra en MLflow (parámetros,
    métricas y el modelo como artefacto). El manifiesto
    del dataset se actualiza con un resumen del
    entrenamiento, siguiendo el mismo patrón utilizado
    por las etapas anteriores del pipeline.
    """

    manifest_path, manifest = load_manifest(
        dataset_id
    )

    # --------------------------------------------------
    # Estado requerido: el dataset debe estar preparado
    # --------------------------------------------------

    preparation = manifest.get("preparation")

    if not preparation:
        raise ValueError(
            "El dataset debe estar preparado antes de "
            "entrenar un modelo. No se encontró "
            "información de 'preparation' en el "
            "manifiesto."
        )

    input_file = (
        PREPARED_DIR
        / preparation["output"]
    )

    if not input_file.exists():
        raise FileNotFoundError(
            f"No existe el dataset preparado: {input_file}"
        )

    # --------------------------------------------------
    # Comprobar integridad del dataset de entrada
    # --------------------------------------------------

    current_hash = sha256_file(input_file)
    expected_hash = preparation["sha256"]

    if current_hash != expected_hash:
        raise ValueError(
            "El dataset preparado fue modificado. "
            "El SHA-256 no coincide con el manifiesto."
        )

    df = pd.read_csv(input_file)

    x, y = build_training_frame(df)

    n_total = len(x)

    if n_total < 10:
        raise ValueError(
            "No hay suficientes registros con nota "
            "para entrenar un modelo."
        )

    # --------------------------------------------------
    # Separar entrenamiento y prueba
    # --------------------------------------------------

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    # --------------------------------------------------
    # Entrenar
    # --------------------------------------------------

    model = LinearRegression()

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    rmse = float(
        mean_squared_error(
            y_test,
            predictions,
        )
        ** 0.5
    )

    r2 = float(
        r2_score(
            y_test,
            predictions,
        )
    )

    # --------------------------------------------------
    # Registrar el experimento en MLflow
    # --------------------------------------------------

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT_NAME
    )

    with mlflow.start_run(
        run_name=f"regression_{dataset_id}"
    ) as run:

        mlflow.set_tag(
            "dataset_id",
            dataset_id,
        )

        mlflow.log_params(
            {
                "model_type": "LinearRegression",
                "features": ",".join(FEATURE_COLUMNS),
                "n_features_encoded": x.shape[1],
                "test_size": test_size,
                "random_state": random_state,
                "n_train": len(x_train),
                "n_test": len(x_test),
            }
        )

        mlflow.log_metrics(
            {
                "rmse": rmse,
                "r2": r2,
            }
        )

        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=x_train.head(3),
        )

        run_id = run.info.run_id
        experiment_id = run.info.experiment_id

    # --------------------------------------------------
    # Guardar una copia del modelo versionada por dataset
    # --------------------------------------------------

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_output = (
        MODELS_DIR
        / f"regression_{dataset_id}.joblib"
    )

    joblib.dump(
        {
            "model": model,
            "feature_columns": list(x.columns),
        },
        model_output,
    )

    # --------------------------------------------------
    # Actualizar manifiesto
    # --------------------------------------------------

    training = {
        "model_type": "LinearRegression",
        "target": TARGET_COLUMN,
        "raw_features": FEATURE_COLUMNS,
        "encoded_features": list(x.columns),
        "test_size": test_size,
        "random_state": random_state,
        "n_train": len(x_train),
        "n_test": len(x_test),
        "metrics": {
            "rmse": rmse,
            "r2": r2,
        },
        "mlflow": {
            "tracking_uri": MLFLOW_TRACKING_URI,
            "experiment_name": MLFLOW_EXPERIMENT_NAME,
            "experiment_id": experiment_id,
            "run_id": run_id,
        },
        "model_output": model_output.name,
        "model_sha256": sha256_file(model_output),
    }

    manifest["training"] = training
    manifest["status"] = "trained"

    save_manifest(
        manifest_path,
        manifest,
    )

    return model_output, training


# ============================================================
# CLI
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Entrena un modelo de regresión que predice "
            "la nota del curso a partir de la asistencia."
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

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporción de datos para prueba (default: 0.2).",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semilla aleatoria (default: 42).",
    )

    args = parser.parse_args()

    try:

        print(
            "\nEntrenando modelo de regresión "
            f"para el dataset: {args.dataset}\n"
        )

        model_output, training = train_regression_model(
            dataset_id=args.dataset,
            test_size=args.test_size,
            random_state=args.random_state,
        )

    except Exception as exc:

        print(
            "ENTRENAMIENTO FALLIDO\n"
        )

        print(exc)

        raise SystemExit(1)

    print(
        "ENTRENAMIENTO CORRECTO\n"
    )

    print(
        f"Modelo guardado en : {model_output}"
    )

    print(
        f"RMSE                : {training['metrics']['rmse']:.4f}"
    )

    print(
        f"R2                  : {training['metrics']['r2']:.4f}"
    )

    print(
        "MLflow run_id       : "
        f"{training['mlflow']['run_id']}"
    )


if __name__ == "__main__":
    main()
