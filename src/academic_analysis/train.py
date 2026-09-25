from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]

PREPARED_DIR = ROOT / "data" / "prepared"
MODELS_DIR = ROOT / "models"
METADATA_DIR = ROOT / "metadata"
RESULTS_DIR = ROOT / "results"

MLFLOW_DB_FILE = ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_FILE}"

FEATURE_COLUMNS = [
    "n_asistencias",
    "Curso",
    "Semestre",
]

TARGET_COLUMN = "Nota Curso"

REGRESSION_EXPERIMENT_NAME = "nota_curso_regresion"
CLASSIFICATION_EXPERIMENT_NAME = "aprobacion_clasificacion"
PASSING_GRADE = 3.0


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


KNOWN_TRAINING_KEYS = (
    "regression",
    "classification",
)


def set_training_entry(
    manifest: dict,
    model_key: str,
    training: dict,
) -> None:
    """
    Guarda el resumen de un entrenamiento bajo
    manifest["training"][model_key], preservando el
    resumen de otros modelos ya entrenados sobre el
    mismo dataset (p. ej. regression y classification
    conviven en el mismo manifiesto).
    """

    existing = manifest.get("training", {})

    training_section = {
        key: value
        for key, value in existing.items()
        if key in KNOWN_TRAINING_KEYS
    }

    training_section[model_key] = training

    manifest["training"] = training_section
    manifest["status"] = "trained"


def load_prepared_dataset(
    dataset_id: str,
) -> tuple[Path, dict, pd.DataFrame]:
    """
    Carga el dataset preparado de un dataset_id,
    comprobando que el manifiesto indique que ya pasó
    por la etapa de preparación y que el archivo no haya
    sido modificado por fuera del pipeline (SHA-256).

    Esta comprobación es común a cualquier modelo que
    entrenemos a partir de los datos preparados.
    """

    manifest_path, manifest = load_manifest(
        dataset_id
    )

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

    current_hash = sha256_file(input_file)
    expected_hash = preparation["sha256"]

    if current_hash != expected_hash:
        raise ValueError(
            "El dataset preparado fue modificado. "
            "El SHA-256 no coincide con el manifiesto."
        )

    df = pd.read_csv(input_file)

    return manifest_path, manifest, df


def encode_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Codifica las variables categóricas (Curso, Semestre)
    mediante one-hot encoding. Se usa igual para el
    modelo de regresión y el de clasificación, de modo
    que ambos parten exactamente de las mismas variables.
    """

    return pd.get_dummies(
        data[FEATURE_COLUMNS],
        columns=["Curso", "Semestre"],
        drop_first=True,
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
    válida.
    """

    data = df[
        df[TARGET_COLUMN].notna()
    ].copy()

    y = data[TARGET_COLUMN].astype(float)

    x = encode_features(data)

    return x, y


def build_classification_frame(
    df: pd.DataFrame,
    passing_grade: float = PASSING_GRADE,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Construye X (features) e y (target) para el modelo
    de clasificación que predice si el estudiante
    aprueba el curso.

    aprueba = 1  si  Nota Curso >= passing_grade
    aprueba = 0  en caso contrario

    Igual que en la regresión, solo se conservan los
    registros que tienen una nota válida: no tendría
    sentido "predecir" la aprobación de un registro cuyo
    resultado real ni siquiera conocemos.
    """

    data = df[
        df[TARGET_COLUMN].notna()
    ].copy()

    y = (
        data[TARGET_COLUMN].astype(float)
        >= passing_grade
    ).astype(int)

    x = encode_features(data)

    return x, y


# ============================================================
# Entrenamiento: regresión
# ============================================================

def train_regression_model(
    dataset_id: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Path, dict]:
    """
    Entrena un modelo de regresión lineal que predice
    'Nota Curso' a partir de la asistencia, el curso y
    el semestre.

    El experimento se registra en MLflow (parámetros,
    métricas y el modelo como artefacto). El manifiesto
    del dataset se actualiza con un resumen del
    entrenamiento, bajo training.regression.
    """

    manifest_path, manifest, df = load_prepared_dataset(
        dataset_id
    )

    x, y = build_training_frame(df)

    if len(x) < 10:
        raise ValueError(
            "No hay suficientes registros con nota "
            "para entrenar un modelo."
        )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
    )

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
        REGRESSION_EXPERIMENT_NAME
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
            "experiment_name": REGRESSION_EXPERIMENT_NAME,
            "experiment_id": experiment_id,
            "run_id": run_id,
        },
        "model_output": model_output.name,
        "model_sha256": sha256_file(model_output),
    }

    set_training_entry(manifest, "regression", training)

    save_manifest(
        manifest_path,
        manifest,
    )

    return model_output, training


# ============================================================
# Entrenamiento: clasificación
# ============================================================

def train_classification_model(
    dataset_id: str,
    test_size: float = 0.2,
    random_state: int = 42,
    passing_grade: float = PASSING_GRADE,
) -> tuple[Path, dict]:
    """
    Entrena un modelo de clasificación (regresión
    logística) que predice si el estudiante aprueba el
    curso (Nota Curso >= passing_grade) a partir de la
    asistencia, el curso y el semestre.

    Usa exactamente las mismas variables de entrada que
    el modelo de regresión, para que ambos experimentos
    sean comparables. El experimento se registra en
    MLflow, incluyendo la matriz de confusión como
    artefacto. El manifiesto se actualiza bajo
    training.classification.
    """

    manifest_path, manifest, df = load_prepared_dataset(
        dataset_id
    )

    x, y = build_classification_frame(
        df,
        passing_grade=passing_grade,
    )

    if len(x) < 10:
        raise ValueError(
            "No hay suficientes registros con nota "
            "para entrenar un modelo."
        )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = LogisticRegression(
        max_iter=1000,
    )

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    accuracy = float(
        accuracy_score(
            y_test,
            predictions,
        )
    )

    f1 = float(
        f1_score(
            y_test,
            predictions,
        )
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    # --------------------------------------------------
    # Matriz de confusión como artefacto (igual al
    # ejemplo de "confusion_matrix.png" de la Unidad 4)
    # --------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    confusion_matrix_output = (
        RESULTS_DIR
        / f"classification_{dataset_id}_confusion_matrix.png"
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["reprueba", "aprueba"],
    )

    fig, ax = plt.subplots(figsize=(5, 5))

    display.plot(ax=ax, colorbar=False)

    ax.set_title(
        "Matriz de confusión — "
        "¿aprueba el curso?"
    )

    fig.tight_layout()

    fig.savefig(
        confusion_matrix_output,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------
    # Registrar el experimento en MLflow
    # --------------------------------------------------

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        CLASSIFICATION_EXPERIMENT_NAME
    )

    with mlflow.start_run(
        run_name=f"classification_{dataset_id}"
    ) as run:

        mlflow.set_tag(
            "dataset_id",
            dataset_id,
        )

        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "features": ",".join(FEATURE_COLUMNS),
                "n_features_encoded": x.shape[1],
                "passing_grade": passing_grade,
                "test_size": test_size,
                "random_state": random_state,
                "n_train": len(x_train),
                "n_test": len(x_test),
            }
        )

        mlflow.log_metrics(
            {
                "accuracy": accuracy,
                "f1": f1,
            }
        )

        mlflow.log_artifact(
            str(confusion_matrix_output)
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
        / f"classification_{dataset_id}.joblib"
    )

    joblib.dump(
        {
            "model": model,
            "feature_columns": list(x.columns),
            "passing_grade": passing_grade,
        },
        model_output,
    )

    # --------------------------------------------------
    # Actualizar manifiesto
    # --------------------------------------------------

    training = {
        "model_type": "LogisticRegression",
        "target": (
            f"{TARGET_COLUMN} >= {passing_grade}"
        ),
        "raw_features": FEATURE_COLUMNS,
        "encoded_features": list(x.columns),
        "test_size": test_size,
        "random_state": random_state,
        "n_train": len(x_train),
        "n_test": len(x_test),
        "metrics": {
            "accuracy": accuracy,
            "f1": f1,
            "confusion_matrix": matrix.tolist(),
        },
        "mlflow": {
            "tracking_uri": MLFLOW_TRACKING_URI,
            "experiment_name": CLASSIFICATION_EXPERIMENT_NAME,
            "experiment_id": experiment_id,
            "run_id": run_id,
        },
        "model_output": model_output.name,
        "model_sha256": sha256_file(model_output),
        "confusion_matrix_plot": confusion_matrix_output.name,
    }

    set_training_entry(manifest, "classification", training)

    save_manifest(
        manifest_path,
        manifest,
    )

    return model_output, training


# ============================================================
# CLI
# ============================================================

def _add_common_arguments(
    subparser: argparse.ArgumentParser,
) -> None:

    subparser.add_argument(
        "--dataset",
        required=True,
        help=(
            "Identificador del dataset, "
            "por ejemplo ING-20260910-101728."
        ),
    )

    subparser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporción de datos para prueba (default: 0.2).",
    )

    subparser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semilla aleatoria (default: 42).",
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Entrena un modelo (regresión o clasificación) "
            "a partir de los datos preparados."
        )
    )

    subparsers = parser.add_subparsers(
        dest="model",
        required=True,
    )

    regression_parser = subparsers.add_parser(
        "regression",
        help="Predice la nota del curso.",
    )

    _add_common_arguments(regression_parser)

    classification_parser = subparsers.add_parser(
        "classification",
        help="Predice si el estudiante aprueba el curso.",
    )

    _add_common_arguments(classification_parser)

    classification_parser.add_argument(
        "--passing-grade",
        type=float,
        default=PASSING_GRADE,
        help=(
            "Nota mínima para considerar aprobado "
            f"(default: {PASSING_GRADE})."
        ),
    )

    args = parser.parse_args()

    try:

        if args.model == "regression":

            print(
                "\nEntrenando modelo de regresión "
                f"para el dataset: {args.dataset}\n"
            )

            model_output, training = train_regression_model(
                dataset_id=args.dataset,
                test_size=args.test_size,
                random_state=args.random_state,
            )

        else:

            print(
                "\nEntrenando modelo de clasificación "
                f"para el dataset: {args.dataset}\n"
            )

            model_output, training = train_classification_model(
                dataset_id=args.dataset,
                test_size=args.test_size,
                random_state=args.random_state,
                passing_grade=args.passing_grade,
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

    for metric_name, metric_value in training["metrics"].items():

        if isinstance(metric_value, (int, float)):
            print(f"{metric_name:<20}: {metric_value:.4f}")

    print(
        "MLflow run_id       : "
        f"{training['mlflow']['run_id']}"
    )


if __name__ == "__main__":
    main()
