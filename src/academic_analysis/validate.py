from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]

INCOMING_DIR = ROOT / "data" / "incoming"
VALIDATED_DIR = ROOT / "data" / "validated"
METADATA_DIR = ROOT / "metadata"
SCHEMA_FILE = ROOT / "config" / "data_schema.yaml"


# ============================================================
# Utilidades
# ============================================================

def sha256_file(path: Path) -> str:
    """Calcula SHA-256."""

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


def load_schema() -> dict:
    """Carga el contrato de datos."""

    if not SCHEMA_FILE.exists():
        raise FileNotFoundError(
            f"No existe {SCHEMA_FILE}"
        )

    with SCHEMA_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:

        schema = yaml.safe_load(f)

    if not schema:
        raise ValueError(
            "El schema está vacío."
        )

    return schema


# ============================================================
# Validación estructural
# ============================================================

def validate_columns(
    df: pd.DataFrame,
    schema: dict,
) -> None:
    """Valida las columnas esperadas."""

    expected = list(
        schema["columns"].keys()
    )

    actual = list(
        df.columns
    )

    missing = (
        set(expected)
        - set(actual)
    )

    extra = (
        set(actual)
        - set(expected)
    )

    if missing:
        raise ValueError(
            "Faltan columnas requeridas: "
            f"{sorted(missing)}"
        )

    if extra:
        raise ValueError(
            "Se encontraron columnas no esperadas: "
            f"{sorted(extra)}"
        )


def validate_nulls(
    df: pd.DataFrame,
    schema: dict,
) -> None:
    """Valida campos que no permiten nulos."""

    for column, rules in (
        schema["columns"].items()
    ):

        if rules.get(
            "nullable",
            True,
        ):
            continue

        invalid = (
            df[column].isna()
            | (
                df[column]
                .astype(str)
                .str.strip()
                == ""
            )
        )

        if invalid.any():

            raise ValueError(
                f"La columna '{column}' "
                f"contiene {int(invalid.sum())} "
                "valores nulos o vacíos."
            )


# ============================================================
# Validación de tipos y reglas
# ============================================================

def validate_student_id(
    df: pd.DataFrame,
) -> None:

    values = (
        df["student_id"]
        .astype(str)
        .str.strip()
    )

    invalid = (
        values == ""
    )

    if invalid.any():

        raise ValueError(
            "Se encontraron student_id vacíos."
        )


def validate_semester(
    df: pd.DataFrame,
) -> None:

    values = (
        df["Semestre"]
        .astype(str)
        .str.strip()
    )

    valid = values.map(
        lambda value:
            bool(
                re.fullmatch(
                    r"\d{4}-[12]",
                    value,
                )
            )
    )

    if not valid.all():

        invalid = sorted(
            values[
                ~valid
            ].unique()
        )

        raise ValueError(
            "Semestres con formato inválido: "
            f"{invalid}"
        )


def validate_grades(
    df: pd.DataFrame,
) -> None:
    """
    Nota Curso puede ser nula.

    Se consideran valores faltantes:
        NaN
        ""
        "-"
        "N/A"
        "NA"

    Cuando existe una nota, debe ser numérica
    y estar entre 0.0 y 5.0.
    """

    original = df["Nota Curso"]

    # --------------------------------------------------
    # Identificar valores que representan ausencia
    # de nota
    # --------------------------------------------------

    text = (
        original
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    missing_values = {
        "",
        "-",
        "N/A",
        "NA",
    }

    is_missing = text.isin(
        missing_values
    )

    # --------------------------------------------------
    # Convertir únicamente las notas existentes
    # --------------------------------------------------

    numeric = pd.to_numeric(
        original.where(~is_missing),
        errors="coerce",
    )

    # --------------------------------------------------
    # Detectar valores no numéricos que no sean
    # representaciones permitidas de ausencia
    # --------------------------------------------------

    invalid_type = (
        ~is_missing
        & numeric.isna()
    )

    if invalid_type.any():

        values = (
            original[
                invalid_type
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Nota Curso contiene valores "
            "no numéricos: "
            f"{values[:10]}"
        )

    # --------------------------------------------------
    # Validar rango
    # --------------------------------------------------

    invalid_range = (
        numeric.notna()
        & (
            (numeric < 0)
            | (numeric > 5)
        )
    )

    if invalid_range.any():

        raise ValueError(
            "Nota Curso debe estar entre "
            "0.0 y 5.0. "
            f"Filas afectadas: "
            f"{int(invalid_range.sum())}"
        )


def validate_attendance_count(
    df: pd.DataFrame,
) -> None:

    numeric = pd.to_numeric(
        df["n_asistencias"],
        errors="coerce",
    )

    if numeric.isna().any():

        raise ValueError(
            "n_asistencias contiene "
            "valores no numéricos."
        )

    non_integer = (
        numeric % 1 != 0
    )

    if non_integer.any():

        raise ValueError(
            "n_asistencias debe contener "
            "números enteros."
        )

    negative = (
        numeric < 0
    )

    if negative.any():

        raise ValueError(
            "n_asistencias no puede "
            "ser negativo."
        )


def normalize_boolean(
    value,
) -> bool | None:
    """
    Convierte representaciones habituales
    de booleanos.
    """

    if isinstance(value, bool):
        return value

    value = (
        str(value)
        .strip()
        .lower()
    )

    if value in {
        "true",
        "1",
    }:
        return True

    if value in {
        "false",
        "0",
    }:
        return False

    return None


def validate_attendance_flag(
    df: pd.DataFrame,
) -> None:

    normalized = (
        df["asistio"]
        .map(normalize_boolean)
    )

    if normalized.isna().any():

        invalid = (
            df.loc[
                normalized.isna(),
                "asistio",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "La columna asistio contiene "
            "valores no booleanos: "
            f"{invalid[:10]}"
        )


def validate_attendance_consistency(
    df: pd.DataFrame,
) -> None:
    """
    Comprueba:

    asistio == (n_asistencias > 0)
    """

    counts = pd.to_numeric(
        df["n_asistencias"],
        errors="raise",
    )

    attended = (
        df["asistio"]
        .map(normalize_boolean)
    )

    expected = (
        counts > 0
    )

    inconsistent = (
        attended != expected
    )

    if inconsistent.any():

        raise ValueError(
            "Inconsistencia entre "
            "n_asistencias y asistio. "
            f"Filas afectadas: "
            f"{int(inconsistent.sum())}"
        )


# ============================================================
# Inspección de duplicados
# ============================================================

def inspect_duplicate_records(
    df: pd.DataFrame,
    dataset_id: str,
) -> dict:
    """
    Detecta múltiples registros para:

        student_id + Curso + Semestre

    Los duplicados no invalidan el dataset.

    Pueden corresponder, por ejemplo, a cancelaciones,
    nuevas matrículas o cursos tomados posteriormente
    dentro del mismo período académico.

    En esta etapa únicamente se detectan y reportan.
    Su resolución corresponde a prepare.py.
    """

    key = [
        "student_id",
        "Curso",
        "Semestre",
    ]

    duplicated = df.duplicated(
        subset=key,
        keep=False,
    )

    duplicate_rows = int(
        duplicated.sum()
    )

    # --------------------------------------------------
    # No hay duplicados
    # --------------------------------------------------

    if duplicate_rows == 0:

        return {
            "duplicate_rows": 0,
            "duplicate_groups": 0,
            "report": None,
        }

    # --------------------------------------------------
    # Extraer registros involucrados
    # --------------------------------------------------

    duplicates = (
        df.loc[duplicated]
        .sort_values(key)
    )

    duplicate_groups = int(
        duplicates[
            key
        ]
        .drop_duplicates()
        .shape[0]
    )

    # --------------------------------------------------
    # Guardar reporte asociado a esta versión
    # del dataset
    # --------------------------------------------------

    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_file = (
        METADATA_DIR
        / f"{dataset_id}_duplicates.csv"
    )

    duplicates.to_csv(
        report_file,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------
    # Mostrar advertencia
    # --------------------------------------------------

    print(
        "\nADVERTENCIA\n"
    )

    print(
        "Se encontraron registros repetidos para "
        "student_id + Curso + Semestre."
    )

    print(
        f"  Filas involucradas: {duplicate_rows}"
    )

    print(
        f"  Grupos repetidos: {duplicate_groups}"
    )

    print(
        f"  Reporte: {report_file}"
    )

    print(
        "\nLos duplicados no invalidan el dataset. "
        "Serán tratados posteriormente durante "
        "la preparación de los datos.\n"
    )

    return {
        "duplicate_rows":
            duplicate_rows,

        "duplicate_groups":
            duplicate_groups,

        "report":
            report_file.name,
    }


# ============================================================
# Validación completa
# ============================================================

def validate_dataset(
    dataset_id: str,
) -> Path:
    """Valida un dataset previamente ingerido."""

    manifest_path, manifest = (
        load_manifest(
            dataset_id
        )
    )

    if (
        manifest.get("status")
        != "received"
    ):

        raise ValueError(
            "El dataset debe tener "
            "status='received'. "
            f"Status actual: "
            f"{manifest.get('status')}"
        )

    # --------------------------------------------------
    # Archivo registrado
    # --------------------------------------------------

    input_file = (
        INCOMING_DIR
        / manifest["filename"]
    )

    if not input_file.exists():

        raise FileNotFoundError(
            f"No existe el archivo registrado: "
            f"{input_file}"
        )

    # --------------------------------------------------
    # Integridad
    # --------------------------------------------------

    current_hash = (
        sha256_file(
            input_file
        )
    )

    if (
        current_hash
        != manifest["sha256"]
    ):

        raise ValueError(
            "El archivo fue modificado después "
            "de la ingesta. "
            "El SHA-256 no coincide."
        )

    # --------------------------------------------------
    # Cargar
    # --------------------------------------------------

    df = pd.read_csv(
        input_file
    )

    schema = load_schema()

    # --------------------------------------------------
    # Aplicar contrato
    #
    # Estas condiciones sí son errores y detienen
    # el pipeline.
    # --------------------------------------------------

    validate_columns(
        df,
        schema,
    )

    validate_nulls(
        df,
        schema,
    )

    validate_student_id(df)
    validate_semester(df)
    validate_grades(df)
    validate_attendance_count(df)
    validate_attendance_flag(df)
    validate_attendance_consistency(df)

    # --------------------------------------------------
    # Inspeccionar condiciones conocidas que no
    # invalidan el dataset.
    #
    # Los duplicados son una ADVERTENCIA.
    # --------------------------------------------------

    duplicate_report = (
        inspect_duplicate_records(
            df,
            dataset_id,
        )
    )

    # --------------------------------------------------
    # Estadísticas de validación
    # --------------------------------------------------

    grades = pd.to_numeric(
        df["Nota Curso"],
        errors="coerce",
    )

    validation = {

        "rows":
            len(df),

        "students":
            int(
                df[
                    "student_id"
                ].nunique()
            ),

        "courses":
            int(
                df[
                    "Curso"
                ].nunique()
            ),

        "semesters":
            int(
                df[
                    "Semestre"
                ].nunique()
            ),

        "missing_grades":
            int(
                grades.isna().sum()
            ),

        "duplicates":
            duplicate_report,
    }

    # --------------------------------------------------
    # Copiar a validated
    #
    # IMPORTANTE:
    # validate.py NO modifica los datos.
    # Los duplicados permanecen en el archivo.
    # --------------------------------------------------

    VALIDATED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        VALIDATED_DIR
        / manifest["filename"]
    )

    shutil.copy2(
        input_file,
        output_file,
    )

    # --------------------------------------------------
    # Actualizar manifiesto
    # --------------------------------------------------

    manifest[
        "validation"
    ] = validation

    manifest[
        "validated_file"
    ] = output_file.name

    manifest[
        "status"
    ] = "validated"

    save_manifest(
        manifest_path,
        manifest,
    )

    return output_file


# ============================================================
# CLI
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Valida una versión ingerida "
            "del dataset."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help=(
            "Identificador del dataset, "
            "por ejemplo "
            "ING-20260910-101728."
        ),
    )

    args = parser.parse_args()

    try:

        output = validate_dataset(
            args.dataset
        )

    except Exception as exc:

        print(
            "\nVALIDACIÓN FALLIDA\n"
        )

        print(exc)

        raise SystemExit(1)

    print(
        "\nVALIDACIÓN CORRECTA\n"
    )

    print(
        f"Dataset validado: {output}"
    )


if __name__ == "__main__":
    main()