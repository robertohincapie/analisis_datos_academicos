from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

VALIDATED_DIR = ROOT / "data" / "validated"
PREPARED_DIR = ROOT / "data" / "prepared"
METADATA_DIR = ROOT / "metadata"


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
    """Carga el manifiesto asociado al dataset."""

    manifest_path = METADATA_DIR / f"{dataset_id}.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"No existe un manifiesto para {dataset_id}.")

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
# Normalización
# ============================================================


def normalize_grades(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normaliza Nota Curso.

    Se consideran valores faltantes:

        ""
        "-"
        "N/A"
        "NA"
        NaN

    Después de esta función Nota Curso contiene
    exclusivamente float o NaN.
    """

    df = df.copy()

    grade_text = df["Nota Curso"].fillna("").astype(str).str.strip().str.upper()

    missing_values = {
        "",
        "-",
        "N/A",
        "NA",
    }

    is_missing = grade_text.isin(missing_values)

    df.loc[
        is_missing,
        "Nota Curso",
    ] = pd.NA

    df["Nota Curso"] = pd.to_numeric(
        df["Nota Curso"],
        errors="raise",
    )

    return df


def normalize_types(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normaliza los tipos utilizados por
    el análisis posterior.
    """

    df = df.copy()

    df["student_id"] = df["student_id"].astype(str).str.strip()

    df["Curso"] = df["Curso"].astype(str).str.strip()

    df["Semestre"] = df["Semestre"].astype(str).str.strip()

    df["n_asistencias"] = pd.to_numeric(
        df["n_asistencias"],
        errors="raise",
    ).astype(int)

    # Reconstruimos asistio desde n_asistencias.
    # La consistencia ya fue comprobada en validate.py.

    df["asistio"] = df["n_asistencias"] > 0

    return df


# ============================================================
# Resolución de duplicados
# ============================================================


def sum_attendance(values: pd.Series) -> float:
    """
    Suma las asistencias de registros duplicados.

    Si todos los valores faltan, el resultado también falta (NaN) en vez
    de 0: no es lo mismo "no asistió" que "no hay información".
    """

    return values.sum(min_count=1)


def resolve_duplicate_records(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Consolida los registros correspondientes a:

        student_id + Curso + Semestre

    Política:

    - Nota Curso:
        se conserva la mayor nota válida.

    - n_asistencias:
        se suman las asistencias de todos los registros.

    - asistio:
        se reconstruye después de la agregación como:

            n_asistencias > 0

    De esta manera cada estudiante tiene una única
    observación por curso y semestre.
    """

    df = df.copy()

    key = [
        "student_id",
        "Curso",
        "Semestre",
    ]

    # --------------------------------------------------
    # Estadísticas antes de consolidar
    # --------------------------------------------------

    rows_before = len(df)

    duplicated = df.duplicated(
        subset=key,
        keep=False,
    )

    duplicate_rows_before = int(duplicated.sum())

    duplicate_groups_before = int(
        df.loc[
            duplicated,
            key,
        ]
        .drop_duplicates()
        .shape[0]
    )

    # --------------------------------------------------
    # Consolidar
    #
    # max() ignora NaN.
    # Si todas las notas del grupo son NaN,
    # el resultado permanece NaN.
    #
    # min_count=1 evita convertir un grupo
    # completamente vacío en cero.
    # --------------------------------------------------

    df = df.groupby(
        key,
        as_index=False,
        dropna=False,
    ).agg(
        {
            "Nota Curso": "max",
            "n_asistencias": sum_attendance,
        }
    )

    # --------------------------------------------------
    # Reconstruir asistio
    # --------------------------------------------------

    df["asistio"] = df["n_asistencias"] > 0

    # --------------------------------------------------
    # Orden estable para facilitar inspección
    # y reproducibilidad
    # --------------------------------------------------

    df = df.sort_values(
        by=[
            "Semestre",
            "Curso",
            "student_id",
        ]
    ).reset_index(drop=True)

    rows_after = len(df)

    report = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "duplicate_rows_detected": duplicate_rows_before,
        "duplicate_groups_detected": duplicate_groups_before,
        "duplicate_rows_removed": rows_before - rows_after,
    }

    return df, report


# ============================================================
# Comprobaciones posteriores
# ============================================================


def validate_prepared_dataset(
    df: pd.DataFrame,
) -> None:
    """
    Comprueba las propiedades que debe satisfacer
    el dataset preparado.
    """

    key = [
        "student_id",
        "Curso",
        "Semestre",
    ]

    # --------------------------------------------------
    # Ya no pueden existir duplicados
    # --------------------------------------------------

    duplicated = df.duplicated(
        subset=key,
        keep=False,
    )

    if duplicated.any():
        raise RuntimeError(
            "La preparación terminó con registros "
            "duplicados para "
            "student_id + Curso + Semestre."
        )

    # --------------------------------------------------
    # Las notas existentes deben estar en rango
    # --------------------------------------------------

    invalid_grade = df["Nota Curso"].notna() & (
        (df["Nota Curso"] < 0) | (df["Nota Curso"] > 5)
    )

    if invalid_grade.any():
        raise RuntimeError("La preparación produjo notas fuera del rango 0-5.")

    # --------------------------------------------------
    # Coherencia de asistencia
    # --------------------------------------------------

    expected_attendance = df["n_asistencias"] > 0

    if not (df["asistio"] == expected_attendance).all():
        raise RuntimeError(
            "La preparación produjo inconsistencias entre asistio y n_asistencias."
        )


# ============================================================
# Preparación completa
# ============================================================


def prepare_dataset(
    dataset_id: str,
) -> Path:
    """
    Prepara un dataset previamente validado.
    """

    manifest_path, manifest = load_manifest(dataset_id)

    # --------------------------------------------------
    # Solo procesamos datasets validados
    # --------------------------------------------------

    status = manifest.get("status")

    if status not in {"validated", "prepared"}:
        raise ValueError(
            "El dataset debe haber sido validado "
            "antes de prepararse. "
            f"Status actual: {status}"
        )

    # --------------------------------------------------
    # Localizar dataset validado
    # --------------------------------------------------

    validated_filename = manifest.get("validated_file")

    if not validated_filename:
        raise ValueError("El manifiesto no contiene 'validated_file'.")

    input_file = VALIDATED_DIR / validated_filename

    if not input_file.exists():
        raise FileNotFoundError(f"No existe el dataset validado: {input_file}")

    # --------------------------------------------------
    # Verificar que el dataset validado sigue siendo
    # exactamente el archivo que fue ingerido.
    #
    # validate.py copia el archivo sin modificarlo.
    # --------------------------------------------------

    current_hash = sha256_file(input_file)

    expected_hash = manifest["sha256"]

    if current_hash != expected_hash:
        raise ValueError(
            "El dataset validado fue modificado "
            "después de la validación. "
            "El SHA-256 no coincide."
        )

    # --------------------------------------------------
    # Cargar
    # --------------------------------------------------

    df = pd.read_csv(input_file)

    # --------------------------------------------------
    # Preparar
    # --------------------------------------------------

    rows_received = len(df)

    df = normalize_grades(df)

    df = normalize_types(df)

    df, duplicate_report = resolve_duplicate_records(df)

    validate_prepared_dataset(df)

    # --------------------------------------------------
    # Estadísticas finales
    # --------------------------------------------------

    rows_with_grade = int(df["Nota Curso"].notna().sum())

    rows_without_grade = int(df["Nota Curso"].isna().sum())

    students_with_attendance = int(
        df.loc[
            df["asistio"],
            "student_id",
        ].nunique()
    )

    students_without_attendance = int(
        df.loc[
            ~df["asistio"],
            "student_id",
        ].nunique()
    )

    # --------------------------------------------------
    # Guardar
    # --------------------------------------------------

    PREPARED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = PREPARED_DIR / (f"academic_performance_prepared_{dataset_id}.csv")

    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
    )

    output_hash = sha256_file(output_file)

    # --------------------------------------------------
    # Registrar preparación
    # --------------------------------------------------

    preparation = {
        "input": input_file.name,
        "output": output_file.name,
        "sha256": output_hash,
        "rows_received": rows_received,
        "rows_prepared": len(df),
        "students": int(df["student_id"].nunique()),
        "courses": int(df["Curso"].nunique()),
        "semesters": int(df["Semestre"].nunique()),
        "rows_with_grade": rows_with_grade,
        "rows_without_grade": rows_without_grade,
        "students_with_attendance": students_with_attendance,
        "students_without_attendance": students_without_attendance,
        "duplicates": duplicate_report,
    }

    manifest["preparation"] = preparation

    manifest["status"] = "prepared"

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
        description=("Prepara un dataset académico previamente validado.")
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help=("Identificador del dataset, por ejemplo ING-20260910-101728."),
    )

    args = parser.parse_args()

    try:
        print(f"\nPreparando dataset: {args.dataset}\n")

        output = prepare_dataset(args.dataset)

    # Borde del comando: cualquier fallo se informa como un error controlado
    # (mensaje claro + código de salida 1) en vez de una traza de Python.
    except Exception as exc:  # noqa: BLE001
        print("PREPARACIÓN FALLIDA\n")

        print(exc)

        raise SystemExit(1)

    print("PREPARACIÓN CORRECTA\n")

    print(f"Dataset preparado: {output}")


if __name__ == "__main__":
    main()
