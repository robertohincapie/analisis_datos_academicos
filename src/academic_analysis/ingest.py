from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

INCOMING_DIR = ROOT / "data" / "incoming"
METADATA_DIR = ROOT / "metadata"


# ============================================================
# Utilidades
# ============================================================

def sha256_file(path: Path) -> str:
    """Calcula el SHA-256 de un archivo."""

    sha256 = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            sha256.update(block)

    return sha256.hexdigest()


def extract_dataset_id(filename: str) -> str:
    """
    Extrae el identificador de la ingesta desde el nombre.

    Ejemplo:

    academic_performance_ING-20260910-101728.csv

    -> ING-20260910-101728
    """

    match = re.search(
        r"(ING-\d{8}-\d{6})",
        filename,
    )

    if not match:
        raise ValueError(
            "No se pudo obtener el dataset_id "
            "a partir del nombre del archivo.\n"
            "Se esperaba un nombre como:\n"
            "academic_performance_"
            "ING-20260910-101728.csv"
        )

    return match.group(1)


# ============================================================
# Ingesta
# ============================================================

def ingest_file(file_path: Path) -> Path:
    """
    Registra formalmente un archivo ubicado en data/incoming.
    """

    file_path = file_path.resolve()
    incoming_dir = INCOMING_DIR.resolve()

    # --------------------------------------------------
    # Verificar existencia
    # --------------------------------------------------

    if not file_path.exists():
        raise FileNotFoundError(
            f"No existe el archivo: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"La ruta no corresponde a un archivo: {file_path}"
        )

    # --------------------------------------------------
    # El archivo debe estar en data/incoming
    # --------------------------------------------------

    if file_path.parent != incoming_dir:
        raise ValueError(
            "El archivo debe encontrarse directamente "
            "en data/incoming/."
        )

    # --------------------------------------------------
    # Solo aceptamos CSV
    # --------------------------------------------------

    if file_path.suffix.lower() != ".csv":
        raise ValueError(
            "El archivo de entrada debe ser CSV."
        )

    # --------------------------------------------------
    # Obtener dataset_id
    # --------------------------------------------------

    dataset_id = extract_dataset_id(
        file_path.name
    )

    # --------------------------------------------------
    # Evitar registrar dos veces el mismo dataset
    # --------------------------------------------------

    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        METADATA_DIR
        / f"{dataset_id}.json"
    )

    if manifest_path.exists():
        raise FileExistsError(
            f"El dataset {dataset_id} ya fue ingerido.\n"
            f"Existe el manifiesto: {manifest_path}"
        )

    # --------------------------------------------------
    # Integridad
    # --------------------------------------------------

    file_hash = sha256_file(
        file_path
    )

    file_size = (
        file_path.stat().st_size
    )

    # --------------------------------------------------
    # Crear manifiesto
    # --------------------------------------------------

    manifest = {
        "dataset_id":
            dataset_id,

        "filename":
            file_path.name,

        "sha256":
            file_hash,

        "size_bytes":
            file_size,

        "ingested_at":
            datetime.now()
            .astimezone()
            .isoformat(),

        "status":
            "received",
    }

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

    return manifest_path


# ============================================================
# CLI
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Registra una nueva versión "
            "del dataset."
        )
    )

    parser.add_argument(
        "--file",
        required=True,
        help=(
            "Archivo CSV ubicado en "
            "data/incoming/."
        ),
    )

    args = parser.parse_args()

    try:

        manifest_path = ingest_file(
            Path(args.file)
        )

    except Exception as exc:

        print(
            "\nINGESTA FALLIDA\n"
        )

        print(exc)

        raise SystemExit(1)

    print(
        "\nINGESTA CORRECTA\n"
    )

    print(
        f"Manifiesto: {manifest_path}"
    )


if __name__ == "__main__":
    main()