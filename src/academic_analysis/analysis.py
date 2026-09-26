from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import ttest_ind

ROOT = Path(__file__).resolve().parents[2]

PREPARED_DIR = ROOT / "data" / "prepared"
RESULTS_DIR = ROOT / "results"
METADATA_DIR = ROOT / "metadata"


# ============================================================
# Utilidades
# ============================================================


def create_analysis_plot(
    results_df: pd.DataFrame,
    output_file: Path,
    alpha: float,
) -> None:
    """
    Genera una gráfica con la nota promedio por curso
    para quienes asisten y quienes no asisten.

    El fondo de cada curso indica el resultado
    de la prueba de hipótesis:

        verde  -> se rechaza H0
        rosado -> no se rechaza H0
    """

    plot_df = results_df.copy()

    # --------------------------------------------------
    # Ordenar alfabéticamente para facilitar lectura
    # --------------------------------------------------

    plot_df = plot_df.sort_values("Curso").reset_index(drop=True)

    n_courses = len(plot_df)

    if n_courses == 0:
        return

    # --------------------------------------------------
    # Posiciones
    # --------------------------------------------------

    x = np.arange(n_courses)

    width = 0.36

    # --------------------------------------------------
    # Tamaño dinámico
    # --------------------------------------------------

    figure_width = max(
        12,
        n_courses * 1.5,
    )

    fig, ax = plt.subplots(figsize=(figure_width, 7))

    # --------------------------------------------------
    # Fondos según prueba de hipótesis
    # --------------------------------------------------

    for i, row in plot_df.iterrows():
        significant = row["significativo"]

        if pd.isna(significant):
            color = "lightgray"

        elif bool(significant):
            color = "lightgreen"

        else:
            color = "mistyrose"

        ax.axvspan(
            i - 0.48,
            i + 0.48,
            color=color,
            alpha=0.35,
            zorder=0,
        )

    # --------------------------------------------------
    # Barras
    # --------------------------------------------------

    bars_attended = ax.bar(
        x - width / 2,
        plot_df["media_asisten"],
        width,
        label="Asisten",
        zorder=3,
    )

    bars_not_attended = ax.bar(
        x + width / 2,
        plot_df["media_no_asisten"],
        width,
        label="No asisten",
        zorder=3,
    )

    # --------------------------------------------------
    # Valores sobre las barras
    # --------------------------------------------------

    def add_labels(bars):

        for bar in bars:
            height = bar.get_height()

            if np.isnan(height):
                continue

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.04,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    add_labels(bars_attended)

    add_labels(bars_not_attended)

    # --------------------------------------------------
    # p-value por curso
    # --------------------------------------------------

    for i, row in plot_df.iterrows():
        p_value = row["p_value"]

        if pd.isna(p_value):
            text = "p = N/A"

        elif p_value < 0.001:
            text = "p < 0.001"

        else:
            text = f"p = {p_value:.3f}"

        ax.text(
            i,
            5.30,
            text,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    # --------------------------------------------------
    # Ejes
    # --------------------------------------------------

    ax.set_ylim(
        0,
        5.55,
    )

    ax.set_ylabel("Nota promedio")

    ax.set_xlabel("Curso")

    ax.set_title("Desempeño promedio según asistencia a actividades académicas")

    ax.set_xticks(x)

    ax.set_xticklabels(
        plot_df["Curso"],
        rotation=45,
        ha="right",
    )

    ax.grid(
        axis="y",
        alpha=0.25,
        zorder=1,
    )

    # --------------------------------------------------
    # Leyenda de barras
    # --------------------------------------------------

    bar_legend = ax.legend(
        handles=[
            bars_attended,
            bars_not_attended,
        ],
        labels=[
            "Asisten",
            "No asisten",
        ],
        loc="upper left",
    )

    ax.add_artist(bar_legend)

    # --------------------------------------------------
    # Leyenda de hipótesis
    # --------------------------------------------------

    hypothesis_legend = [
        Patch(
            facecolor="lightgreen",
            alpha=0.35,
            label=f"Se rechaza H₀ (p < {alpha})",
        ),
        Patch(
            facecolor="mistyrose",
            alpha=0.35,
            label=f"No se rechaza H₀ (p ≥ {alpha})",
        ),
        Patch(
            facecolor="lightgray",
            alpha=0.35,
            label="Prueba no disponible",
        ),
    ]

    ax.legend(
        handles=hypothesis_legend,
        loc="upper right",
    )

    # --------------------------------------------------
    # Nota metodológica
    # --------------------------------------------------

    fig.text(
        0.5,
        0.01,
        ("Prueba t de Welch bilateral. H₀: las medias de ambos grupos son iguales."),
        ha="center",
        fontsize=9,
    )

    # --------------------------------------------------
    # Guardar
    # --------------------------------------------------

    fig.tight_layout(
        rect=[
            0,
            0.05,
            1,
            1,
        ]
    )

    fig.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


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
# Análisis por curso
# ============================================================


def analyze_course(
    course_df: pd.DataFrame,
    course_name: str,
    alpha: float = 0.05,
) -> dict:
    """
    Compara las notas de estudiantes que asistieron
    al menos una vez contra quienes no asistieron.

    Se utiliza una prueba t de Welch para muestras
    independientes.
    """

    # --------------------------------------------------
    # Conservar únicamente registros con nota
    # --------------------------------------------------

    data = course_df[course_df["Nota Curso"].notna()].copy()

    # --------------------------------------------------
    # Separar los grupos
    # --------------------------------------------------

    attended = data.loc[
        data["asistio"],
        "Nota Curso",
    ].astype(float)

    not_attended = data.loc[
        ~data["asistio"],
        "Nota Curso",
    ].astype(float)

    n_attended = len(attended)
    n_not_attended = len(not_attended)

    mean_attended = attended.mean() if n_attended > 0 else None

    mean_not_attended = not_attended.mean() if n_not_attended > 0 else None

    difference = (
        mean_attended - mean_not_attended
        if (mean_attended is not None and mean_not_attended is not None)
        else None
    )

    # --------------------------------------------------
    # Necesitamos al menos dos observaciones
    # en cada grupo para realizar la prueba.
    # --------------------------------------------------

    if n_attended < 2 or n_not_attended < 2:
        return {
            "Curso": course_name,
            "n_con_nota": len(data),
            "n_asisten": n_attended,
            "n_no_asisten": n_not_attended,
            "media_asisten": mean_attended,
            "media_no_asisten": mean_not_attended,
            "diferencia_medias": difference,
            "t_statistic": None,
            "p_value": None,
            "alpha": alpha,
            "significativo": None,
            "resultado": "Muestras insuficientes para realizar la prueba.",
        }

    # --------------------------------------------------
    # Prueba t de Welch
    #
    # H0: mu_asisten = mu_no_asisten
    # H1: mu_asisten != mu_no_asisten
    # --------------------------------------------------

    test = ttest_ind(
        attended,
        not_attended,
        equal_var=False,
        nan_policy="omit",
    )

    t_statistic = float(test.statistic)

    p_value = float(test.pvalue)

    significant = p_value < alpha

    # --------------------------------------------------
    # Interpretación
    # --------------------------------------------------

    if significant:
        result = (
            "Se rechaza H0. "
            "Existe evidencia estadística de una "
            "diferencia entre las medias de los grupos."
        )

    else:
        result = (
            "No se rechaza H0. "
            "No existe evidencia estadística suficiente "
            "para afirmar que las medias sean diferentes."
        )

    return {
        "Curso": course_name,
        "n_con_nota": len(data),
        "n_asisten": n_attended,
        "n_no_asisten": n_not_attended,
        "media_asisten": mean_attended,
        "media_no_asisten": mean_not_attended,
        "diferencia_medias": difference,
        "t_statistic": t_statistic,
        "p_value": p_value,
        "alpha": alpha,
        "significativo": significant,
        "resultado": result,
    }


# ============================================================
# Análisis completo
# ============================================================


def analyze_dataset(
    dataset_id: str,
    alpha: float = 0.05,
) -> tuple[Path, Path]:
    """
    Ejecuta el análisis para todos los cursos
    del dataset preparado.
    """

    manifest_path, manifest = load_manifest(dataset_id)

    # --------------------------------------------------
    # Estado requerido
    # --------------------------------------------------

    if manifest.get("status") not in {
        "prepared",
        "analyzed",
    }:
        raise ValueError(
            "El dataset debe estar preparado "
            "antes de analizarse. "
            f"Status actual: {manifest.get('status')}"
        )

    preparation = manifest.get("preparation")

    if not preparation:
        raise ValueError("El manifiesto no contiene información de preparation.")

    # --------------------------------------------------
    # Localizar dataset preparado
    # --------------------------------------------------

    input_file = PREPARED_DIR / preparation["output"]

    if not input_file.exists():
        raise FileNotFoundError(f"No existe el dataset preparado: {input_file}")

    # --------------------------------------------------
    # Comprobar integridad
    # --------------------------------------------------

    current_hash = sha256_file(input_file)

    expected_hash = preparation["sha256"]

    if current_hash != expected_hash:
        raise ValueError(
            "El dataset preparado fue modificado. "
            "El SHA-256 no coincide con el manifiesto."
        )

    # --------------------------------------------------
    # Cargar
    # --------------------------------------------------

    df = pd.read_csv(input_file)

    df["Nota Curso"] = pd.to_numeric(
        df["Nota Curso"],
        errors="coerce",
    )

    # CSV carga los booleanos normalmente como bool,
    # pero garantizamos la representación.

    if df["asistio"].dtype != bool:
        df["asistio"] = (
            df["asistio"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(
                {
                    "true": True,
                    "false": False,
                    "1": True,
                    "0": False,
                }
            )
        )

    if df["asistio"].isna().any():
        raise ValueError("Se encontraron valores inválidos en la columna asistio.")

    # --------------------------------------------------
    # Ejecutar análisis por curso
    # --------------------------------------------------

    results = []

    for course_name, course_df in df.groupby("Curso"):
        result = analyze_course(
            course_df,
            course_name,
            alpha,
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    # --------------------------------------------------
    # Ordenar por p-value
    #
    # Los resultados con p-value más pequeño
    # aparecen primero.
    # --------------------------------------------------

    results_df = results_df.sort_values(
        by="p_value",
        na_position="last",
    ).reset_index(drop=True)

    # --------------------------------------------------
    # Guardar resultados
    # --------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_output = RESULTS_DIR / f"analysis_{dataset_id}.csv"

    json_output = RESULTS_DIR / f"analysis_{dataset_id}.json"

    plot_output = RESULTS_DIR / f"analysis_{dataset_id}.png"
    results_df.to_csv(
        csv_output,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------
    # Resultado JSON con contexto del experimento
    # --------------------------------------------------

    analysis_document = {
        "dataset_id": dataset_id,
        "analyzed_at": datetime.now().astimezone().isoformat(),
        "method": "Welch independent two-sample t-test",
        "hypothesis": {
            "H0": "La nota promedio de quienes asisten "
            "es igual a la nota promedio de quienes "
            "no asisten.",
            "H1": "La nota promedio de quienes asisten "
            "es diferente de la nota promedio de "
            "quienes no asisten.",
        },
        "alpha": alpha,
        "courses": results,
    }

    with json_output.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            analysis_document,
            f,
            ensure_ascii=False,
            indent=2,
        )
    create_analysis_plot(
        results_df=results_df,
        output_file=plot_output,
        alpha=alpha,
    )
    # --------------------------------------------------
    # Actualizar manifiesto
    # --------------------------------------------------

    analysis = {
        "method": "Welch independent two-sample t-test",
        "alpha": alpha,
        "courses_analyzed": len(results_df),
        "courses_significant": int((results_df["significativo"] == True).sum()),
        "csv_output": csv_output.name,
        "json_output": json_output.name,
        "plot_output": plot_output.name,
        "csv_sha256": sha256_file(csv_output),
        "json_sha256": sha256_file(json_output),
        "plot_sha256": sha256_file(plot_output),
    }

    manifest["analysis"] = analysis

    manifest["status"] = "analyzed"

    save_manifest(
        manifest_path,
        manifest,
    )

    return (
        csv_output,
        json_output,
        plot_output,
    )


# ============================================================
# CLI
# ============================================================


def main() -> None:

    parser = argparse.ArgumentParser(
        description=("Analiza la relación entre asistencia y desempeño académico.")
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help=("Identificador del dataset, por ejemplo ING-20260910-101728."),
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help=("Nivel de significancia. Valor por defecto: 0.05."),
    )

    args = parser.parse_args()

    try:
        print(f"\nAnalizando dataset: {args.dataset}\n")

        csv_output, json_output, plot_output = analyze_dataset(
            dataset_id=args.dataset,
            alpha=args.alpha,
        )

    # Borde del comando: cualquier fallo se informa como un error controlado
    # (mensaje claro + código de salida 1) en vez de una traza de Python.
    except Exception as exc:  # noqa: BLE001
        print("ANÁLISIS FALLIDO\n")

        print(exc)

        raise SystemExit(1)

    print("ANÁLISIS CORRECTO\n")

    print(f"Resultados CSV : {csv_output}")

    print(f"Resultados JSON: {json_output}")
    print(f"Gráfica        : {plot_output}")


if __name__ == "__main__":
    main()
