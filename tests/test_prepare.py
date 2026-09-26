import pandas as pd
import pytest

from academic_analysis.prepare import (
    normalize_grades,
    normalize_types,
    resolve_duplicate_records,
)


class TestNormalizeGrades:
    @pytest.mark.parametrize(
        "value",
        ["", "-", "N/A", "NA", None],
    )
    def test_converts_missing_representations_to_na(self, value):

        df = pd.DataFrame({"Nota Curso": [value]})

        result = normalize_grades(df)

        assert pd.isna(result["Nota Curso"].iloc[0])

    def test_keeps_valid_numeric_grade(self):

        df = pd.DataFrame({"Nota Curso": ["3.75"]})

        result = normalize_grades(df)

        assert result["Nota Curso"].iloc[0] == pytest.approx(3.75)

    def test_result_column_is_numeric_dtype(self):

        df = pd.DataFrame({"Nota Curso": ["3.5", "-", "4.0"]})

        result = normalize_grades(df)

        assert pd.api.types.is_numeric_dtype(result["Nota Curso"])

    def test_does_not_mutate_input(self):

        df = pd.DataFrame({"Nota Curso": ["3.5"]})

        normalize_grades(df)

        # El valor original sigue siendo texto: la función
        # debe trabajar sobre una copia.
        assert df["Nota Curso"].iloc[0] == "3.5"


class TestNormalizeTypes:
    def test_reconstructs_asistio_from_attendance_count(self):

        df = pd.DataFrame(
            {
                "student_id": [1, 2],
                "Curso": ["Cálculo", "Cálculo"],
                "Semestre": ["2026-1", "2026-1"],
                "n_asistencias": ["0", "3"],
                "asistio": [True, False],
            }
        )

        result = normalize_types(df)

        assert result["asistio"].tolist() == [False, True]

    def test_casts_ids_to_stripped_strings(self):

        df = pd.DataFrame(
            {
                "student_id": [" s1 "],
                "Curso": [" Cálculo "],
                "Semestre": [" 2026-1 "],
                "n_asistencias": [1],
                "asistio": [True],
            }
        )

        result = normalize_types(df)

        assert result["student_id"].iloc[0] == "s1"
        assert result["Curso"].iloc[0] == "Cálculo"
        assert result["Semestre"].iloc[0] == "2026-1"


class TestResolveDuplicateRecords:
    def test_keeps_highest_grade_among_duplicates(self):

        df = pd.DataFrame(
            {
                "student_id": ["s1", "s1"],
                "Curso": ["Cálculo", "Cálculo"],
                "Semestre": ["2026-1", "2026-1"],
                "Nota Curso": [3.0, 4.5],
                "n_asistencias": [1, 1],
            }
        )

        result, _ = resolve_duplicate_records(df)

        assert len(result) == 1
        assert result["Nota Curso"].iloc[0] == 4.5

    def test_sums_attendance_across_duplicates(self):

        df = pd.DataFrame(
            {
                "student_id": ["s1", "s1"],
                "Curso": ["Cálculo", "Cálculo"],
                "Semestre": ["2026-1", "2026-1"],
                "Nota Curso": [3.0, None],
                "n_asistencias": [2, 3],
            }
        )

        result, _ = resolve_duplicate_records(df)

        assert result["n_asistencias"].iloc[0] == 5
        assert result["asistio"].iloc[0] == True

    def test_all_missing_grades_stay_missing(self):

        df = pd.DataFrame(
            {
                "student_id": ["s1", "s1"],
                "Curso": ["Cálculo", "Cálculo"],
                "Semestre": ["2026-1", "2026-1"],
                "Nota Curso": [None, None],
                "n_asistencias": [0, 0],
            }
        )

        result, _ = resolve_duplicate_records(df)

        assert pd.isna(result["Nota Curso"].iloc[0])

    def test_non_duplicate_rows_are_left_alone(self):

        df = pd.DataFrame(
            {
                "student_id": ["s1", "s2"],
                "Curso": ["Cálculo", "Cálculo"],
                "Semestre": ["2026-1", "2026-1"],
                "Nota Curso": [3.0, 4.0],
                "n_asistencias": [1, 2],
            }
        )

        result, report = resolve_duplicate_records(df)

        assert len(result) == 2
        assert report["duplicate_rows_removed"] == 0

    def test_report_counts_match_expectations(self):

        df = pd.DataFrame(
            {
                "student_id": ["s1", "s1", "s2"],
                "Curso": ["Cálculo", "Cálculo", "Cálculo"],
                "Semestre": ["2026-1", "2026-1", "2026-1"],
                "Nota Curso": [3.0, 4.0, 2.0],
                "n_asistencias": [1, 1, 1],
            }
        )

        _, report = resolve_duplicate_records(df)

        assert report["rows_before"] == 3
        assert report["rows_after"] == 2
        assert report["duplicate_rows_detected"] == 2
        assert report["duplicate_groups_detected"] == 1
        assert report["duplicate_rows_removed"] == 1
