import pandas as pd
import pytest

from academic_analysis.validate import (
    normalize_boolean,
    validate_attendance_consistency,
    validate_attendance_count,
    validate_attendance_flag,
    validate_columns,
    validate_grades,
    validate_nulls,
    validate_semester,
    validate_student_id,
)

SCHEMA = {
    "columns": {
        "student_id": {"nullable": False},
        "Curso": {"nullable": False},
        "Semestre": {"nullable": False},
        "Nota Curso": {"nullable": True},
        "n_asistencias": {"nullable": False},
        "asistio": {"nullable": False},
    }
}


def make_df(**overrides) -> pd.DataFrame:

    base = {
        "student_id": ["s1", "s2"],
        "Curso": ["Cálculo", "Cálculo"],
        "Semestre": ["2026-1", "2026-1"],
        "Nota Curso": [3.5, None],
        "n_asistencias": [2, 0],
        "asistio": [True, False],
    }

    base.update(overrides)

    return pd.DataFrame(base)


class TestValidateColumns:
    def test_passes_with_exact_columns(self):
        validate_columns(make_df(), SCHEMA)

    def test_raises_when_column_missing(self):

        df = make_df().drop(columns=["asistio"])

        with pytest.raises(ValueError, match="Faltan"):
            validate_columns(df, SCHEMA)

    def test_raises_when_extra_column(self):

        df = make_df()
        df["columna_sobrante"] = 1

        with pytest.raises(ValueError, match="no esperadas"):
            validate_columns(df, SCHEMA)


class TestValidateNulls:
    def test_passes_when_required_fields_present(self):
        validate_nulls(make_df(), SCHEMA)

    def test_allows_null_in_nullable_column(self):
        # Nota Curso es nullable=True: no debe fallar.
        validate_nulls(make_df(), SCHEMA)

    def test_raises_when_required_field_is_null(self):

        df = make_df(student_id=["s1", None])

        with pytest.raises(ValueError, match="student_id"):
            validate_nulls(df, SCHEMA)

    def test_raises_when_required_field_is_blank_string(self):

        df = make_df(Curso=["Cálculo", "   "])

        with pytest.raises(ValueError, match="Curso"):
            validate_nulls(df, SCHEMA)


class TestValidateStudentId:
    def test_passes_with_valid_ids(self):
        validate_student_id(make_df())

    def test_raises_on_blank_id(self):

        df = make_df(student_id=["s1", "  "])

        with pytest.raises(ValueError):
            validate_student_id(df)


class TestValidateSemester:
    @pytest.mark.parametrize(
        "value",
        ["2026-1", "2026-2", "1999-1"],
    )
    def test_accepts_valid_formats(self, value):

        df = make_df(Semestre=[value, value])
        validate_semester(df)

    @pytest.mark.parametrize(
        "value",
        ["2026-3", "26-1", "2026", "2026/1", ""],
    )
    def test_rejects_invalid_formats(self, value):

        df = make_df(Semestre=[value, value])

        with pytest.raises(ValueError):
            validate_semester(df)


class TestValidateGrades:
    @pytest.mark.parametrize(
        "value",
        ["", "-", "N/A", "NA", "n/a", None],
    )
    def test_accepts_missing_value_representations(self, value):

        df = make_df(**{"Nota Curso": [value, value]})
        validate_grades(df)

    def test_accepts_values_in_range(self):

        df = make_df(**{"Nota Curso": [0.0, 5.0]})
        validate_grades(df)

    @pytest.mark.parametrize("value", [-0.5, 5.1, 10])
    def test_rejects_out_of_range_values(self, value):

        df = make_df(**{"Nota Curso": [value, 3.0]})

        with pytest.raises(ValueError, match="0.0 y 5.0"):
            validate_grades(df)

    def test_rejects_non_numeric_text(self):

        df = make_df(**{"Nota Curso": ["excelente", 3.0]})

        with pytest.raises(ValueError, match="no numéricos"):
            validate_grades(df)


class TestValidateAttendanceCount:
    def test_accepts_non_negative_integers(self):

        df = make_df(n_asistencias=[0, 5])
        validate_attendance_count(df)

    def test_rejects_negative_values(self):

        df = make_df(n_asistencias=[-1, 5])

        with pytest.raises(ValueError, match="negativo"):
            validate_attendance_count(df)

    def test_rejects_non_integer_values(self):

        df = make_df(n_asistencias=[1.5, 2])

        with pytest.raises(ValueError, match="enteros"):
            validate_attendance_count(df)


class TestNormalizeBoolean:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (True, True),
            (False, False),
            ("true", True),
            ("True", True),
            ("1", True),
            ("false", False),
            ("0", False),
        ],
    )
    def test_recognizes_common_representations(self, value, expected):
        assert normalize_boolean(value) is expected

    def test_returns_none_for_unrecognized_value(self):
        assert normalize_boolean("tal vez") is None


class TestValidateAttendanceFlag:
    def test_passes_with_valid_booleans(self):
        validate_attendance_flag(make_df())

    def test_raises_on_unrecognized_value(self):

        df = make_df(asistio=[True, "quizas"])

        with pytest.raises(ValueError, match="no booleanos"):
            validate_attendance_flag(df)


class TestValidateAttendanceConsistency:
    def test_passes_when_consistent(self):

        df = make_df(
            n_asistencias=[2, 0],
            asistio=[True, False],
        )

        validate_attendance_consistency(df)

    def test_raises_when_inconsistent(self):

        df = make_df(
            n_asistencias=[2, 0],
            # Asistió 2 veces pero queda marcado como False.
            asistio=[False, False],
        )

        with pytest.raises(ValueError, match="Inconsistencia"):
            validate_attendance_consistency(df)
