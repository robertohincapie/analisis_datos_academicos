import pandas as pd

from academic_analysis.train import (
    build_classification_frame,
    build_training_frame,
    encode_features,
)


def make_df() -> pd.DataFrame:

    return pd.DataFrame(
        {
            "student_id": ["s1", "s2", "s3", "s4"],
            "Curso": [
                "Cálculo",
                "Cálculo",
                "Física",
                "Física",
            ],
            "Semestre": [
                "2026-1",
                "2026-1",
                "2026-2",
                "2026-2",
            ],
            "Nota Curso": [4.0, None, 2.0, 3.5],
            "n_asistencias": [5, 0, 1, 3],
        }
    )


class TestEncodeFeatures:

    def test_drops_first_category_per_column(self):

        df = make_df()

        encoded = encode_features(df)

        # drop_first=True: una de las dos categorías de
        # Curso y una de Semestre no deberían aparecer
        # como columna dummy.
        curso_columns = [
            c for c in encoded.columns if c.startswith("Curso_")
        ]
        semestre_columns = [
            c
            for c in encoded.columns
            if c.startswith("Semestre_")
        ]

        assert len(curso_columns) == 1
        assert len(semestre_columns) == 1

    def test_keeps_numeric_column_unchanged(self):

        df = make_df()

        encoded = encode_features(df)

        assert encoded["n_asistencias"].tolist() == [
            5,
            0,
            1,
            3,
        ]


class TestBuildTrainingFrame:

    def test_drops_rows_without_grade(self):

        df = make_df()

        x, y = build_training_frame(df)

        # Solo una fila (s2) tiene Nota Curso nula.
        assert len(x) == 3
        assert len(y) == 3
        assert y.isna().sum() == 0

    def test_target_matches_source_grades(self):

        df = make_df()

        _, y = build_training_frame(df)

        assert sorted(y.tolist()) == [2.0, 3.5, 4.0]


class TestBuildClassificationFrame:

    def test_applies_default_passing_grade(self):

        df = make_df()

        _, y = build_classification_frame(df)

        # Con umbral 3.0: 4.0 -> aprueba, 2.0 -> no,
        # 3.5 -> aprueba. La fila sin nota (s2) se excluye.
        assert sorted(y.tolist()) == [0, 1, 1]

    def test_respects_custom_passing_grade(self):

        df = make_df()

        _, y = build_classification_frame(
            df,
            passing_grade=4.0,
        )

        # Con umbral 4.0: solo la fila con 4.0 aprueba.
        assert sorted(y.tolist()) == [0, 0, 1]

    def test_target_is_binary_integer(self):

        df = make_df()

        _, y = build_classification_frame(df)

        assert set(y.unique()).issubset({0, 1})
