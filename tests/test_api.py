import pytest
from fastapi import HTTPException

from academic_analysis import api


@pytest.fixture
def loaded_state():
    """
    Simula el estado que deja load_deployment_bundle(), sin
    necesitar un deploy/model.joblib real en disco. Las mismas
    columnas que produciría encode_features() para un dataset
    con dos cursos y dos semestres.
    """

    api._state.clear()

    api._state["feature_columns"] = [
        "n_asistencias",
        "Curso_Física",
        "Semestre_2026-2",
    ]

    api._state["categories"] = {
        "Curso": ["Cálculo", "Física"],
        "Semestre": ["2026-1", "2026-2"],
    }

    yield api._state

    api._state.clear()


class TestBuildModelInput:

    def test_known_reference_category_becomes_all_zero_dummies(
        self, loaded_state
    ):

        # "Cálculo" y "2026-1" fueron la categoría de
        # referencia (drop_first) en entrenamiento: no
        # generan columna dummy, así que deben quedar en 0.
        request = api.PredictionRequest(
            n_asistencias=5,
            Curso="Cálculo",
            Semestre="2026-1",
        )

        model_input = api._build_model_input(request)

        assert model_input["Curso_Física"].iloc[0] == 0
        assert model_input["Semestre_2026-2"].iloc[0] == 0
        assert model_input["n_asistencias"].iloc[0] == 5

    def test_non_reference_category_activates_its_dummy(
        self, loaded_state
    ):

        request = api.PredictionRequest(
            n_asistencias=2,
            Curso="Física",
            Semestre="2026-2",
        )

        model_input = api._build_model_input(request)

        assert model_input["Curso_Física"].iloc[0] == 1
        assert model_input["Semestre_2026-2"].iloc[0] == 1

    def test_output_columns_match_feature_columns_order(
        self, loaded_state
    ):

        request = api.PredictionRequest(
            n_asistencias=1,
            Curso="Física",
            Semestre="2026-1",
        )

        model_input = api._build_model_input(request)

        assert list(model_input.columns) == [
            "n_asistencias",
            "Curso_Física",
            "Semestre_2026-2",
        ]


class TestValidateCategory:

    def test_accepts_known_value(self, loaded_state):
        api._validate_category("Curso", "Física")

    def test_rejects_unknown_value(self, loaded_state):

        with pytest.raises(HTTPException) as exc_info:
            api._validate_category("Curso", "Curso Inventado")

        assert exc_info.value.status_code == 422
