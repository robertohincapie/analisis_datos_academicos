import hashlib

import pytest

from academic_analysis.ingest import extract_dataset_id, sha256_file


class TestExtractDatasetId:

    def test_extracts_from_expected_filename(self):

        result = extract_dataset_id(
            "academic_performance_ING-20260910-101728.csv"
        )

        assert result == "ING-20260910-101728"

    def test_extracts_even_with_extra_text_around(self):

        result = extract_dataset_id(
            "copia_final_ING-20260910-101728_v2.csv"
        )

        assert result == "ING-20260910-101728"

    def test_raises_when_pattern_missing(self):

        with pytest.raises(ValueError):
            extract_dataset_id("academic_performance.csv")

    def test_raises_on_malformed_dataset_id(self):

        with pytest.raises(ValueError):
            extract_dataset_id("academic_performance_ING-2026-101728.csv")


class TestSha256File:

    def test_matches_hashlib_reference(self, tmp_path):

        file_path = tmp_path / "sample.txt"
        content = "contenido de prueba para sha256" * 100
        file_path.write_text(content, encoding="utf-8")

        expected = hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()

        assert sha256_file(file_path) == expected

    def test_same_content_same_hash(self, tmp_path):

        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"

        a.write_text("mismo contenido", encoding="utf-8")
        b.write_text("mismo contenido", encoding="utf-8")

        assert sha256_file(a) == sha256_file(b)

    def test_different_content_different_hash(self, tmp_path):

        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"

        a.write_text("contenido A", encoding="utf-8")
        b.write_text("contenido B", encoding="utf-8")

        assert sha256_file(a) != sha256_file(b)
