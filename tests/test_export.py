"""Tests for export module."""

import csv
import json
from io import StringIO
from pathlib import Path

import pytest

from noid_rag.export import _detect_format, export, format_data
from noid_rag.models import Chunk, Document


class TestDetectFormat:
    def test_detect_json(self):
        assert _detect_format(Path("out.json")) == "json"

    def test_detect_jsonl(self):
        assert _detect_format(Path("out.jsonl")) == "jsonl"

    def test_detect_csv(self):
        assert _detect_format(Path("out.csv")) == "csv"

    def test_detect_unknown_raises_valueerror(self):
        with pytest.raises(ValueError, match="Cannot detect format"):
            _detect_format(Path("out.xyz"))

    def test_detect_case_insensitive(self):
        assert _detect_format(Path("OUT.JSON")) == "json"


class TestFormatJson:
    def test_json_output(self, sample_chunks):
        result = format_data(sample_chunks, "json")
        parsed = json.loads(result)
        assert len(parsed) == 2
        assert parsed[0]["text"] == "First section text."
        assert parsed[1]["text"] == "Second section text."

    def test_json_is_indented(self, sample_chunks):
        result = format_data(sample_chunks, "json")
        # Indented JSON should contain newlines
        assert "\n" in result


class TestFormatJsonl:
    def test_jsonl_output(self, sample_chunks):
        result = format_data(sample_chunks, "jsonl")
        lines = [line for line in result.strip().split("\n") if line]
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["text"] == "First section text."

    def test_jsonl_ends_with_newline(self, sample_chunks):
        result = format_data(sample_chunks, "jsonl")
        assert result.endswith("\n")


class TestFormatCsv:
    def test_csv_output(self, sample_chunks):
        result = format_data(sample_chunks, "csv")
        lines = result.strip().split("\n")
        assert len(lines) == 3  # header + 2 rows

    def test_csv_flattens_metadata(self):
        chunk = Chunk(
            text="test",
            document_id="doc_1",
            metadata={"source_type": "pdf", "page": 3},
        )
        result = format_data([chunk], "csv")
        reader = csv.DictReader(StringIO(result))
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["metadata.source_type"] == "pdf"
        assert rows[0]["metadata.page"] == "3"

    def test_csv_empty_records(self):
        result = format_data([], "csv")
        assert result == ""


class TestFormatUnsupported:
    def test_unsupported_format_raises(self):
        with pytest.raises(ValueError, match="Unsupported format"):
            format_data([], "xml")


class TestExportToFile:
    def test_export_json_file(self, sample_chunks, tmp_path):
        out = tmp_path / "output.json"
        result_path = export(sample_chunks, out)
        assert out.exists()
        assert result_path == str(out)
        data = json.loads(out.read_text())
        assert len(data) == 2

    def test_export_jsonl_file(self, sample_chunks, tmp_path):
        out = tmp_path / "output.jsonl"
        export(sample_chunks, out)
        assert out.exists()
        lines = [line for line in out.read_text().strip().split("\n") if line]
        assert len(lines) == 2

    def test_export_csv_file(self, sample_chunks, tmp_path):
        out = tmp_path / "output.csv"
        export(sample_chunks, out)
        assert out.exists()

    def test_export_auto_detects_format(self, sample_chunks, tmp_path):
        out = tmp_path / "output.json"
        export(sample_chunks, out)
        # Should auto-detect json from .json extension
        data = json.loads(out.read_text())
        assert isinstance(data, list)

    def test_export_creates_parent_dirs(self, sample_chunks, tmp_path):
        out = tmp_path / "nested" / "dir" / "output.json"
        export(sample_chunks, out)
        assert out.exists()

    def test_export_explicit_format_overrides_extension(self, sample_chunks, tmp_path):
        out = tmp_path / "output.txt"
        export(sample_chunks, out, fmt="json")
        data = json.loads(out.read_text())
        assert len(data) == 2


class TestExportWithDifferentTypes:
    def test_export_document_objects(self):
        doc = Document(source="test.pdf", content="Hello", metadata={"k": "v"})
        result = format_data([doc], "json")
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["content"] == "Hello"
        assert "_docling_doc" not in parsed[0]

    def test_export_chunk_objects(self, sample_chunks):
        result = format_data(sample_chunks, "json")
        parsed = json.loads(result)
        assert parsed[0]["document_id"] == "doc_test123456"

    def test_export_search_result_objects(self, sample_search_results):
        result = format_data(sample_search_results, "json")
        parsed = json.loads(result)
        assert len(parsed) == 2
        assert parsed[0]["score"] == 0.95
        assert parsed[0]["chunk_id"] == "chk_aaa111"

    def test_export_strips_embedding(self):
        chunk = Chunk(text="test", document_id="doc_1", embedding=[0.1, 0.2])
        result = format_data([chunk], "json")
        parsed = json.loads(result)
        assert "embedding" not in parsed[0]

    def test_export_strips_docling_doc(self):
        doc = Document(source="f.txt", content="c", _docling_doc=object())
        result = format_data([doc], "json")
        parsed = json.loads(result)
        assert "_docling_doc" not in parsed[0]
